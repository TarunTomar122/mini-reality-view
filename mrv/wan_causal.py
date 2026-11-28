"""PLAN-video S2 groundwork: block-causal attention + per-layer K/V cache for the Wan transformer (council/review-20261128-1430-
astra.md and -fable.md, the parts both seats agree on). NEW file: wraps diffusers' WanTransformer3DModel or mrv.wan_cond.WanCond
by swapping each block's SELF-attention processor (attn1); no frozen file is edited. Warm start, losses and the critic are NOT
here (undecided).

Block = ONE latent plane: plane 0 = tick 0, every later plane = 4 ticks (the Wan VAE's 1 + 4k layout); a plane at 512x256 is
16 x 32 = 512 spatial tokens (patch 1 x 2 x 2).
  - self-attention: full spatial attention inside a plane, plus at most `window` (4) preceding FINISHED planes of generated
    history (sliding window, oldest evicted first). The rotary time index is ABSOLUTE inside a drive (it keeps counting after an
    eviction) and restarts at 0 at a drive boundary / discontinuity, where everything resets.
  - camera conditioning stays restricted to the current group's ticks with no extra code: guide planes enter through the
    patch conv (time kernel 1, so per plane) and raw-view tokens through mrv.wan_cond.DualCrossAttn, which already attends per
    latent frame; text cross-attention is the same for every plane. Only self-attention mixes planes, and it is masked / cached
    here.
Modes (CausalWan):
  masked(...)  : one forward over several planes with the block-causal sliding-window mask (training windows, and the SLOW
                 full-prefix reference); per-plane absolute positions and segment ids (no attention across a reset).
  step(...)    : one denoising substep of the CURRENT plane: its K/V are scratch (used, never stored); history K/V come from
                 the cache.
  commit(...)  : after the final substep, the plane's final latent is re-encoded at `t_commit` (default 0 = clean; Fable's
                 small fixed noise level is the same call with another t) and its K/V are appended ONCE per layer, detached.
  reset()      : drive boundary / discontinuity: cache emptied, time index back to 0.
Cached decoding equals the slow masked full-prefix recomputation (same eviction semantics: each plane attends to its 4
predecessors, recursively) up to floating-point order; see scripts/wan_causal_tests.py.
Gradient checkpointing must be OFF while the cache is used (a recomputed block would read a later cache state).
"""
import torch
import torch.nn.functional as F

WINDOW = 4          # finished planes of history attended (Astra / Fable: four preceding planes)
TICKS_FIRST, TICKS_PER_PLANE = 1, 4


def plane_ticks(n_planes):
    """tick count of each plane of a drive: [1, 4, 4, ...] (first plane = tick 0)"""
    return [TICKS_FIRST] + [TICKS_PER_PLANE] * max(0, n_planes - 1)


def rotary_for(rope, positions, pph, ppw):
    """(cos, sin) [1, n*pph*ppw, 1, D] exactly as diffusers' WanRotaryPosEmbed.forward builds them, but for arbitrary ABSOLUTE
    time positions (one per plane) instead of 0..n-1"""
    pos = torch.as_tensor(positions, dtype=torch.long, device=rope.freqs_cos.device)
    if int(pos.max()) >= rope.max_seq_len or int(pos.min()) < 0:
        raise ValueError(f"time position {int(pos.max())} outside the rotary table ({rope.max_seq_len})")
    n = pos.numel()
    split = [rope.t_dim, rope.h_dim, rope.w_dim]
    out = []
    for tab in (rope.freqs_cos, rope.freqs_sin):
        f, h, w = tab.split(split, dim=1)
        ff = f[pos].view(n, 1, 1, -1).expand(n, pph, ppw, -1)
        hh = h[:pph].view(1, pph, 1, -1).expand(n, pph, ppw, -1)
        ww = w[:ppw].view(1, 1, ppw, -1).expand(n, pph, ppw, -1)
        out.append(torch.cat([ff, hh, ww], dim=-1).reshape(1, n * pph * ppw, 1, -1))
    return out[0], out[1]


def apply_rotary(x, cos, sin):
    """diffusers WanAttnProcessor's rotary arithmetic (interleaved pairs), unchanged"""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    c = cos[..., 0::2]
    s = sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out.type_as(x)


def block_mask(plane_of_token, seg_of_token, window=WINDOW):
    """bool [L, L]: query token may attend key token iff same segment and 0 <= plane_q - plane_k <= window"""
    d = plane_of_token[:, None] - plane_of_token[None, :]
    return (d >= 0) & (d <= window) & (seg_of_token[:, None] == seg_of_token[None, :])


class CausalState:
    """shared by the processors of one wrapped model"""

    def __init__(s, n_layers, window=WINDOW):
        s.n_layers, s.window = n_layers, window
        s.mode = None            # "masked" | "cache"
        s.pph = s.ppw = None
        s.positions = None       # masked: absolute position per plane
        s.segments = None        # masked: segment id per plane
        s.append = False         # cache: store this forward's K/V (commit)
        s.next_pos = 0           # cache: absolute time index of the current plane
        s.rot = None
        s.reset()

    def reset(s):
        s.k = [[] for _ in range(s.n_layers)]     # per layer: list of [B, P, H, D] (rotated), oldest first
        s.v = [[] for _ in range(s.n_layers)]
        s.pos = [[] for _ in range(s.n_layers)]   # per layer: absolute position of each cached plane
        s.appends = [0] * s.n_layers              # per layer: appends since the last reset
        s.next_pos = 0


class CausalSelfAttnProcessor:
    """replacement for WanAttnProcessor on a block's attn1 (self-attention); same projections / norms / rotary arithmetic"""

    def __init__(s, state, layer):
        s.state, s.layer = state, layer

    def __call__(s, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kw):
        st = s.state
        if st.mode is None:
            raise RuntimeError("CausalWan: call through masked() / step() / commit(), not the bare model")
        if getattr(attn, "fused_projections", False):
            q, k, v = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        q, k = attn.norm_q(q), attn.norm_k(k)
        q, k, v = (t.unflatten(2, (attn.heads, -1)) for t in (q, k, v))      # [B, L, H, D]
        cos, sin = st.rot                                                    # absolute positions (the model's 0-based rope is ignored)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        tpp = st.pph * st.ppw
        if st.mode == "masked":
            L = q.shape[1]
            pl = torch.arange(L, device=q.device) // tpp
            seg = torch.as_tensor(st.segments, device=q.device)[pl]
            mask = block_mask(pl, seg, st.window)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        else:  # cache: the current plane's scratch K/V + cached finished planes (already rotated at their absolute positions)
            if q.shape[1] != tpp:
                raise RuntimeError("CausalWan cache mode runs ONE plane per forward")
            kk = torch.cat(st.k[s.layer] + [k], dim=1)
            vv = torch.cat(st.v[s.layer] + [v], dim=1)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2))
            if st.append:  # one append per block: detached (no gradient across past K/V), oldest evicted first
                st.k[s.layer].append(k.detach()); st.v[s.layer].append(v.detach()); st.pos[s.layer].append(st.next_pos)
                st.appends[s.layer] += 1
                while len(st.k[s.layer]) > st.window:
                    st.k[s.layer].pop(0); st.v[s.layer].pop(0); st.pos[s.layer].pop(0)
        o = o.transpose(1, 2).flatten(2, 3).type_as(q)
        return attn.to_out[1](attn.to_out[0](o))


class CausalWan:
    """block-causal wrapper. `model` is a diffusers WanTransformer3DModel or an mrv.wan_cond.WanCond (its .tf is the transformer);
    masked() / step() / commit() pass their positional / keyword arguments straight to `model` (WanCond: x_t, extra, timestep,
    tok, ray, tok_keep; bare transformer: hidden_states=..., timestep=..., encoder_hidden_states=...). x_t carries the planes on
    its frame axis (dim 2). Timesteps may be per token ([B, L], supported by diffusers 0.43) for mixed-noise prefixes."""

    def __init__(s, model, window=WINDOW):
        s.model = model
        s.tf = model.tf if hasattr(model, "tf") else model
        s.state = CausalState(len(s.tf.blocks), window)
        s._orig = []
        for i, blk in enumerate(s.tf.blocks):
            s._orig.append(blk.attn1.processor)
            blk.attn1.processor = CausalSelfAttnProcessor(s.state, i)
        s.patch = tuple(s.tf.config.patch_size)

    def uninstall(s):
        for blk, p in zip(s.tf.blocks, s._orig):
            blk.attn1.processor = p

    # ------------------------------------------------------------------------------------------------ helpers
    def _geom(s, x):
        _, _, n, h, w = x.shape
        if s.patch[0] != 1:
            raise ValueError("block = one latent plane needs a time patch of 1")
        s.state.pph, s.state.ppw = h // s.patch[1], w // s.patch[2]
        return n

    def _x(s, args, kwargs):
        return args[0] if args else kwargs["hidden_states"]

    def _call(s, args, kwargs):
        if torch.is_grad_enabled() and getattr(s.tf, "gradient_checkpointing", False):
            raise RuntimeError("disable gradient checkpointing while using the causal cache")
        out = s.model(*args, **kwargs)
        return out[0] if isinstance(out, tuple) else (out.sample if hasattr(out, "sample") else out)

    # ------------------------------------------------------------------------------------------------ modes
    def masked(s, *args, positions=None, segments=None, **kwargs):
        """one forward over n planes: block-causal sliding-window mask; positions (absolute, per plane; default 0..n-1) and
        segments (per plane; no attention across a reset) as given"""
        st = s.state
        n = s._geom(s._x(args, kwargs))
        st.positions = list(range(n)) if positions is None else list(positions)
        st.segments = [0] * n if segments is None else list(segments)
        st.mode, st.rot = "masked", rotary_for(s.tf.rope, st.positions, st.pph, st.ppw)
        try:
            return s._call(args, kwargs)
        finally:
            st.mode = None

    def step(s, *args, **kwargs):
        """one denoising substep of the current plane (absolute index next_pos); nothing is stored"""
        return s._run(False, args, kwargs)

    def commit(s, *args, **kwargs):
        """re-encode the current plane's FINAL latent (pass it with the commit timestep, default clean) and append its K/V once
        per layer; advances the time index"""
        out = s._run(True, args, kwargs)
        s.state.next_pos += 1
        return out

    def _run(s, append, args, kwargs):
        st = s.state
        if s._geom(s._x(args, kwargs)) != 1:
            raise ValueError("step / commit take exactly one latent plane")
        st.mode, st.append = "cache", append
        st.rot = rotary_for(s.tf.rope, [st.next_pos], st.pph, st.ppw)
        try:
            return s._call(args, kwargs)
        finally:
            st.mode, st.append = None, False

    def reset(s):
        """drive boundary / discontinuity: empty every layer's cache, time index back to 0"""
        s.state.reset()

    # ------------------------------------------------------------------------------------------------ introspection
    def cached_positions(s, layer=0):
        return list(s.state.pos[layer])

    def appends(s):
        return list(s.state.appends)
