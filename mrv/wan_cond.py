"""PLAN-video S1: Wan2.1-T2V-1.3B conditioned on the six car cameras (council/PLAN-video.md, signed + owner decisions).

Two input paths, both trained from update 0 with INDEPENDENT dropout (guide 20 % + random holes, tokens 10 %), so the end-of-S1
ablation can render hybrid / guide-only / tokens-only / none from the SAME checkpoint at fixed noise:

(a) GUIDE (main road). Per tick a chase-view guide RGB + validity mask (mrv/guide.py -> data/guides_v1/<scene>.pt). The guide video
    is encoded with the FROZEN Wan VAE exactly like the target (same 4-tick grouping, same per-channel normalisation) -> 16 guide
    latent channels; the mask is area-pooled 8x8 to the latent grid and its 4 ticks per latent frame are folded into 4 channels
    (Wan-I2V layout: a single-frame first latent repeats its tick 4x). Both are concatenated to the noisy latent:
        input = [x_t (16) | guide latent (16) | mask (4)] = 36 channels.
    The stock patch embedding (Conv3d 16 -> 1536, kernel / stride (1, 2, 2)) is kept FROZEN and a new Conv3d 20 -> 1536 (no bias,
    ZERO-initialised, trained fully) is added on the extra channels: embed(x) = base(x[:, :16]) + extra(x[:, 16:]).
    Step 0 is therefore exactly stock Wan on the latent channels. A dropped guide = zero guide latent AND zero mask (mask 0 says
    "no guide here"; random holes zero both inside rectangles).

(b) RAW-VIEW TOKENS (second path). Frozen per-view encoder = the ViT-S/14 (DINOv2-small) backbone of Depth Anything V2-Small, the
    same network the guide uses for depth (default = mrv/guide.py PRIMARY, depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf,
    the transformers port of the Apache-2.0-tagged Metric-VKITTI-Small), so at run time ONE encoder pass per view serves depth
    (guide) and tokens. Last-layer patch tokens
    (384-d, layer-normed) at 448x252 input -> 32x18 = 576 UNPOOLED tokens per view, 3,456 per tick (scripts/wan_tokens.py).
    Each token gets its source ray as Pluecker coordinates (d, o x d) expressed in the CHASE camera frame (target-relative; the
    judge's camera: 7.5 m behind, 2.4 m up, pitch -8 deg). A small trainable projector (LayerNorm + Linear 384->1024, ray MLP
    6->1024, learned view and tick-slot embeddings, SiLU, Linear 1024->4096) maps them to the 4096-d umT5 text width, then the
    STOCK (frozen) Wan text_embedder (4096 -> 1536) and each block's cross-attention (LoRA'd) read them.
    Per-latent-frame alignment (Fable 06:35 item b3): latent frame f attends ONLY to the tokens of its own ticks
    (block-diagonal; implemented as a [B*F] reshape, not a dense mask). Which of the 4 ticks of a group: --tok_ticks
    last (newest tick, 3,456 tokens / latent), ends (first + newest, 6,912; default: speed inside the group observable),
    all (4 ticks, 13,824).
    The token attention is DECOUPLED from the text slot (IP-Adapter style): out = to_out(Attn(q, prompt) + g * Attn_f(q, tokens)),
    g a per-block, per-channel gate initialised to ZERO -> step 0 == stock. umT5 is NOT used (not downloaded on the cluster):
    the text slot holds a fixed all-zero 4096-d embedding (stock Wan pads every prompt with zero rows, so this is the stock
    "nothing but padding" input; 512 identical zero rows attend exactly like 1 row, so 1 row is kept). A cached umT5 embedding
    can be passed instead (prompt=...).

LoRA rank 32 (peft) on attn1 / attn2 to_q, to_k, to_v, to_out.0 and ffn.net.0.proj / ffn.net.2 in all 30 blocks; base
transformer, text_embedder and VAE frozen; new modules (extra patch conv, projector, gates) trained fully, in fp32.

Wan latent conventions (diffusers): z_norm = (vae.encode(x).latent_dist.mode() - latents_mean) / latents_std per channel;
x in [-1, 1], [B, 3, F, H, W], F = 4k + 1; latent frames L = 1 + (F - 1) / 4; latent j >= 1 covers frames 4j-3 .. 4j.
Flow matching: x_sigma = (1 - sigma) x0 + sigma eps, model predicts v = eps - x0, timestep input = 1000 sigma.
"""
import hashlib
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WAN_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
TOK_ENCODER_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"  # = mrv/guide.py PRIMARY
LORA_TARGET_REGEX = r".*blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)$|.*blocks\.\d+\.ffn\.net\.(0\.proj|2)$"
N_EXTRA = 20          # 16 guide-latent channels + 4 folded mask channels
TEXT_DIM = 4096
TOK_DIM = 384
N_VIEWS = 6
TOK_GRID = (18, 32)   # rows, cols of ViT-S/14 patches at 448x252
SLOTS = {"last": 1, "ends": 2, "all": 4}
# the judge's chase camera (mrv/carmap.py): ego frame x fwd, y left, z up
CHASE_POS = np.array([-7.5, 0.0, 2.4])
_cp, _sp = math.cos(math.radians(8)), math.sin(math.radians(8))
CHASE_R = np.array([[0.0, -1.0, 0.0],      # camera x = right  (= -ego y)
                    [-_sp, 0.0, -_cp],     # camera y = down
                    [_cp, 0.0, -_sp]])     # camera z = forward (pitched 8 deg down)


# ----------------------------------------------------------------------------------------------------------------- geometry
def plucker_chase(d_ego, o_ego):
    """unit ray directions d (N, 3) and the camera origin o (3,) in the ego frame -> Pluecker (N, 6) in the chase-camera frame"""
    d = d_ego @ CHASE_R.T
    o = CHASE_R @ (np.asarray(o_ego, np.float64) - CHASE_POS)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return np.concatenate([d, np.cross(np.broadcast_to(o, d.shape), d)], 1)


# -------------------------------------------------------------------------------------------------------- latent helpers
def vae_stats(vae, device, dtype=torch.float32):
    zd = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, zd, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, zd, 1, 1, 1)
    return mean, std


@torch.no_grad()
def wan_encode(vae, x):
    """x [B, 3, F, H, W] in [-1, 1] (F = 4k + 1) -> normalised latents [B, 16, L, H/8, W/8] (deterministic: posterior mode)"""
    assert (x.shape[2] - 1) % 4 == 0, x.shape
    p = next(vae.parameters())
    mean, std = vae_stats(vae, x.device)
    z = vae.encode(x.to(p.dtype)).latent_dist.mode().float()
    return (z - mean) / std


@torch.no_grad()
def wan_decode(vae, z):
    """normalised latents [B, 16, L, h, w] -> RGB [B, 3, 4L-3, 8h, 8w] in [-1, 1]"""
    p = next(vae.parameters())
    mean, std = vae_stats(vae, z.device)
    return vae.decode((z.float() * std + mean).to(p.dtype)).sample.float().clamp(-1, 1)


def mask_lowres(mask):
    """guide validity [T, 1, 256, 512] (bool / uint8 0-1 or 0-255 / float) -> fraction valid per latent cell [T, 32, 64] float"""
    m = mask.float()
    if m.max() > 1.0:
        m = m / 255.0
    return F.avg_pool2d(m, 8)[:, 0]


def latent_spans(L, first_single=True):
    """frame offsets covered by each latent frame of a clip: first_single (stream / clip start): [0]*4, then 4j-3..4j;
    mid-stream (a window that starts at a regular latent j0 of a longer stream, offsets relative to frame 4*j0-3): 4 frames each"""
    if first_single:
        return [[0, 0, 0, 0]] + [[4 * j - 3 + k for k in range(4)] for j in range(1, L)]
    return [[4 * j + k for k in range(4)] for j in range(L)]


def fold_mask(mask_lr, spans):
    """mask_lr [T, 32, 64] (frame-indexed), spans = per-latent frame lists of 4 -> [4, L, 32, 64]"""
    return torch.stack([torch.stack([mask_lr[f] for f in sp]) for sp in spans], 1)


def token_slots(span, mode):
    """the ticks of one latent frame whose camera tokens it attends to"""
    if mode == "last":
        return [span[-1]]
    if mode == "ends":
        return [span[0], span[-1]]
    return list(span)


def sigma_shift(u, shift):
    """SD3 / Wan time shift: sigma = s u / (1 + (s - 1) u)"""
    return shift * u / (1 + (shift - 1) * u)


# -------------------------------------------------------------------------------------------------------------- modules
class PatchEmbedExt(nn.Module):
    """frozen stock patch embedding on the 16 latent channels + zero-init trainable conv on the extra channels"""

    def __init__(s, base: nn.Conv3d, n_extra: int):
        super().__init__()
        s.base, s.n_base = base, base.in_channels
        s.extra = nn.Conv3d(n_extra, base.out_channels, base.kernel_size, base.stride, bias=False)
        nn.init.zeros_(s.extra.weight)

    def forward(s, x):
        return s.base(x[:, :s.n_base]) + s.extra(x[:, s.n_base:].to(s.extra.weight.dtype)).to(s.base.weight.dtype)


class TokenProjector(nn.Module):
    """per-view patch tokens + chase-relative Pluecker rays -> 4096-d text-slot embeddings"""

    def __init__(s, n_slots, d_tok=TOK_DIM, d=1024, d_out=TEXT_DIM, n_views=N_VIEWS):
        super().__init__()
        s.tok = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, d))
        s.ray = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        s.view = nn.Parameter(torch.zeros(n_views, d))
        s.slot = nn.Parameter(torch.zeros(n_slots, d))
        nn.init.normal_(s.view, std=0.02); nn.init.normal_(s.slot, std=0.02)
        s.out = nn.Sequential(nn.SiLU(), nn.Linear(d, d_out))
        s.n_slots, s.n_views = n_slots, n_views

    def forward(s, tok, ray):
        """tok [B, F, slots, V, P, 384]; ray [B, F, slots, V, P, 6] -> [B, F * slots * V * P, 4096]"""
        B, Fr, S, V, P, _ = tok.shape
        ray = torch.cat([ray[..., :3], ray[..., 3:] / 10.0], -1)  # moments are metres: bring them to O(1)
        h = s.tok(tok.float()) + s.ray(ray.float()) + s.view.view(1, 1, 1, V, 1, -1) + s.slot[:S].view(1, 1, S, 1, 1, -1)
        return s.out(h).reshape(B, Fr * S * V * P, -1)


class DualCrossAttn(nn.Module):
    """replaces a block's attn2: stock cross-attention on the text slot + a gated, per-latent-frame attention on camera tokens.
    The extra inputs travel through a shared dict (set by WanCond.forward), so the stock block / checkpointing code is unchanged."""

    def __init__(s, inner, state, dim):
        super().__init__()
        s.inner, s.state = inner, state
        s.gate = nn.Parameter(torch.zeros(dim))  # per-channel, zero-init: the token path starts switched off

    def forward(s, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kw):
        a, st = s.inner, s.state
        n_p = st["n_prompt"]; h = a.heads
        B, Lq, C = hidden_states.shape
        sh = lambda x: x.unflatten(-1, (h, -1)).transpose(1, 2)  # [b, L, C] -> [b, h, L, d]
        q = a.norm_q(a.to_q(hidden_states))
        ep = encoder_hidden_states[:1, :n_p] if st.get("prompt_shared", True) else encoder_hidden_states[:, :n_p]
        kp, vp = a.norm_k(a.to_k(ep)), a.to_v(ep)
        kp, vp = sh(kp).expand(B, -1, -1, -1).contiguous(), sh(vp).expand(B, -1, -1, -1).contiguous()
        o = F.scaled_dot_product_attention(sh(q), kp.to(q.dtype), vp.to(q.dtype)).transpose(1, 2).flatten(2)
        keep = st.get("keep")
        if keep is not None and encoder_hidden_states.shape[1] > n_p:
            Fr = st["F"]
            et = encoder_hidden_states[:, n_p:]
            kt, vt = a.norm_k(a.to_k(et)), a.to_v(et)
            qf = q.reshape(B * Fr, Lq // Fr, C)
            kt = kt.reshape(B * Fr, -1, C); vt = vt.reshape(B * Fr, -1, C)
            ot = F.scaled_dot_product_attention(sh(qf), sh(kt).to(q.dtype), sh(vt).to(q.dtype)).transpose(1, 2).flatten(2)
            ot = ot.reshape(B, Lq, C)
            o = o + (ot.float() * s.gate.float() * keep.view(B, 1, 1).float()).to(o.dtype)
        o = a.to_out[0](o)
        return a.to_out[1](o)


class WanCond(nn.Module):
    """Wan transformer (LoRA) + guide channels + camera-token cross-attention. forward() returns the flow prediction v."""

    def __init__(s, transformer, tok_ticks="ends", use_tokens=True, prompt=None):
        super().__init__()
        s.tf = transformer
        s.state = {"n_prompt": 1, "F": 1, "keep": None, "prompt_shared": True}
        s.tf.patch_embedding = PatchEmbedExt(s.tf.patch_embedding, N_EXTRA)
        s.xattn = []
        dim = s.tf.config.num_attention_heads * s.tf.config.attention_head_dim
        for blk in s.tf.blocks:
            blk.attn2 = DualCrossAttn(blk.attn2, s.state, dim)
            s.xattn.append(blk.attn2)
        s.use_tokens, s.tok_ticks = use_tokens, tok_ticks
        s.proj = TokenProjector(SLOTS[tok_ticks]) if use_tokens else None
        if prompt is None:
            prompt = torch.zeros(1, 1, TEXT_DIM)  # = stock zero padding rows (umT5 not available on the cluster)
        s.register_buffer("prompt", prompt.float().reshape(1, -1, TEXT_DIM), persistent=False)

    def forward(s, x_t, extra, timestep, tok=None, ray=None, tok_keep=None):
        """x_t [B,16,F,h,w]; extra [B,20,F,h,w]; timestep [B] (= 1000 sigma); tok [B,F,slots,V,P,384]; ray same with 6; tok_keep [B]"""
        B, _, Fr = x_t.shape[:3]
        dt = s.tf.patch_embedding.base.weight.dtype
        pr = s.prompt.to(dt).expand(B, -1, -1)
        s.state.update(n_prompt=s.prompt.shape[1], F=Fr, keep=None)
        if s.use_tokens and tok is not None and tok_keep is not None and bool(tok_keep.any()):
            enc = torch.cat([pr, s.proj(tok, ray).to(dt)], 1)
            s.state["keep"] = tok_keep.to(x_t.device)
        else:
            enc = pr
        x = torch.cat([x_t, extra], 1).to(dt)
        return s.tf(hidden_states=x, timestep=timestep, encoder_hidden_states=enc, return_dict=False)[0]

    # --------------------------------------------------------------------------------------------- bookkeeping
    def trainable(s):
        return [(n, p) for n, p in s.named_parameters() if p.requires_grad]

    def trainable_state(s):
        return {n: p.detach().float().cpu().clone() for n, p in s.trainable()}

    def load_trainable(s, sd):
        names = {n for n, _ in s.trainable()}
        if set(sd) != names:
            raise SystemExit(f"trainable state mismatch: missing {sorted(names - set(sd))[:4]} unexpected {sorted(set(sd) - names)[:4]}")
        own = dict(s.named_parameters())
        with torch.no_grad():
            for n, v in sd.items():
                own[n].copy_(v.to(own[n].dtype))

    def frozen(s):
        return [(n, p) for n, p in s.named_parameters() if not p.requires_grad]


def build(wan=WAN_ID, rank=32, tok_ticks="ends", use_tokens=True, dtype=torch.bfloat16, device="cuda", prompt=None,
          grad_ckpt=False, parity=False, lora=True):
    """load the stock transformer (local files only), add LoRA, wrap. parity=True also returns max|stock - wrapped| on one random
    5-latent input at init (must be ~0: zero-init extension, zero gates, LoRA B = 0)."""
    from diffusers import WanTransformer3DModel
    tf = WanTransformer3DModel.from_pretrained(wan, subfolder="transformer", torch_dtype=dtype, local_files_only=True)
    tf.requires_grad_(False); tf.to(device)
    par = None
    if parity:
        g = torch.Generator().manual_seed(7)
        x = torch.randn(1, 16, 5, 32, 64, generator=g).to(device, dtype); t = torch.tensor([600.0], device=device)
        ex = torch.randn(1, N_EXTRA, 5, 32, 64, generator=g).to(device, dtype)
        pr = (prompt if prompt is not None else torch.zeros(1, 1, TEXT_DIM)).reshape(1, -1, TEXT_DIM)
        stock_ctx = torch.cat([pr, torch.zeros(1, 512 - pr.shape[1], TEXT_DIM)], 1).to(device, dtype) if pr.shape[1] < 512 else pr.to(device, dtype)
        with torch.no_grad():
            ref = tf(hidden_states=x, timestep=t, encoder_hidden_states=stock_ctx, return_dict=False)[0].float()
    if lora:
        from peft import LoraConfig
        tf.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights=True, target_modules=LORA_TARGET_REGEX))
    m = WanCond(tf, tok_ticks=tok_ticks, use_tokens=use_tokens, prompt=prompt).to(device)
    for n, p in m.named_parameters():  # trainable = LoRA + new modules, all fp32; everything else frozen in `dtype`
        new = ("lora_" in n) or n.startswith("proj.") or ".patch_embedding.extra." in n or n.endswith(".attn2.gate")
        p.requires_grad_(new)
        if new:
            p.data = p.data.float()
    if grad_ckpt:
        tf.enable_gradient_checkpointing()
    if parity:
        S = SLOTS[tok_ticks]
        tok = torch.randn(1, 5, S, N_VIEWS, TOK_GRID[0] * TOK_GRID[1], TOK_DIM, generator=g).to(device)
        ray = torch.randn(1, 5, S, N_VIEWS, TOK_GRID[0] * TOK_GRID[1], 6, generator=g).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            out = m(x, ex, t, tok if use_tokens else None, ray if use_tokens else None, torch.ones(1, dtype=torch.bool, device=device)).float()
        par = {"max_abs": (out - ref).abs().max().item(), "ref_std": ref.std().item()}
    return (m, par) if parity else m


def tensor_sha(named):
    """sha256 over (name, raw bytes) of an iterable of (name, tensor), order-independent of dict order (sorted by name)"""
    h = hashlib.sha256()
    for n, t in sorted(named, key=lambda x: x[0]):
        t = t.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        h.update(n.encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def base_sha(model):
    return tensor_sha(model.frozen())


def vae_sha(vae):
    return tensor_sha(vae.named_parameters())
