"""PLAN-video S1: Wan2.1-T2V-1.3B conditioned on the six car cameras (council/PLAN-video.md, signed + owner decisions; revised for
Astra's review council/codereview-wan-s1-astra.md and the guide with flat-ground underlay, 415c470).

Two input paths, both trained from update 0 with INDEPENDENT dropout (guide 20 % + random holes, tokens 10 %), so the end-of-S1
ablation renders hybrid / guide-only / tokens-only / none from the SAME checkpoint at fixed noise:

(a) GUIDE (main road; mrv/guide.py -> data/guides_v1/<scene>.pt): per tick a chase-view RGB splat (depth points win, the flat-
    ground underlay fills the rest) and a per-pixel SOURCE map src (1 = depth, 2 = ground prior, 0 = nothing). The guide video is
    encoded by the FROZEN Wan VAE exactly like the target (16 latent channels, same normalisation). The source map becomes two
    planes (depth share, ground share per 8x8 latent cell) and the 4 ticks of each latent frame are folded into channels
    (Wan-I2V layout; a single-frame first latent repeats its tick 4x): 2 x 4 = 8 channels. Input:
        [x_t (16) | guide latent (16) | depth share x4 ticks (4) | ground share x4 ticks (4)] = 40 channels (N_EXTRA = 24).
    Stock patch embedding (Conv3d 16 -> 1536, kernel (1,2,2)) FROZEN; new Conv3d 24 -> 1536, no bias, ZERO-init, trained fully:
    embed(x) = base(x[:, :16]) + extra(x[:, 16:]). A dropped guide = zero latent AND zero source planes; holes zero both.

(b) RAW-VIEW TOKENS (second path), SHARED with the guide's depth pass (Astra item 1): the depth network of the guide (same
    checkpoint, same preprocessing: mrv.guide.load_views at the guide's depth resolution 294x518, ImageNet normalisation, fp16)
    is run ONCE with output_hidden_states; tokens = backbone.layernorm(last hidden state) without CLS -> 21 x 37 = 777 patch
    tokens x 384 per view, 4,662 per tick (scripts/wan_tokens.py). Each token's ray uses THAT tick's per-camera K and cam2tick
    (written by the guide builder), expressed as Pluecker (d, o x d) in the chase-camera frame (mrv.guide.chase_cam_to_ego,
    including the guide's chase_dz). A trainable projector (LayerNorm + Linear 384->1024, ray MLP 6->1024, view + tick-slot
    embeddings, SiLU, Linear 1024->4096) maps them to the umT5 text width; the STOCK frozen text_embedder (4096 -> 1536) and each
    block's (LoRA'd) cross-attention read them. Latent frame f attends ONLY to its own ticks' tokens ([B*F] reshape =
    block-diagonal mask). --tok_ticks last (newest tick of the group; default after Astra item 6: 4,662 rows / latent),
    ends (first + newest; 9,324), all (4 ticks; 18,648).
    Decoupled from the text slot: out = to_out(Attn(q, text) + g * Attn_f(q, tokens)), g per block / channel, ZERO-init.
    umT5 is not used (not on the cluster): the text slot is one all-zero 4096-d row (stock pads prompts with zero rows; 512
    identical rows attend exactly like one: tested by build(parity=True)).

LoRA rank 32 (peft) on attn1 / attn2 to_q, to_k, to_v, to_out.0 and ffn.net.0.proj / ffn.net.2 = 10 targets x 30 blocks
(asserted); base transformer, text_embedder, VAE frozen; new modules + LoRA trained in fp32. Initialisation is seeded.
Latents: z = (vae.encode(x).latent_dist.mode() - latents_mean) / latents_std; flow matching x_s = (1 - s) x0 + s eps,
v = eps - x0, timestep 1000 s.
"""
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mrv.wan_plan import latent_spans, token_slots  # noqa: F401  (re-exported for the scripts)

WAN_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
N_GUIDE_LAT = 16
N_SRC = 2                       # source planes: depth, ground prior
N_EXTRA = N_GUIDE_LAT + 4 * N_SRC  # 24
TEXT_DIM = 4096
TOK_DIM = 384
N_VIEWS = 6
PS = 14
DEPTH_HW = (294, 518)           # the guide's depth input (multiples of 14); tokens come from the same pass
TOK_GRID = (DEPTH_HW[0] // PS, DEPTH_HW[1] // PS)  # (21, 37)
TOK_P = TOK_GRID[0] * TOK_GRID[1]                 # 777
FULL_WH = (1600, 900)
SLOTS = {"last": 1, "ends": 2, "all": 4}
LORA_TARGET_REGEX = r".*blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)$|.*blocks\.\d+\.ffn\.net\.(0\.proj|2)$"
LORA_TARGETS10 = ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k", "attn2.to_v",
                  "attn2.to_out.0", "ffn.net.0.proj", "ffn.net.2"]


# ----------------------------------------------------------------------------------------------------------------- geometry
def token_rays(K, cam2tick, chase_cam2ego, grid=TOK_GRID, in_wh=(DEPTH_HW[1], DEPTH_HW[0]), full_wh=FULL_WH):
    """per-tick Pluecker rays of the token patch centres, in the chase-camera frame.
    K [6,3,3] (full-res px), cam2tick [6,4,4], chase_cam2ego [4,4] (chase camera -> tick ego) -> [6, h*w, 6] float32"""
    K, c2t = np.asarray(K, np.float64), np.asarray(cam2tick, np.float64)
    h, w = grid
    u = (np.arange(w) + 0.5) * PS * full_wh[0] / in_wh[0]
    v = (np.arange(h) + 0.5) * PS * full_wh[1] / in_wh[1]
    uu, vv = np.meshgrid(u, v)  # row-major like the tokens
    pix = np.stack([uu.ravel(), vv.ravel(), np.ones(h * w)], 1)
    E2C = np.linalg.inv(np.asarray(chase_cam2ego, np.float64))  # tick ego -> chase camera
    out = []
    for c in range(K.shape[0]):
        d = (np.linalg.inv(K[c]) @ pix.T).T @ c2t[c, :3, :3].T        # tick-ego directions
        o = c2t[c, :3, 3]
        d = d @ E2C[:3, :3].T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        oc = E2C[:3, :3] @ o + E2C[:3, 3]
        out.append(np.concatenate([d, np.cross(np.broadcast_to(oc, d.shape), d)], 1))
    return np.stack(out).astype(np.float32)


def patch_centre_pixels(K, grid=TOK_GRID, in_wh=(DEPTH_HW[1], DEPTH_HW[0]), full_wh=FULL_WH):
    """full-res pixel coordinates of the token patch centres [h*w, 2] (same convention as token_rays)"""
    h, w = grid
    u = (np.arange(w) + 0.5) * PS * full_wh[0] / in_wh[0]; v = (np.arange(h) + 0.5) * PS * full_wh[1] / in_wh[1]
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu.ravel(), vv.ravel()], 1)


# ------------------------------------------------------------------------------------------------------------ guide intake
GUIDE_VERSION_KEYS = ("depth_model_id", "scale_sha", "code_sha", "depth_res_hw", "splat", "rmax", "chase_dz", "tick_pose", "underlay",
                      "every", "lidar_ceiling")


def guide_version(prov):
    v = {k: prov.get(k) for k in GUIDE_VERSION_KEYS}
    v["depth_weights_sha"] = (prov.get("depth_info") or {}).get("weights_sha")
    return v


def check_guide(path, G, side, ticks, hard, scene, expect=None):
    """consumer checks for one guide drive (Astra item 2 + coordinator 415c470). Returns its version dict; raises SystemExit.
    expect: a version dict every drive must equal (train / dev / rollout consistency)."""
    low = path.lower()
    if "lidar" in low or "preview" in low:
        raise SystemExit(f"{path}: LiDAR-ceiling / preview guides are never model inputs")
    if not side.get("complete"):
        raise SystemExit(f"{path}: guide sidecar not marked complete")
    prov = G.get("provenance") or {}
    if prov.get("lidar_ceiling") or prov.get("note") != "camera-only":
        raise SystemExit(f"{path}: not a camera-only guide ({prov.get('note')})")
    if not prov.get("underlay"):
        raise SystemExit(f"{path}: guide without the flat-ground underlay (C9) is not the production guide")
    if prov.get("every") != 1:
        raise SystemExit(f"{path}: sparse guide (every={prov.get('every')})")
    if "src" not in G:
        raise SystemExit(f"{path}: no source map 'src' (old guide version)")
    if G.get("scene", scene) != scene or side.get("scene", scene) != scene:
        raise SystemExit(f"{path}: scene mismatch")
    gt = [int(t) for t in G["ticks"].tolist()]
    if gt != [int(t) for t in ticks]:
        raise SystemExit(f"{path}: guide ticks != enc ticks ({len(gt)} vs {len(ticks)})")
    if [bool(x) for x in G["hard"].tolist()] != [bool(x) for x in hard]:
        raise SystemExit(f"{path}: guide hard flags != enc hard flags")
    built = G["built"].tolist()
    miss = [int(ticks[i]) for i in range(len(ticks)) if not hard[i] and not built[i]]
    if miss:
        raise SystemExit(f"{path}: {len(miss)} non-hard ticks not built, e.g. {miss[:5]}")
    v = guide_version(prov)
    if not v["scale_sha"] or not v["depth_model_id"]:
        raise SystemExit(f"{path}: provenance lacks depth model / scale sha")
    for k_ in ("depth_model_id", "scale_sha", "underlay", "code_sha", "depth_res_hw"):
        if side.get(k_) != prov.get(k_):
            raise SystemExit(f"{path}: sidecar {k_} {side.get(k_)} != file provenance {prov.get(k_)}")
    if expect is not None and v != expect:
        diff = {k: (v.get(k), expect.get(k)) for k in set(v) | set(expect) if v.get(k) != expect.get(k)}
        raise SystemExit(f"{path}: guide version differs from the expected one: {diff}")
    return v


# -------------------------------------------------------------------------------------------------------- latent helpers
def vae_stats(vae, device, dtype=torch.float32):
    zd = vae.config.z_dim
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=dtype).view(1, zd, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device, dtype=dtype).view(1, zd, 1, 1, 1)
    return mean, std


@torch.no_grad()
def wan_encode(vae, x):
    """x [B, 3, F, H, W] in [-1, 1] (F = 4k + 1) -> normalised latents [B, 16, L, H/8, W/8] (posterior mode)"""
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


def src_lowres(src):
    """guide source map [T, 1, 256, 512] uint8 (1 depth, 2 ground prior, 0 none) -> shares per latent cell [T, 2, 32, 64]"""
    s = src[:, 0]
    planes = torch.stack([(s == 1).float(), (s == 2).float()], 1)
    return F.avg_pool2d(planes, 8)


def fold_src(src_lr, spans):
    """src_lr [T, 2, 32, 64] frame-indexed, spans = per-latent lists of 4 frames -> [8, L, 32, 64]
    (channels 0-3 depth share of the 4 ticks, 4-7 ground share)"""
    return torch.stack([torch.cat([src_lr[sp, c] for c in range(N_SRC)], 0) for sp in spans], 1)


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

    def forward(s, tok, ray):
        """tok [B, F, slots, V, P, 384]; ray [B, F, slots, V, P, 6] -> [B, F * slots * V * P, 4096]"""
        B, Fr, S, V, P, _ = tok.shape
        ray = torch.cat([ray[..., :3], ray[..., 3:] / 10.0], -1)  # moments are metres: bring them to O(1)
        h = s.tok(tok.float()) + s.ray(ray.float()) + s.view.view(1, 1, 1, V, 1, -1) + s.slot[:S].view(1, 1, S, 1, 1, -1)
        return s.out(h).reshape(B, Fr * S * V * P, -1)


class DualCrossAttn(nn.Module):
    """replaces a block's attn2: stock cross-attention on the text slot + a gated per-latent-frame attention on camera tokens.
    Extra inputs travel through a shared dict set by WanCond.forward (stock block / checkpointing code unchanged)."""

    def __init__(s, inner, state, dim):
        super().__init__()
        s.inner, s.state = inner, state
        s.gate = nn.Parameter(torch.zeros(dim))

    def forward(s, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None, **kw):
        a, st = s.inner, s.state
        n_p = st["n_prompt"]; h = a.heads
        B, Lq, C = hidden_states.shape
        sh = lambda x: x.unflatten(-1, (h, -1)).transpose(1, 2)  # [b, L, C] -> [b, h, L, d]
        q = a.norm_q(a.to_q(hidden_states))
        ep = encoder_hidden_states[:1, :n_p]  # the text slot is identical for every row
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
            o = o + (ot.reshape(B, Lq, C).float() * s.gate.float() * keep.view(B, 1, 1).float()).to(o.dtype)
        return a.to_out[1](a.to_out[0](o))


class WanCond(nn.Module):
    """Wan transformer (LoRA) + guide channels + camera-token cross-attention. forward() returns the flow prediction v."""

    def __init__(s, transformer, tok_ticks="last", use_tokens=True, prompt=None):
        super().__init__()
        s.tf = transformer
        s.state = {"n_prompt": 1, "F": 1, "keep": None}
        s.tf.patch_embedding = PatchEmbedExt(s.tf.patch_embedding, N_EXTRA)
        dim = s.tf.config.num_attention_heads * s.tf.config.attention_head_dim
        s.xattn = []
        for blk in s.tf.blocks:
            blk.attn2 = DualCrossAttn(blk.attn2, s.state, dim)
            s.xattn.append(blk.attn2)
        s.use_tokens, s.tok_ticks = use_tokens, tok_ticks
        s.proj = TokenProjector(SLOTS[tok_ticks]) if use_tokens else None
        s.set_prompt(prompt)

    def set_prompt(s, prompt):
        p = torch.zeros(1, 1, TEXT_DIM) if prompt is None else prompt.float().reshape(1, -1, TEXT_DIM)
        dev = s.tf.patch_embedding.base.weight.device
        s.register_buffer("prompt", p.to(dev), persistent=False)

    def forward(s, x_t, extra, timestep, tok=None, ray=None, tok_keep=None):
        """x_t [B,16,F,h,w]; extra [B,24,F,h,w]; timestep [B] (= 1000 sigma); tok [B,F,slots,V,P,384]; ray same with 6; tok_keep [B]"""
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


def lora_census(model):
    """{target: number of blocks carrying LoRA A and B on it}; must be 30 for each of the 10 targets"""
    names = {n for n, _ in model.named_parameters()}
    nb = len(model.tf.blocks); res = {}
    for t in LORA_TARGETS10:
        a_, b_ = t.split(".", 1)
        pre = lambda i: f"tf.blocks.{i}.{a_}.inner.{b_}" if a_ == "attn2" else f"tf.blocks.{i}.{t}"
        res[t] = sum(1 for i in range(nb) if f"{pre(i)}.lora_A.default.weight" in names and f"{pre(i)}.lora_B.default.weight" in names)
    return res


def _rand_inputs(device, dtype, tok_ticks, seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 16, 5, 32, 64, generator=g).to(device, dtype)
    ex = torch.randn(1, N_EXTRA, 5, 32, 64, generator=g).to(device, dtype)
    S = SLOTS[tok_ticks]
    tok = torch.randn(1, 5, S, N_VIEWS, TOK_P, TOK_DIM, generator=g).to(device)
    ray = torch.randn(1, 5, S, N_VIEWS, TOK_P, 6, generator=g).to(device)
    return x, ex, tok, ray


def build(wan=WAN_ID, rank=32, tok_ticks="last", use_tokens=True, dtype=torch.bfloat16, device="cuda", prompt=None,
          grad_ckpt=False, parity=False, lora=True, seed=0):
    """load the stock transformer (local files only), add LoRA (seeded), wrap, assert the 10 x 30 LoRA targets.
    parity=True returns diagnostics (Astra item 4), all under the SAME autocast and the SAME 512-row padded text context:
      stock_vs_wrapped : stock transformer vs wrapped model at init (zero-init extension, zero gates, LoRA B = 0)
      one_row_shortcut : wrapped with the 512-row padded context vs the 1-row context the model uses
      guide_inert / tokens_inert : output change from random vs zero guide channels / tokens on vs off at init (must be 0)
      finite           : every output finite (NaN would pass a max-diff threshold)"""
    from diffusers import WanTransformer3DModel
    tf = WanTransformer3DModel.from_pretrained(wan, subfolder="transformer", torch_dtype=dtype, local_files_only=True)
    tf.requires_grad_(False); tf.to(device)
    t = torch.tensor([600.0], device=device)
    pr1 = torch.zeros(1, 1, TEXT_DIM) if prompt is None else prompt.float().reshape(1, -1, TEXT_DIM)
    pr512 = torch.cat([pr1, torch.zeros(1, max(0, 512 - pr1.shape[1]), TEXT_DIM)], 1)
    diag = {}
    if parity:
        x, ex, tok, ray = _rand_inputs(device, dtype, tok_ticks)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            ref = tf(hidden_states=x, timestep=t, encoder_hidden_states=pr512.to(device, dtype), return_dict=False)[0].float()
    if lora:
        from peft import LoraConfig
        torch.manual_seed(seed)  # seeded LoRA A init (kaiming), projector / embeddings below
        tf.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights=True, target_modules=LORA_TARGET_REGEX))
    torch.manual_seed(seed + 1)
    m = WanCond(tf, tok_ticks=tok_ticks, use_tokens=use_tokens, prompt=prompt).to(device)
    for n, p in m.named_parameters():  # trainable = LoRA + new modules, all fp32; everything else frozen in `dtype`
        new = ("lora_" in n) or n.startswith("proj.") or ".patch_embedding.extra." in n or n.endswith(".attn2.gate")
        p.requires_grad_(new)
        if new:
            p.data = p.data.float()
    if lora:
        cen = lora_census(m)
        bad = {k: v for k, v in cen.items() if v != len(m.tf.blocks)}
        if bad or len(m.tf.blocks) != 30:
            raise SystemExit(f"LoRA targets missing: {bad} (blocks {len(m.tf.blocks)})")
    if grad_ckpt:
        tf.enable_gradient_checkpointing()
    if parity:
        on = torch.ones(1, dtype=torch.bool, device=device)
        def run(extra, tk, keep, prompt_):
            m.set_prompt(prompt_)
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                return m(x, extra, t, tk, ray if tk is not None else None, keep).float()
        w512 = run(ex, tok if use_tokens else None, on, pr512)
        w1 = run(ex, tok if use_tokens else None, on, pr1)
        w1_zero_guide = run(torch.zeros_like(ex), tok if use_tokens else None, on, pr1)
        w1_no_tok = run(ex, None, None, pr1)
        m.set_prompt(prompt)
        diag = {"stock_vs_wrapped": (w512 - ref).abs().max().item(), "one_row_shortcut": (w1 - w512).abs().max().item(),
                "guide_inert": (w1 - w1_zero_guide).abs().max().item(), "tokens_inert": (w1 - w1_no_tok).abs().max().item(),
                "ref_std": ref.std().item(), "finite": bool(all(torch.isfinite(v).all() for v in (ref, w512, w1, w1_zero_guide, w1_no_tok)))}
    return (m, diag) if parity else m


def check_parity(diag, tol_rel=0.02):
    """raise unless the init diagnostics hold: finite, stock-equal within tol x output std, guide / tokens exactly inert"""
    if not diag.get("finite"):
        raise SystemExit(f"non-finite output at init: {diag}")
    lim = tol_rel * diag["ref_std"]
    if not (diag["stock_vs_wrapped"] <= lim and diag["one_row_shortcut"] <= lim):
        raise SystemExit(f"init is not stock Wan: {diag}")
    # guide: exactly 0 (zero conv). tokens: the stock text_embedder then runs on a longer sequence, and a GEMM may pick another
    # tiling for the text row (last-bit differences), so a 1e-3 x std bound; the gated term itself is exactly 0
    if diag["guide_inert"] != 0.0 or diag["tokens_inert"] > 1e-3 * diag["ref_std"]:
        raise SystemExit(f"guide / token inputs are not inert at init: {diag}")


@torch.no_grad()
def sensitivity(model, x, extra, t, tok, ray, dtype=torch.bfloat16):
    """mean |v(full inputs) - v(guide off)| and |v(full) - v(tokens off)| on one batch: 0 at init, > 0 once trained (item 4)"""
    B = x.shape[0]; on = torch.ones(B, dtype=torch.bool, device=x.device)
    with torch.autocast("cuda", dtype=dtype):
        full = model(x, extra, t, tok, ray, on).float()
        ng = model(x, torch.zeros_like(extra), t, tok, ray, on).float()
        nt = model(x, extra, t, tok, ray, ~on).float()
    return {"guide": (full - ng).abs().mean().item(), "tokens": (full - nt).abs().mean().item(), "v_abs": full.abs().mean().item()}


def tensor_sha(named):
    """sha256 over (name, raw bytes), sorted by name"""
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


def file_sha(path, n=64):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()[:n]


def code_sha(*paths):
    return hashlib.sha1(b"".join(open(p_, "rb").read() for p_ in paths)).hexdigest()[:12]
