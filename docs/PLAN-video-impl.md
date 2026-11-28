# PLAN-video S1: how to run the Wan teacher and the dev panel

This guide covers the code for stage S1 in `council/PLAN-video.md` (signed plan plus owner decisions): the Wan2.1-T2V-1.3B
bidirectional teacher, conditioned on a chase-view guide and on raw camera tokens. All model downloads are read from
`~/mini-reality-view/hf` with `local_files_only=True`, and jobs run with `HF_HUB_OFFLINE=1`. The umT5 text encoder is not
used.

| File | What it does |
|---|---|
| `mrv/wan_cond.py` | The conditioned model: guide channels, camera-token cross-attention, LoRA. Also the latent, mask and geometry helpers. |
| `scripts/wan_tokens.py` | Builds the camera-token cache, `data/wan_tok_v1/<scene>.pt` (tokens of the frozen Depth Anything V2-Small ViT plus chase-relative Plücker rays). |
| `scripts/wan_cache.py` | Builds the latent cache, `data/wan_cache_v1/<scene>.pt`: Wan-VAE latents of the answer keys and guides, as 4-phase whole-segment streams plus true-start clips. |
| `scripts/train_wan_teacher.py` | The S1 trainer. |
| `scripts/wan_rollout.py` | Renders whole drives with the teacher. Writes RGB for `scripts/score_rgb.py`, plus optional SD latents and an MP4. |
| `slurm/wan_teacher.sh` | Runs the trainer on the H200 slice. Modes: timing, overfit, train, extend. |

## Order of work

1. **Guides.** `mrv/guide.py` (written separately) produces `data/guides_v1/<scene>.pt` for the 144 train drives and the 6
   dev drives. Check one pinned drive by eye first.
2. **Token cache.** Run on 2080 Tis, sharded; it needs the nuScenes devkit and `data/nuscenes_full`:
   `sbatch --gres=gpu:nvidia_geforce_rtx_2080_ti:1 -J wtok slurm/py.sbatch scripts/wan_tokens.py --splits train,dev --shard i/12`
   Use the same `--encoder` checkpoint as `mrv/guide.py`, so that one encoder pass per view serves both depth and tokens at
   run time. The default is `mrv/guide.py`'s primary, `depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf`, which
   is already cached (S0b loaded it). Disk use is 2.65 MB per tick: about 56 GB for train and about 60 GB with dev.
3. **Latent cache.** Run after the guides: `slurm/py.sbatch scripts/wan_cache.py --splits train,dev --shard i/12`. It writes
   about 6.5 GB for train. A missing key stops the job, so no grey frames ever enter the video codec.
4. **Timing.** `sbatch ... --export=ALL,MODE=timing slurm/wan_teacher.sh` runs 30 updates. Read `step_s`, `data_s`,
   `wait_s` and `mem_gb` in `log.jsonl`. If batch 8 runs out of memory, append `--micro 4` (2-step accumulation, same
   update). Then restate the 24 h budget as a number of updates.
5. **Bug gate.** `MODE=overfit` trains on 8 fixed clips for 2,000 updates with no dropout. The loss must collapse;
   otherwise there is a bug.
6. **S1.** `MODE=train` writes to `runs/wan_s1`, can be resubmitted with the same line, and stops at 24 h cumulative.
   Verdicts come only after at least 12k updates (A1). `MODE=extend` is the one 12 h extension, run only on the council's
   reading.
7. **Dev panel.** Run at the numbered checkpoints, 5k-style cheap checks, and the final checkpoint:
   `slurm/py.sbatch scripts/wan_rollout.py --ckpt runs/wan_s1/ckpt_12000.pt --split dev --seeds 0,1 --out runs/wroll/s1_12k_dev --mp4 --mp4_keys`
   Then score it: `slurm/py.sbatch scripts/score_rgb.py --rgb runs/wroll/s1_12k_dev --out runs/wroll/m_s1_12k_dev.json`.
   The ablation uses the same checkpoint and the same noise with `--cond guide | tokens | none` and separate `--out`
   folders. The camera-swap check uses `--swap --swap_at 50`. The sampling-shift sweep runs on one drive with
   `--drive_list one.txt --shift 1.5 | 2 | 3 | 5`.

## Design choices

**Guide input.** The guide RGB video goes through the frozen Wan VAE exactly like the target, giving 16 latent channels. The
validity mask is area-pooled 8x8, and the 4 ticks of each latent frame are folded into 4 channels (Wan-I2V layout; a
single-frame first latent repeats its tick 4 times). Together that is 20 extra input channels.

**Patch embedding.** `embed(x) = stock_conv(x[:, :16]) + new_conv(x[:, 16:])`. The stock conv is frozen; the new conv has
no bias, starts at zero and is trained fully. So step 0 is exactly stock Wan.

**Camera tokens.** These use the frozen ViT-S/14 of Depth Anything V2-Small: last-layer patch tokens at 448x252, giving
576 unpooled tokens per view and 3,456 per tick. Each token gets a Plücker ray in the chase-camera frame. A trainable
projector (384 to 1024 to 4096, plus ray MLP, view and tick-slot embeddings) feeds the stock frozen `text_embedder`, which
reads the 4096-d text width.
- **Cross-attention.** It is decoupled from the text slot: `to_out(Attn(q, text) + gate * Attn_f(q, tokens))`. The gate is
  per block and per channel, and starts at zero.
- **Per-frame alignment.** Each latent frame attends only to the tokens of its own ticks, via a `[B*F]` reshape (equivalent
  to a block-diagonal mask).
- **Ticks per latent.** `--tok_ticks ends` is the default: the first and newest tick, 6,912 tokens per latent. `last` is
  3,456 and `all` is 13,824.
- **Text slot.** It holds one all-zero row. Stock Wan pads every prompt with zero rows, and 512 identical rows attend
  exactly like one.

**Training clips.** 85% are mid-stream clips: 5 regular latents, 20 ticks, cut from whole-segment streams. Each segment is
encoded in 4 phases (start offsets 0 to 3), so the causal VAE always has real history. The other 15% are true starts:
17-tick clips, first latent a single frame. This follows Fable 06:35 b2 and c3.

**Objective.** Rectified flow, as Wan uses: `x_s = (1-s) x0 + s eps`, target `eps - x0`, timestep `1000 s`. Sigma is drawn
shifted-uniform with a shift of 2.0. The reasoning is in the trainer's docstring:
- Wan uses 3.0 at 480p.
- The SD3 square-root token rule gives 1.7 when scaled by spatial tokens per frame, and 0.84 when scaled by whole-clip
  tokens.
- We keep more high-noise mass than the rule because car placement is decided there.

The sampling shift is swept on one dev drive at the first checkpoint.

**Dropout.** Independent per clip: guide 20% (otherwise random latent-grid holes with probability 0.5), tokens 10%, both
about 2% (the CFG unconditional row).

**Trainable parts.** LoRA rank 32 on attn1 and attn2 (q, k, v, out) and the FFN in all 30 blocks (about 44M parameters).
The new modules add about 6M. The base transformer (bf16) and the VAE are frozen. The first-update guard checks:
- the base and VAE are bit-identical after a real optimizer step;
- the extra conv, the gates and LoRA-B moved;
- attn2 q/k LoRA did not move. This is expected while the gates are 0 and the text slot is one row.

A stock-parity check runs at init.

**Rollout.** For each segment, the guide is encoded as one stream. Latents are generated in windows of 5: the first is a
true start, later ones are mid-stream, and an overlapping tail window covers the end. The concatenated stream is decoded
once. `restart=True` is set on each window's first new tick: no information crosses a window border, so the drift check is
not computed for S1, which is bidirectional.

## Memory and speed (estimates, unmeasured until `MODE=timing`)

These are for the H200-3g slice (71 GB, about 3/7 of an H200: roughly 400 dense bf16 TFLOP/s peak, assumed about 40%
achieved).

| Item | Estimate |
|---|---|
| One update, batch 8 (8 x 2,560 video tokens), forward | DiT about 53 TFLOP; token K/V at `ends` (8 x 5 x 6,912 = 276k context tokens through 30 blocks' k/v plus the 4096-d detour) about 85 TFLOP |
| Whole update | about 3x forward with frozen-base backward and per-block checkpointing |
| Step time at `ends` | about 2-2.5 s |
| Step time at `last` | about 1.5 s |
| Step time, guide only | about 0.9 s |
| Updates in the 24 h budget | about 35-45k at `ends` (the 12k floor fits easily) |
| Weights | 2.6 GB |
| Optimizer state | about 0.6 GB |
| Checkpointed block inputs | about 2 GB |
| Token path | projector activations about 6 GB, text-embedder output and its gradient about 2 GB, per-block K/V about 2 GB during recompute |
| Expected peak, batch 8 | 20-30 GB (fits; `--micro 4` if not) |
| Data per update | about 210 MB of memory-mapped tokens, prefetched in a thread; check `wait_s` |
| Dev panel, per window | 30 UniPC steps x (1-2 rows) x 5 latents, about 5 s |
| Dev panel, whole | about 1 min per 200-tick drive and seed, about 15 min for 6 drives x 2 seeds |

## Open points

- `scripts/score_rgb.py` globs only `*_s0.pt`, while the signed comparison pairs seeds 0 and 1. It needs `*_s*.pt` before
  the paired panel, and Fable's guard against `--cold` / `--lag_ref`.
- The guide file format is taken from the brief (uint8 rgb, mask, ticks). Mask values are accepted as bool, 0-1 or 0-255.
- Starting the projector: with the gate at 0 it receives no gradient until the gates move (Flamingo-style). This is
  expected, and `gate_norm` is logged.
- The guide files also carry inverse depth (`invd`). Fable's C9 listed inverse depth and age as guide planes; they are not
  fed yet. Adding them would mean more folded channels in `N_EXTRA` and a cache rebuild.
- Car-weighted loss (Fable c2) is not implemented. It needs key car masks on the latent grid for the train drives.
