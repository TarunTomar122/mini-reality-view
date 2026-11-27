# Causal video, round 2: swap (27 Nov ~18:30)

Read council/causal-video-brief.md, then BOTH council/causal-video-s1-astra.md and council/causal-video-s1-fable.md. Converge on ONE
design we implement tonight. New owner constraint: every code change will be reviewed by Astra against the agreed design before it
trains (owner worried about Claude silently building something other than what it says), so the design must be concrete enough
to check line by line.

## Agreed already
- Freeze A@20k (E3 UNet + LoRA + CameraAdapter at runs/longA/ckpt_20000.pt); train only new zero-init temporal layers after
  the 16 spatial transformers; "temporal off" = A exactly, so A@20k's signed dev renders are the cold/control.
- Strictly causal (no future ticks), reset at hard breaks, DDIM 25, CFG 2, signed scorer and compare tools, same dev gate vs A
  (smaller of ID / W reduction > 9.4% with both eligible, car F1 >= A - 0.03, late gap <= +0.03), checks at 2k / 4k / 8k (12k).
- Self Forcing / few-step distillation is a later stage; a 2080 Ti timing run (latency, VRAM) before any "live" claim.

## Differences to settle
1. Window: K = 4 (Astra) vs 8 (Fable) past ticks.
2. Training noise on the past frames:
   - Astra: Diffusion-Forcing style; target t ~ U[1, 961]; prefix (past) frames at low independent noise {50, 100, 200}; 20%
     stale-history swap (most recent past replaced by an older key) to teach distrust; frozen "write" pass with temporal
     residuals off so cached features do not depend on older hidden states.
   - Fable: Live2Diff style; the whole clip shares ONE noise level (past as noisy as the current frame), true keys only, one
     stream cache per DDIM step at inference; argues a clean past re-creates today's copy shortcut.
   Settle with evidence: which is more likely to avoid copying while still smoothing? Can a cheap ablation decide it (e.g. both
   variants for 2k updates each on the A6000, ~2 x 40 min)? Owner rule: "spend more compute testing one good thing than 3 okay
   things" — prefer one design unless the evidence is genuinely split.
3. Attention form: Astra's query-current / K-V-from-frozen-past-features with age bias + null slot vs Fable's own attention with
   relative tick positions, band mask and stream cache. Pick one that is simple to implement and to verify.
4. Inference cost and cache: what is cached, per DDIM step or once per frame.

Write council/causal-video-s2-<your name>.md (<= 40 lines): agree / object per point, then the merged design as an implementation
checklist (files, functions, tensor shapes, masks, noise sampling, loss, what is frozen, logging, smoke tests incl. "temporal off
== A" parity and "changing future cameras does not change past outputs"), budget and gate. Reply with a 5-line summary.
Read-only: no code edits, no jobs, no commits.
