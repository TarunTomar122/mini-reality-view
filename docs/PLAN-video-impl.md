# PLAN-video S1: how to run the Wan teacher and the dev panel

This guide covers the code for stage S1 in `council/PLAN-video.md`: the Wan2.1-T2V-1.3B bidirectional teacher, conditioned on
the chase-view guide and on raw camera tokens. It is revised for Astra's review (`council/codereview-wan-s1-astra.md`) and
for the guide with the flat-ground underlay (415c470). Every model load uses `local_files_only=True` and jobs run with
`HF_HUB_OFFLINE=1`. umT5 is not used.

| File | What it does |
|---|---|
| `mrv/wan_cond.py` | The model: guide channels, gated per-frame token cross-attention, LoRA r32 on 10 targets x 30 blocks (asserted), seeded init. Also init diagnostics, sensitivity probe, guide intake checks and per-tick token rays. |
| `mrv/wan_plan.py` | Torch-free rules: segments, streams, clip candidates, windows, swap snapping, S1 decision, slice-hour ledger. |
| `scripts/wan_tests.py` | CPU tests of `wan_plan`: soft reject, hard break, gap, short tail, candidate counts, decisions, ledger. Run with plain `python3`. |
| `scripts/wan_tokens.py` | Camera tokens taken from the guide's own depth pass (shared), with per-tick Plücker rays from the guide's K / cam2tick. |
| `scripts/wan_token_check.py` | Ray geometry check on one pinned tick: road photo colours vs the guide, with a wrong-camera control. |
| `scripts/wan_cache.py` | Wan-VAE latents of keys and guides: 4-phase whole-segment streams through soft rejects, plus true starts and source planes. |
| `scripts/train_wan_teacher.py` | The trainer, with timing / overfit / resume / train modes and the ledger budget. |
| `scripts/wan_rollout.py` | Renders whole drives in windows, at fixed noise. Covers the ablation modes and the swap (snapped to a window start), and writes RGB (+ SD latents). |
| `scripts/score_rgb.py` | Runs the signed scorer on RGB. Handles donor and cold references (real SD decode), receiver / post-swap scoring, A@20k replay parity, and a fresh roll directory. |
| `scripts/wan_panel.py`, `slurm/wan_panel.sh` | The full dev panel through the signed `temporal_compare.py`, plus the swap check and the S1 decision. |
| `scripts/wan_resume_check.py` | Checks that resumed training equals uninterrupted training. |
| `slurm/wan_teacher.sh` | Modes: timing, overfit, resume_test, train, extend. |

## Order of work

Every GPU step from 4 onwards debits `runs/wan_s1_ledger.jsonl`, the 24 h S1 budget.

1. **Guides.** `scripts/build_guides.py --drives train,dev` writes `data/guides_v1` with the underlay, `src`, coverage and
   `complete` sidecars. Pin a version with `--expect_version` if shards run at different times.
2. **Tokens.** Run on a 2080 Ti, sharded:
   `slurm/py.sbatch scripts/wan_tokens.py --splits train,dev --shard i/12`. Disk use is about 3.7 MB per tick, about 78 GB
   for train. Then check the geometry on a pinned dev tick:
   `scripts/wan_token_check.py --scene <dev drive> --tick <t> --out runs/wan_checks/token_geom`. It must print PASS; look at
   the PNG.
3. **Latent cache.** `slurm/py.sbatch scripts/wan_cache.py --splits train,dev --shard i/12`. Existing files are skipped only
   if every hash still matches.
4. **Timing.** `MODE=timing` writes `runs/wan_timing_<job>/timing.json`. It records:
   - each microbatch's wall time, forward+backward, Adam, data wait and peak allocated / reserved memory;
   - probe, save, a 30-step window, VAE encode / decode, SD encode;
   - an 81-tick rollout and its RGB scoring;
   - a forecast against the 12k floor.

   Choose `--micro` from it.
5. **Bug gate.** `MODE=overfit` (exit 4 = FAIL) and `MODE=resume_test` (`RESUME_CHECK PASS`).
6. **S1.** `MODE=train`, resubmitted as needed. It stops when the ledger plus the 2 h panel reserve reaches 24 h.
7. **Panels.** `slurm/wan_panel.sh runs/wan_s1/ckpt_<k>.pt s1_<k>k check|24h|final`. Training must not hold the slice at
   the same time. Stage meanings:
   - Below 12k updates the result is unresolved.
   - At `24h`: pass, or one 12 h extension (`MODE=extend`) only for car F1 in [0.4156, 0.4456) with every other gate passed,
     otherwise stop and re-council.
   - At `final`: the bar stays at 0.4456.

## Design (decisions and why)

**Guide input.**
- Guide RGB goes through the frozen Wan VAE (16 channels).
- The source map becomes depth-share and ground-share planes per latent cell, and the 4 ticks of each latent frame are folded
  into channels (2 x 4 = 8).
- Input is 16 + 24 = 40 channels. The stock patch conv is frozen; the new conv takes the 24 extra channels, has no bias and
  starts at zero.
- A dropped guide zeroes the latent and both source planes.

**Tokens.**
- They come from the guide's depth network in a single forward with `output_hidden_states`: last layer through the backbone
  layernorm, CLS dropped. That gives 21 x 37 = 777 tokens per view at the guide's 294x518 preprocessing, so the live system
  needs one encoder pass per view for both depth and tokens. The builder records the depth difference against the guide's
  own `DepthNet.raw`.
- Rays use each tick's per-camera K and cam2tick (each camera at its own capture pose), as Plücker coordinates in the chase
  frame.
- Default `--tok_ticks last`: the newest tick of each 4-tick group, 4,662 rows per latent. `ends` doubles the cost (Astra
  item 6).
- Cross-attention is decoupled from the text slot, with per-block zero-init gates and a per-latent-frame block mask. The text
  slot is one zero row; this is tested against the 512-row padded context.

**Streams.**
- Segments break only at hard ticks and tick gaps; soft rejects stay in the codec stream.
- Training clips need every tick ok.
- 85% of clips are mid-stream (5 regular latents) from 4-phase streams, 15% are true 17-tick starts.
- Candidate counts per drive go into `args.json`.

**Objective.** Rectified flow, shift 2.0. The reasoning is in the trainer docstring; the sampling shift is swept at the first
checkpoint.

**Dropout.** Guide 20% (else holes 50%), tokens 10%. Updates 1–2 have none, which makes the guards deterministic.

**Init diagnostics (item 4).** All under the same precision and padded context:
- stock vs wrapped;
- the 1-row shortcut;
- guide and token inertness;
- explicit finiteness.

The rollout refuses a checkpoint of 1k+ updates whose guide or tokens are inert.

**Guards (item 5).** After update 1:
- base and VAE are bit-identical;
- the extra conv, 30 gates and LoRA-B of 8 targets x 30 blocks moved.

At update 2, the projector and attn2 q/k LoRA-B have gradients in all 30 blocks.

**Budget (item 11).** One ledger is debited by every S1 job, by heartbeat (preempted work is still counted). The remaining
hours are checked before each update. A resumed run must end on the same chained data hash as an uninterrupted one.

**Panel (items 9, 10, 12).**
- The A@20k replay through the RGB path must match the signed metrics field by field.
- Four modes at fixed noise go through the signed comparator exactly as specified.
- The swap is scored three ways:
  - lag against the donor's A@20k cold reference, decoded with the real SD VAE;
  - post-swap ticks vs the donor keys;
  - post-swap ticks vs the receiver keys.
- Pre-registered here for council confirmation: donor-following = donor F1 ≥ receiver F1 + 0.10 and median lag start ≤ 1.

## Changes for Astra's second review (`council/codereview-wan-s1b-astra.md`)

**Tokens (item 1).** A drive's tokens count as complete only if:
- the depth-weight hash is non-empty and equals the guide's;
- the shared-depth check passes (finite, at most 1e-3 m).

`MODE=timing` first runs the pinned geometry check with its wrong-camera control and exits 5 if it fails.

**Hash matching and splits (item 2).** The latent and token caches of each drive must record the same guide and enc hashes,
and those must match the files on disk. This is checked at intake and again per drive at rollout.
`wan_plan.check_split_membership` checks every resolved scene id, including explicit names and list files. `guide.py` is
unchanged.

**Sensitivity (item 3).** Token values are perturbed at a fixed context length (views cyclically shifted), next to an init
control (guide conv and gates zeroed). A checkpoint of 1k+ updates must give a finite change above
max(10 × control, 1e-3 × |v|) on each path.

**Timing (item 4).** Forward and backward are timed separately. After an OOM, the fastest microbatch that fitted is restored.
The production prefetch-queue wait is measured.

**Forecast (item 5).** `scripts/wan_forecast.py` runs after the 81-tick scorer and deducts:
- ledger hours;
- the pending overfit gate and resume test;
- recurring probes and saves;
- numbered saves;
- trainer starts;
- `--n_panels` measured full panels.

**Overfit gate (item 6).**
- Both gates are read only at update 2,000 (a shorter job prints OVERFIT_PENDING).
- The update-0 eval loss is stored in every checkpoint.
- The 8 clips are sampled in the fitting microbatch.
- The round-trip car F1 must be positive.

**Panel parity (item 7).** The A@20k replay goes through the real-SD-to-RGB path. The cached pass is bound to the shas of the
reference metrics, the reference rollouts, `score_rgb.py`, `temporal_metrics.py` and the SD VAE files. Any change triggers a
replay.

**Swap (items 8–9).** The check uses paired evidence and blocks unless all of these hold:
- all 12 run ids are present;
- donor and receiver keys are scored on the same frames (receiver and donor ticks both ≡ 0 mod 5);
- each run's windows and noise equal the unswapped hybrid run's, with no donor truncation;
- every drive has lag-eligible evidence.

Exclusions are reported and non-recoveries count as infinity. The rule itself: donor F1 ≥ receiver F1 + 0.10 and median lag
start ≤ 1.

**Ledger (item 10).** One outer heartbeat per allocation (`python -m mrv.wan_plan beat`, started by both slurm scripts)
charges the whole allocation once, including the timing scorer. A job killed without an end beat is also charged its last
interval. Panels refuse to start over budget and stop at the cap. The resume check uses per-tensor tolerances (weights and
Adam moments).

**Decision (item 11).** A verdict needs:
- a production checkpoint (global batch 8, not timing / overfit / test);
- all four conditioning modes at full length with complete 6 × 2 panels.

Below 12k updates the result is unresolved. The 24 h read happens once (marker `runs/wan_s1/S1_24H_DECISION.json`). The final
read requires that marker to say EXTEND, and the bar stays 0.4456.

## Before MODE=train (Fable 07:30)

**Save on signal.** The trainer catches SIGTERM and SIGUSR1. It writes the atomic rolling checkpoint at the next update
boundary and exits cleanly, so a time limit loses less than one update. `slurm/wan_teacher.sh` asks Slurm for USR1 300 s before
the limit (`#SBATCH --signal=B:USR1@300`) and forwards USR1 / TERM to the trainer (it runs in the background under
`run_fg`). Timing is unchanged.

**Joint dropout.** One draw per clip, defaults `--p_both 0.10 --p_guide_only 0.10 --p_tok_only 0.10`, recorded in
`args.json` → `dropout`.
- Marginals: guide 20% (as signed in C9), tokens 20% (signed: 10%).
- The both-dropped case (the CFG / S2 unconditional branch) rises from 2% to 10%.
- The token marginal has to exceed the both share, or the tokens-dropped / guide-kept case (the guide-only ablation) would
  never be trained.

**Overfit controls.** The same 8 clips and noise are also sampled tokens-only and with no inputs, and car F1 is reported for
all three. The gate itself is read on hybrid. Pre-registered:
- the loss ratio is missed while placement passes, or while the fixed-noise eval loss is still falling (each of the last 3
  evals below the one before) → `NOT_REACHED`, exit 6: rerun longer or with lr 3e-4. This is not a bug.
- `FAIL` is exit 4.

The eval history is saved in every checkpoint. If a signal or budget stop ends the run before 2,000 updates, it reads as
`OVERFIT_PENDING`.

**Text slot.** The default is still one zero row. The trainer logs a NOTE (also stored in `args.json` → `prompt_note`)
suggesting an off-cluster umT5 embedding of a plain scene caption via `--prompt_embed` before production.

## Open points

- **Donor-following rule.** The swap margin (0.10) and lag rule above are mine, not signed.
- **Token disk.** About 78 GB for the train tokens; the cluster quota is unchecked. Pooling would cut it 4x, but the tokens
  would no longer match the shared pass's grid.
- **Underlay-off guides.** These (`--no_underlay`) are refused as model inputs. Ablating the underlay needs a separate cache
  version.
- **Car-weighted loss (Fable c2).** Not implemented.
