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

**Joint dropout.** One draw per clip, defaults `--p_drop_both 0.10 --p_drop_guide_only 0.10 --p_drop_tokens_only 0.10`,
recorded in `args.json` → `dropout`.
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

## Fable 08:30 additions

**Overfit reading.** In order:
- `FAIL` if the round trip has no cars.
- `FAIL` if hybrid car F1 ≤ no-inputs car F1 + 0.02, whatever the loss says.
- `PASS` if both declared gates hold.
- `NOT_REACHED` if the loss ratio is missed while placement passes or the loss is still falling. This is allowed once per gate
  key; a second one for the same key is `FAIL`.
- `FAIL` otherwise.

**Gate key.** A sha of the trainer code, the model recipe (rank, token ticks, tokens on/off, shift, sigma distribution, lr,
warm-up, prompt, Wan id, batch), the frozen base, the VAE and the guide version. Lists, caches, dropout and steps are excluded,
so the gates can run on the 48 S0c drives with dropout off.

**Gate enforcement.**
- Every overfit verdict and every resume check is appended to `runs/wan_gates/gates.jsonl`.
- `MODE=train` and `MODE=extend` refuse to start without both an overfit PASS and a resume_test PASS for their gate key.
- An overfit run stopped before 2,000 updates (signal or job limit) exits 7 (PENDING), so an `afterok` chain cannot treat it
  as success.
- The signal handler only sets a flag.

**Flags.** The dropout flags are renamed `--p_drop_both`, `--p_drop_guide_only` and `--p_drop_tokens_only` (shares
10 / 10 / 10).

**Splits.** `scripts/build_guides.py` (train / dev; sealed drives only with `--unseal`) and `scripts/fit_depth_scale.py`
(train only) now pass their resolved drives through `mrv.wan_plan.check_split_membership`. That's a two-line change in each.

## Astra / Neon 08:30 production fixes

**Decision.** `decide_s1` returns INVALID whenever the evidence is incomplete:
- swap evidence missing or blocked;
- the compare panel incomplete or not finite;
- not a production checkpoint;
- the four mode panels missing.

No 24 h marker is written for INVALID. STOP is only returned for complete evidence that misses a signed threshold.

**Swap donors.** `scripts/wan_swap_preflight.py` (CPU; the panel runs it first and stops on failure) uses
`mrv.wan_plan.plan_donors` to pick each dev receiver's donor.
- **Full mode:** the conventional pairing if it qualifies, else the donor closest in length that is at least as long as the
  receiver.
- **Shared mode:** used only if no full-length donor qualifies. The receiver is cut to the longest qualifying donor's length n,
  and the panel renders an unswapped hybrid receiver run of that same n for the comparison.

Every assignment must have:
- no truncation;
- segments, windows and noise identical to its unswapped reference;
- common placement frames;
- structurally lag-eligible windows.

The rollout reads the plan via `--donors`. Donor-keyed post-swap files are named `<donor>_s<seed>x<receiver>`, so a donor can
serve more than one receiver.

**Panel budget.**
- The reserve is the measured `forecast_final.panel_h` from the newest `timing.json` (or `--panel_estimate_h`); there is no
  zero default.
- The ledger is polled every 30 s while each step runs, and the step is terminated at the cap.

**Resume check.** Both checkpoints must be at step 16, with per-parameter Adam step counts equal and reaching 16, plus the
per-tensor tolerances.

**Extend.** `MODE=extend` exits 8 before using the allocation unless `runs/wan_s1/S1_24H_DECISION.json` holds an EXTEND
verdict. The trainer also refuses `--budget_hours` above 24 without that marker.

**Forecast.** Charges:
- the overfit verdict in all three sampled modes, plus decodes and segmentation;
- the recurring fixed-noise evals (22 × 3 forwards);
- start-up scaled from the timing drive list to the production one (×144/48);
- overfit and resume-test runs as pending unless the gate registry holds a PASS.

**Preempted overfit.** Resubmit with `--export=ALL,MODE=overfit,OUT=<that run dir>`. This reuses the directory with
`--resume`.

**Dropout marginals.** Both 20% drop marginals (guide, tokens) are recorded with every mode comparison and with the swap in
`<name>_panel.json`, and in each rollout file (`train_dropout`).

## Astra S1c fixes

1. **Eligibility.** ID and W `can_pass` (from the signed comparator) are part of the evidence. `can_pass=False` gives
   INVALID with no marker, never STOP.
2. **Scorer-exact preflight.** Lag windows are counted exactly as `temporal_metrics.py` counts them. The range runs from the
   swap to the first restart after it. Every position must be accepted and have a donor key, donor ticks must be consecutive,
   and no restart may fall inside a window (the swap position is itself a restart). Common placement frames also need the
   receiver's own acceptance. The rollout stores `recv_ok` / `recv_hard`, and `score_rgb.py` applies the same rule.
3. **Preflight first.** The panel runs the preflight before parity and before any rendering, using the rollout's actual
   `--window_latents`. All six assignments must pass.
4. **Panel price and trainer reserve.**
   - The panel price is the measured base cost plus one unswapped same-length rollout (2 seeds) per shared-mode receiver.
   - The forecast records `panel_base_h` and `panel_h_total` from the real donor plan.
   - Train and extend reserve `panel_h_total × --reserve_panels`, not a fixed 2 h / 1 h.
   - The cap is checked again right before the 24 h marker is published.
5. **Forecast gate filter.** Gate costs are skipped only for PASS records whose gate key equals the production key (recorded
   by timing).
6. **Parity before training.** `MODE=train` requires the bound A@20k replay-parity PASS. Produce it with
   `slurm/wan_panel.sh parity`.
7. **Resume test.** It checks that leg B saved update 8 before leg three runs.

## Astra S1d fixes

**Window size (item 3).** `--rollout_args` is parsed once by `mrv.wan_plan.parse_rollout_extra`. That function mirrors the
exact option set of `wan_rollout.py`; a CPU test checks the mirror against the rollout source. So abbreviations such as
`--window=1` resolve exactly as the rollout resolves them. The canonical argument list carries `--window_latents` explicitly to
every rollout call, and the same value goes to the preflight. Options outside steps / sampler / shift / cfg / window / sd_lat /
mp4 / dtype are refused, abbreviated or not.

**Trainer reserve (item 4).** `mrv.wan_plan.train_reserve_h` takes the reserve only from the measured `panel_h_total` of a
complete plan: six valid assignments, shared-mode references priced in. There is no legacy `panel_h` fallback. `MODE=train`
and `MODE=extend` refuse otherwise. `wan_forecast.py` writes `panel_h_total` (and `panel_h`) as null when the donor plan is
incomplete.

## Astra S1e fixes (item 4)

**4a, no lowering the reserve.** `mrv.wan_plan.production_reserve_h` is used for every train / extend start, including runs
with overrides.
- `--reserve_panels < 1` is refused.
- An explicit `--reserve_hours` below the measured total is refused; it may only raise the reserve.
- The forecast is validated in every case.

**4b, reserve bound to the panel settings.** `wan_forecast.py --window_latents W --swap_at S` prices the donor plan with the
panel's own settings and records them as `panel_settings`.
- `wan_panel.py` refuses a measured cost made for another width or swap position.
- The trainer does the same via `--panel_window_latents` / `--panel_swap_at`.
- `slurm/wan_teacher.sh` takes one `PANEL_WIN` (default 5) for both the timing forecast and the train / extend reserve.
- Astra's repro (A–F 200/200/180/180/180/180, B ticks 50–56 rejected): width 5 needs no shared reference; width 2 needs one
  180-tick reference at two seeds. This is a CPU test.

## Open points

- **Donor-following rule.** The swap margin (0.10) and lag rule above are mine, not signed.
- **Token disk.** About 78 GB for the train tokens; the cluster quota is unchecked. Pooling would cut it 4x, but the tokens
  would no longer match the shared pass's grid.
- **Underlay-off guides.** These (`--no_underlay`) are refused as model inputs. Ablating the underlay needs a separate cache
  version.
- **Car-weighted loss (Fable c2).** Not implemented.
