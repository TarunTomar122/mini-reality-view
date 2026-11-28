# Next arm after vpriv — design round r1 (28 Nov 02:40 UTC). Astra + Fable; Neon audits facts.
Goal (owner): a SMOOTH and LIVE (causal) chase view, camera-only, accurate cars. Base: frozen A@20k + causal temporal blocks
(council/AGREED-causal-r1.md, A1-A3, note N1). vpriv (both clips private training noise, rollout rho 0.95) is training: stage 1
done at 12k, 16k expected ~03:30-03:45 if the 8k decision (~03:05) is safe. After that the H200-3g slice is idle unless a signed
arm is ready. This round prepares that arm NOW, conditional on vpriv not passing (if vpriv passes: exam first, N1 applies).

## Evidence so far (read PROGRESS.md top entries; raw on the cluster)
- vlive (shared rho 0.95 training noise): jumpier than A at every gated check (16k ID +57.7%, W +14.9%); probes: blocks keyed on
  the shared noise.
- vpriv 2k (informational): vs A ID -8.8%, W +2.3%; vs own cold ID -27.2%, W -4.0%; costs: car F1 -0.0285 (CI to -0.067), road
  -0.023, lingering ghosts 12.0 vs 4.33, early gap +0.016.
- Two-noise probe @8k, t 300, on vs off: private probe noise -12.5% (past content used; improves 2k -> 8k), rollout-like shared
  noise +30% (worsens 2k -> 8k), shared noise but fresh noise on the past ticks only -13.5%. => rollout harm comes from the
  shared noise sitting in the past ticks' features.
- A's off-renders: lower rollout rho makes A itself jumpier (rho 0.5: ID x1.50, W x1.27 vs signed A@20k; 0.95 chosen by A2).
- Timing: 2080 Ti 3.4 s / frame at 25 steps (0.60 s at 4 steps); no merged-LoRA result yet.

## Candidate directions raised (rank, refine or replace; cite papers)
1. Clean past: blocks read past ticks from the finished (last DDIM step / predicted x0) features instead of same-step noisy
   features (Fable 02:30 idea 1; no extra UNet passes). Needs an amendment (AGREED r1 forbids a clean past in training).
2. Rollout noise for the past decoupled from the current tick (e.g. past rings filled from a private-noise pass) — costs passes.
3. Ego-motion-warped memory (Fable 00:40; BEVFormer / FLATTEN): align past features by the known ego shift.
4. Cheap no-training probes first: DDIM eta 1 in late steps; blocks only in early steps; averaged weights; vpriv rolled at rho 0.5.

## Questions (<= 30 lines each, to council/next-arm-r1-<seat>.md)
(1) Which ONE arm should train next if vpriv misses (exact training + rollout recipe, what changes vs AGREED r1, why, cost)?
(2) Which cheap probe(s) on the existing vpriv 8k / 12k checkpoints (<= 1 2080 Ti-hour each) should run BEFORE that arm to
de-risk it, with a pre-registered reading (what result kills / keeps the arm)?
(3) Bars: keep the AGREED gate unchanged? anything to add (N1 already covers ghosts / road for an exam)?
(4) Anything in the evidence above you think is wrong or over-read.
Read-only: write only your answer file; no code edits, no cluster jobs, no commits. End with "PROPOSAL: <one line>".
