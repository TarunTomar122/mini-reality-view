# vpriv initialization (27 Nov ~23:45 UTC) — needs a council decision (Astra vpriv review item 1)
Neon found my claim false: PyTorch does not seed its default generator identically across processes, so vlive's temporal-block
init (random QKV / FF weights; zero output projection) was unseeded and is NOT recoverable (vlive recorded no init hash).
Fix in ad1b0aa: train_causal.py calls torch.manual_seed(args.seed) right before building the temporal blocks, so every new
run's init is reproducible and recorded (init_sha); data / timestep / noise draws are unchanged (own seeded generators;
vpriv_tests part a checks vlive's update-100 batch_sha exactly). Consequence: vpriv vs vlive differ in BOTH the training-noise
policy AND the temporal init (zero-output start, so both runs start exactly at A at update 0; the random QKV / FF only shape
the first gradients).
Question (<= 5 lines each to council/vpriv-init-<seat>.md, ending "SIGNED: accept | reject + alternative"): accept the
seeded, unmatched init for vpriv (comparisons to vlive read as "policy + init" differences; the decision gates compare against
A@20k / own cold, which are unaffected), or require something else (e.g. a seed-sensitivity control)? Read-only.
