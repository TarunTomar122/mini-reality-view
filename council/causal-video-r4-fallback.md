# Causal video, round 4 (27 Nov ~20:20 UTC): pre-register the fallback BEFORE the rho 0.8 scores land

Facts (signed scorer, 6 dev drives x 2 seeds, renders verified from reviewed code + A@20k digest):
- All tests T1-T7 PASS on the final code (logs2/ctests-h200-3755243.out, ct2-3755181.out).
- rho 0.5 (frozen A, blocks off, eps_tick = sqrt(0.5) z0 + sqrt(0.5) z_tick) vs signed A@20k (rho 1.0 shared noise):
  ID ratio 1.496 (0.1572 vs 0.1051, CI of diff 0.033..0.073), W ratio 1.270 (37.54 vs 29.55) -> fails the <= 1.20 rule.
- rho 0.8 scores land in ~10 min (tmet2-Aoff-0.8); the backup launcher then re-runs the ordered rule.
- AGREED r1 rule: if neither 0.5 nor 0.8 meets, "back to council". The owner is asleep; he asked that nothing be judged on
  short runs and that work continue overnight.

Question (Astra and Fable, <= 12 lines each, to council/causal-video-r4-<seat>.md): pre-register ONE fallback that applies
automatically if rho 0.8 also fails, signed by both now (before the 0.8 numbers are known). Options to weigh:
 (a) extend the ordered rule to rho 0.9 (then 0.95), same 1.20 test, renders made the same way (cost ~45 min of 2080 Tis);
 (b) Fable's progressive AR(1) noise across ticks (PYoCo, arXiv 2305.10474): eps_t = a eps_{t-1} + sqrt(1-a^2) z_t, with a
     chosen by the same off test (needs a small code change in trainer + rollout -> re-test T1-T7 + Astra review);
 (c) Fable's earlier vote: train at rho 0.8 anyway under an override (bars unchanged), accepting a jumpier cold baseline;
 (d) something else.
Also say: if 0.8 MEETS, anything to change before training starts? Read-only. Reply with a 3-line summary.
