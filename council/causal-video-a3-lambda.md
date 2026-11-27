# A3 last point: the difference-loss weight lambda (27 Nov ~22:00 UTC; A3 cannot launch before ~23:30)
Agreed (council/causal-video-a3-swap-astra.md + -fable.md): A3 = fresh arm from frozen A@20k at rho 0.95, loss = MSE +
lambda x adjacent-tick difference loss over pairs i = 2..15, no Min-SNR; starts only after vlive finishes 16k with no joint
pass and both safety bars met, same H200-3g slice; probes diagnostic.
Open: lambda. Astra: fixed 1. Fable: computed by a script (the ratio that makes the two loss terms comparable), capped at 10;
argues lambda 1 could leave the arm nearly identical to vlive.
New evidence (scripts/noise_probe.py on vlive ckpt_2000, pulled/noise_2000.json): t300 on .0459 / off .0919 / noise_only
.0374 / content_only .3688 (fresh past noise -> 4x worse than blocks off); difference loss diff_on .0470 vs diff_off .0984.
Answer in <= 6 lines to council/causal-video-a3-lambda-<seat>.md ending "SIGNED lambda: <rule>" (exact formula / value, when it
is computed, and what is recorded). Read-only. 1-line summary.
