# Signature request: temporal scorer content hash 764dd3ec95db (git fded34e, scripts/temporal_metrics.py)

Implements Fable's five conditions (drift check 4) and Astra's check-4 fixes: placement on manifest tick % 5 (dev
placement ticks verified, e.g. scene-0043_s0: 0, 5, 10, ... 22 ticks); ghost tracks with lifetimes, per-tick ghost
counts, ghost growth (late 150-169 minus early 5-24 window) and >= 5-tick ghosts compared; cold-tick and 170-consecutive-
tick asserts; scorer hash + link mode stamped and --compare refuses mixed / mismatched versions; lag windows only
before the first post-swap reset, 5 consecutive distinct donor ticks; interim lag outcome rule (median start <= 1 pass,
> 3 fail, else undecided -> owner). Linking = signed raw IoU >= 0.5 (coverage 65.6% of 5,112 edge cars, reported).

Dev controls re-run on exactly this version (runs/troll/v4_*_dev.json; vs independent; paired drive bootstrap):
| method | ID | W | flips | car F1 | road | ghosts/tick | >=5-tick ghosts/run |
|---|---|---|---|---|---|---|---|
| key video | 0.000 vs 0.107 | 11.4 vs 30.0 | 0.00 vs 0.19 | 1.00 vs 0.48 | 1.00 vs 0.85 | 0.00 vs 1.61 | 0.0 vs 2.4 |
| VAE round trip | 0.064 vs 0.110 | 13.7 vs 29.4 | 0.21 vs 0.19 | 0.75 vs 0.48 | 0.93 vs 0.85 | 0.82 vs 1.61 | 0.0 vs 2.4 |
| fresh noise | 0.157 vs 0.102 | 41.1 vs 30.1 | 0.24 vs 0.19 | 0.47 vs 0.48 | 0.85 vs 0.85 | 1.69 vs 1.61 | 0.4 vs 2.4 |
| warm 0.7 | 0.083 vs 0.101 [-0.027, -0.010] | 30.8 vs 29.8 | 0.17 vs 0.19 | 0.42 vs 0.48 [-0.113, -0.014] | 0.80 vs 0.85 | 2.25 vs 1.61 | 10.3 vs 2.4 |
| warm 0.5 | 0.067 vs 0.091 | 35.0 vs 29.8 | 0.14 vs 0.19 | 0.30 vs 0.48 | 0.69 vs 0.85 | 3.73 vs 1.61 | 30.1 vs 2.4 |
Ordering required by the stop rule holds (key video / VAE best on ID and W, fresh noise worst). Late LPIPS gap: VAE
-0.147, warm 0.7 +0.086 [+0.038, +0.137], warm 0.5 +0.281, fresh +0.007.

Question: do you SIGN 764dd3ec95db for non-swap rollouts (the six held decision metric jobs: independent and warm 0.7,
30 decision drives x seeds 0/1, then every later method scored once on it)? The lag block is reported, not gated, until
the end-vs-centre question is settled (interim rule above). Reply SIGN / NO + reason in one or two lines, to
council/scorer-sign-764dd3ec95db-<you>.md.
