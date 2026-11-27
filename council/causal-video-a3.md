# Amendment A3 pre-registration (27 Nov ~21:40 UTC; BEFORE any 8k number exists)

Context: council/review-20261127-2130-{astra,fable,neon}.md. All three: keep training runs/vlive at rho 0.95 under A1; the
copy probe suggests the blocks read the shared past noise (noise share 1.17 t300 / 1.52 t700 at 2k). Offline probe
scripts/noise_probe.py on ckpt_2000 is running (arms on / off / noise_only / content_only / neither + Fable's difference
loss), results ~22:00 in runs/probe/noise_2000.json. Fable proposes pre-registering A3 now with the next arm "rho 0.95 +
difference loss + Min-SNR"; a cleaner-past arm would need a full council round.

Question (<= 12 lines each, to council/causal-video-a3-<seat>.md, ending "SIGNED: <A3 text>"):
1. What, exactly, triggers A3 (e.g. the 8k / 12k gated outcome under A1, and / or the noise probe showing content_gain <= 0)?
2. What runs under A3 (Fable's arm or another), from which checkpoint (fresh from A@20k or continue vlive), with what change
   (loss formula; Min-SNR gamma; any change to rho), what tests (T1-T7 + new fixtures) and whose signature, and what bars
   (unchanged AGREED bars; A1 timing)?
3. Does anything about the CURRENT run change before 8k? (Default: no.)
Read-only. Reply with a 2-line summary.
