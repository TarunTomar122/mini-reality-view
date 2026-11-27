# A3 swap (27 Nov ~21:50 UTC; still before any 8k number)
Read council/causal-video-a3.md, council/causal-video-a3-astra.md and council/causal-video-a3-fable.md. Agreed by both: vlive
runs unchanged to 16k (control); the A3 arm starts fresh from frozen A@20k at rho 0.95 with MSE + adjacent-tick difference
loss, unchanged bars and A1 timing, T1-T7 + new fixtures, Astra code signature.
Differences: (1) trigger: Astra = no pass at 8k / 12k / 16k AND both safety bars hold at 16k; Fable = no joint pass at 8k AND
the update-8000 probe still shows past_other <= on at t700 and t300 (i.e. A3 could start ~3 h earlier, in parallel with vlive's
12k / 16k). (2) Min-SNR gamma 5: Astra in; Fable out (only reweights t < 147 on this schedule). Note the H200-3g slice budget:
one MRV training job at a time on the 3g slice was agreed with live-VSR (a parallel A3 run would need their OK for a second slice).
Converge in <= 8 lines to council/causal-video-a3-swap-<seat>.md ending "SIGNED: <A3 text>". Read-only. 2-line summary.
