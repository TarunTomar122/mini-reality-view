# BLOCK

1. **Camera-only eligibility.** Replace P2’s final-model condition with “Audit detector alignment, per-camera transforms and 10 Hz box/grid propagation end to end. Use fixed calibration and causal camera-derived motion throughout. After any motion-source change, rebuild affected caches, pass the 0.397 hull gate and repeat renderer adoption gates with the deployed feed.” Hull F1 alone cannot validate the renderer after that change.

2. **X3 control.** Replace the sentence beginning “Base frozen” with “Freeze the entire temporal winner, retaining its locked inference settings, and train only the map branch. Compare against that same winner with the branch disabled and identical noise.” Arm A’s 2,000 extra updates cannot control 3,000 updates on B/C.

3. **Oracle attribution.** Replace the oracle bullet with “Report paired rendered F1 with predicted versus annotated boxes, holding cameras, noise and any grid fixed. A low oracle-input score alone cannot distinguish branch capacity from sensitivity to the change in inputs.”

4. **Winner selection.** Replace the grid bullet with “Select the sole passing arm if only one passes. If both pass, select X2 when either stated incremental threshold passes with positive lower bound and the other point difference is nonnegative. Otherwise select X1.” The existing tie rule can reverse a qualifying road gain.

5. **Confirmation.** Replace “Confirmed once on the final 24” with “Lock the temporal winner, map, comparators and gates before either round opens the final 24, then confirm jointly. If temporal confirmation already influenced selection, reserve 24 fresh drives excluded from calibration and both rounds’ decisions.”

6. **Cache coverage.** Append to P1 “Verify map coverage for every X3 singles draw from all training drives excluding dev, plus every eligible dense tick for the selected box/grid feed before X3.” P1’s 144 training drives cover only the dense subset.
