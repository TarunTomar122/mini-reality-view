# Astra · Round 4

1. **Accept.** Train both the adapter and LoRA during B’s 2k sim updates, as C does on sim rows. Freezing LoRA in B would make a B–C difference harder to attribute to the training schedule.

2. **Accept.** On trusted keys, shift labels sideways on the ground plane by 0, 0.5, 1 and 2 m, then reproject and score against fixed predictions. Require pooled vehicle F1 to fall at each step and to drop by at least 0.15 from 0 to 1 m. A single 24 px shift can drive F1 near zero while revealing little about sensitivity to smaller placement errors. The 0.15 drop is a proposed pass threshold.
