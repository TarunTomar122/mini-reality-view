# Astra · Round 3

1. **Include C now.** The same 1,280 verified, ego-hidden replay pairs serve B and C. Give C the same 16 real rows as A and B, plus two sim-camera and two real-camera rows targeting the replay key. Split those sources only if C passes.

2. **Choose 8 GPU-hours.** Placement was still imprecise at 5k, making a 3k tie hard to interpret. From identical initialization, give B 2k sim updates, then all arms 5k paired real updates. Extend A to 7k for B’s equal-update comparison. Estimated training costs 6.1 GPU-hours, leaving 1.9 for preparation and scoring.

3. **Freeze Mask2Former Swin-Large Cityscapes panoptic.** Use car instances (confidence ≥0.5) and road pixels. Within lidar-supported visible regions out to 30 m, match car masks to projected nuScenes car hulls ≥8 px tall. Maximize one-to-one matches at IoU ≥0.3, breaking ties by total IoU. Pool matches, extras and misses for F1. Compare road edges to projected drivable-area boundaries by symmetric nearest-edge distance in ground metres. Exclude occlusion and crop borders. Cap errors and charge missing edges at 2 m.

   Score style 1 on four frames from each of 30 validation scenes outside the final 30, with two paired seeds. Require both metrics to worsen under a 24 px label shift on trusted keys and under camera shuffling at fixed noise on baseline outputs. Otherwise withhold judgment. Advance for ≥0.05 F1 gain or ≥10% kerb-error reduction over A at 5k, with a paired scene-bootstrap 95% interval excluding zero for the gain, no regression on the other metric and masked splat LPIPS increase ≤0.01.

**Difference from Fable’s round two:** One-to-one matching counts the second car in a merged prediction as missed.
