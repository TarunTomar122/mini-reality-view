# Long bet: final pick (Fable)

**Pick: C at full dose, one history frame. Not the 16-frame model, not a second frame.**

**Why.** C is the only arm with dev evidence: placement and drift inside the gates, jumps 12-19% lower. Its one failure is
ghosts lasting 5+ ticks, and its cache cannot teach those away: clips are 8 ticks, so own history is at most 6 ticks
deep and a long-lived ghost almost never appears in training. A dose fault, not a method fault.

**Recipe**
- Fork C 4k (optimiser state kept), train to 20k. A to 20k on the same row draws.
- Own rows: whole-drive rollouts (`temporal_rollout.py --split train --method hist`) of 48 train drives, 12 each at
  s = 0, 0.1, 0.3, 0.5. The 6 own rows per batch come from all their ticks.
- Refresh from snapshots 4k, 8k, 12k, 16k (about 8 GPU-h each, sharded). Training continues on the previous cache
  meanwhile.
- Unchanged: 6 own, 6 true, 4 singles; 10% no-history; s ~ U[0, 0.7]; learning rates.

**5k checks** (6 dev drives, both seeds, s = 0.3, against A at the same step). The owner's number: smoothness = the
worse of jumps and wobble against A (4k: 9% better). Beside it: lingering fake cars per drive (4k: +4.4) and the video.

**Stop early**
- Car F1 below -0.03 or late drift above +0.03, two checks running.
- Lingering ghosts at B's level (+9), two checks running.
- Smoothness at 10k and 15k no better than 4k: next is a second history frame.
