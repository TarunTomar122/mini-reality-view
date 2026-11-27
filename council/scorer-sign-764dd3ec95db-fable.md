# Scorer signature 764dd3ec95db: Fable

**SIGN** for non-swap rollouts (the six held decision metric jobs, then every later method scored once on it). All five of my check-4 conditions hold. No further round from me.

**What I checked (read-only; the six `v4_*_dev.json` files copied from the cluster, nothing run there)**

- The file is the one named: `git show fded34e:scripts/temporal_metrics.py`, the working copy at HEAD and the cluster copy all hash to 764dd3ec95db.
- (1) Ghosts: per-tick counts and lifetimes are stored; `ghosts`, `ghost_growth` and `ghost_ge5` are in `--compare` (`temporal_metrics.py:64`, `:344-346`, `:373-376`).
- (2) Tick asserts: cold tick list must equal the rollout's (`:357`); the drift window needs 170 consecutive ticks with no restart (`:352-353`). On all 60 dev runs that have a drift window, ghost growth recomputed from the tick-keyed counts equals the stored position-based value, so position 150 is tick 150.
- (3) Stamp: all 72 dev runs carry `764dd3ec95db` and `iou`. `--compare` refused all four cases I fed it: other hash, other link mode, missing hash, mixed hashes in one file.
- (4) Controls: I re-ran `--compare` locally with this exact file. Every figure in the request table reproduces, intervals included. Ordering holds: key video ID 0.000 / W 11.4, VAE 0.064 / 13.7, fresh noise 0.157 / 41.1 (worst on both, intervals exclude zero).
- (5) HEAD is identical to fded34e.

**Two conditions on how the outputs are read (no code change, no re-sign)**

1. **"Ghosts not growing" is read as the paired difference against independent, never the raw number, and on lingering ghosts as well as growth.** Raw growth is set by the scene (VAE round trip of the keys alone: -3.1 on scene-0285, +2.7 on scene-0393). Growth by itself is also weak: it did not separate warm 0.5 from independent on dev (+1.29, interval [-0.51, +3.82]), while >= 5-tick ghosts did (+27.7 [+13.3, +42.4]). A method does not clear the gate on growth alone.
2. **State the drive count beside every late gap and ghost growth figure.** A missing cold file or a broken chain drops a run silently (`:355`); on the decision set the count should be 22. If it is lower, find out why before reading the gate.

**Notes**

- Flips do not rank the controls (VAE round trip 0.208 vs independent 0.191, interval spans zero). Keep flips descriptive, as the plan already has it (secondary); do not use it as a tie-break.
- Lag block: the reset fix is in (`:394-400`), so I sign it for reporting under the interim rule. A run that has eligible windows but does not recover before the reset counts as not recovered (infinity) rather than excluded; that can only hurt a method, so I accept it. The gating position stays open as agreed.
- The stamp is a whole-file hash, so a later edit inside the swap block will also make `--compare` refuse to mix old and new outputs. Non-swap outputs stay valid on 764dd3ec95db; re-score the swap rollouts on the new hash and compare them only with each other.
