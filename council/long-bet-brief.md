# ONE long bet (owner decision, 27 Nov ~06:50). ONE round + converge. This replaces the "next video step" round.

Owner, in their words: "jumps gone, video safe (no ghost cars), then further training to make it smooth"; "going forward
spend more compute testing one good thing then 3 okay things"; "if this works then we will figure out how to get rid of
fake cars later"; "if something is proven in research and doesnt work for us then maybe we are not doing it correctly
instead of jumping to a new thing"; "ask it if c is the right thing or not - that decision is on the council".
=> Train ONE candidate long (~20k updates, ~5 h on one A6000) + arm A (no memory) at the same length as the control.
During the run: every 5k, the 6 dev drives + one side-by-side video. ONE full 30-drive exam + one council review at the
end. Selection now weighs SMOOTHNESS first; lingering fake cars are measured and shown but do not block (separate step).

## Evidence (scorer 764dd3ec95db; vs A 4k independent; 6 dev drives x seeds 0/1; council/s_lock_B.json, s_lock_C.json)
| arm @4k | ID (A=0.099-0.109) | W (A=31-32) | car F1 diff | recall diff | own late LPIPS gap | late F1 gap | >=5-tick ghosts / run vs A |
| B s=0..0.5 | 0.081-0.092 (-9..-23%) | 28.3-30.0 (-5..-9%) | -0.034..-0.069 | -0.017..-0.055 | +0.044..+0.073 | -0.039..-0.070 | +7.6..+14.2 |
| C s=0..0.5 | 0.088-0.093 (-12..-19%) | 28.1-29.2 (-9..-11%) | -0.011..-0.020 | -0.007..-0.018 | +0.007..+0.010 | +0.001..+0.007 | +3.9..+6.0 |
C passes every dev clause except the lingering-ghost clause (ghost_ge5); B fails placement, recall, drift and ghosts.
Warm start 0.7 (decision set): ID -22.5%, W +3%, car F1 0.258 vs 0.317, drift +0.058, ghosts 13.7 vs 5.1 -> rejected.
Undertraining: B's history-conv norm 0.85 / 1.27 / 1.37 / 1.55 at 1k / 2k / 2.5k / 4k (still growing); C's ended at
1.16. All arms were cut 6k -> 4k for budget. C's own-history dose was small (128 clips x 8 ticks, caches at 1k / 2.5k).
Fidelity round (council/fidelity-*.md): Astra - one-frame history lacks motion (GameNGen: 64 frames, biggest gain 1->2);
Fable - B never sees its own errors in training, so it learns to keep the last frame; fix = C at full dose (whole-drive
own rollouts, refreshed). The C-vs-B gap above supports Fable's reading.
Also on the table (both reviewers' earlier swap): a joint 16-frame video model (code written: mrv/video.py,
scripts/train_video.py, scripts/video_rollout.py; not causal).

## Question - give ONE final pick and recipe (<= 250 words)
Which arm to train long (C, a modified C, the 16-frame model, other), with which recipe changes (e.g. own-rollout share,
refresh schedule and depth, more history frames, history dropout), the update count, the 5k cheap checks (what number
the owner sees), and what would make you stop early. Write to council/long-bet-<you>.md.
