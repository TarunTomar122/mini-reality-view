# Council round 2: converge

Both seats wrote round-1 plans: `council/s1-astra.md` and `council/s1-fable.md` (brief: `council/sim2real-brief.md`).
They agree on ScenarioNet nuScenes replays first, skipping MetaDrive procedural maps, CARLA only after a positive pilot,
and a cheap pilot before any big run. They differ on, at least:
- training recipe: sim pretrain -> real fine-tune (the owner's idea) vs keeping ~20% sim rows in the mix with their own
  style token ("twin keys": a ScenarioNet-rendered complete key of the same real moment, paired with the real cameras);
- the geometry metric (car-placement F1 / kerb thresholds vs projecting the nuScenes map + 3D boxes into the drone view
  and comparing with a segmenter on our renders);
- pilot size, go/no-go rule and budget (~3.6 vs ~8 GPU-hours).
New facts since round 1: the low-drone lidar model scores L1 23.9 / LPIPS 0.348 on 30 val scenes (mean image
30.1 / 0.615); sd_run1 is at ~8k of 40k steps and will be paused at 10k to add phone-style token copies, then continue.
Fable flagged two things to verify first: per-camera FOV (nuScenes intrinsics ~65 deg for five cameras, ~89 deg for the
back camera; our test scripts used 70 deg for all) and the ScenarioNet ego origin (rear axle; length/width swap).

## Your job
Read BOTH round-1 plans (and the brief). Critique the other seat's plan honestly: what is right, what is wrong or
unsupported, what is missing. Then write ONE converged plan that you would sign: the single recipe we should run first
(data, model changes, training phases, steps, mixing, style tokens), the geometry metric (exact definition, how to
validate it), the pilot with its go/no-go rule and GPU-hours, and the order of work. Where you still disagree, say so in
one line each and say what evidence would settle it. Keep citations from round 1; add new ones only if you checked them.

## Output
Write ONE file `council/s2-<your seat>.md` (astra or fable), at most ~1.5 pages: (1) critique of the other plan,
(2) converged plan, (3) remaining disagreements. Do not edit any other file, run jobs or touch the cluster.
Finish by stating the file path you wrote.
