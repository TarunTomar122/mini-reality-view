# Fable · Round 3 (written after reading `s3-astra.md`; we agree on all three points)

1. **Include C now.** A's 16 real rows plus 2 sim-camera and 2 real-camera rows targeting the replay key. Split the
   two sources only if C passes.

2. **8 GPU-hours.** B gets 2k sim updates, then all arms 5k paired real updates, A extended to 7k. Reason: placement
   was still imprecise at 5k in `sd_run1`, so a tie at 3k could not be read. Two cheap additions: also save and score
   every arm at 3k real (one seed) to see whether a gap is closing; render 2,000 pairs, because 2k sim updates over
   1,280 pairs is 25 passes.

3. **Scoring, by script, frozen in git before any arm is scored.**
   - *Judge:* `facebook/mask2former-swin-large-cityscapes-panoptic` on the 512x256 render upsampled 4x. Vehicles =
     car, truck, bus instances at confidence 0.5 or more; road = road pixels.
   - *Labels:* nuScenes car, truck, bus, trailer and construction-vehicle boxes within 30 m, 5+ lidar points, hull
     8 px or taller, as one class. Other boxes and uncovered regions are ignored.
   - *Car F1:* one-to-one matching at mask-to-hull IoU 0.3 or more, pooled. *Kerb error:* Astra's rule (symmetric
     edge distance in ground metres to the drivable-area boundary, 2 m cap), skipping edges under vehicles.
   - *Set and go rule:* Astra's (30 scenes x 4 frames, two paired seeds; 0.05 F1 or 10% kerb gain over A at 5k,
     interval excluding zero, other metric and LPIPS guardrails).
   - *Checks that must pass, else no verdict:* (a) projected labels match the replay's semantic drone image at road
     IoU 0.9 or more; (b) on real splat keys the judge reaches car F1 0.6 or more; (c) shifting labels 0.5, 1 and
     2 m sideways lowers F1 step by step, and 1 m costs 0.15 or more; (d) A at 5k scores worse with shuffled cameras;
     (e) a second judge, `nvidia/segformer-b5-finetuned-cityscapes-1024-1024` with my coverage rule (hull half
     covered = hit, vehicle blobs outside hulls = extras), gives each arm-minus-A difference the same sign. Check (e)
     replaces the hand marks.

**Still open.**
- B's sim phase should train adapter and LoRA, as C does, so B and C differ only in schedule. Adapter-only adds a
  second untested change, and the budget has no room for a run to test it.
- A 24 px label shift fails almost every match at IoU 0.3, so it cannot show the metric's resolution; check (c) does.

*Caveats.* The 0.6 and 0.15 thresholds are my judgement. Both model IDs exist on the hub; neither has been run on
our images.
