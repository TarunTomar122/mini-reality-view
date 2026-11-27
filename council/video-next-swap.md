# Next video step: swap (ONE round, then converge). Read council/video-next-astra.md and council/video-next-fable.md.

- Astra: causal two-tick camera-token memory (mutual-NN patch matching to the last two ticks' DINO patches, zero-init
  residual MLP before the frozen CameraAdapter; 4k updates; no output feedback). Live-capable.
- Fable: joint multi-frame model (16 zero-init temporal attention blocks in frozen E3, 16-tick clips, PYoCo mixed noise,
  sliding windows of 16 with stride 8 at inference, 12k updates, ~4-5 A6000-h). Not causal (sees ~1 s ahead): fine for
  rendered video, not live.

New input from the owner (06:30): "i hope u have accounted for undertraining cz we might be undertraining". Data: arms
A / B / C were cut 6k -> 4k for the slice budget; B's history-conv weight norm was still growing at the stop (0.85 / 1.27 /
1.37 / 1.55 at 1k / 2k / 2.5k / 4k); a B 2k-vs-4k dev comparison is queued to see whether ghosts / drift shrink with more
training. The owner's priority: "jumps go away, video safe, then further training to make it smooth"; they will watch
side-by-side videos.

Please give your FINAL pick in <= 150 words: Fable's, Astra's, or a merged plan (e.g. run both as two arms if compute
allows), the update budget (in light of the undertraining point), and whether "not causal" is acceptable for the owner's
goal (rendered chase-cam video now; the phone demo is low priority). Write to council/video-next-swap-<you>.md.
