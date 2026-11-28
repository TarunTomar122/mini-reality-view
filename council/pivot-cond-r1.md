# PIVOT conditioning round r1 — the best way to connect our 6 camera images to a pretrained video model (28 Nov 06:30 UTC)
Owner (verbatim): "make sure we find the best way to connect our images to the video model dont try to reuse and loose".
Runs in parallel with council/pivot-video-r1.md (backbone choice: ~1.3-2B open video model, e.g. Wan 2.1/2.2 1.3B, LTX-Video 2B,
Cosmos-Predict2 2B). Answer for the leading backbones; say where the answer depends on the backbone.

## Task
Input per tick (10 Hz): 6 synchronized nuScenes surround cameras (1600x900 originals; we store 448x252 crops + DINOv2-small tokens
and per-camera ray maps), exact intrinsics / extrinsics, ego pose over time. Output: a game-style chase-camera view (512x256 now),
a virtual camera ~6 m behind and above our car, as a live causal video; cars must be in the right place (car F1, road IoU) and
the video smooth (car jumps, wobble), no lingering ghosts. Camera-only at run time. Training data: 144 nuScenes drives (~20 s each,
~21k usable ticks) with game-render answer keys; 6 dev / 30 exam drives. NO reuse bias: our CameraAdapter (DINOv2-small tokens +
rays -> cross-attention tokens) is only one candidate and must win on merit.

## Questions (each seat <= 45 lines, to council/pivot-cond-r1-<seat>.md; cite papers / results with links)
1. Candidate families, with the strongest published evidence for each (driving / multi-view / novel-view synthesis):
   a) geometric: unproject the 6 views (estimated depth, or learned) and warp / splat them into the chase camera, then condition the
      video model on the warped guide (+ mask) by channel concat / ControlNet (e.g. Gen3C, ViewCrafter, ReconX, StreetCrafter,
      FreeVS, DriveDreamer4D, Cosmos transfer-style control);
   b) attention: multi-view tokens (per-view encoder: backbone VAE latents, DINOv2-L/G, SigLIP 2, ...) with camera embeddings
      (Plücker rays relative to the target camera) fed by cross-attention or token concatenation (e.g. CAT3D, MagicDrive-V2,
      SV4D, Stable Virtual Camera / SEVA, 4DiM);
   c) BEV / 3D latent lifting (e.g. BEVFormer-style or tri-plane) then decode to the target view;
   d) hybrids (warped guide + attention to raw views for what warping misses).
2. For OUR geometry (target camera behind / above the car, mostly seeing the road ahead and around, much of it visible in the 6
   views but from a very different height; our own car occluding the bottom): which family should place cars most accurately and
   stay temporally stable, and why? What depth source would a geometric route use at run time (camera-only: monocular / multi-view
   depth such as Metric3D v2, Depth Pro, UniDepth, VGGT / DUSt3R-family), and its cost per tick?
3. How the chosen conditioning enters the backbone (which layers, zero-init, how much of the backbone to fine-tune, LoRA vs full),
   how it is made causal (per-tick conditioning with a streaming KV cache), and its live cost per tick.
4. A cheap discriminating EXPERIMENT (<= ~4 H200-slice hours or 2080 Ti hours) to choose between the top two families on our dev
   drives with our signed metrics before committing the full training budget, with a pre-registered reading.
End with "PROPOSAL: <family + concrete design> — <one line>". Read-only: write only your answer file; no code edits, no cluster
jobs, no commits. Neon: answer "FACT CHECK" (<= 30 lines, to council/pivot-cond-r1-neon.md): verify the cited methods exist, what
they report, code / weight availability and licences, with links; flag doubtful claims.
