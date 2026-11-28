# Astra vmix review of `3a8ed08`
- **2 closes.** The trainer hashes and loads one A@20k buffer. The gate compares the full A1 decision returned by the verifier with the decision parsed from the bytes it records.
- **4 closes once pinned.** The suite hashes the pins file. The verifier checks the reference drive, path and digest against it before measuring whole-drive parity. The current `PENDING` value blocks both gate paths.
- **5 closes in code.** All six full-file dev digests are pinned. Seed verification requires five matching ticks and hard flags, at least one non-hard tick, and max latent difference below 0.01 across every non-hard tick.
- **T4 keeps exact zero.** Both calls now gather the same four rows at tick 6. PyTorch uses `warn_only=True`, so the full-model H200 result still decides whether the fix works.
- The pending-cache gate checks digest length only. The requested passed-cache gate also verifies the digest format and the reference bytes. Cluster results remain unverified here.
SIGNED: if the committed parity pin matches the independent scene-0005 render from job 3756349, cache_check passes, the clean suite passes on the final hashes with H200 T4 exactly zero, the passed-cache gate record re-verifies, and the 100-update timing records step time and peak memory with the frozen check passing.
