A. Launch vmix with `--cache_pending` once the existing pre-update-1 checks pass and the pending gate record re-verifies.
Updates 1–2000 use the signed pure key + jitter recipe. The trainer checkpoints and exits at 2000 unless it verifies the rebuilt cache, including scene-0005 max |latent diff| < 0.01, before update 2001.
SIGNED: A
