"""Run a script with deterministic GPU kernels (tests only): python scripts/det_run.py <script.py> [args...]
Sets cudnn.deterministic / benchmark off / no TF32 before executing the target as __main__ (its __file__ stays the target path,
so fingerprints computed from __file__ are unchanged)."""
import runpy, sys
import torch
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
target = sys.argv[1]; sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
