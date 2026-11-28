"""Profile the S1 data producer's token reads (timing 3756964: producer 10.7 s / update vs GPU 3.6 s). CPU only.
Mimics clip(): per clip, 5 ticks (tok_ticks last) of one drive's token cache [T,6,777,384] f16 + rays, from mmap'd torch.load.
Measures: cold batch of 8 clips (1 thread), the same batch again (warm page cache), a cold batch with 4 / 8 threads, and a plain
sequential read of one token file (MB/s). Prints PROF lines.
  python scripts/prof_producer.py --lists data/lists_s0c48.json
"""
import argparse, json, os, random, threading, time
import torch
p = argparse.ArgumentParser(); p.add_argument("--lists", default="data/lists_s0c48.json"); p.add_argument("--tokens", default="data/wan_tok_v1")
a = p.parse_args()
drives = sorted(json.load(open(a.lists))["train"]); random.seed(0); random.shuffle(drives)
torch.set_num_threads(1)
T = {}
t0 = time.time()
for sc in drives:
    T[sc] = torch.load(f"{a.tokens}/{sc}.pt", map_location="cpu", weights_only=False, mmap=True)
print("PROF open_mmap_s", round(time.time() - t0, 2), "drives", len(T), "shape", tuple(T[drives[0]]["tokens"].shape), T[drives[0]]["tokens"].dtype, flush=True)


def one_clip(sc, rng):
    n = T[sc]["tokens"].shape[0]; j = rng.randrange(0, max(1, n - 20))
    idx = torch.tensor([j + 4 * k for k in range(5)])
    return T[sc]["tokens"][idx].clone(), T[sc]["rays"][idx].float()


def batch(ds, seed, nthreads=1):
    rng = random.Random(seed); jobs = [(d, random.Random(seed * 31 + i)) for i, d in enumerate(ds)]
    t0 = time.time()
    if nthreads == 1:
        out = [one_clip(d, r) for d, r in jobs]
    else:
        out = [None] * len(jobs)
        def w(k):
            for i in range(k, len(jobs), nthreads):
                out[i] = one_clip(*jobs[i])
        th = [threading.Thread(target=w, args=(k,)) for k in range(nthreads)]
        [x.start() for x in th]; [x.join() for x in th]
    mb = sum(o[0].numel() * 2 + o[1].numel() * 4 for o in out) / 1e6
    return time.time() - t0, mb


for rep in range(3):  # cold: fresh drives each time
    ds = drives[rep * 8:(rep + 1) * 8]
    s, mb = batch(ds, rep); s2, _ = batch(ds, rep)
    print(f"PROF cold_1thread_s {s:.2f} warm_same_s {s2:.3f} MB {mb:.0f}", flush=True)
for nt, rep in ((4, 3), (8, 4)):
    ds = drives[rep * 8:(rep + 1) * 8] if (rep + 1) * 8 <= len(drives) else drives[-8:]
    s, mb = batch(ds, 100 + rep, nt)
    print(f"PROF cold_{nt}threads_s {s:.2f} MB {mb:.0f}", flush=True)
f = f"{a.tokens}/{drives[-1]}.pt"; t0 = time.time(); n = 0
with open(f, "rb") as fh:
    while True:
        b = fh.read(64 << 20)
        if not b:
            break
        n += len(b)
s = time.time() - t0
print(f"PROF seq_read_MBps {n / 1e6 / s:.0f} ({n / 1e6:.0f} MB in {s:.1f} s)", flush=True)
