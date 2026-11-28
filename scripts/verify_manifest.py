"""F2 (council/s1-timing-r1-CONVERGED.md item 3): ONE CPU job hashes every latent cache (data/wan_cache_v1), token cache
(data/wan_tok_v1), guide (data/guides_v1) and enc (data/enc_tpv10) file of the train + dev drives in --lists, plus the cache /
token / guide sidecars and the Wan HF snapshot files the trainer loads (transformer/, vae/, model_index.json), and checks each
against what its sidecar records:
  cache .pt == cache sidecar sha256; token .pt == token sidecar sha256; guide .pt == the cache sidecar's guide_sha; enc .pt ==
  its enc_sha; the token sidecar must name the same guide / enc files; both sidecars complete; the token sidecar carries its
  shared-depth / weight-hash evidence. Sidecars and Wan files have no recorded sha: they are hashed and pinned here.
Any mismatch / missing file -> exit 1 and NO manifest. Otherwise writes ONE manifest json (path, size, mtime, sha256 per file),
made read-only, and <out>.sha256 (the manifest's own sha256; it goes into the gate key and beside DATA_SHA in every run record).
Hashing runs in --nproc processes. Never overwrites an existing manifest (pick a new --out, e.g. data/wan_manifest_v2.json).
  sbatch -w saxa --gres=gpu:h200_1g.18gb:0 --cpus-per-task=8 --mem=16G -t 3:00:00 -J verify-manifest -o logs2/%x-%j.out \
         slurm/py.sbatch scripts/verify_manifest.py --out data/wan_manifest_v1.json --nproc 8
"""
import argparse, json, os, socket, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrv import wan_stage as WS

p = argparse.ArgumentParser()
p.add_argument("--out", required=True)
p.add_argument("--lists", default="data/temporal_lists.json")
p.add_argument("--cache", default="data/wan_cache_v1")
p.add_argument("--tokens", default="data/wan_tok_v1")
p.add_argument("--guides", default="data/guides_v1")
p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--use_tokens", type=int, default=1)
p.add_argument("--wan", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", help="HF repo id (mrv.wan_cond.WAN_ID)")
p.add_argument("--wan_dir", default="", help="override: local snapshot folder of --wan (default: found under $HF_HOME/hub)")
p.add_argument("--nproc", type=int, default=8)
a = p.parse_args()


def _hash(path):
    st = os.stat(path)
    return path, {"size": st.st_size, "mtime": st.st_mtime, "sha256": WS.sha256_file(path)}


def main():
    t0 = time.time()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists (manifests are immutable): choose a new --out")
    L = json.load(open(a.lists))
    roots = {"cache": WS.norm(a.cache), "tokens": WS.norm(a.tokens), "guides": WS.norm(a.guides), "enc": WS.norm(a.enc)}
    split = {"train": sorted(L["train"]), "dev": sorted(L["dev"])}
    if set(split["train"]) & set(split["dev"]):
        raise SystemExit("train / dev overlap in --lists")
    drives = split["train"] + split["dev"]
    files, kinds, exp, bad = [], {}, {}, []
    for sc in drives:
        for k, f in WS.drive_files(sc, roots, bool(a.use_tokens)):
            if not os.path.exists(f):
                bad.append(f"{f}: missing"); continue
            files.append(f); kinds[f] = (k, sc)
        e_, b_ = WS.expected_shas(sc, roots, bool(a.use_tokens))
        exp.update(e_); bad += b_
    snap = a.wan_dir or WS.hf_snapshot_dir(a.wan)
    if not snap or not os.path.isdir(snap):
        bad.append(f"Wan snapshot of {a.wan} not found (HF_HOME={os.environ.get('HF_HOME')}); pass --wan_dir")
        wrel = []
    else:
        wrel = WS.wan_files(snap)
        if not any(r.startswith("transformer/") for r in wrel) or not any(r.startswith("vae/") for r in wrel):
            bad.append(f"{snap}: no transformer/ or vae/ files")
    if bad:
        raise SystemExit(f"VERIFY_MANIFEST FAIL before hashing: {len(bad)} problems, e.g. {bad[:5]}")
    wabs = [os.path.realpath(os.path.join(snap, r)) for r in wrel]  # HF snapshot entries are symlinks into blobs/
    todo = files + wabs
    print(f"hashing {len(files)} data files of {len(drives)} drives + {len(wabs)} Wan files with {a.nproc} processes", flush=True)
    res = {}
    with Pool(a.nproc) as pool:
        for i, (f, e) in enumerate(pool.imap_unordered(_hash, todo, chunksize=1)):
            res[f] = e
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)} files, {time.time() - t0:.0f} s", flush=True)
    for f in files:
        want = exp.get(f)
        if want is not None and res[f]["sha256"] != want:
            bad.append(f"{f}: sha256 {res[f]['sha256'][:12]} != sidecar {want[:12]}")
    for f, e in res.items():  # changed while hashing -> refuse (the hash may describe neither version)
        st = os.stat(f)
        if st.st_size != e["size"] or st.st_mtime != e["mtime"]:
            bad.append(f"{f}: changed while hashing")
    if bad:
        raise SystemExit(f"VERIFY_MANIFEST FAIL: {len(bad)} mismatches, e.g. {bad[:5]}; no manifest written")
    out = {"version": WS.MANIFEST_VERSION, "created": time.time(), "host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"),
           "lists": WS.norm(a.lists), "lists_sha256": WS.sha256_file(a.lists), "split": split, "roots": roots, "use_tokens": bool(a.use_tokens),
           "files": {f: {**res[f], "kind": kinds[f][0], "drive": kinds[f][1], "sidecar_sha": exp.get(f)} for f in sorted(files)},
           "wan": {"id": a.wan, "dir": snap, "files": {r: res[w] for r, w in zip(wrel, wabs)}},
           "totals": {"files": len(todo), "bytes": sum(e["size"] for e in res.values()), "seconds": round(time.time() - t0, 1)}}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = a.out + f".tmp{os.getpid()}"
    json.dump(out, open(tmp, "w"), indent=1, sort_keys=True)
    os.chmod(tmp, 0o444); os.replace(tmp, a.out)
    sha = WS.sha256_file(a.out)
    with open(a.out + ".sha256", "w") as f:
        f.write(f"{sha}  {os.path.basename(a.out)}\n")
    os.chmod(a.out + ".sha256", 0o444)
    print("VERIFY_MANIFEST PASS", json.dumps({"out": a.out, "sha256": sha, **out["totals"], "drives": len(drives)}), flush=True)


if __name__ == "__main__":
    main()
