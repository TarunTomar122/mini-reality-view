"""PLAN-video S1 data trust + staging (council/s1-timing-r1-CONVERGED.md items 2, 3, 7). Torch-free, stdlib only (tested by
scripts/wan_tests.py with plain python3).

One rule (F2): a byte is trusted only if it was hashed.
  - scripts/verify_manifest.py (CPU job) hashes every latent cache, token cache, guide and enc file of the train + dev drives
    (+ the cache / token / guide sidecars and the Wan HF snapshot files) against the sidecars and writes ONE read-only manifest
    {path: size, mtime, sha256} + <manifest>.sha256; any mismatch -> no manifest.
  - every training job copies the token + latent caches it uses (and the Wan weights) to a job-local stage dir (/dev/shm),
    hashing while it copies (stage_files, 8 streams); each hash must equal the manifest or the job refuses.
  - files a job never opens (guide / enc in the trainer) are covered by the CPU job; the trainer checks their size / mtime as a
    CHANGE ALARM only (mismatch -> refuse, "re-run verify_manifest"); a match is never proof.
Memory (F1, Neon 11:30 P0): files in /dev/shm count against the job's memory cgroup. The request is
    --mem >= (staged bytes + 45 GB measured trainer RSS allowance) / 0.80, rounded UP to a multiple of 10 G   (need_bytes / mem_request_g)
(timing 3756964: trainer RSS ~38.5 GiB; staged 97.56 GB -> 178 GB -> --mem=170G, so peak ~139 GB stays under the 85 % rule).
copy_hash drops the SOURCE file's page cache after each copy, so memory.peak counts only the staged tmpfs + process memory.
Peak read from memory.peak (v2) / memory.max_usage_in_bytes (v1).
  python -m mrv.wan_stage need --manifest data/wan_manifest_v1.json [--lists data/temporal_lists.json] [--no_dev]  -> e.g. 142G
  python -m mrv.wan_stage memcheck --manifest data/wan_manifest_v1.json  (exit 9 if the request < (staged + 45 GB) / 0.80)
"""
import hashlib, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor

RSS_ALLOW_GB = 45.0   # measured trainer RSS allowance (timing 3756964: ~38.5 GiB = 41.3 GB) on top of the staged files
REQUEST_FRAC = 0.80   # staged + RSS must fit in 80 % of the request (the re-time rule allows 85 %)
ROUND_G = 10          # --mem rounded up to a multiple of 10 G
PEAK_FRAC = 0.85      # re-time rule: cgroup peak incl. staged files <= 85 % of the request
CHUNK = 1 << 24       # 16 MiB
MANIFEST_VERSION = 1
KINDS = ("cache", "cache_json", "tokens", "tokens_json", "guide", "guide_json", "enc")


def norm(p):
    """manifest key of a data path: normalised, relative to the repo root as given (e.g. data/wan_cache_v1/scene-0001.pt)"""
    return os.path.normpath(p)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def stat_entry(path):
    st = os.stat(path)
    return {"size": st.st_size, "mtime": st.st_mtime}


def drive_files(sc, roots, use_tokens=True):
    """every file of one drive the manifest covers: [(kind, path)]"""
    out = [("cache", f"{roots['cache']}/{sc}.pt"), ("cache_json", f"{roots['cache']}/{sc}.json")]
    if use_tokens:
        out += [("tokens", f"{roots['tokens']}/{sc}.pt"), ("tokens_json", f"{roots['tokens']}/{sc}.json")]
    out += [("guide", f"{roots['guides']}/{sc}.pt"), ("guide_json", f"{roots['guides']}/{sc}.json"), ("enc", f"{roots['enc']}/{sc}.pt")]
    return [(k, norm(p)) for k, p in out]


def expected_shas(sc, roots, use_tokens=True, load=None):
    """sha256 each file must have according to the sidecars (None = no sidecar records it: hashed and recorded only).
    cache / tokens .pt <- their own sidecar "sha256"; guide / enc .pt <- the CACHE sidecar's guide_sha / enc_sha, and the token
    sidecar must name the same guide / enc (the two caches were built from the same files). Returns (exp, problems)."""
    load = load or (lambda f: json.load(open(f)))
    exp, bad = {}, []
    try:
        jc = load(f"{roots['cache']}/{sc}.json")
    except Exception as e:  # noqa: BLE001
        return exp, [f"{sc}: cache sidecar unreadable ({type(e).__name__})"]
    if not jc.get("complete"):
        bad.append(f"{sc}: cache sidecar not complete")
    exp[norm(f"{roots['cache']}/{sc}.pt")] = jc.get("sha256")
    exp[norm(f"{roots['guides']}/{sc}.pt")] = jc.get("guide_sha")
    exp[norm(f"{roots['enc']}/{sc}.pt")] = jc.get("enc_sha")
    if use_tokens:
        try:
            jt = load(f"{roots['tokens']}/{sc}.json")
        except Exception as e:  # noqa: BLE001
            return exp, bad + [f"{sc}: token sidecar unreadable ({type(e).__name__})"]
        if not jt.get("complete"):
            bad.append(f"{sc}: token sidecar not complete")
        if not (jt.get("shared_pass_ok") and jt.get("weights_sha")):
            bad.append(f"{sc}: token sidecar lacks the shared-depth / weight-hash evidence")
        if jt.get("guide_sha") != jc.get("guide_sha") or jt.get("enc_sha") != jc.get("enc_sha"):
            bad.append(f"{sc}: latent cache and token cache were built from different guide / enc files")
        exp[norm(f"{roots['tokens']}/{sc}.pt")] = jt.get("sha256")
    for k, v in list(exp.items()):
        if not v:
            bad.append(f"{k}: sidecar records no sha")
    return exp, bad


def hf_snapshot_dir(repo_id, hf_home=None):
    """local snapshot folder of an HF repo (offline): $HF_HOME/hub/models--<org>--<name>/snapshots/<rev> (refs/main if present,
    else the only snapshot); None if not found"""
    hub = os.path.join(hf_home or os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    base = os.path.join(hub, "models--" + repo_id.replace("/", "--"))
    ref = os.path.join(base, "refs", "main")
    if os.path.exists(ref):
        d = os.path.join(base, "snapshots", open(ref).read().strip())
        if os.path.isdir(d):
            return d
    snaps = sorted(os.listdir(os.path.join(base, "snapshots"))) if os.path.isdir(os.path.join(base, "snapshots")) else []
    return os.path.join(base, "snapshots", snaps[0]) if len(snaps) == 1 else None


WAN_SUBDIRS = ("transformer", "vae")


def wan_files(snap):
    """relative paths of the Wan snapshot files the trainer loads (transformer + vae folders and model_index.json)"""
    out = []
    for sub in WAN_SUBDIRS:
        for r, _, fs in os.walk(os.path.join(snap, sub), followlinks=True):
            for f in fs:
                out.append(os.path.relpath(os.path.join(r, f), snap))
    if os.path.exists(os.path.join(snap, "model_index.json")):
        out.append("model_index.json")
    return sorted(out)


def manifest_sha(path):
    return sha256_file(path)


def load_manifest(path):
    """-> (manifest dict, its sha256); refuses if <path>.sha256 exists and differs (the manifest was edited after writing)"""
    if not path or not os.path.exists(path):
        raise SystemExit(f"--manifest {path!r} missing: run scripts/verify_manifest.py (CPU job) first")
    sha = manifest_sha(path)
    side = path + ".sha256"
    if os.path.exists(side) and open(side).read().split()[0] != sha:
        raise SystemExit(f"{path} does not match {side}: the manifest changed after it was written; re-run verify_manifest")
    m = json.load(open(path))
    if m.get("version") != MANIFEST_VERSION or not isinstance(m.get("files"), dict):
        raise SystemExit(f"{path}: not a v{MANIFEST_VERSION} manifest")
    return m, sha


def entry(man, path):
    e = man["files"].get(norm(path))
    if e is None:
        raise SystemExit(f"{path} is not in the manifest: re-run verify_manifest for this drive list")
    return e


def change_alarm(man, paths):
    """size / mtime of files the job will NOT open vs the manifest. Returns the list of changed / missing paths (empty = no alarm).
    A match is not proof (only the CPU job's hash is); a mismatch means the file changed since it was hashed."""
    bad = []
    for p in paths:
        e = man["files"].get(norm(p))
        if e is None:
            bad.append(f"{p}: not in manifest"); continue
        try:
            st = os.stat(p)
        except OSError:
            bad.append(f"{p}: missing"); continue
        if st.st_size != e["size"] or abs(st.st_mtime - e["mtime"]) > 1e-3:
            bad.append(f"{p}: size / mtime changed ({st.st_size}, {st.st_mtime} vs {e['size']}, {e['mtime']})")
    return bad


def drop_page_cache(fd):
    """POSIX_FADV_DONTNEED on a file we finished reading (no-op where the platform lacks posix_fadvise, e.g. macOS)"""
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            return True
        except OSError:
            return False
    return False


def copy_hash(src, dst, chunk=CHUNK):
    """ONE read does both: copy src -> dst and return (sha256 of the bytes read, bytes)"""
    h = hashlib.sha256(); n = 0
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    with open(src, "rb") as fi, open(dst + ".part", "wb") as fo:
        for b in iter(lambda: fi.read(chunk), b""):
            h.update(b); fo.write(b); n += len(b)
        drop_page_cache(fi.fileno())  # Neon 11:30 P0: the Lustre source pages would otherwise count in the job's memory.peak
    os.replace(dst + ".part", dst)
    return h.hexdigest(), n


def stage_files(jobs, streams=8, log=None):
    """jobs: [(src, dst, expected_sha256, expected_size)] -> record {files, bytes, seconds, mb_s, streams, mismatches}.
    Copies with `streams` parallel workers, hashing while copying. Any mismatch (sha or size) -> SystemExit after all copies
    finish (the record lists every mismatch). A file whose size already differs from the manifest is refused before copying."""
    t0 = time.time(); bad = []; done = {"bytes": 0, "n": 0}; lock = threading.Lock()
    for src, _, _, size in jobs:
        try:
            if os.path.getsize(src) != size:
                bad.append(f"{src}: size {os.path.getsize(src)} != manifest {size}")
        except OSError:
            bad.append(f"{src}: missing")
    if bad:
        raise SystemExit(f"staging refused: {len(bad)} files differ from the manifest before copying, e.g. {bad[:3]}; re-run verify_manifest")

    def one(job):
        src, dst, sha, size = job
        got, n = copy_hash(src, dst)
        with lock:
            done["bytes"] += n; done["n"] += 1
            if log and done["n"] % 50 == 0:
                log(f"staged {done['n']}/{len(jobs)} files, {done['bytes'] / 1e9:.1f} GB, {time.time() - t0:.0f} s")
        if got != sha or n != size:
            return f"{src}: sha {got[:12]} / {n} B != manifest {str(sha)[:12]} / {size} B"
        return None

    with ThreadPoolExecutor(max_workers=max(1, streams)) as ex:
        bad = [r for r in ex.map(one, jobs) if r]
    dt = time.time() - t0
    rec = {"files": len(jobs), "bytes": done["bytes"], "seconds": round(dt, 1), "mb_s": round(done["bytes"] / 1e6 / max(dt, 1e-9), 1),
           "streams": streams, "mismatches": bad, "hash_equals_manifest": not bad}
    if bad:
        raise SystemExit(f"staging refused: {len(bad)} staged hashes differ from the manifest, e.g. {bad[:3]}; re-run verify_manifest")
    return rec


def rehash_files(jobs, streams=8):
    """hash already-staged destinations in parallel -> [(job, sha or None)]"""
    def one(j):
        try:
            return j, sha256_file(j[1])
        except OSError:
            return j, None
    with ThreadPoolExecutor(max_workers=max(1, streams)) as ex:
        return list(ex.map(one, jobs))


def stage_reuse(jobs, receipt_path, job_id, manifest_sha, streams=8, log=None):
    """Astra t1 item 9: stage `jobs`, reusing files an earlier process of the SAME allocation staged (receipt: job id + manifest
    sha) only after RE-HASHING every reused destination; a reused file whose bytes differ is copied and hashed again. A receipt
    is never proof. -> record (stage_files fields + reused / rehashed / recopied counts)"""
    rc = {}
    if os.path.exists(receipt_path):
        try:
            rc = json.load(open(receipt_path))
        except Exception:  # noqa: BLE001
            rc = {}
    reuse = bool(job_id) and rc.get("job") == job_id and rc.get("manifest_sha") == manifest_sha
    have = (rc.get("files") or {}) if reuse else {}
    cand = [j for j in jobs if j[1] in have and os.path.exists(j[1])]
    t0 = time.time()
    rh = rehash_files(cand, streams)
    good = {j[1] for j, sha in rh if sha == j[2] and os.path.getsize(j[1]) == j[3]}
    recopy = [j for j, sha in rh if j[1] not in good]
    todo = [j for j in jobs if j[1] not in good]
    rehash_s = time.time() - t0
    rec = stage_files(todo, streams, log) if todo else {"files": 0, "bytes": 0, "seconds": 0.0, "mb_s": None, "streams": streams,
                                                       "mismatches": [], "hash_equals_manifest": True}
    rec.update(reused_files=len(good), rehashed_files=len(cand), recopied_after_rehash=len(recopy), rehash_s=round(rehash_s, 1),
               cold=len(good) == 0 and not cand)
    json.dump({"job": job_id, "manifest_sha": manifest_sha, "files": {j[1]: j[2] for j in jobs}}, open(receipt_path + ".tmp", "w"))
    os.replace(receipt_path + ".tmp", receipt_path)
    return rec


def stage_plan(man, drives, roots, stage_dir, use_tokens=True, include_wan=True):
    """-> (jobs, layout): the files a training job opens (cache + token .pt / .json per drive, Wan snapshot files) and where they
    go under stage_dir (cache/, tokens/, wan/). Guide / enc files are NOT staged (the trainer never opens them)."""
    jobs = []
    for sc in drives:
        for kind, sub in (("cache", "cache"), ("tokens", "tokens")):
            if kind == "tokens" and not use_tokens:
                continue
            for ext in ("pt", "json"):
                src = norm(f"{roots[kind]}/{sc}.{ext}")
                e = entry(man, src)
                jobs.append((src, os.path.join(stage_dir, sub, f"{sc}.{ext}"), e["sha256"], e["size"]))
    if include_wan:
        w = man.get("wan") or {}
        if not w.get("files"):
            raise SystemExit("manifest has no Wan snapshot entry: re-run verify_manifest (it hashes the HF snapshot files)")
        for rel, e in sorted(w["files"].items()):
            jobs.append((os.path.join(w["dir"], rel), os.path.join(stage_dir, "wan", rel), e["sha256"], e["size"]))
    return jobs, {"cache": os.path.join(stage_dir, "cache"), "tokens": os.path.join(stage_dir, "tokens"), "wan": os.path.join(stage_dir, "wan")}


def staged_bytes(man, drives, roots, use_tokens=True, include_wan=True):
    jobs, _ = stage_plan(man, drives, roots, "/nonexistent", use_tokens, include_wan)
    return sum(j[3] for j in jobs)


def need_bytes(staged, rss_gb=RSS_ALLOW_GB, frac=REQUEST_FRAC):
    """minimum request in bytes = (staged bytes + RSS allowance) / frac"""
    return int(-(-(staged + rss_gb * 1e9) // frac))


def mem_request_g(staged, rss_gb=RSS_ALLOW_GB, frac=REQUEST_FRAC, round_g=ROUND_G):
    """--mem for sbatch in Slurm G (1024^3): (staged + 45 GB) / 0.80, rounded UP to a multiple of 10 G"""
    g = need_bytes(staged, rss_gb, frac) / (1 << 30)
    return int(-(-g // round_g) * round_g)


# ------------------------------------------------------------------------------------------------------------- cgroup
def _cg_paths():
    """candidate directories of this process's memory cgroup (v2 unified first, then v1 memory controller)"""
    out = []
    try:
        for line in open("/proc/self/cgroup"):
            hid, ctrl, path = line.rstrip("\n").split(":", 2)
            if hid == "0" and ctrl == "":
                out.append(("v2", "/sys/fs/cgroup" + path))
            elif "memory" in ctrl.split(","):
                out.append(("v1", "/sys/fs/cgroup/memory" + path))
    except OSError:
        pass
    return out


def _read_int(f):
    try:
        v = open(f).read().strip()
    except OSError:
        return None
    return None if v in ("", "max") else int(v)


def cgroup_mem(paths=None):
    """{"limit", "peak", "current", "version", "dir"} in bytes (None where unreadable). Walks up the hierarchy for the limit:
    Slurm sets it on the job / step cgroup, the process may sit in a child (task_0)."""
    for ver, d in (paths if paths is not None else _cg_paths()):
        if not os.path.isdir(d):
            continue
        if ver == "v2":
            names = {"limit": "memory.max", "peak": "memory.peak", "current": "memory.current"}
        else:
            names = {"limit": "memory.limit_in_bytes", "peak": "memory.max_usage_in_bytes", "current": "memory.usage_in_bytes"}
        rec = {"version": ver, "dir": d}
        for k, n in names.items():
            rec[k] = _read_int(os.path.join(d, n))
        up = d
        while rec["limit"] is None or rec["limit"] >= (1 << 60):  # unlimited at this level: look at the parents
            nup = os.path.dirname(up)
            if nup == up or not nup.startswith("/sys/fs/cgroup"):
                rec["limit"] = None; break
            up = nup
            rec["limit"] = _read_int(os.path.join(up, names["limit"]))
            if rec["limit"] is not None and rec["limit"] < (1 << 60):
                rec["limit_dir"] = up; break
        if rec.get("limit_dir"):  # usage where the limit is enforced (the job cgroup: trainer + shell + heartbeat + staged files)
            for k in ("peak", "current"):
                v = _read_int(os.path.join(rec["limit_dir"], names[k]))
                if v is not None:
                    rec[k] = v
        if rec["peak"] is None:  # kernels < 5.19 have no memory.peak: the caller uses PeakSampler (never a single read as a peak)
            rec["peak_note"] = "memory.peak unavailable (kernel < 5.19): use the sampled maximum of memory.current"
        return rec
    return {"limit": None, "peak": None, "current": None, "version": None, "dir": None}


def slurm_requested_bytes(env=None, scontrol_text=None):
    """Astra t1 item 8: the memory this job REQUESTED (the denominator of the 85 % rule): SLURM_MEM_PER_NODE (MB), else
    SLURM_MEM_PER_CPU x SLURM_CPUS_ON_NODE, else `scontrol show job` (MinMemoryNode / mem= in TRES). None if unknown."""
    env = os.environ if env is None else env
    if env.get("SLURM_MEM_PER_NODE"):
        return int(float(env["SLURM_MEM_PER_NODE"])) * (1 << 20)
    if env.get("SLURM_MEM_PER_CPU") and env.get("SLURM_CPUS_ON_NODE"):
        return int(float(env["SLURM_MEM_PER_CPU"])) * int(env["SLURM_CPUS_ON_NODE"]) * (1 << 20)
    if scontrol_text is None and env.get("SLURM_JOB_ID"):
        try:
            import subprocess
            scontrol_text = subprocess.run(["scontrol", "show", "job", env["SLURM_JOB_ID"]], capture_output=True, text=True, timeout=30).stdout
        except Exception:  # noqa: BLE001
            scontrol_text = None
    if scontrol_text:
        import re
        m = re.search(r"MinMemoryNode=(\d+(?:\.\d+)?)([KMGT]?)", scontrol_text) or re.search(r"\bmem=(\d+(?:\.\d+)?)([KMGT]?)", scontrol_text)
        if m:
            mult = {"": 1 << 20, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}[m.group(2)]
            return int(float(m.group(1)) * mult)
    return None


def job_cgroup_dir(paths=None):
    """the JOB-level memory cgroup (Slurm: a path component job_<id>), so the peak covers every process of the allocation
    (batch shell, heartbeat, trainer, copy workers, staged tmpfs pages); falls back to this process's own cgroup"""
    for ver, d in (paths if paths is not None else _cg_paths()):
        parts = d.split("/")
        for i in range(len(parts) - 1, 0, -1):
            if parts[i].startswith("job_"):
                return ver, "/".join(parts[:i + 1])
        return ver, d
    return None, None


def _job_current():
    ver, d = job_cgroup_dir()
    if not d:
        return None
    return _read_int(os.path.join(d, "memory.current" if ver == "v2" else "memory.usage_in_bytes"))


class PeakSampler:
    """max of the cgroup's current usage, sampled every `interval` s in a daemon thread (a peak where memory.peak is missing)"""

    def __init__(s, interval=1.0, reader=None):
        s.interval, s.max, s.n = interval, None, 0
        s.reader = reader or _job_current
        s._stop = threading.Event()

    def _run(s):
        while not s._stop.is_set():
            try:
                v = s.reader()
            except Exception:  # noqa: BLE001
                v = None
            if v is not None:
                s.max = v if s.max is None else max(s.max, v); s.n += 1
            s._stop.wait(s.interval)

    def start(s):
        threading.Thread(target=s._run, daemon=True).start()
        return s

    def stop(s):
        s._stop.set()
        return s.max


def sample_once(rec, d, names):
    """one allocation-sampler read (Astra t2 item 8): n counts every attempt, n_valid only reads that returned a job-level value"""
    cur = _read_int(os.path.join(d, names[1])) if d else None
    pk = _read_int(os.path.join(d, names[0])) if d else None
    if cur is not None:
        rec["max_current"] = cur if rec["max_current"] is None else max(rec["max_current"], cur)
    if pk is not None:
        rec["peak"] = pk if rec["peak"] is None else max(rec["peak"], pk)
    rec["n"] += 1
    rec["n_valid"] = rec.get("n_valid", 0) + int(cur is not None or pk is not None)
    rec["t"] = time.time()
    if cur is not None or pk is not None:  # Astra T3 item 8: the time of the last SUCCESSFUL job-level read
        rec["t_valid"] = rec["t"]
    return rec


def _sample_main(out, interval=1.0):
    """python -m mrv.wan_stage sample --out <json>: started by slurm at ALLOCATION start; records the job cgroup's memory.peak (if
    the kernel has it) and the max of memory.current every `interval` s until killed; rewrites <out.json> atomically"""
    ver, d = job_cgroup_dir()
    names = ("memory.peak", "memory.current") if ver == "v2" else ("memory.max_usage_in_bytes", "memory.usage_in_bytes")
    rec = {"t0": time.time(), "dir": d, "version": ver, "max_current": None, "peak": None, "n": 0, "n_valid": 0,
           "job": os.environ.get("SLURM_JOB_ID")}
    while True:
        sample_once(rec, d, names)
        json.dump(rec, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
        time.sleep(interval)


def mem_check(staged, limit, rss_gb=RSS_ALLOW_GB):
    """-> (ok, message). ok is None when the request / limit is unreadable (cannot check: the caller records it)."""
    need = need_bytes(staged, rss_gb)
    f_ = f"(staged {staged / 1e9:.1f} GB + {rss_gb:.0f} GB RSS) / {REQUEST_FRAC} = {need / 1e9:.1f} GB"
    if limit is None:
        return None, f"memory request unreadable; need {f_}"
    if limit < need:
        return False, f"memory request {limit / 1e9:.1f} GB < {f_}: resubmit with --mem={mem_request_g(staged, rss_gb)}G"
    return True, f"memory request {limit / 1e9:.1f} GB >= {f_}"


def peak_rule(peak, limit, frac=PEAK_FRAC):
    """re-time rule: cgroup peak incl. staged files <= frac x request -> (ok | None if unreadable, ratio)"""
    if not peak or not limit:
        return None, None
    return peak <= frac * limit, round(peak / limit, 4)


# ------------------------------------------------------------------------------------------------------- forecast helpers
def block_stats(rows, block=32):
    """per-update production rows [{wall_s, gpu_s, queue_wait_s}] -> consecutive full blocks of `block` updates:
    [{updates, wall_s (mean / update), gpu_s, queue_wait_s, queue_frac = wait / wall, wall_over_gpu}] (partial tail dropped)"""
    out = []
    for i in range(0, len(rows) - block + 1, block):
        b = rows[i:i + block]
        w = sum(r["wall_s"] for r in b); g = sum(r["gpu_s"] for r in b); q = sum(r["queue_wait_s"] for r in b)
        out.append({"first": i, "updates": block, "wall_s": round(w / block, 4), "gpu_s": round(g / block, 4), "queue_wait_s": round(q / block, 4),
                    "queue_frac": round(q / w, 4) if w else None, "wall_over_gpu": round(w / g, 4) if g else None})
    return out


def forecast_wall(rows, block=32):
    """s / update the forecast uses: the SLOWEST of the full-run mean and the means of the last two consecutive full blocks"""
    if not rows:
        return None, {}
    full = sum(r["wall_s"] for r in rows) / len(rows)
    bl = block_stats(rows, block)
    last2 = [b["wall_s"] for b in bl[-2:]]
    return max([full] + last2), {"full_run_mean": round(full, 4), "last_two_blocks": last2}


# ------------------------------------------------------------------------------------------------------------- CLI
def _cli(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m mrv.wan_stage")
    ap.add_argument("cmd", choices=["need", "memcheck", "copy", "sample"])
    ap.add_argument("--manifest", default="")
    ap.add_argument("--dst", default="", help="copy: destination folder (files keep their manifest-relative folder name)")
    ap.add_argument("--out", default="", help="sample: output json")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--lists", default="data/temporal_lists.json")
    ap.add_argument("--no_dev", action="store_true", help="the job does not open the dev drives (no probe)")
    ap.add_argument("--use_tokens", type=int, default=1)
    ap.add_argument("--rss_gb", type=float, default=RSS_ALLOW_GB)
    ap.add_argument("--drive_list", default="", help="override: whitespace-separated train drives (default: lists['train'])")
    a = ap.parse_intermixed_args(argv)  # positionals around options: Python 3.11 parse_args refuses files after --dst
    if a.cmd == "sample":
        _sample_main(a.out)
        return 0
    man, _ = load_manifest(a.manifest)
    if a.cmd == "copy":  # Astra t1 item 9: copy + hash files a job will open (e.g. the geometry check's guide / tokens) to --dst
        jobs = []
        for f in a.files:
            e = entry(man, f)
            jobs.append((norm(f), os.path.join(a.dst, os.path.basename(os.path.dirname(norm(f))), os.path.basename(f)), e["sha256"], e["size"]))
        rec = stage_files(jobs, streams=min(8, max(1, len(jobs))))
        print("STAGE_COPY", json.dumps({k: v for k, v in rec.items() if k != "mismatches"}))
        return 0
    L = json.load(open(a.lists))
    drives = sorted(open(a.drive_list).read().split()) if a.drive_list else sorted(L["train"])
    if not a.no_dev:
        drives += sorted(L["dev"])
    sb = staged_bytes(man, drives, man["roots"], bool(a.use_tokens), True)
    if a.cmd == "need":
        print(f"{mem_request_g(sb, a.rss_gb)}G")
        return 0
    cg = cgroup_mem()
    req = slurm_requested_bytes()
    ok, msg = mem_check(sb, req if req else cg["limit"], a.rss_gb)  # the request is what Slurm grants; an ancestor limit may be larger
    print(f"MEMCHECK {'PASS' if ok else ('UNKNOWN' if ok is None else 'FAIL')} {msg} (staged {len(drives)} drives + Wan, cgroup {cg.get('dir')})")
    return 0 if ok in (True, None) else 9


if __name__ == "__main__":
    import sys
    sys.exit(_cli(sys.argv[1:]))
