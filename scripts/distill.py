"""Stage 7: distil a few-step teacher into a 1-step student.

Teacher: a trained renderer sampled with --teacher_steps Euler steps from noise eps -> x0_T.
Student: starts as a copy of the teacher; in one step it maps the same eps to x0_S = eps - v(eps, t=1).
Loss: ||x0_S - x0_T||^2 in latent space. Same noise -> same answer, so the student learns the teacher's whole path in one jump.
"""
import argparse, copy, glob, json, os, time
import torch
from mrv.model import Renderer

p = argparse.ArgumentParser()
p.add_argument("--teacher", required=True)
p.add_argument("--enc", default="data/enc_full")
p.add_argument("--teacher_steps", type=int, default=4)
p.add_argument("--steps", type=int, default=15000)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--lr", type=float, default=5e-5)
p.add_argument("--out", required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda"
AMP = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
ck = torch.load(a.teacher, map_location=dev)
ar, lat_std = ck["args"], ck["lat_std"]
mk = lambda: Renderer(d=ar["d"], depth=ar["depth"], heads=ar["heads"], prev=ar["prev"], n_styles=2 if ar.get("enc2") else 0).to(dev)
teacher = mk().eval().requires_grad_(False); teacher.load_state_dict(ck["ema"])
student = mk(); student.load_state_dict(ck["ema"])
ema = copy.deepcopy(student).eval().requires_grad_(False)
from nuscenes.utils.splits import val
names = [os.path.basename(f)[:-3] for f in sorted(glob.glob(f"{a.enc}/*.pt"))]
T = torch.cat([torch.load(f"{a.enc}/{n}.pt")["tokens"] for n in names if n not in set(val)])
print("train frames", len(T), flush=True)
opt = torch.optim.AdamW(student.parameters(), lr=a.lr)
scaler = torch.amp.GradScaler(enabled=AMP == torch.float16)
ts = torch.linspace(1, 0, a.teacher_steps + 1, device=dev)
log = open(f"{a.out}/log.jsonl", "a"); t0 = time.time()
for step in range(a.steps):
    tok = T[torch.randint(0, len(T), (a.bs,))].to(dev).float()
    eps = torch.randn(a.bs, 4, 18, 32, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=AMP):
        ctx = teacher.context(tok)
        x = eps
        for i in range(a.teacher_steps):
            x = x + (ts[i + 1] - ts[i]) * teacher(x, ts[i].expand(a.bs), None, ctx=ctx).float()
    with torch.autocast("cuda", dtype=AMP):
        v = student(eps, torch.ones(a.bs, device=dev), tok)
    loss = ((eps - v.float() - x) ** 2).mean()
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); scaler.step(opt); scaler.update()
    with torch.no_grad():
        for pe, pm in zip(ema.parameters(), student.parameters()):
            pe.lerp_(pm, 1 - 0.999)
    if step % 200 == 0:
        r = {"step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)}
        print(r, flush=True); log.write(json.dumps(r) + "\n"); log.flush()
out = {"model": student.state_dict(), "ema": ema.state_dict(), "step": a.steps, "lat_std": lat_std, "args": {**ar, "distilled_from": a.teacher}}
torch.save(out, f"{a.out}/ckpt.pt")
print("DISTILL_DONE")
