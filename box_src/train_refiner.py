#!/usr/bin/env python3
"""POSE — LEARNED Pose-Refinement (GigaPose/MegaPose-Stil, render-and-compare).

Problem: der Template-Bank-Matcher hat ein hohes Oracle (richtige Lage IST in der
Bank, ~3°) trifft sie aber nicht zuverlaessig — Domain-Gap zwischen synthetischem
Template und Isaac-Render-Crop + handgewichtete NCC/IoU/Gradient-Metrik. Loesung:
ein gelerntes Embedding, das Query-Tiefen-Crop UND Bank-Tiefen-Templates in denselben
Raum bettet, so dass das embedding-naechste Template das pose-naechste ist.

Trainingsdaten (vollsynthetisch, schnell, mit GT): pro Teil die physisch stabilen
Ruhelagen (trimesh.compute_stable_poses) x kontinuierlicher Yaw x kleiner Kipp-
Stoerung; orthographisches Top-Down-Tiefen-Rendering (derselbe Rasterizer wie die
Bank) + Domain-Randomization (Tiefen-Rauschen, Pixel-Dropout, Skalen-Jitter, Rand-
Occlusion). Label = die GT-Rotation R (world=R@body).

Loss: für jeden Sample wird das Bank-Template mit minimalem sym-bewusstem
Rotationsfehler zum Sample-GT als POSITIV gewählt; InfoNCE zieht das Query-Embedding
zu diesem Positiv und stösst die übrigen Bank-Templates ab. Sym-aware: Templates die
(sym-bewusst) ebenfalls < tol Grad von GT sind, werden NICHT als Negative bestraft
(unbeobachtbare DoF nicht bestrafen).

Output: refiner_<part>.pt = {state_dict, arch, size}. Wird in e2e_infer.py geladen;
match_template_bank re-rankt die Bank-Kandidaten mit dem Embedding statt der
Handmetrik.

    python3 train_refiner.py --part Getriebegehaeuse_typ4 \
        --usd /path/parts/Getriebegehaeuse_typ4.usdz \
        --bank /mnt/data/.../templates/Getriebegehaeuse_typ4/bank.npz \
        --out /mnt/data/.../models --epochs 30 --samples 6000
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, "/mnt/data/kip_pose/faces")
sys.path.insert(0, "/mnt/data/kip_pose/sim_code")
from usd_mesh import load_usd_mesh

SIZE = 96
PAD_FRAC = 0.12


def log(m): print(f"[refiner {time.strftime('%T')}] {m}", flush=True)


def rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rand_axis_angle(rng, max_deg):
    """Kleine zufaellige Rotation (Kipp-Stoerung): Achse uniform, Winkel <= max_deg."""
    ax = rng.normal(size=3); ax /= (np.linalg.norm(ax) + 1e-9)
    th = np.radians(rng.uniform(0, max_deg))
    x, y, z = ax; c, s = np.cos(th), np.sin(th); C = 1 - c
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])


def rasterize_topdown(V, F, size=SIZE, pad=PAD_FRAC):
    """Orthographische Top-Down-Tiefenkarte (Kamera entlang -Z, +Y oben). Identisch
    zum Bank-Renderer, damit Query und Bank like-with-like sind."""
    x, y = V[:, 0], V[:, 1]
    minx, maxx = x.min(), x.max(); miny, maxy = y.min(), y.max()
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    span = max(maxx - minx, maxy - miny)
    span = span * (1.0 + 2 * pad) if span > 1e-9 else 1.0

    def to_px(px, py):
        u = (px - cx) / span * (size - 1) + (size - 1) / 2.0
        v = (size - 1) / 2.0 - (py - cy) / span * (size - 1)
        return u, v

    depth = np.full((size, size), -np.inf, np.float32)
    for tri in F:
        p0, p1, p2 = V[tri[0]], V[tri[1]], V[tri[2]]
        u0, v0 = to_px(p0[0], p0[1]); u1, v1 = to_px(p1[0], p1[1]); u2, v2 = to_px(p2[0], p2[1])
        z0, z1, z2 = p0[2], p1[2], p2[2]
        umin = max(int(np.floor(min(u0, u1, u2))), 0); umax = min(int(np.ceil(max(u0, u1, u2))), size - 1)
        vmin = max(int(np.floor(min(v0, v1, v2))), 0); vmax = min(int(np.ceil(max(v0, v1, v2))), size - 1)
        if umax < umin or vmax < vmin: continue
        denom = (v1 - v2) * (u0 - u2) + (u2 - u1) * (v0 - v2)
        if abs(denom) < 1e-9: continue
        uu, vv = np.meshgrid(np.arange(umin, umax + 1), np.arange(vmin, vmax + 1))
        aa = ((v1 - v2) * (uu - u2) + (u2 - u1) * (vv - v2)) / denom
        bb = ((v2 - v0) * (uu - u2) + (u0 - u2) * (vv - v2)) / denom
        cc = 1.0 - aa - bb
        inside = (aa >= -1e-4) & (bb >= -1e-4) & (cc >= -1e-4)
        if not inside.any(): continue
        zb = aa * z0 + bb * z1 + cc * z2
        sub = depth[vmin:vmax + 1, umin:umax + 1]
        upd = inside & (zb > sub); sub[upd] = zb[upd]
        depth[vmin:vmax + 1, umin:umax + 1] = sub
    sil = np.isfinite(depth)
    if sil.sum() == 0:
        return np.zeros((size, size), np.float32)
    d = depth.copy(); zmin = d[sil].min(); zmax = d[sil].max()
    d[~sil] = zmin; d = (d - zmin) / (zmax - zmin + 1e-9); d[~sil] = 0.0
    return d.astype(np.float32)


def domain_randomize(d, rng):
    """DR auf eine normierte Tiefenkarte (0..1): Tiefen-Rauschen, Pixel-Dropout,
    leichter Gauss-Blur-Ersatz (Box), Rand-Occlusion. Schliesst den Gap synthetisch
    sauber -> Isaac-noisy."""
    out = d.copy()
    m = out > 0.02
    # additives Tiefen-Rauschen nur auf dem Teil
    out[m] += rng.normal(0, rng.uniform(0.0, 0.06), size=int(m.sum()))
    # globaler Offset/Kontrast (Isaac far/near variiert)
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05)
    # Pixel-Dropout (Reflexionen/Tiefenlöcher bei Metallteilen)
    if rng.random() < 0.7:
        drop = rng.random(out.shape) < rng.uniform(0.0, 0.10)
        out[drop & m] = 0.0
    # Rand-Occlusion: ein zufälliger Streifen genullt (anderes Teil verdeckt)
    if rng.random() < 0.4:
        H, W = out.shape
        if rng.random() < 0.5:
            w = rng.integers(0, W // 3);
            if rng.random() < 0.5: out[:, :w] = 0.0
            else: out[:, W - w:] = 0.0
        else:
            h = rng.integers(0, H // 3)
            if rng.random() < 0.5: out[:h, :] = 0.0
            else: out[H - h:, :] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _sym_variants(spec):
    """Liste aller diskreten Symmetrie-Operationen G (3x3) als (M,3,3)-Array:
    Welt-Z-Flips x (Körperachsen-Rotationen, falls kontinuierlich). Ein Sample-GT R
    ist sym-äquivalent zu R@G für jedes G. Damit lässt sich der sym-Fehler als
    min_G geo(Ra^T @ Rb @ G) BATCH-vektorisiert berechnen (G vorab, kein Python-Loop
    pro Template)."""
    Gs = []
    axis = spec.get("axis")
    body_rots = [np.eye(3)]
    if axis is not None:
        ax = np.asarray(axis, float); ax /= np.linalg.norm(ax) + 1e-9
        for ang in np.arange(0, 360, 6.0):
            t = np.radians(ang); x, y, z = ax; c, s = np.cos(t), np.sin(t); C = 1 - c
            body_rots.append(np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                                       [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                                       [z*x*C-y*s, z*y*C+x*s, c+z*z*C]]))
    # diskrete In-Plane-Flips wirken auf der WELT-Seite (Rz @ Rb); kontinuierliche
    # Symmetrie auf der KÖRPER-Seite (Rb @ Rbody). Wir falten beides in ein G auf der
    # Körperseite NICHT direkt — daher zwei getrennte Sätze, kombiniert beim Scoren.
    return ([rz(np.radians(d)) for d in spec.get("discrete", [0])], body_rots)


def sym_err_to_bank(R, bank_R, world_ops, body_ops):
    """sym-bewusster Rotationsfehler von einem Sample-GT R (3x3) zu ALLEN Bank-
    Templates (T,3,3) — vektorisiert. Liefert (T,) Grad. Für jede Welt-Flip Rz und
    jede Körper-Rotation Rbody: err = geo(R^T @ (Rz @ Rb @ Rbody)). Min über alle Ops."""
    T = bank_R.shape[0]
    best = np.full(T, 1e9)
    Rt = R.T
    for Rz_ in world_ops:
        # (Rz @ Rb): (T,3,3)
        RzRb = np.einsum("ij,tjk->tik", Rz_, bank_R)
        for Rb_ in body_ops:
            M = np.einsum("tij,jk->tik", RzRb, Rb_)        # (T,3,3)
            D = np.einsum("ij,tjk->tik", Rt, M)            # R^T @ M
            tr = D[:, 0, 0] + D[:, 1, 1] + D[:, 2, 2]
            ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
            best = np.minimum(best, ang)
    return best


PART_SYMMETRY = {
    "Anker_Lang": {"axis": (0, 1, 0), "discrete": [0, 180]},
    "Anker_Kurz": {"axis": (0, 1, 0), "discrete": [0, 180]},
    "Ringmagnet": {"axis": (0, 0, 1), "discrete": [0, 90, 180, 270]},
    "Zahnrad": {"axis": None, "discrete": [k * 360.0 / 14 for k in range(14)]},
}
_DEFAULT_SYM = {"axis": None, "discrete": [0, 90, 180, 270]}


# Worker-State (per-Prozess, read-only nach fork) + Worker-Funktion auf Modul-Ebene
# (Pickling-Anforderung von multiprocessing). Jeder Worker zieht ein Sample.
_G = {}


def _gen_one(seed):
    g = _G
    rng = np.random.default_rng(seed * 2654435761 % (2**32))
    k = rng.choice(len(g["Rstab"]), p=g["Pstab"])
    Rs = g["Rstab"][k]
    yaw = rng.uniform(0, 360)
    R = rz(np.radians(yaw)) @ Rs
    if rng.random() < 0.85:
        R = rand_axis_angle(rng, g["tilt"]) @ R
    Vw = g["Vr"] @ R.T
    d = rasterize_topdown(Vw, g["Fr"], size=g["S"])
    d = domain_randomize(d, rng)
    errs = sym_err_to_bank(R, g["bank_R"], g["world_ops"], g["body_ops"])
    pos = int(np.argmin(errs))
    ok = errs <= max(g["pos_tol"], errs[pos] + 2.0)
    return d.astype(np.float32), pos, ok


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn

    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--usd", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tilt", type=float, default=12.0, help="max Kipp-Stoerung (Grad)")
    ap.add_argument("--pos-tol", type=float, default=8.0, help="sym-Toleranz fuer Positive (Grad)")
    ap.add_argument("--scale", type=float, default=0.001)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--render-tris", type=int, default=1200,
                    help="Render-Mesh-Triangle-Cap fuer Query-Sampling (Speed)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(7)
    spec = PART_SYMMETRY.get(a.part, _DEFAULT_SYM)
    log(f"part={a.part} dev={dev} sym={spec}")

    # ---- Bank laden (Templates + ihre R) ----
    bd = np.load(a.bank)
    bank_depth = bd["depth"].astype(np.float32)              # (T,S,S)
    bank_R = bd["R_world"].astype(np.float32).reshape(-1, 3, 3)
    T = len(bank_R); S = int(bd["size"])
    log(f"bank: {T} Templates, size {S}")
    bank_depth_t = torch.from_numpy(bank_depth)[:, None].to(dev)   # (T,1,S,S)

    # ---- Mesh + stabile Lagen (fuer Query-Sampling) ----
    import trimesh
    V, F = load_usd_mesh(a.usd)
    V = np.asarray(V, np.float64) * a.scale; V = V - V.mean(0, keepdims=True)
    F = np.asarray(F, np.int64)
    tm = trimesh.Trimesh(vertices=V, faces=F, process=True)
    Ts, Ps = trimesh.poses.compute_stable_poses(tm, n_samples=4, threshold=0.0)
    order = np.argsort(-Ps)[:24]
    Rstab = [np.asarray(Ts[i][:3, :3], float) for i in order]
    Pstab = np.array([Ps[i] for i in order]); Pstab = Pstab / Pstab.sum()
    cap = a.render_tris
    Vr = V
    Fr = F if len(F) <= cap else F[np.random.default_rng(0).choice(len(F), cap, replace=False)]
    log(f"{len(Rstab)} stabile Lagen fuer Query-Sampling, render-mesh {len(Fr)} tris")

    # ---- Trainings-Samples generieren (depth-crop, GT R, positiv-template-idx) ----
    world_ops, body_ops = _sym_variants(spec)         # vorab, vektorisiertes Scoren
    log(f"sym-ops: {len(world_ops)} Welt-Flips x {len(body_ops)} Körper-Rot")

    # Sample-Generierung über CPU-Kerne parallelisieren (Python-Rasterizer ist der
    # Flaschenhals). Worker erben den read-only State über globals (fork).
    _G.update(dict(Rstab=Rstab, Pstab=Pstab, Vr=Vr, Fr=Fr, S=S, tilt=a.tilt,
                   bank_R=bank_R, world_ops=world_ops, body_ops=body_ops,
                   pos_tol=a.pos_tol))

    log(f"generiere {a.samples} Trainings-Samples (multiprocess) ...")
    Xd = np.zeros((a.samples, S, S), np.float32)
    Pos = np.zeros(a.samples, np.int64)
    OkMask = np.zeros((a.samples, T), bool)
    t0 = time.time()
    import multiprocessing as mp
    nproc = min(mp.cpu_count(), 16)
    with mp.Pool(nproc) as pool:
        for i, (d, pos, ok) in enumerate(pool.imap_unordered(_gen_one, range(a.samples), chunksize=32)):
            Xd[i] = d; Pos[i] = pos; OkMask[i] = ok
            if (i + 1) % 1000 == 0:
                log(f"  {i+1}/{a.samples} ({(time.time()-t0)/(i+1)*1000:.1f} ms/sample, {nproc} proc)")
    Xd_t = torch.from_numpy(Xd)[:, None]
    Pos_t = torch.from_numpy(Pos)
    Ok_t = torch.from_numpy(OkMask)

    # ---- Embedding-Netz (shared encoder fuer Query und Templates) ----
    def conv(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co),
                             nn.ReLU(True), nn.Conv2d(co, co, 3, padding=1),
                             nn.BatchNorm2d(co), nn.ReLU(True), nn.MaxPool2d(2))

    class Enc(nn.Module):
        def __init__(self, emb):
            super().__init__()
            self.f = nn.Sequential(conv(1, 32), conv(32, 64), conv(64, 96), conv(96, 128))
            self.h = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                   nn.Linear(128, emb))

        def forward(self, x):
            z = self.h(self.f(x))
            return Fnn.normalize(z, dim=1)

    net = Enc(a.emb).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    tau = 0.1
    N = a.samples
    log("training ...")
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(N)
        tot = 0.0; nb = 0; correct = 0; seen = 0
        # template-embeddings einmal pro epoch (encoder shared)
        for bi in range(0, N, a.batch):
            idx = perm[bi:bi + a.batch]
            xq = Xd_t[idx].to(dev)
            zt = net(bank_depth_t)                  # (T,emb)
            zq = net(xq)                            # (B,emb)
            logits = zq @ zt.t() / tau              # (B,T)
            pos = Pos_t[idx].to(dev)
            ok = Ok_t[idx].to(dev)                  # (B,T) korrekte (nicht bestrafen)
            # InfoNCE: maskiere alle OK-ausser-pos aus den Negativen (-inf)
            mask = ok.clone()
            mask.scatter_(1, pos[:, None], False)   # pos bleibt erlaubt
            logits = logits.masked_fill(mask, float("-inf"))
            loss = Fnn.cross_entropy(logits, pos)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
            with torch.no_grad():
                pred = logits.argmax(1)
                # "korrekt" = pred liegt im OK-Set
                correct += int(ok.gather(1, pred[:, None]).sum()); seen += len(idx)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            log(f"  ep {ep+1}/{a.epochs} loss={tot/nb:.4f} top1-in-OK={correct/seen:.3f}")

    os.makedirs(a.out, exist_ok=True)
    ckpt = os.path.join(a.out, f"refiner_{a.part}.pt")
    net.eval()
    with torch.no_grad():
        zt = net(bank_depth_t).cpu().numpy()
    torch.save({"state": net.state_dict(), "emb": a.emb, "size": S,
                "part": a.part, "bank_emb": zt.astype(np.float32)}, ckpt)
    log(f"saved -> {ckpt}  (bank_emb {zt.shape})")


if __name__ == "__main__":
    main()
