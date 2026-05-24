#!/usr/bin/env python3
"""T-038 smoke proof: visualise that the re-gen renders (a) EMPTY 0-GT scenes
cleanly and (b) parts spread across the WHOLE TABLE (Marc-style spawn extent),
not the narrow inner-table zone.

Reads a smoke render dir (gt_raw_*.json + rgb_*.png) and writes one montage:
  - left:  scatter of every part's world XY (from T_obj2world) over the full
           table, with Marc's SPAWN_BOUNDS rectangle + the OLD narrow inner-table
           rectangle drawn for comparison. Empty scenes contribute 0 points.
  - right: the RGB of the EMPTY scene (proves the bare cell renders, 0 GT).

No torch / Isaac needed — pure json + matplotlib + PIL. Run in any venv with
matplotlib + pillow (faces-venv works).
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# Marc datagenerationscript.py SPAWN_BOUNDS (random mode) = full table
MARC = {"x": (0.05732, 0.78332), "y": (0.04157, 0.53698)}
# old narrow inner-table zone we replaced
OLD = {"x": (0.18, 0.52), "y": (0.08, 0.50)}
FOCUS = {"Anker_Kurz", "Anker_Lang", "Zahnrad"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gts = sorted(glob.glob(os.path.join(a.smoke_dir, "gt_raw_*.json")))
    assert gts, f"no gt_raw_*.json in {a.smoke_dir}"

    pts_focus, pts_distract = [], []
    per_scene = []
    empty_idx = None
    for g in gts:
        d = json.load(open(g))
        sid = d["image_id"]
        insts = d.get("instances", [])
        n = len(insts)
        per_scene.append((sid, n))
        if n == 0 and empty_idx is None:
            empty_idx = sid
        for it in insts:
            T = np.array(it["T_obj2world"], float)  # 4x4 row-major obj->world
            x, y = float(T[0, 3]), float(T[1, 3])   # world translation XY (metres)
            (pts_focus if it["label"] in FOCUS else pts_distract).append((x, y))

    pf = np.array(pts_focus) if pts_focus else np.zeros((0, 2))
    pd = np.array(pts_distract) if pts_distract else np.zeros((0, 2))

    print("=== SMOKE PROOF ===")
    for sid, n in per_scene:
        tag = "  <-- EMPTY 0-GT" if n == 0 else ""
        print(f"  scene {sid:04d}: {n} parts{tag}")
    allpts = np.vstack([pf, pd]) if (len(pf) + len(pd)) else np.zeros((0, 2))
    if len(allpts):
        print(f"  part XY extent: x [{allpts[:,0].min():.3f}, {allpts[:,0].max():.3f}] "
              f"y [{allpts[:,1].min():.3f}, {allpts[:,1].max():.3f}]  (n={len(allpts)})")
    print(f"  empty scene present: {empty_idx is not None} (idx={empty_idx})")

    has_empty = empty_idx is not None
    fig, axes = plt.subplots(1, 2 if has_empty else 1,
                             figsize=(15 if has_empty else 8, 6))
    if not has_empty:
        axes = [axes]
    ax = axes[0]
    # full-table marc bounds rectangle
    ax.add_patch(Rectangle((MARC["x"][0], MARC["y"][0]),
                           MARC["x"][1] - MARC["x"][0], MARC["y"][1] - MARC["y"][0],
                           fill=False, edgecolor="green", lw=2.5,
                           label="NEW full-table (Marc SPAWN_BOUNDS)"))
    ax.add_patch(Rectangle((OLD["x"][0], OLD["y"][0]),
                           OLD["x"][1] - OLD["x"][0], OLD["y"][1] - OLD["y"][0],
                           fill=False, edgecolor="red", lw=1.8, ls="--",
                           label="OLD narrow inner-table"))
    if len(pd):
        ax.scatter(pd[:, 0], pd[:, 1], c="tab:gray", s=45, alpha=0.7,
                   edgecolors="k", linewidths=0.4, label="distractor parts")
    if len(pf):
        ax.scatter(pf[:, 0], pf[:, 1], c="tab:blue", s=60, alpha=0.85,
                   edgecolors="k", linewidths=0.5, label="focus parts")
    ax.set_xlabel("world X (m)"); ax.set_ylabel("world Y (m)")
    ax.set_title(f"Settled part positions over the table\n"
                 f"{len(allpts)} parts across {len(per_scene)} smoke scenes")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlim(0.0, 0.83); ax.set_ylim(0.0, 0.58)

    if has_empty:
        rgb_p = os.path.join(a.smoke_dir, f"rgb_{empty_idx:04d}.png")
        ax2 = axes[1]
        if os.path.exists(rgb_p):
            ax2.imshow(Image.open(rgb_p))
        ax2.set_title(f"EMPTY scene rgb_{empty_idx:04d}.png\n"
                      f"(bare cell, arm visible, 0 GT — renders clean, no crash)")
        ax2.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    plt.savefig(a.out, dpi=110, bbox_inches="tight")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
