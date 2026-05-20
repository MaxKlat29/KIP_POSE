#!/usr/bin/env python3
"""
Minimal headless smoke test for an Isaac Sim 5.x install.

Boots SimulationApp in headless mode, creates a trivial scene, runs the
Replicator RGB annotator for a few frames and writes one PNG. If this passes,
the synthetic-data-generation pipeline has everything it needs (the real
pipeline just swaps the dummy scene for the USD tray + parts).

Run inside the Isaac Sim venv:
    /mnt/data/isaacsim-venv/bin/python scripts/verify_isaacsim.py
"""
import os

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

OUT = os.environ.get("VERIFY_OUT", "/mnt/data/isaacsim_smoke")
os.makedirs(OUT, exist_ok=True)

# Trivial scene: a light, a camera, a cube.
rep.create.light(light_type="dome")
cube = rep.create.cube(position=(0, 0, 0), scale=0.2)
cam = rep.create.camera(position=(2, 2, 2), look_at=(0, 0, 0))
rp = rep.create.render_product(cam, (640, 480))

rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

# A few warm-up steps so the renderer settles, then capture.
for _ in range(10):
    rep.orchestrator.step(rt_subframes=4)

data = rgb.get_data()
if data is not None and getattr(data, "size", 0) > 0:
    img = Image.fromarray(data[..., :3].astype(np.uint8))
    path = os.path.join(OUT, "smoke_rgb.png")
    img.save(path)
    print(f"[verify] OK -> wrote {path} shape={data.shape}")
else:
    print("[verify] FAILED -> annotator returned no data")

simulation_app.close()
