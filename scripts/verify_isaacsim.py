#!/usr/bin/env python3
"""
Headless smoke test for an Isaac Sim 5.x install.

Boots SimulationApp headless, builds a trivial scene (light + cube + camera),
renders a few frames via Replicator and writes one PNG. Verbose + unbuffered so
the exact step is visible in the log — the first RTX render triggers a full
shader compilation that can take several minutes (GPU may look idle).

Run inside the Isaac Sim venv:
    /mnt/data/isaacsim-venv/bin/python -u scripts/verify_isaacsim.py
"""
import os
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"


def log(msg):
    print(f"[verify {time.strftime('%T')}] {msg}", flush=True)


from isaacsim import SimulationApp  # noqa: E402

log("booting SimulationApp (headless) ...")
simulation_app = SimulationApp({"headless": True})
log("SimulationApp ready")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import carb  # noqa: E402

# Force synchronous rendering — async paths can deadlock on a headless capture.
_s = carb.settings.get_settings()
_s.set("/app/asyncRendering", False)
_s.set("/app/asyncRenderingLowLatency", False)
_s.set("/omni/replicator/asyncRendering", False)

OUT = os.environ.get("VERIFY_OUT", "/mnt/data/isaacsim_smoke")
os.makedirs(OUT, exist_ok=True)

log("creating new stage")
omni.usd.get_context().new_stage()
for _ in range(10):
    simulation_app.update()

log("building scene: distant light + cube + camera")
rep.create.light(light_type="distant", intensity=3000)
rep.create.cube(position=(0, 0, 0), scale=0.3)
cam = rep.create.camera(position=(2.0, 2.0, 2.0), look_at=(0, 0, 0))
render_product = rep.create.render_product(cam, (512, 512))

log("attaching rgb annotator")
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([render_product])

log("warmup app.update() x30")
for i in range(30):
    simulation_app.update()
log("warmup done — calling rep.orchestrator.step() "
    "(first call compiles RTX shaders, can take minutes) ...")
rep.orchestrator.step(rt_subframes=16)
log("rep.orchestrator.step() returned")

data = rgb.get_data()
log(f"annotator data: type={type(data).__name__} shape={getattr(data, 'shape', None)}")
if data is not None and getattr(data, "size", 0) > 0:
    img = Image.fromarray(data[..., :3].astype(np.uint8))
    path = os.path.join(OUT, "smoke_rgb.png")
    img.save(path)
    log(f"OK -> wrote {path}")
else:
    log("FAILED -> annotator returned no data")

simulation_app.close()
log("closed")
