# yolo-obb-svc — lightweight OBB seg source (Pipeline-A seg, combo 1 -> GDRNPP).
#
# Unlike fp-svc/gigapose-svc/yolo-svc this does NOT need the heavy ampere/blackwell
# base images (no pytorch3d / nvdiffrast / xformers source builds). Ultralytics +
# stock PyTorch CUDA wheels are enough — the official torch CUDA build already
# ships sm_86 (Ampere / RTX 3090) kernels, so no recompile is required.
#
# Base: pytorch runtime (cu124, torch 2.5) already has torch + CUDA + cudnn.
# We only add ultralytics + FastAPI on top -> small, fast build.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# opencv-python-headless needs libGL/glib at runtime even in headless mode.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Avoid ultralytics trying to write config/telemetry to a non-writable HOME.
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    MPLCONFIGDIR=/tmp/mpl

EXPOSE 8011
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8011"]
