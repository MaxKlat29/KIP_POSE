FROM gigapose:blackwell

WORKDIR /svc

COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV GP_REPO=/workspace/GigaPose \
    GP_ROOT_DIR=/workspace/GigaPose \
    GP_DATASET=kip2 \
    PYTHONPATH=/workspace/GigaPose \
    TORCH_HOME=/workspace/GigaPose/torch_cache \
    XFORMERS_DISABLED=1 \
    CONDA_PREFIX=/opt/venv \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    GP_ENABLE_REFINER=1
# GP_ENABLE_REFINER=1 -> full coarse + MegaPose "GenFlow" refiner (+ ICP/Kabsch).
# The refiner's panda3d offscreen rendering previously deadlocked headless because
# the multiprocess Panda3dBatchRenderer forks EGL worker processes inside
# forward_refiner. It now uses the in-process Panda3dSingleProcessRenderer
# (GigaPose: src/megapose/panda3d_renderer/panda3d_single_renderer.py, selected by
# GP_REFINER_RENDERER=single, the default), validated on Blackwell. Set
# GP_ENABLE_REFINER=0 for coarse-only, or GP_REFINER_RENDERER=batch for the old
# (deadlocking) multiprocess renderer.

EXPOSE 8003

CMD ["/opt/venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
