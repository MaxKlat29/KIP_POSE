FROM foundationpose:blackwell

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV FP_REPO=/workspace/FoundationPose \
    FP_MESH_DIR=/assets/meshes \
    PYTHONPATH=/workspace/FoundationPose \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYOPENGL_PLATFORM=egl \
    OPENCV_IO_ENABLE_OPENEXR=1

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
