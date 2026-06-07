# Ampere (RTX 3090, sm_86) build. The original Yannic Dockerfile used a
# foundationpose:blackwell-sam3 base that only exists on the Blackwell box and
# that does NOT exist here. foundationpose:ampere (torch 2.11.0+cu128, sm_86
# kernels — S-001/S-006) is the right local base; it has NO transformers, so
# SAM3 (Sam3Model/Sam3Processor, landed in transformers v5) is installed here.
# transformers 5.10.2 imports Sam3Model/Sam3Processor cleanly against this torch
# (verified on the box before baking). Build:  docker compose build sam3-svc.
FROM foundationpose:ampere

# Keep the base image's prebuilt venv (torch cu128) on PATH.
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# requirements.txt pins transformers>=5 (Sam3*) + the FastAPI/cv stack. torch
# stays the base image's cu128 build — never reinstalled here.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8004
# Default offline so the container never reaches the network even if env is
# mis-set; the gated facebook/sam3 snapshot is mounted read-only (compose).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
