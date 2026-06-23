# Blackwell (RTX 5070) needs CUDA 12.8+. Use NVIDIA's CUDA 12.8 runtime base;
# the NVIDIA Container Toolkit on the host exposes the GPU at run time.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # ultralytics writes config/weights here; keep it inside the image.
    YOLO_CONFIG_DIR=/opt/ultralytics

# ffmpeg/ffprobe for frame extraction; python + pip; libGL for opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv (Ubuntu 24.04 marks the system python externally-managed).
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install CUDA 12.8 torch first so ultralytics binds to the GPU build.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision

COPY requirements.txt .
RUN pip install -r requirements.txt

# Writable ultralytics config dir, and bake the YOLO weights into the image so
# ephemeral `docker compose run --rm worker` never re-downloads them at runtime.
RUN mkdir -p "$YOLO_CONFIG_DIR" \
    && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY config.yaml pipeline.py ./

# Default: run the full pipeline. SD_MOUNT/DATA_DIR come from compose env.
ENTRYPOINT ["python", "pipeline.py"]
CMD ["run"]
