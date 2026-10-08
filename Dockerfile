# Serving image: CPU-only torch for small size. Artifacts mounted or copied in.
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
# Optional: bake prebuilt artifacts (index + encoder) into the image.
# Comment out if you mount them at deploy time instead.
COPY artifacts/ artifacts/

ENV INDEX_PATH=artifacts/index.faiss \
    METADATA_PATH=artifacts/metadata.json \
    ENCODER_PATH=artifacts/pointnet.pt \
    MODEL=pointnet \
    NUM_POINTS=1024

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
