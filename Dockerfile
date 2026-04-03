FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod facenet-pytorch opencv-python-headless Pillow

# Pre-download FaceNet + MTCNN models into image (avoids 107MB download on cold start)
RUN python -c "from facenet_pytorch import MTCNN, InceptionResnetV1; InceptionResnetV1(pretrained='vggface2'); MTCNN(); print('Models cached')"

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
