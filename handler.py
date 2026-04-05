"""
StegoAudio — Adversarial Face Worker v2
Per-frame optimization: each frame gets its own perturbation.
"""

import runpod
import torch
import torch.nn.functional as F
import numpy as np
import subprocess
import tempfile
import os
import base64
import urllib.request
import cv2
import time
import traceback

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INIT] Device: {DEVICE}")

print("[INIT] Loading FaceNet + MTCNN...")
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

MTCNN_MODEL = MTCNN(image_size=160, margin=20, device=DEVICE, post_process=True, keep_all=False)
FACENET = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Warmup
print("[INIT] Warming up...")
with torch.no_grad():
    _ = FACENET(torch.randn(1, 3, 160, 160).to(DEVICE))
print("[INIT] Ready.")


def optimize_face(frame_bgr, epsilon, iterations, target_similarity):
    """Optimize perturbation for a single frame. Returns noise_bgr (float32) and similarity."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Detect face
    face_tensor = MTCNN_MODEL(pil_img)
    if face_tensor is None:
        return None, None, 1.0

    boxes, _ = MTCNN_MODEL.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return None, None, 1.0

    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box
    margin = 30
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)

    # Original embedding
    with torch.no_grad():
        orig_emb = FACENET(face_tensor.unsqueeze(0).to(DEVICE)).squeeze()
        orig_emb = F.normalize(orig_emb, dim=0)

    # Optimize
    face_input = face_tensor.unsqueeze(0).to(DEVICE).clone().detach()
    delta = torch.nn.Parameter(torch.randn_like(face_input) * 0.01)
    optimizer = torch.optim.Adam([delta], lr=0.05)

    best_sim = 1.0
    best_delta = torch.zeros_like(face_input)

    for i in range(iterations):
        optimizer.zero_grad()
        adv_face = torch.clamp(face_input + delta, -1, 1)
        with torch.enable_grad():
            adv_emb = FACENET(adv_face).squeeze()
        sim = torch.dot(orig_emb.detach(), F.normalize(adv_emb, dim=0))
        sim.backward()
        optimizer.step()
        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)
        sv = sim.item()
        if sv < best_sim:
            best_sim = sv
            best_delta = delta.data.clone()
        if best_sim < target_similarity:
            break

    # Convert to pixel-space noise
    with torch.no_grad():
        adv_aligned = torch.clamp(face_input + best_delta, -1, 1)
        orig_np = ((face_input.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
        adv_np = ((adv_aligned.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
        diff = adv_np - orig_np
        rh, rw = y2 - y1, x2 - x1
        noise = cv2.resize(diff, (rw, rh), interpolation=cv2.INTER_LINEAR)
        noise = noise[:, :, ::-1].copy()  # RGB to BGR
        # Soft elliptical mask
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float64)
        cx, cy = rw / 2, rh / 2
        mask = np.exp(-((xx - cx) ** 2 / (cx * 0.85) ** 2 + (yy - cy) ** 2 / (cy * 0.85) ** 2))
        noise = (noise * mask[:, :, np.newaxis]).astype(np.float32)

    return (x1, y1, x2, y2), noise, best_sim


def handler(job):
    t0 = time.time()
    try:
        input_data = job["input"]
        video_b64 = input_data.get("video_b64")
        video_url = input_data.get("video_url")
        epsilon = input_data.get("epsilon", 0.25)
        iterations = input_data.get("iterations", 300)
        target_similarity = input_data.get("target_similarity", 0.35)

        if not video_b64 and not video_url:
            return {"error": "Missing video_b64 or video_url"}

        if video_url:
            print(f"[{time.time()-t0:.1f}s] Downloading video...")
            tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            urllib.request.urlretrieve(video_url, tmp.name)
            video_path = tmp.name
        else:
            print(f"[{time.time()-t0:.1f}s] Decoding video...")
            video_bytes = base64.b64decode(video_b64)
            tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            tmp.write(video_bytes)
            tmp.close()
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[{time.time()-t0:.1f}s] Video: {w}x{h} @ {fps:.0f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

        # Optimize every N frames (1 per second), interpolate between
        opt_interval = max(1, int(fps))  # 1 per second
        opt_frames = list(range(0, total_frames, opt_interval))
        print(f"[{time.time()-t0:.1f}s] Optimizing {len(opt_frames)} key frames...")

        # Read key frames
        key_data = {}  # frame_idx -> (box, noise, similarity)
        for idx in opt_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            box, noise, sim = optimize_face(frame, epsilon, iterations, target_similarity)
            if box is not None:
                key_data[idx] = (box, noise, sim)
                if len(key_data) % 5 == 0:
                    print(f"  [{time.time()-t0:.1f}s] Optimized {len(key_data)}/{len(opt_frames)} frames, last sim={sim:.4f}")

        if not key_data:
            cap.release()
            os.unlink(video_path)
            return {"error": "No faces detected"}

        sims = [v[2] for v in key_data.values()]
        avg_sim = np.mean(sims)
        print(f"[{time.time()-t0:.1f}s] Done. Avg similarity: {avg_sim:.4f} ({len(key_data)} frames)")

        # Apply perturbation to ALL frames
        print(f"[{time.time()-t0:.1f}s] Applying to all {total_frames} frames...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        temp_out = video_path + '_out.mp4'
        writer = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))

        opt_indices = sorted(key_data.keys())
        for fi in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Find nearest optimized frame
            nearest = min(opt_indices, key=lambda k: abs(k - fi))
            box, noise, _ = key_data[nearest]
            x1, y1, x2, y2 = box

            # Resize noise if needed
            rh, rw = y2 - y1, x2 - x1
            if noise.shape[0] != rh or noise.shape[1] != rw:
                noise_r = cv2.resize(noise, (rw, rh), interpolation=cv2.INTER_LINEAR)
            else:
                noise_r = noise

            # Apply
            region = frame[y1:y2, x1:x2].astype(np.float32) + noise_r
            frame[y1:y2, x1:x2] = np.clip(region, 0, 255).astype(np.uint8)
            writer.write(frame)

        cap.release()
        writer.release()

        # Re-encode with H.264 + original audio
        print(f"[{time.time()-t0:.1f}s] Re-encoding...")
        final_out = video_path + '_final.mp4'
        probe = subprocess.run(['ffprobe', '-v', 'quiet', '-select_streams', 'a', '-show_entries', 'stream=codec_type', video_path], capture_output=True, text=True)
        has_audio = 'audio' in probe.stdout

        if has_audio:
            subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-i', video_path, '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-map', '0:v:0', '-map', '1:a:0', '-shortest', final_out], capture_output=True)
        else:
            subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-an', final_out], capture_output=True)

        if not os.path.exists(final_out):
            subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-c:v', 'libx264', '-crf', '18', '-an', final_out], capture_output=True)

        print(f"[{time.time()-t0:.1f}s] Reading output...")
        with open(final_out, 'rb') as f:
            result_b64 = base64.b64encode(f.read()).decode()

        for p in [video_path, temp_out, final_out]:
            try: os.unlink(p)
            except: pass

        elapsed = time.time() - t0
        print(f"[{elapsed:.1f}s] DONE! Avg similarity: {avg_sim:.4f}, {len(key_data)} key frames")

        return {
            "video_b64": result_b64,
            "avg_similarity_before": 1.0,
            "avg_similarity_after": round(float(avg_sim), 4),
            "success": bool(avg_sim < target_similarity + 0.1),
            "epsilon": epsilon,
            "iterations": iterations,
            "frames_processed": total_frames,
            "key_frames_optimized": len(key_data),
            "execution_seconds": round(elapsed, 1),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[FATAL] {e}\n{tb}")
        return {"error": str(e), "traceback": tb[:1500]}


runpod.serverless.start({"handler": handler})
