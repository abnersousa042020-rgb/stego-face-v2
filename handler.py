"""
StegoAudio — Adversarial Face Worker v3
Pixel-space optimization: perturbation applied directly to frame pixels.
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

print("[INIT] Warming up...")
with torch.no_grad():
    _ = FACENET(torch.randn(1, 3, 160, 160).to(DEVICE))
print("[INIT] Ready.")


def optimize_frame_pixel(frame_bgr, orig_emb, epsilon, iterations, target_similarity):
    """
    Optimize perturbation in PIXEL SPACE (not aligned space).
    This survives H.264 re-encoding because the perturbation maps 1:1 to pixels.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    boxes, _ = MTCNN_MODEL.detect(pil)
    if boxes is None or len(boxes) == 0:
        return None, None, 1.0

    box = boxes[0].astype(int)
    h, w = frame_bgr.shape[:2]
    x1 = max(0, box[0] - 20)
    y1 = max(0, box[1] - 20)
    x2 = min(w, box[2] + 20)
    y2 = min(h, box[3] + 20)
    rh, rw = y2 - y1, x2 - x1

    # Crop face from frame, resize to 160x160, convert to tensor
    face_crop = frame_bgr[y1:y2, x1:x2].copy()
    crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    crop_160 = cv2.resize(crop_rgb, (160, 160)).astype(np.float32) / 127.5 - 1.0
    crop_tensor = torch.from_numpy(crop_160).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Optimize perturbation on the raw crop (pixel space)
    delta = torch.nn.Parameter(torch.randn_like(crop_tensor) * 0.01)
    optimizer = torch.optim.Adam([delta], lr=0.05)

    best_sim = 1.0
    best_delta = torch.zeros_like(delta)

    for i in range(iterations):
        optimizer.zero_grad()
        adv = torch.clamp(crop_tensor + delta, -1, 1)
        with torch.enable_grad():
            emb = FACENET(adv).squeeze()
        sim = torch.dot(orig_emb.detach(), F.normalize(emb, dim=0))
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

    # Convert delta to pixel-space noise for the face region
    with torch.no_grad():
        delta_np = best_delta.squeeze().permute(1, 2, 0).cpu().numpy() * 127.5
        delta_full = cv2.resize(delta_np, (rw, rh))
        delta_bgr = delta_full[:, :, ::-1].copy()
        # Soft elliptical mask
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float64)
        mask = np.exp(-((xx - rw/2)**2 / (rw*0.42)**2 + (yy - rh/2)**2 / (rh*0.42)**2))
        delta_bgr = (delta_bgr * mask[:, :, np.newaxis]).astype(np.float32)

    return (x1, y1, x2, y2), delta_bgr, best_sim


def handler(job):
    t0 = time.time()
    try:
        input_data = job["input"]
        video_b64 = input_data.get("video_b64")
        video_url = input_data.get("video_url")
        epsilon = input_data.get("epsilon", 0.30)
        iterations = input_data.get("iterations", 200)
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

        # Get original embedding from first good frame
        print(f"[{time.time()-t0:.1f}s] Getting original embedding...")
        orig_emb = None
        for fi in range(0, min(total_frames, int(fps * 5)), max(1, int(fps))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = MTCNN_MODEL(Image.fromarray(rgb))
            if face is not None:
                with torch.no_grad():
                    orig_emb = F.normalize(FACENET(face.unsqueeze(0).to(DEVICE)).squeeze(), dim=0)
                break

        if orig_emb is None:
            cap.release()
            os.unlink(video_path)
            return {"error": "No face detected for embedding"}

        # Optimize every second
        opt_interval = max(1, int(fps))
        opt_frames = list(range(0, total_frames, opt_interval))
        print(f"[{time.time()-t0:.1f}s] Optimizing {len(opt_frames)} key frames (pixel-space)...")

        key_data = {}
        for idx in opt_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            box, noise, sim = optimize_frame_pixel(frame, orig_emb, epsilon, iterations, target_similarity)
            if box is not None:
                key_data[idx] = (box, noise, sim)
            if len(key_data) % 5 == 0 and len(key_data) > 0:
                print(f"  [{time.time()-t0:.1f}s] {len(key_data)}/{len(opt_frames)} frames, last sim={sim:.4f}")

        if not key_data:
            cap.release()
            os.unlink(video_path)
            return {"error": "No faces optimized"}

        sims = [v[2] for v in key_data.values()]
        avg_sim = float(np.mean(sims))
        print(f"[{time.time()-t0:.1f}s] Optimization done. Avg sim: {avg_sim:.4f}")

        # Apply to all frames
        print(f"[{time.time()-t0:.1f}s] Applying to all frames...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        temp_out = video_path + '_out.mp4'
        writer = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))

        opt_indices = sorted(key_data.keys())
        for fi in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            nearest = min(opt_indices, key=lambda k: abs(k - fi))
            box, noise, _ = key_data[nearest]
            bx1, by1, bx2, by2 = box
            rh, rw = by2 - by1, bx2 - bx1
            n = noise
            if n.shape[0] != rh or n.shape[1] != rw:
                n = cv2.resize(n, (rw, rh))
            region = frame[by1:by2, bx1:bx2].astype(np.float32) + n
            frame[by1:by2, bx1:bx2] = np.clip(region, 0, 255).astype(np.uint8)
            writer.write(frame)

        cap.release()
        writer.release()

        # Re-encode
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

        with open(final_out, 'rb') as f:
            result_b64 = base64.b64encode(f.read()).decode()

        for p in [video_path, temp_out, final_out]:
            try: os.unlink(p)
            except: pass

        elapsed = time.time() - t0
        print(f"[{elapsed:.1f}s] DONE! Avg sim: {avg_sim:.4f}, {len(key_data)} key frames")

        return {
            "video_b64": result_b64,
            "avg_similarity_before": 1.0,
            "avg_similarity_after": round(avg_sim, 4),
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
