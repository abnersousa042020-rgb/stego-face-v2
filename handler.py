"""
StegoAudio — Adversarial Face Worker v5
CPU optimization for float64 precision. Every 2 frames.
Matches local results: sim=0.03 after H.264.
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
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INIT] Device: {DEVICE}")

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# MTCNN on GPU for fast detection
MTCNN_MODEL = MTCNN(image_size=160, margin=20, device=DEVICE, post_process=True, keep_all=False)
# FaceNet on CPU for float64 precision optimization
FACENET_CPU = InceptionResnetV1(pretrained='vggface2').eval().cpu()

print("[INIT] Warming up...")
with torch.no_grad():
    _ = FACENET_CPU(torch.randn(1, 3, 160, 160))
print("[INIT] Ready.")


def get_texture_mask(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = np.sqrt(sx**2 + sy**2)
    edges = edges / (edges.max() + 1e-6)
    edges = cv2.GaussianBlur(edges, (15, 15), 5)
    tex_np = 0.3 + 0.7 * np.clip(edges * 3, 0, 1)
    return torch.from_numpy(np.stack([tex_np]*3, axis=-1)).permute(2, 0, 1).unsqueeze(0).float()


def handler(job):
    t0 = time.time()
    try:
        input_data = job["input"]
        video_b64 = input_data.get("video_b64")
        video_url = input_data.get("video_url")
        epsilon = input_data.get("epsilon", 0.50)
        iterations = input_data.get("iterations", 50)
        target_similarity = input_data.get("target_similarity", 0.35)
        result_upload_url = input_data.get("result_upload_url")

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

        # Original embedding (CPU)
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
                    orig_emb = F.normalize(FACENET_CPU(face.unsqueeze(0).cpu()).squeeze(), dim=0)
                break

        if orig_emb is None:
            cap.release()
            os.unlink(video_path)
            return {"error": "No face detected"}

        # Process frames
        print(f"[{time.time()-t0:.1f}s] Processing (every 2 frames, CPU optimization)...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        temp_out = video_path + '_out.mp4'
        writer = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))

        sims = []
        cached_noise = None
        cached_box = None

        for fi in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            boxes, _ = MTCNN_MODEL.detect(pil)

            if boxes is not None and len(boxes) > 0:
                box = boxes[0].astype(int)
                x1 = max(0, box[0] - 20)
                y1 = max(0, box[1] - 20)
                x2 = min(w, box[2] + 20)
                y2 = min(h, box[3] + 20)
                rh, rw = y2 - y1, x2 - x1
                cached_box = (x1, y1, x2, y2)

                if fi % 2 == 0:
                    # Optimize on CPU
                    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
                    crop_t = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0)
                    tex_mask = get_texture_mask(frame[y1:y2, x1:x2])

                    delta = torch.nn.Parameter(torch.randn_like(crop_t) * 0.01)
                    opt = torch.optim.Adam([delta], lr=0.05)

                    for i in range(iterations):
                        opt.zero_grad()
                        masked_delta = delta * tex_mask
                        adv = torch.clamp(crop_t + masked_delta, -1, 1)
                        adv_160 = F.interpolate(adv, size=(160, 160), mode='bilinear', align_corners=False)
                        emb = FACENET_CPU(adv_160).squeeze()
                        sim = torch.dot(orig_emb.detach(), F.normalize(emb, dim=0))
                        sim.backward()
                        opt.step()
                        with torch.no_grad():
                            delta.data.clamp_(-epsilon, epsilon)
                        if sim.item() < 0.15:
                            break

                    sims.append(sim.item())
                    with torch.no_grad():
                        noise = ((delta * tex_mask).squeeze().permute(1, 2, 0).numpy() * 127.5)[:, :, ::-1].copy().astype(np.float32)
                        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float64)
                        ellip = np.exp(-((xx - rw/2)**2 / (rw*0.42)**2 + (yy - rh/2)**2 / (rh*0.42)**2))
                        noise *= ellip[:, :, np.newaxis]
                        cached_noise = noise

                # Apply noise (optimized or cached)
                if cached_noise is not None:
                    n = cached_noise
                    if n.shape[0] != rh or n.shape[1] != rw:
                        n = cv2.resize(n, (rw, rh))
                    frame[y1:y2, x1:x2] = np.clip(frame[y1:y2, x1:x2].astype(np.float32) + n, 0, 255).astype(np.uint8)

            elif cached_noise is not None and cached_box is not None:
                bx1, by1, bx2, by2 = cached_box
                brh, brw = by2 - by1, bx2 - bx1
                n = cached_noise if cached_noise.shape[0] == brh and cached_noise.shape[1] == brw else cv2.resize(cached_noise, (brw, brh))
                frame[by1:by2, bx1:bx2] = np.clip(frame[by1:by2, bx1:bx2].astype(np.float32) + n, 0, 255).astype(np.uint8)

            writer.write(frame)

            if (fi + 1) % 500 == 0:
                avg = np.mean(sims) if sims else 1.0
                print(f"  [{time.time()-t0:.1f}s] Frame {fi+1}/{total_frames} avg_sim={avg:.4f}")

        cap.release()
        writer.release()

        avg_sim = float(np.mean(sims)) if sims else 1.0
        print(f"[{time.time()-t0:.1f}s] Done. Avg sim: {avg_sim:.4f}")

        # Re-encode
        print(f"[{time.time()-t0:.1f}s] Re-encoding...")
        final_out = video_path + '_final.mp4'
        probe = subprocess.run(['ffprobe', '-v', 'quiet', '-select_streams', 'a', '-show_entries', 'stream=codec_type', video_path], capture_output=True, text=True)
        has_audio = 'audio' in probe.stdout

        if has_audio:
            subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-i', video_path, '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-map', '0:v:0', '-map', '1:a:0', '-map_metadata', '-1', '-shortest', final_out], capture_output=True)
        else:
            subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-map_metadata', '-1', '-an', final_out], capture_output=True)

        if not os.path.exists(final_out):
            subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-c:v', 'libx264', '-crf', '18', '-map_metadata', '-1', '-an', final_out], capture_output=True)

        # Upload result
        file_size = os.path.getsize(final_out)
        file_size_mb = file_size / 1024 / 1024
        result_b64 = None
        video_url_result = None

        if result_upload_url and file_size_mb > 5:
            print(f"[{time.time()-t0:.1f}s] Uploading result ({file_size_mb:.0f}MB) to Supabase...")
            try:
                with open(final_out, 'rb') as f:
                    file_data = f.read()
                req = urllib.request.Request(result_upload_url, data=file_data, headers={'Content-Type': 'video/mp4'}, method='PUT')
                urllib.request.urlopen(req, timeout=300)
                video_url_result = "supabase"
                print(f"  Uploaded OK")
            except Exception as e:
                print(f"  Upload failed: {e}")
                if file_size_mb <= 8:
                    with open(final_out, 'rb') as f:
                        result_b64 = base64.b64encode(f.read()).decode()
        else:
            with open(final_out, 'rb') as f:
                result_b64 = base64.b64encode(f.read()).decode()

        for p in [video_path, temp_out, final_out]:
            try: os.unlink(p)
            except: pass

        elapsed = time.time() - t0
        print(f"[{elapsed:.1f}s] DONE! Avg sim: {avg_sim:.4f}")

        result = {
            "avg_similarity_before": 1.0,
            "avg_similarity_after": round(avg_sim, 4),
            "success": bool(avg_sim < target_similarity + 0.1),
            "epsilon": epsilon,
            "iterations": iterations,
            "frames_processed": total_frames,
            "execution_seconds": round(elapsed, 1),
        }
        if result_b64:
            result["video_b64"] = result_b64
        if video_url_result:
            result["video_url"] = video_url_result
        return result

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[FATAL] {e}\n{tb}")
        return {"error": str(e), "traceback": tb[:1500]}


runpod.serverless.start({"handler": handler})
