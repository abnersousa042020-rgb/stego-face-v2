"""
StegoAudio — Adversarial Face Worker v1 (DEBUG BUILD)
Simplified to isolate the crash.
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
for p in FACENET.parameters():
    p.requires_grad = False

# Warmup
print("[INIT] Warming up...")
with torch.no_grad():
    dummy = torch.randn(1, 3, 160, 160).to(DEVICE)
    _ = FACENET(dummy)
    del dummy
print("[INIT] Ready.")


def handler(job):
    t0 = time.time()
    try:
        input_data = job["input"]
        video_b64 = input_data.get("video_b64")
        epsilon = input_data.get("epsilon", 0.15)
        iterations = input_data.get("iterations", 100)

        if not video_b64:
            return {"error": "Missing video_b64"}

        print(f"[{time.time()-t0:.1f}s] Decoding video...")
        video_bytes = base64.b64decode(video_b64)
        tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        tmp.write(video_bytes)
        tmp.close()
        video_path = tmp.name

        print(f"[{time.time()-t0:.1f}s] Opening video...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[{time.time()-t0:.1f}s] Video: {w}x{h} @ {fps}fps, {total_frames} frames")

        # Read middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            os.unlink(video_path)
            return {"error": "Could not read frame"}

        print(f"[{time.time()-t0:.1f}s] Detecting face...")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        face_tensor = MTCNN_MODEL(pil_img)

        if face_tensor is None:
            os.unlink(video_path)
            return {"error": "No face detected", "time": time.time()-t0}

        print(f"[{time.time()-t0:.1f}s] Getting embedding...")
        with torch.no_grad():
            orig_emb = FACENET(face_tensor.unsqueeze(0).to(DEVICE)).squeeze()
            orig_emb = F.normalize(orig_emb, dim=0)

        print(f"[{time.time()-t0:.1f}s] Running {iterations} optimization steps...")
        face_input = face_tensor.unsqueeze(0).to(DEVICE).clone().detach()
        delta = torch.zeros_like(face_input, requires_grad=True, device=DEVICE)
        optimizer = torch.optim.Adam([delta], lr=0.01)

        best_sim = 1.0
        best_delta = None

        for i in range(iterations):
            optimizer.zero_grad()
            adv_face = torch.clamp(face_input + delta, -1, 1)
            adv_emb = FACENET(adv_face).squeeze()
            adv_emb_norm = F.normalize(adv_emb, dim=0)
            similarity = torch.dot(orig_emb, adv_emb_norm)
            loss = similarity + 0.05 * delta.norm()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon)
            sim_val = similarity.item()
            if sim_val < best_sim:
                best_sim = sim_val
                best_delta = delta.clone().detach()
            if (i+1) % 25 == 0:
                print(f"  Step {i+1}: sim={sim_val:.4f} best={best_sim:.4f}")

        print(f"[{time.time()-t0:.1f}s] Optimization done. Best sim: {best_sim:.4f}")

        # Apply perturbation to ALL frames and write output
        print(f"[{time.time()-t0:.1f}s] Applying to video frames...")

        # Get face box for the frame
        boxes, probs = MTCNN_MODEL.detect(pil_img)
        if boxes is None:
            os.unlink(video_path)
            return {"error": "No face box", "best_similarity": best_sim}

        box = boxes[0].astype(int)
        x1, y1, x2, y2 = box
        margin = 30
        x1, y1 = max(0, x1-margin), max(0, y1-margin)
        x2, y2 = min(w, x2+margin), min(h, y2+margin)

        # Compute noise in pixel space
        with torch.no_grad():
            adv_aligned = torch.clamp(face_input + best_delta, -1, 1)
            orig_np = ((face_input.squeeze().permute(1,2,0).cpu().numpy() + 1) * 127.5)
            adv_np = ((adv_aligned.squeeze().permute(1,2,0).cpu().numpy() + 1) * 127.5)
            diff = adv_np - orig_np
            rh, rw = y2-y1, x2-x1
            noise = cv2.resize(diff, (rw, rh), interpolation=cv2.INTER_LINEAR)
            # RGB to BGR
            noise = noise[:,:,::-1].copy()
            # Soft mask
            yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float64)
            cx, cy = rw/2, rh/2
            mask = np.exp(-((xx-cx)**2/(cx*0.85)**2 + (yy-cy)**2/(cy*0.85)**2))
            noise = (noise * mask[:,:,np.newaxis]).astype(np.float32)

        print(f"[{time.time()-t0:.1f}s] Writing output video...")
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        temp_out = video_path + '_out.mp4'
        writer = cv2.VideoWriter(temp_out, fourcc, fps, (w, h))

        while True:
            ret, f = cap.read()
            if not ret:
                break
            face_area = f[y1:y2, x1:x2].astype(np.float32) + noise
            f[y1:y2, x1:x2] = np.clip(face_area, 0, 255).astype(np.uint8)
            writer.write(f)

        cap.release()
        writer.release()

        # Re-encode with H.264 + audio
        print(f"[{time.time()-t0:.1f}s] Re-encoding...")
        final_out = video_path + '_final.mp4'

        # Check for audio
        probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-select_streams', 'a', '-show_entries', 'stream=codec_type', video_path],
            capture_output=True, text=True
        )
        has_audio = 'audio' in probe.stdout

        if has_audio:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_out, '-i', video_path,
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-b:a', '128k', '-map', '0:v:0', '-map', '1:a:0', '-shortest',
                final_out
            ], capture_output=True)
        else:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_out,
                '-c:v', 'libx264', '-crf', '18', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                '-an', final_out
            ], capture_output=True)

        if not os.path.exists(final_out):
            # Fallback
            subprocess.run(['ffmpeg', '-y', '-i', temp_out, '-c:v', 'libx264', '-crf', '18', '-an', final_out], capture_output=True)

        print(f"[{time.time()-t0:.1f}s] Reading output...")
        with open(final_out, 'rb') as f:
            result_b64 = base64.b64encode(f.read()).decode()

        # Cleanup
        for p in [video_path, temp_out, final_out]:
            try: os.unlink(p)
            except: pass

        elapsed = time.time() - t0
        print(f"[{elapsed:.1f}s] DONE! Similarity: {best_sim:.4f}")

        return {
            "video_b64": result_b64,
            "avg_similarity_before": 1.0,
            "avg_similarity_after": round(best_sim, 4),
            "success": best_sim < 0.45,
            "epsilon": epsilon,
            "iterations": iterations,
            "frames_processed": total_frames,
            "execution_seconds": round(elapsed, 1),
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[FATAL] {e}\n{tb}")
        return {"error": str(e), "traceback": tb[:1500]}


runpod.serverless.start({"handler": handler})
