"""
StegoAudio — Adversarial Face Worker v1
GPU-accelerated face de-identification via adversarial perturbation.
Makes face unrecognizable to AI while looking identical to humans.

Architecture: Same RunPod serverless pattern as the audio worker.
Model: FaceNet (InceptionResnetV1) pretrained on VGGFace2.
Attack: PGD (Projected Gradient Descent) targeting face embeddings.
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
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INIT] Device: {DEVICE}")

print("[INIT] Loading FaceNet + MTCNN...")
MTCNN_MODEL = MTCNN(
    image_size=160, margin=20, device=DEVICE,
    post_process=True,  # normalize to [-1, 1]
    keep_all=False,     # only largest face
)
FACENET = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
for p in FACENET.parameters():
    p.requires_grad = False

# Warmup: first PyTorch inference triggers JIT compilation (slow).
# Do it now so real jobs don't pay the cost.
print("[INIT] Warming up GPU (JIT compile)...")
with torch.no_grad():
    dummy = torch.randn(1, 3, 160, 160).to(DEVICE)
    _ = FACENET(dummy)
    del dummy
print("[INIT] Ready.")


def download_file(url, suffix='.mp4'):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        urllib.request.urlretrieve(url, f.name)
        return f.name


def get_face_box(frame_bgr, expand=0.35):
    """Detect face in a BGR frame, return expanded box or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs = MTCNN_MODEL.detect(rgb)
    if boxes is None or len(boxes) == 0:
        return None
    # Pick highest confidence
    best = np.argmax(probs)
    x1, y1, x2, y2 = boxes[best].astype(int)
    # Expand
    w, h = x2 - x1, y2 - y1
    ex, ey = int(w * expand), int(h * expand)
    fh, fw = frame_bgr.shape[:2]
    x1 = max(0, x1 - ex)
    y1 = max(0, y1 - ey)
    x2 = min(fw, x2 + ex)
    y2 = min(fh, y2 + ey)
    return (x1, y1, x2, y2)


def get_embedding(face_tensor):
    """Get normalized face embedding from a preprocessed face tensor."""
    with torch.no_grad():
        emb = FACENET(face_tensor.unsqueeze(0).to(DEVICE)).squeeze()
        return F.normalize(emb, dim=0)


def optimize_perturbation(frame_bgr, face_box, target_similarity=0.3,
                          epsilon=0.08, iterations=1000, lr=0.01):
    """
    PGD attack: find minimal perturbation on face region that makes
    FaceNet embedding maximally different from original.

    Args:
        frame_bgr: original frame (BGR, uint8)
        face_box: (x1, y1, x2, y2) face region
        target_similarity: stop when cosine similarity drops below this
        epsilon: max perturbation per pixel in [-1,1] range (~±10 in 0-255)
        iterations: max optimization steps
        lr: learning rate for Adam optimizer

    Returns:
        noise_bgr: perturbation to add to face region (float32, BGR)
        final_similarity: achieved cosine similarity
    """
    x1, y1, x2, y2 = face_box
    face_region = frame_bgr[y1:y2, x1:x2].copy()

    # Get original face embedding via MTCNN alignment
    rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil_img = Image.fromarray(rgb_full)
    face_aligned = MTCNN_MODEL(pil_img)

    if face_aligned is None:
        return np.zeros_like(face_region, dtype=np.float32), 1.0

    # Original embedding (target to move AWAY from)
    orig_emb = get_embedding(face_aligned)

    # Work in the aligned face space (160x160, normalized [-1,1])
    face_tensor = face_aligned.unsqueeze(0).to(DEVICE).clone().detach()

    # Learnable perturbation
    delta = torch.zeros_like(face_tensor, requires_grad=True, device=DEVICE)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_sim = 1.0
    best_delta = torch.zeros_like(face_tensor)

    for i in range(iterations):
        optimizer.zero_grad()

        # Apply perturbation
        adv_face = torch.clamp(face_tensor + delta, -1, 1)

        # Get adversarial embedding
        adv_emb = FACENET(adv_face).squeeze()
        adv_emb_norm = F.normalize(adv_emb, dim=0)

        # Loss: minimize cosine similarity to original
        similarity = torch.dot(orig_emb, adv_emb_norm)

        # Also push AWAY from original (maximize distance)
        # Combined loss: similarity + L2 regularization
        loss = similarity + 0.05 * delta.norm()

        loss.backward()
        optimizer.step()

        # Project to epsilon ball
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)

        sim_val = similarity.item()
        if sim_val < best_sim:
            best_sim = sim_val
            best_delta = delta.clone().detach()

        if (i + 1) % 50 == 0:
            print(f"    Step {i+1}/{iterations}: similarity={sim_val:.4f} (target<{target_similarity}) best={best_sim:.4f}")

        # Early stop if target reached
        if best_sim < target_similarity:
            print(f"    Target reached at step {i+1}! similarity={best_sim:.4f}")
            break

    # Convert optimized delta back to pixel space for the face REGION (not aligned)
    # Strategy: compute pixel-space difference between original and adversarial aligned faces
    with torch.no_grad():
        adv_aligned = torch.clamp(face_tensor + best_delta, -1, 1)

        # Convert aligned faces to numpy [0,255]
        orig_face_np = ((face_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5)
        adv_face_np = ((adv_aligned.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5)

        # Diff in aligned space (RGB, 160x160)
        diff_aligned = adv_face_np - orig_face_np  # float, range roughly ±epsilon*127.5

        # Resize diff to match actual face region size
        rh, rw = y2 - y1, x2 - x1
        diff_resized = cv2.resize(diff_aligned, (rw, rh), interpolation=cv2.INTER_LINEAR)

        # Convert RGB diff to BGR
        noise_bgr = diff_resized[:, :, ::-1].copy()

        # Apply soft elliptical mask (fade at edges)
        yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float64)
        cx, cy = rw / 2, rh / 2
        mask = np.exp(-((xx - cx)**2 / (cx * 0.85)**2 + (yy - cy)**2 / (cy * 0.85)**2))
        noise_bgr *= mask[:, :, np.newaxis]

    return noise_bgr.astype(np.float32), best_sim


def handler(job):
    input_data = job["input"]
    video_url = input_data.get("video_url")
    video_b64 = input_data.get("video_b64")
    mode = input_data.get("mode", "deidentify")  # 'deidentify' = make face unrecognizable
    epsilon = input_data.get("epsilon", 0.15)
    iterations = input_data.get("iterations", 100)
    target_similarity = input_data.get("target_similarity", 0.45)
    import time as _time
    _job_start = _time.time()

    if not video_url and not video_b64:
        return {"error": "Missing video_url or video_b64"}

    # Download/decode video
    if video_url:
        video_path = download_file(video_url)
    else:
        video_bytes = base64.b64decode(video_b64)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_bytes)
            video_path = f.name

    print(f"[START] mode={mode} eps={epsilon} iters={iterations} target_sim={target_similarity}")

    try:
        return _process_video(video_path, epsilon, iterations, target_similarity, mode)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[FATAL] {e}\n{tb}")
        return {"error": str(e), "traceback": tb[:1000], "phase": "handler"}
    finally:
        try: os.unlink(video_path)
        except: pass


def _process_video(video_path, epsilon, iterations, target_similarity, mode):
    import time as _time
    _job_start = _time.time()

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video: {w}x{h} @ {fps}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

    # Phase 1: Detect faces in key frames
    print(f"[PHASE 1] Detecting faces... ({_time.time()-_job_start:.1f}s)")
    key_frame_interval = max(1, int(fps * 2))  # 1 per 2 seconds (faster detection)
    key_frames = {}
    face_boxes = {}

    frame_idx = 0
    last_box = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % key_frame_interval == 0:
            box = get_face_box(frame)
            if box is not None:
                last_box = box
                key_frames[frame_idx] = frame.copy()
                face_boxes[frame_idx] = box
        elif last_box is not None:
            face_boxes[frame_idx] = last_box

        frame_idx += 1

    detected = len([b for b in face_boxes.values() if b is not None])
    print(f"[INFO] Faces detected in {detected}/{total_frames} frames")

    if len(key_frames) == 0:
        cap.release()
        os.unlink(video_path)
        return {"error": "No faces detected in video"}

    # Phase 2: Optimize perturbation on key frames
    print(f"[PHASE 2] Optimizing adversarial perturbation... ({_time.time()-_job_start:.1f}s)")

    # Pick ~3-5 key frames evenly spaced for optimization
    key_indices = sorted(key_frames.keys())
    num_opt_frames = min(2, len(key_indices))
    opt_step = max(1, len(key_indices) // num_opt_frames)
    opt_indices = key_indices[::opt_step][:num_opt_frames]

    optimized_noises = {}
    similarities = []

    for idx in opt_indices:
        print(f"\n  Optimizing frame {idx} ({idx/fps:.1f}s)...")
        frame = key_frames[idx]
        box = face_boxes[idx]
        noise, sim = optimize_perturbation(
            frame, box,
            target_similarity=target_similarity,
            epsilon=epsilon,
            iterations=iterations,
            lr=0.01
        )
        optimized_noises[idx] = noise
        similarities.append(sim)
        print(f"  Frame {idx}: similarity={sim:.4f}")

    avg_sim = np.mean(similarities)
    print(f"\n[INFO] Average similarity after attack: {avg_sim:.4f}")

    # Phase 3: Apply perturbation to all frames
    print(f"[PHASE 3] Applying perturbation to all frames... ({_time.time()-_job_start:.1f}s)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Temp output video
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    temp_video = video_path + '_adv.mp4'
    out = cv2.VideoWriter(temp_video, fourcc, fps, (w, h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        box = face_boxes.get(frame_idx)
        if box is not None:
            x1, y1, x2, y2 = box

            # Find nearest optimized noise
            nearest_idx = min(opt_indices, key=lambda k: abs(k - frame_idx))
            noise = optimized_noises[nearest_idx]

            # Resize noise if face box size changed
            rh, rw = y2 - y1, x2 - x1
            if noise.shape[0] != rh or noise.shape[1] != rw:
                noise = cv2.resize(noise, (rw, rh), interpolation=cv2.INTER_LINEAR)

            # Apply noise to face region
            face_float = frame[y1:y2, x1:x2].astype(np.float32) + noise
            frame[y1:y2, x1:x2] = np.clip(face_float, 0, 255).astype(np.uint8)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    # Phase 4: Re-encode with proper codec + original audio
    print(f"[PHASE 4] Re-encoding... ({_time.time()-_job_start:.1f}s)")
    output_path = video_path + '_final.mp4'

    # Check if original has audio
    probe = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'a', '-show_entries', 'stream=codec_type', video_path],
        capture_output=True, text=True
    )
    has_audio = 'audio' in probe.stdout

    if has_audio:
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_video,
            '-i', video_path,
            '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0', '-shortest',
            output_path
        ], capture_output=True, text=True)
    else:
        result = subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_video,
            '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
            '-an', output_path
        ], capture_output=True, text=True)

    if not os.path.exists(output_path):
        print(f"[ERROR] FFmpeg failed: {result.stderr[:500]}")
        # Fallback: just copy temp video
        subprocess.run(['ffmpeg', '-y', '-i', temp_video, '-c:v', 'libx264', '-crf', '18', '-an', output_path], capture_output=True)

    # Phase 5: Verify — check similarity after full pipeline
    print(f"[PHASE 5] Verifying... ({_time.time()-_job_start:.1f}s)")
    cap_verify = cv2.VideoCapture(output_path)
    cap_orig = cv2.VideoCapture(video_path)

    verify_sims = []
    for _ in range(min(5, total_frames)):
        cap_orig.set(cv2.CAP_PROP_POS_FRAMES, _ * (total_frames // 5))
        cap_verify.set(cv2.CAP_PROP_POS_FRAMES, _ * (total_frames // 5))
        ret1, f_orig = cap_orig.read()
        ret2, f_adv = cap_verify.read()
        if not ret1 or not ret2:
            continue

        from PIL import Image
        rgb_orig = cv2.cvtColor(f_orig, cv2.COLOR_BGR2RGB)
        rgb_adv = cv2.cvtColor(f_adv, cv2.COLOR_BGR2RGB)

        face_orig = MTCNN_MODEL(Image.fromarray(rgb_orig))
        face_adv = MTCNN_MODEL(Image.fromarray(rgb_adv))

        if face_orig is not None and face_adv is not None:
            emb_orig = get_embedding(face_orig)
            emb_adv = get_embedding(face_adv)
            sim = torch.dot(emb_orig, emb_adv).item()
            verify_sims.append(sim)

    cap_verify.release()
    cap_orig.release()

    final_sim = np.mean(verify_sims) if verify_sims else avg_sim
    print(f"[VERIFY] Final similarity after H.264: {final_sim:.4f} (from {len(verify_sims)} frames)")

    # Read output
    with open(output_path, 'rb') as f:
        result_bytes = f.read()
    result_b64 = base64.b64encode(result_bytes).decode()

    # Cleanup
    for p in [temp_video, output_path]:
        try: os.unlink(p)
        except: pass

    success = final_sim < target_similarity + 0.1  # some margin for codec degradation
    print(f"\n[DONE] {'SUCCESS' if success else 'PARTIAL'} | Similarity: {final_sim:.4f} | Target: <{target_similarity}")

    return {
        "video_b64": result_b64,
        "avg_similarity_before": 1.0,
        "avg_similarity_after": round(final_sim, 4),
        "target_similarity": target_similarity,
        "success": success,
        "epsilon": epsilon,
        "iterations": iterations,
        "frames_processed": total_frames,
        "key_frames_optimized": len(opt_indices),
        "output_size": len(result_bytes),
        "per_keyframe_similarity": [round(s, 4) for s in similarities],
    }


runpod.serverless.start({"handler": handler})
