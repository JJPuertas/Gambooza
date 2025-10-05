# app/transcode.py
import os, uuid, shutil, subprocess
import cv2

def _can_read(path: str) -> bool:
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return False
    ok = True
    for _ in range(3):
        ret, _ = cap.read()
        if not ret:
            ok = False
            break
    cap.release()
    return ok

def ensure_h264_fast(src_path: str) -> str:
    # Si se puede leer, no convertimos (rápido).
    if _can_read(src_path):
        return src_path

    # Si no, convertir con preset rápido.
    out = os.path.join(os.path.dirname(src_path), f"{uuid.uuid4()}_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_path,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an", out
    ]
    subprocess.run(cmd, check=True)
    return out
