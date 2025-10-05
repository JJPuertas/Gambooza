# app/transcode.py
import os, uuid, shutil, subprocess, tempfile

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def transcode_to_h264(src_path: str) -> str:
    """
    Convierte incondicionalmente a H.264 (yuv420p) para evitar problemas con HEVC.
    Devuelve la ruta del mp4 convertido.
    """
    if not have_ffmpeg():
        raise RuntimeError("ffmpeg no está instalado (conda install -c conda-forge ffmpeg)")
    base = os.path.dirname(src_path)
    out = os.path.join(base, f"{uuid.uuid4()}_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", src_path,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an", out
    ]
    subprocess.run(cmd, check=True)
    return out
