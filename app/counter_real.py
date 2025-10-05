# app/counter_real.py
# Conteo por eventos usando dos ROIs (A y B) y frame differencing + umbral + suavizado.
# Sin DB, sin hilos: función pura que devuelve (A, B).

import cv2
import numpy as np

class Params:
    target_fps = 12
    resize_width = 640
    # Histeresis y filtros anti-falsos
    th_high = 0.35   # umbral alto (proporción 0..1 de actividad normalizada)
    th_low  = 0.20   # umbral bajo
    min_duration_s = 1.5
    gap_close_s = 0.30
    smooth_win = 5   # frames

# ROIs por defecto (porcentuales sobre ancho/alto redimensionados)
# Ajusta si tu vídeo sitúa los grifos en otro sitio.
ROI_A = (0.506, 0.328, 0.039, 0.269)  # Grifo A (izquierda del conjunto)
ROI_B = (0.554, 0.328, 0.039, 0.269)  # Grifo B (derecha del conjunto)

def _roi_rect(frame_shape, roi_pct):
    H, W = frame_shape[:2]
    x, y, w, h = roi_pct
    return (int(x*W), int(y*H), int(w*W), int(h*H))

def _iter_frames(path, target_w=640, target_fps=12):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el vídeo")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / target_fps)), 1)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            h, w = frame.shape[:2]
            if w != target_w:
                scale = target_w / w
                frame = cv2.resize(frame, (target_w, int(h*scale)))
            yield frame
        idx += 1
    cap.release()

def _signal_from_roi(frames, roi_pct):
    prev = None
    vals = []
    rx, ry, rw, rh = None, None, None, None
    for f in frames:
        if rx is None:
            rx, ry, rw, rh = _roi_rect(f.shape, roi_pct)
        roi = f[ry:ry+rh, rx:rx+rw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        if prev is None:
            prev = gray
            vals.append(0.0)
            continue
        diff = cv2.absdiff(gray, prev)
        prev = gray
        # actividad: cantidad de pixeles con cambio fuerte
        _, th = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        activity = float(np.count_nonzero(th)) / th.size  # 0..1
        vals.append(activity)
    # suavizado
    if len(vals) >= 3:
        k = Params.smooth_win
        kernel = np.ones(k) / k
        vals = np.convolve(vals, kernel, mode="same")
    return np.array(vals, dtype=np.float32)

def _segments_from_signal(sig, fps, th_hi, th_lo, min_dur, gap_close):
    # histéresis + unión de pequeños huecos
    on = False
    start = None
    raw_segments = []
    for t, x in enumerate(sig):
        if not on and x >= th_hi:
            on = True
            start = t
        elif on and x <= th_lo:
            end = t
            if end > start:
                raw_segments.append([start, end])
            on = False
    if on:  # si quedó abierto
        raw_segments.append([start, len(sig)-1])

    # cerrar huecos cortos
    merged = []
    for seg in raw_segments:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            gap = (seg[0] - prev[1]) / fps
            if gap <= gap_close:
                prev[1] = seg[1]
            else:
                merged.append(seg)

    # filtrar por duración mínima
    out = []
    for s, e in merged:
        dur = (e - s) / fps
        if dur >= min_dur:
            out.append((s, e))
    return out

def count_beers(video_path: str):
    # 1) muestreo de frames
    frames = list(_iter_frames(video_path, Params.resize_width, Params.target_fps))
    if len(frames) < 3:
        return 0, 0
    # 2) señales por grifo
    sigA = _signal_from_roi(frames, ROI_A)
    sigB = _signal_from_roi(frames, ROI_B)
    # 3) segmentación → eventos
    eventsA = _segments_from_signal(sigA, Params.target_fps, Params.th_high, Params.th_low,
                                    Params.min_duration_s, Params.gap_close_s)
    eventsB = _segments_from_signal(sigB, Params.target_fps, Params.th_high, Params.th_low,
                                    Params.min_duration_s, Params.gap_close_s)
    # 4) conteo por eventos válidos
    return len(eventsA), len(eventsB)
