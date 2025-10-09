# app/counter_real.py
# Conteo robusto de tiradas A/B por eventos en dos ROIs.
# Señal compuesta = movimiento (frame differencing) + color (espuma/amarillo).
# Sin dependencias externas aparte de OpenCV y NumPy.

import os
import cv2
import numpy as np

# ----------------- Parámetros -----------------
class Params:
    # Rendimiento (puedes bajar más si quieres más velocidad)
    target_fps = 10         # antes 12
    resize_width = 540      # antes 640

    # Pesos de la señal compuesta
    w_motion = 0.70
    w_color  = 0.30
    w_flow   = 0.00         # desactivado (caro). Déjalo en 0.0

    # Segmentación por histéresis
    th_high = 0.32          # umbral alto
    th_low  = 0.20          # umbral bajo
    min_duration_s = 1.3    # duración mínima del evento
    gap_close_s    = 0.30   # une huecos cortos
    min_peak       = 0.35   # pico mínimo dentro del segmento

    # Artefactos de depuración (desactivados en prod)
    write_debug_artifacts = False

# ROIs (x, y, w, h) en % sobre el frame redimensionado
# Ajustadas según tu captura (grifos bajo letras A y B):
ROI_A = (0.506, 0.328, 0.039, 0.269)
ROI_B = (0.554, 0.328, 0.039, 0.269)

# ----------------- Utilidades -----------------
def _roi_rect(frame_shape, roi_pct):
    H, W = frame_shape[:2]
    x, y, w, h = roi_pct
    return (int(x*W), int(y*H), int(w*W), int(h*H))

def _iter_frames(path, target_w=540, target_fps=10):
    """Itera frames redimensionados y submuestreados a ~target_fps usando FFmpeg."""
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass

    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not (5.0 <= src_fps <= 120.0):
        src_fps = 25.0

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

def _hsv_foam_mask(bgr_roi):
    """Máscara para espuma/blanco y amarillo claro (cerveza). Ajustable si cambia la iluminación."""
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    # Amarillo pálido / cerveza clara
    lower1 = np.array([10,  10, 160], dtype=np.uint8)
    upper1 = np.array([35, 140, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Espuma blanca (S bajo, V alto)
    lower2 = np.array([0,   0, 180], dtype=np.uint8)
    upper2 = np.array([179, 60, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    return mask

def _signal_components(frames, roi_pct):
    """Devuelve (motion, color, flow) por frame dentro del ROI."""
    prev_gray = None
    vals_motion, vals_color, vals_flow = [], [], []

    rx = ry = rw = rh = None
    for f in frames:
        if rx is None:
            rx, ry, rw, rh = _roi_rect(f.shape, roi_pct)
        roi = f[ry:ry+rh, rx:rx+rw]

        # Movimiento (frame differencing binarizado)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        if prev_gray is None:
            prev_gray = gray
            vals_motion.append(0.0)
        else:
            diff = cv2.absdiff(gray, prev_gray)
            prev_gray = gray
            _, th = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            motion = float(np.count_nonzero(th)) / th.size
            vals_motion.append(motion)

        # Color (proporción de pixeles en máscara espuma/amarilla)
        mask = _hsv_foam_mask(roi)
        color = float(np.count_nonzero(mask)) / mask.size
        vals_color.append(color)

        # Flujo óptico desactivado (dejar 0.0)
        vals_flow.append(0.0)

    return (np.array(vals_motion, np.float32),
            np.array(vals_color,  np.float32),
            np.array(vals_flow,   np.float32))

def _smooth(x, k=5):
    if len(x) < 3:
        return x
    k = max(3, int(k))
    kernel = np.ones(k, np.float32) / k
    return np.convolve(x, kernel, mode="same")

def _compose_signal(motion, color, flow):
    """Normaliza cada canal por percentiles y mezcla con pesos."""
    def norm01(v):
        a = np.percentile(v, 5)
        b = np.percentile(v, 95)
        if b <= a + 1e-6:
            return np.zeros_like(v)
        vv = (v - a) / (b - a)
        return np.clip(vv, 0, 1)

    m = norm01(motion)
    c = norm01(color)
    f = norm01(flow)
    s = Params.w_motion*m + Params.w_color*c + Params.w_flow*f
    return np.clip(s, 0, 1)

def _segments_from_signal(sig, fps):
    """Histéresis + cierre de huecos + mínimos de duración y pico."""
    TH_HI, TH_LO = Params.th_high, Params.th_low
    on = False
    start = None
    raw = []
    for t, x in enumerate(sig):
        if not on and x >= TH_HI:
            on = True
            start = t
        elif on and x <= TH_LO:
            end = t
            if end > start:
                raw.append([start, end])
            on = False
    if on:
        raw.append([start, len(sig)-1])

    # gap closing
    merged = []
    for seg in raw:
        if not merged:
            merged.append(seg)
        else:
            prev = merged[-1]
            gap = (seg[0] - prev[1]) / fps
            if gap <= Params.gap_close_s:
                prev[1] = seg[1]
            else:
                merged.append(seg)

    # filtros finales
    out = []
    for s, e in merged:
        dur = (e - s) / fps
        peak = float(np.max(sig[s:e+1])) if e > s else 0.0
        if dur >= Params.min_duration_s and peak >= Params.min_peak:
            out.append((s, e))
    return out

def _write_debug(path_dir, first_frame, sigA, sigB, fps):
    """Artefactos de depuración sencillos (ROI + señales)."""
    try:
        H, W = first_frame.shape[:2]
        fr = first_frame.copy()

        def draw_roi(img, roi, color):
            x,y,w,h = roi
            x,y,w,h = int(x*W), int(y*H), int(w*W), int(h*H)
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)

        draw_roi(fr, ROI_A, (0,255,0))
        draw_roi(fr, ROI_B, (255,0,0))
        cv2.imwrite(os.path.join(path_dir, "roi_check.jpg"), fr)

        csv_path = os.path.join(path_dir, "signals_A_B.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("t_frame,t_sec,sigA,sigB\n")
            for i in range(max(len(sigA), len(sigB))):
                t_sec = i / fps
                a = sigA[i] if i < len(sigA) else ""
                b = sigB[i] if i < len(sigB) else ""
                f.write(f"{i},{t_sec:.3f},{a},{b}\n")
    except Exception as e:
        print("[DEBUG] Error escribiendo artefactos:", e)

# ----------------- API pública -----------------
def count_beers(video_path: str):
    """Devuelve (A, B): número de tiradas detectadas en cada grifo."""
    frames = list(_iter_frames(video_path, Params.resize_width, Params.target_fps))
    if len(frames) < 3:
        return 0, 0

    motA, colA, floA = _signal_components(frames, ROI_A)
    motB, colB, floB = _signal_components(frames, ROI_B)

    sigA = _compose_signal(_smooth(motA,5), _smooth(colA,5), _smooth(floA,5))
    sigB = _compose_signal(_smooth(motB,5), _smooth(colB,5), _smooth(floB,5))

    if Params.write_debug_artifacts:
        _write_debug(os.path.dirname(video_path), frames[0], sigA, sigB, Params.target_fps)

    evA = _segments_from_signal(sigA, Params.target_fps)
    evB = _segments_from_signal(sigB, Params.target_fps)

    return len(evA), len(evB)

# Ejecutable para depurar por línea de comandos
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        p = sys.argv[1]
        a, b = count_beers(p)
        print("A:", a, "B:", b)
        print("Artefactos (si activados) se guardan en:", os.path.dirname(p))
