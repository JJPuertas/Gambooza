# app/main.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid, os, time
from datetime import datetime
from sqlalchemy import func
from sqlmodel import select

from app.transcode import ensure_h264_fast
from app.counter_real import count_beers
from app.jobs import submit, get_executor, shutdown as jobs_shutdown
from app.db import init_db, get_session, Video, Event



# -------------------------------------------------------------------
# Configuración básica
# -------------------------------------------------------------------
app = FastAPI(title="Gambooza - Conteo de tiradas A/B")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

STORAGE_DIR = "app/storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Inicializa la base de datos
init_db()

@app.on_event("startup")
def on_startup():
    init_db()        # crea tablas si no existen
    get_executor()   # inicializa el pool de hilos

@app.on_event("shutdown")
def on_shutdown():
    jobs_shutdown()  # cierra el pool limpio

# -------------------------------------------------------------------
# Página principal
# -------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Renderiza la interfaz web."""
    return templates.TemplateResponse("video.html", {"request": request})

# -------------------------------------------------------------------
# Subida de vídeo
# -------------------------------------------------------------------
@app.post("/videos")
async def upload_video(file: UploadFile = File(...)):
    """Sube un vídeo al servidor y crea registro en la BD."""
    vid = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    raw_path = os.path.join(STORAGE_DIR, f"{vid}{ext}")

    # Guardado por chunks (más rápido y evita truncados)
    with open(raw_path, "wb") as f:
        while True:
            chunk = await file.read(4 * 1024 * 1024)  # 4 MB
            if not chunk:
                break
            f.write(chunk)

    # Registro en la BD
    with get_session() as s:
        s.add(Video(id=vid, filename=raw_path))
        s.commit()

    return {"id": vid, "status": "pending"}

# -------------------------------------------------------------------
# Procesamiento (en background)
# -------------------------------------------------------------------
@app.post("/videos/{vid}/process")
def process_video(vid: str):
    # Abrimos sesión y cogemos todo lo que necesitamos ANTES de cerrarla
    with get_session() as s:
        v = s.get(Video, vid)
        if not v:
            return JSONResponse({"error": "not found"}, status_code=404)
        if v.status == "running":
            return {"id": vid, "status": "running"}

        # Guarda lo necesario en variables locales
        src_filename = v.filename

        # Marca running y persiste
        v.status = "running"
        s.add(v)
        s.commit()

    # Fuera de la sesión ya NO usamos 'v', usamos 'src_filename'
    def _job(path):
        safe = ensure_h264_fast(path)  # convierte si hace falta
        A, B = count_beers(safe)
        return {"A": A, "B": B}

    # Usa el pool (hilos) y pasa la ruta local
    fut = submit(_job, src_filename)

    # Callback que reabre sesión y actualiza la fila
    def _done(f):
        try:
            res = f.result()
            with get_session() as s2:
                v2 = s2.get(Video, vid)
                if not v2:
                    return
                v2.count_a = res["A"]
                v2.count_b = res["B"]
                v2.status = "done"
                s2.add(v2)
                s2.commit()
        except Exception as e:
            with get_session() as s2:
                v2 = s2.get(Video, vid)
                if not v2:
                    return
                v2.status = "error"
                v2.error_message = str(e)
                s2.add(v2)
                s2.commit()

    fut.add_done_callback(_done)
    return {"id": vid, "status": "running"}

# -------------------------------------------------------------------
# Consulta de estado/resultados
# -------------------------------------------------------------------
@app.get("/videos/{vid}")
def get_video(vid: str):
    """Devuelve el estado y conteos de un vídeo."""
    with get_session() as s:
        v = s.get(Video, vid)
        if not v:
            return JSONResponse({"error": "not found"}, status_code=404)
        return {
            "id": v.id,
            "status": v.status,
            "counts": {
                "A": v.count_a,
                "B": v.count_b,
                "total": v.count_a + v.count_b,
            },
            "error": v.error_message,
        }

# -------------------------------------------------------------------
# Estadísticas globales
# -------------------------------------------------------------------
@app.get("/stats")
def stats():
    """Devuelve el total de tiradas por día."""
    with get_session() as s:
        rows = s.exec(
            select(
                func.date(Video.created_at),
                func.sum(Video.count_a),
                func.sum(Video.count_b),
            )
            .where(Video.status == "done")
            .group_by(func.date(Video.created_at))
            .order_by(func.date(Video.created_at))
        ).all()

        out = []
        for d, a, b in rows:
            out.append(
                {
                    "date": str(d),
                    "A": int(a or 0),
                    "B": int(b or 0),
                    "total": int((a or 0) + (b or 0)),
                }
            )
        return {"days": out}

# -------------------------------------------------------------------
# Punto de arranque (modo directo)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
