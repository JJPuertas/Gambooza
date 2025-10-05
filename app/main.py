from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uuid, os
# from app.counter_stub import count_beers
from app.counter_real import count_beers
from app.transcode import transcode_to_h264

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DB = {}  # memoria: video_id -> {filename, status, counts}
STORAGE_DIR = "app/storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("video.html", {"request": request})

@app.post("/videos")
async def upload_video(file: UploadFile = File(...)):
    vid = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    raw_path = os.path.join(STORAGE_DIR, f"{vid}_raw{ext}")
    # Escribir por chunks (evita archivos truncados)
    with open(raw_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB
            if not chunk:
                break
            f.write(chunk)

    # ★ Convertir SIEMPRE a H.264 al subir
    try:
        safe_path = transcode_to_h264(raw_path)
    except Exception as e:
        # Limpieza y error claro
        try:
            os.remove(raw_path)
        except FileNotFoundError:
            pass
        return JSONResponse({"error": f"Transcodificación falló: {e}"}, status_code=400)

    # (Opcional) borrar el original para ahorrar espacio
    try:
        os.remove(raw_path)
    except FileNotFoundError:
        pass

    # Guardamos directamente la ruta H.264
    DB[vid] = {"filename": safe_path, "status": "pending", "counts": {"A":0,"B":0,"total":0}}
    return {"id": vid, "status": "pending"}

@app.post("/videos/{vid}/process")
def process_video(vid: str):
    if vid not in DB:
        return JSONResponse({"error":"not found"}, status_code=404)
    DB[vid]["status"] = "running"
    A, B = count_beers(DB[vid]["filename"])  # ★ contador stub
    DB[vid]["counts"] = {"A": A, "B": B, "total": A+B}
    DB[vid]["status"] = "done"
    return {"id": vid, "status": "done"}

@app.get("/videos/{vid}")
def get_video(vid: str):
    if vid not in DB:
        return JSONResponse({"error":"not found"}, status_code=404)
    return {"id": vid, "status": DB[vid]["status"], "counts": DB[vid]["counts"]}
