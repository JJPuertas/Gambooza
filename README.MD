# Gambooza · Conteo de tiradas de cerveza A/B - V 1.0.0

Descripción
Aplicación para el caso práctico técnico de Gambooza. Detecta cuántas cervezas se sirven desde dos grifos (A y B) en un vídeo con cámara fija. Usa FastAPI, OpenCV y SQLite. Interfaz web incluida; el procesamiento corre en background.

## Requisitos

Python 3.10 o superior

FFmpeg instalado (para decodificar/convertir vídeos HEVC/H.265)

Si usas Conda: conda install -c conda-forge ffmpeg -y

Alternativa: https://ffmpeg.org/download.html

Dependencias: archivo requirements.txt

## Instalación
a) Clonar o descargar el proyecto
git clone https://github.com/JJPuertas/Gambooza.git
cd Gambooza

b) Crear y activar entorno virtual
python -m venv .venv
(Windows) .venv\Scripts\activate
(macOS/Linux) source .venv/bin/activate

c) Instalar dependencias
pip install -r requirements.txt

d) Iniciar la aplicación
uvicorn app.main:app --port 8000

e) Abrir el navegador
http://localhost:8000

## Uso

Sube un vídeo (.mp4/.mov) con los grifos A y B visibles (cámara fija).

Pulsa “Procesar vídeo”.

Si el códec no es compatible (p. ej. HEVC), se convierte automáticamente a H.264 al procesar.

El procesamiento corre en segundo plano (estado running).

Al terminar: verás tiradas por A, por B y el total. Estado done.

El panel lateral muestra estadísticas diarias agregadas.

## Consultar la base de datos (SQLite)
Ruta de la base de datos:
app/storage/gambooza.db

### Opción A: sqlite3 (línea de comandos)
sqlite3 app/storage/gambooza.db
Dentro de sqlite3:
.tables
.schema video
.schema event
SELECT id, status, count_a, count_b, datetime(created_at) FROM video ORDER BY created_at DESC LIMIT 50;
SELECT video_id, tap, t_start_s, t_end_s FROM event ORDER BY id DESC LIMIT 50;
Salir: .exit

### Opción B: Python interactivo
from sqlmodel import Session, select
from app.db import engine, Video, Event
with Session(engine) as s:
for v in s.exec(select(Video)):
print(v.id, v.status, v.count_a, v.count_b)
for e in s.exec(select(Event).order_by(Event.id.desc()).limit(20)):
print(e.video_id, e.tap, e.t_start_s, e.t_end_s)

### Opción C: Herramienta visual

DB Browser for SQLite

VSCode con la extensión “SQLite Viewer”
Abrir: app/storage/gambooza.db


## Detalles técnicos

Framework: FastAPI + Jinja2 (UI ligera)

Visión por computador: OpenCV (señales por ROI: movimiento + color)

Persistencia: SQLite con SQLModel (tablas video y event)

Ejecución: procesamiento en background mediante ThreadPoolExecutor

Compatibilidad: Windows, macOS y Linux

## Solución de problemas
Problema: No abre vídeos HEVC/H.265
Causa: Falta FFmpeg o códec no soportado
Solución: Instalar FFmpeg (conda install -c conda-forge ffmpeg -y). El sistema convertirá al procesar.

Problema: La app parece quedarse “cargando” al iniciar
Causa: Reloader en Windows
Solución: Ejecutar sin --reload: uvicorn app.main:app --port 8000

Problema: Tarda mucho en procesar
Causa: Vídeo muy largo o muy grande
Solución: Reducir target_fps y resize_width en app/counter_real.py

Problema: No detecta tiradas
Causa: ROIs/umbrales no ajustados
Solución: Ajustar ROI_A, ROI_B, th_high, th_low, min_duration_s en counter_real.py

Problema: “database is locked”
Causa: Varias instancias usando la base a la vez
Solución: Cerrar instancias previas o esperar a que termine el proceso en curso

# Ejecución rápida (resumen)
python -m venv .venv 

 .venv\Scripts\activate


pip install -r requirements.txt


uvicorn app.main:app --port 8000


Abrir: http://localhost:8000


Si aparece un error de python-multipart añadir este comando y volver a abrir: pip install python-multipart  