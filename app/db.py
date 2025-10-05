# app/db.py
from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine, Session

# En Windows + FastAPI (hilos), mejor desactivar check_same_thread
engine = create_engine(
    "sqlite:///app/storage/gambooza.db",
    echo=False,
    connect_args={"check_same_thread": False},
)

class StatusEnum(str, Enum):
    pending = "pending"
    running = "running"
    done    = "done"
    error   = "error"

class TapEnum(str, Enum):
    A = "A"
    B = "B"

class Video(SQLModel, table=True):
    id: str = Field(primary_key=True)
    filename: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: StatusEnum = Field(default=StatusEnum.pending)
    count_a: int = 0
    count_b: int = 0
    error_message: Optional[str] = None

class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    video_id: str = Field(index=True, foreign_key="video.id")
    tap: TapEnum
    t_start_s: float
    t_end_s: float

def init_db() -> None:
    SQLModel.metadata.create_all(engine)

def get_session() -> Session:
    # evita que las instancias expiren al hacer commit
    return Session(engine, expire_on_commit=False)
