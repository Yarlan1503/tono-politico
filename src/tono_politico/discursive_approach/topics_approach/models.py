"""DTOs de topics_approach (enfoques de tono)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..argument_shape.models import Argumento
from ..topics_cluster.models import TopicoInfo


@dataclass
class PerfilTonoArgumento:
    """Snapshot de tono de un argumento (independiente de SegmentoConTono)."""

    stance: str  # apoyo | rechazo
    intensidad: int  # 1-5
    logica_dominante: str
    sentimiento_dominante: str
    estilo_dominante: str
    funcion_dominante: str
    raw: Any | None = None  # opcional: objeto Tono completo


@dataclass
class EnfoqueInfo:
    id: int
    topico_id: int
    nombre: str
    palabras_clave: list[str] = field(default_factory=list)
    num_argumentos: int = 0
    fecha_primera: str | None = None
    fecha_ultima: str | None = None
    stance_dominante: str | None = None
    intensidad_media: float | None = None
    logica_dominante: str | None = None
    sentimiento_dominante: str | None = None
    estilo_dominante: str | None = None
    funcion_dominante: str | None = None


@dataclass
class ArgumentoConEnfoque:
    argumento: Argumento
    topico_id: int
    enfoque_id: int
    probabilidad_topico: float
    probabilidad_enfoque: float = 1.0
    tono: PerfilTonoArgumento | None = None


@dataclass
class ResultadoEnfoquesTema:
    topico: TopicoInfo
    enfoques: list[EnfoqueInfo] = field(default_factory=list)
    argumentos: list[ArgumentoConEnfoque] = field(default_factory=list)


@dataclass
class ResultadoEnfoques:
    por_tema: list[ResultadoEnfoquesTema] = field(default_factory=list)
    num_temas: int = 0
    num_enfoques_total: int = 0
