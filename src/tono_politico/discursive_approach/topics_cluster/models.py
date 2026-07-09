"""DTOs de topics_cluster (Argumento en lugar de Segmento)."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..argument_shape.models import Argumento


@dataclass
class ArgumentoTematizado:
    argumento: Argumento
    topico_id: int
    probabilidad: float


@dataclass
class TopicoInfo:
    id: int
    nombre: str
    palabras_clave: list[str] = field(default_factory=list)
    num_argumentos: int = 0
    representatividad: float = 0.0


@dataclass
class ResultadoTemas:
    argumentos: list[ArgumentoTematizado] = field(default_factory=list)
    topicos: list[TopicoInfo] = field(default_factory=list)
    num_topicos: int = 0
