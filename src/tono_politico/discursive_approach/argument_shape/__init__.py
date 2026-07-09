"""argument_shape: oraciones → argumentos semánticos (1 audio)."""

from __future__ import annotations

from .models import Argumento, Oracion
from .service import ArgumentShapeService

__all__ = ["Argumento", "Oracion", "ArgumentShapeService"]
