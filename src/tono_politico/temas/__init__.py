"""DTOs de temas — ResultadoTemas, TopicoInfo, SegmentoTematizado.

Los DTOs se conservan porque discursive_approach/topics_approach los referencia.
"""

from .models import ResultadoTemas, SegmentoTematizado, TopicoInfo

__all__ = [
    "ResultadoTemas",
    "TopicoInfo",
    "SegmentoTematizado",
]
