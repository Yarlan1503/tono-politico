"""DTOs de filtrado — CriterioFiltrado, SegmentoFiltrado, ResultadoFiltrado.

Los DTOs se conservan porque tono/service.py y
discursive_approach/topics_approach/adapter.py los referencian.
"""

from .models import CriterioFiltrado, ResultadoFiltrado, SegmentoFiltrado

__all__ = [
    "CriterioFiltrado",
    "ResultadoFiltrado",
    "SegmentoFiltrado",
]
