"""DTOs de segmentación — Segmento, Oracion.

Los DTOs se conservan porque tono/ y discursive_approach/topics_approach los
referencian vía models.py.
"""

from .models import Oracion, Segmento

__all__ = ["Segmento", "Oracion"]
