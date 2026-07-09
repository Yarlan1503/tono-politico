"""Umbrella discursive_approach: shape → cluster → approaches.

El service compone TopicsApproach/Tono; se expone lazy para que importar
submódulos de datos/serialización no cargue numpy/torch durante el parse CLI.
"""

from __future__ import annotations

__all__ = ["DiscursiveApproachService"]


def __getattr__(name: str):
    if name == "DiscursiveApproachService":
        from .service import DiscursiveApproachService

        return DiscursiveApproachService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
