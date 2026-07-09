"""topics_approach: enfoques de tono por tema × tiempo.

El service carga TonoService; se expone de forma lazy para que importar el
paquete/CLI no cargue numpy/torch antes de ejecutar la etapa.
"""

from __future__ import annotations

from .models import (
    ArgumentoConEnfoque,
    EnfoqueInfo,
    PerfilTonoArgumento,
    ResultadoEnfoques,
    ResultadoEnfoquesTema,
)
from .serializacion import guardar_resultado_enfoques, resultado_enfoques_to_dict

__all__ = [
    "ArgumentoConEnfoque",
    "EnfoqueInfo",
    "PerfilTonoArgumento",
    "ResultadoEnfoques",
    "ResultadoEnfoquesTema",
    "TopicsApproachService",
    "guardar_resultado_enfoques",
    "resultado_enfoques_to_dict",
]


def __getattr__(name: str):
    if name == "TopicsApproachService":
        from .service import TopicsApproachService

        return TopicsApproachService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
