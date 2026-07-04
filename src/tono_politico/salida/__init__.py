"""Componente 6: Salida — informe final del pipeline."""

from .agregacion import (
    dimension_dominante,
    generar_perfil,
    intensidad_promedio,
    stance_dominante,
)
from .models import InformeTono, PerfilActor, Provenance
from .serializacion import (
    generar_json,
    generar_markdown,
    perfil_a_dict,
    provenance_a_dict,
    segmento_a_dict,
)
from .service import SalidaService

__all__ = [
    "InformeTono",
    "PerfilActor",
    "Provenance",
    "SalidaService",
    "dimension_dominante",
    "generar_json",
    "generar_markdown",
    "generar_perfil",
    "intensidad_promedio",
    "perfil_a_dict",
    "provenance_a_dict",
    "segmento_a_dict",
    "stance_dominante",
]
