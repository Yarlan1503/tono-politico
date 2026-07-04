"""DTOs del Componente 6: Salida.

Modela la salida final del pipeline: perfil agregado del actor,
segmentos detallados y metadata de provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..tono.models import SegmentoConTono


@dataclass
class Provenance:
    """Metadata de cómo se generó el análisis.

    Todo informe debe declarar qué modelos se usaron y bajo qué marco
    teórico, para que el lector pueda interpretar los scores correctamente.

    Attributes:
        pipeline: Nombre y versión del pipeline (ej. "tono-politico v0.1.0").
        modelos: Lista de modelos de ML usados.
        fecha: Timestamp ISO de generación.
        marco_teorico: Ruta al documento de marco teórico.
        advertencia_confianza: Advertencia sobre la interpretación de scores.
    """

    pipeline: str
    modelos: list[str]
    fecha: str
    marco_teorico: str = "docs/marco-teorico.md"
    advertencia_confianza: str = (
        "Los scores de similitud coseno son medidas relativas de "
        "proximidad semántica, no probabilidades calibradas."
    )


@dataclass
class PerfilActor:
    """Perfil agregado de un actor político sobre un tema.

    Colapsa el análisis de todos los segmentos en un perfil interpretable.

    Attributes:
        actor: Nombre del actor político.
        tema: Tema evaluado.
        n_segmentos: Número de segmentos analizados.
        stance_dominante: "apoyo" o "rechazo" (mayoritario).
        intensidad_promedio: Promedio de intensidad antagónica (1-5).
        logica_dominante: Label de lógica política con mayor score promedio.
        sentimiento_dominante: Emoción dominante promedio.
        estilo_dominante: Estilo discursivo dominante promedio.
        funcion_dominante: Función discursiva dominante promedio.
    """

    actor: str
    tema: str
    n_segmentos: int
    stance_dominante: str
    intensidad_promedio: float
    logica_dominante: str
    sentimiento_dominante: str
    estilo_dominante: str
    funcion_dominante: str


@dataclass
class InformeTono:
    """Salida final del pipeline completo.

    Attributes:
        perfil: Perfil agregado del actor.
        segmentos: Lista de segmentos con análisis de tono detallado.
        provenance: Metadata de cómo se generó el análisis.
    """

    perfil: PerfilActor
    segmentos: list[SegmentoConTono] = field(default_factory=list)
    provenance: Provenance | None = None
