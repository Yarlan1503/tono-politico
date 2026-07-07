"""Configuración tipada del pipeline tono-politico."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"
DEFAULT_FALLBACK_PIPELINE = "pyannote-community/speaker-diarization-community-1"


@dataclass(frozen=True)
class ProjectConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
    idioma: str = "es"
    random_state: int = 42


@dataclass(frozen=True)
class IngestaConfig:
    whisper_model: str = "large-v3-turbo"
    idioma: str = "es"


@dataclass(frozen=True)
class DiarizacionConfig:
    actor_objetivo: str = "Lilly Téllez"
    video_ref_id: str = "su9nURIj9XQ"
    pipeline: str = DEFAULT_PIPELINE
    fallback_pipeline: str | None = DEFAULT_FALLBACK_PIPELINE
    umbral_match: float = 0.5
    umbral_ambiguo: float = 0.7
    device: str = "auto"


@dataclass(frozen=True)
class SegmentacionConfig:
    spacy_model: str = "es_core_news_lg"
    embedding_model: str = "LiquidAI/LFM2.5-Embedding-350M"
    breakpoint_percentile: int = 95
    min_oraciones: int = 2
    max_oraciones: int = 8
    max_palabras: int = 150


@dataclass(frozen=True)
class TemasConfig:
    embedding_model: str = "LiquidAI/LFM2.5-Embedding-350M"
    min_topic_size: int = 3
    n_neighbors: int = 10
    n_components: int = 5


@dataclass(frozen=True)
class FiltradoConfig:
    min_relevancia: float = 0.35
    incluir_outliers: bool = False


@dataclass(frozen=True)
class SalidaConfig:
    formatos: list[str] = field(default_factory=lambda: ["json", "markdown"])
    incluir_provenance: bool = True
    redondear_scores: int = 4


@dataclass(frozen=True)
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    ingesta: IngestaConfig = field(default_factory=IngestaConfig)
    diarizacion: DiarizacionConfig = field(default_factory=DiarizacionConfig)
    segmentacion: SegmentacionConfig = field(default_factory=SegmentacionConfig)
    temas: TemasConfig = field(default_factory=TemasConfig)
    filtrado: FiltradoConfig = field(default_factory=FiltradoConfig)
    salida: SalidaConfig = field(default_factory=SalidaConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Config:
        project_data = _section(data, "project")
        ingesta_data = _section(data, "ingesta")
        diarizacion_data = _section(data, "diarizacion")
        referencia_voz = _section(diarizacion_data, "referencia_voz")
        segmentacion_data = _section(data, "segmentacion")
        temas_data = _section(data, "temas")
        filtrado_data = _section(data, "filtrado")
        salida_data = _section(data, "salida")

        project = ProjectConfig(
            data_dir=_path(project_data.get("data_dir", "data")),
            output_dir=_path(project_data.get("output_dir", "output")),
            idioma=str(project_data.get("idioma", "es")),
            random_state=int(project_data.get("random_state", 42)),
        )
        _validar_data_dir_ingesta(ingesta_data, project.data_dir)

        return cls(
            project=project,
            ingesta=IngestaConfig(
                whisper_model=str(ingesta_data.get("whisper_model", "large-v3-turbo")),
                idioma=str(ingesta_data.get("idioma", project.idioma)),
            ),
            diarizacion=DiarizacionConfig(
                actor_objetivo=str(diarizacion_data.get("actor_objetivo", "Lilly Téllez")),
                video_ref_id=str(referencia_voz.get("video_id", "su9nURIj9XQ")),
                pipeline=str(diarizacion_data.get("pipeline", DEFAULT_PIPELINE)),
                fallback_pipeline=_optional_str(
                    diarizacion_data.get("fallback_pipeline", DEFAULT_FALLBACK_PIPELINE)
                ),
                umbral_match=float(diarizacion_data.get("umbral_match", 0.5)),
                umbral_ambiguo=float(diarizacion_data.get("umbral_ambiguo", 0.7)),
                device=str(diarizacion_data.get("device", "auto")),
            ),
            segmentacion=SegmentacionConfig(
                spacy_model=str(segmentacion_data.get("spacy_model", "es_core_news_lg")),
                embedding_model=str(
                    segmentacion_data.get("embedding_model", "LiquidAI/LFM2.5-Embedding-350M")
                ),
                breakpoint_percentile=int(segmentacion_data.get("breakpoint_percentile", 95)),
                min_oraciones=int(segmentacion_data.get("min_oraciones", 2)),
                max_oraciones=int(segmentacion_data.get("max_oraciones", 8)),
                max_palabras=int(segmentacion_data.get("max_palabras", 150)),
            ),
            temas=TemasConfig(
                embedding_model=str(
                    temas_data.get("embedding_model", "LiquidAI/LFM2.5-Embedding-350M")
                ),
                min_topic_size=int(temas_data.get("min_topic_size", 3)),
                n_neighbors=int(temas_data.get("n_neighbors", 10)),
                n_components=int(temas_data.get("n_components", 5)),
            ),
            filtrado=FiltradoConfig(
                min_relevancia=float(filtrado_data.get("min_relevancia", 0.35)),
                incluir_outliers=bool(filtrado_data.get("incluir_outliers", False)),
            ),
            salida=SalidaConfig(
                formatos=[str(fmt) for fmt in salida_data.get("formatos", ["json", "markdown"])],
                incluir_provenance=bool(salida_data.get("incluir_provenance", True)),
                redondear_scores=int(salida_data.get("redondear_scores", 4)),
            ),
        )

    def as_legacy_dict(self) -> dict[str, Any]:
        """Representación dict temporal para integraciones no migradas."""
        return {
            "project": {
                "data_dir": str(self.project.data_dir),
                "output_dir": str(self.project.output_dir),
                "idioma": self.project.idioma,
                "random_state": self.project.random_state,
            },
            "ingesta": {
                "data_dir": str(self.project.data_dir),
                "whisper_model": self.ingesta.whisper_model,
                "idioma": self.ingesta.idioma,
            },
            "diarizacion": {
                "pipeline": self.diarizacion.pipeline,
                "fallback_pipeline": self.diarizacion.fallback_pipeline,
                "actor_objetivo": self.diarizacion.actor_objetivo,
                "umbral_match": self.diarizacion.umbral_match,
                "umbral_ambiguo": self.diarizacion.umbral_ambiguo,
                "device": self.diarizacion.device,
                "referencia_voz": {"video_id": self.diarizacion.video_ref_id},
            },
            "segmentacion": {
                "spacy_model": self.segmentacion.spacy_model,
                "embedding_model": self.segmentacion.embedding_model,
                "breakpoint_percentile": self.segmentacion.breakpoint_percentile,
                "min_oraciones": self.segmentacion.min_oraciones,
                "max_oraciones": self.segmentacion.max_oraciones,
                "max_palabras": self.segmentacion.max_palabras,
            },
            "temas": {
                "embedding_model": self.temas.embedding_model,
                "min_topic_size": self.temas.min_topic_size,
                "n_neighbors": self.temas.n_neighbors,
                "n_components": self.temas.n_components,
            },
            "filtrado": {
                "min_relevancia": self.filtrado.min_relevancia,
                "incluir_outliers": self.filtrado.incluir_outliers,
            },
            "salida": {
                "formatos": list(self.salida.formatos),
                "incluir_provenance": self.salida.incluir_provenance,
                "redondear_scores": self.salida.redondear_scores,
            },
        }


def load_config(path: Path) -> Config:
    """Carga y valida un archivo YAML de configuración."""
    if not path.exists():
        raise FileNotFoundError(f"Config no encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config debe ser un mapping YAML: {path}")
    return Config.from_mapping(raw)


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Sección '{key}' debe ser un mapping")
    return value


def _path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _validar_data_dir_ingesta(ingesta_data: Mapping[str, Any], project_data_dir: Path) -> None:
    ingesta_data_dir = ingesta_data.get("data_dir")
    if ingesta_data_dir is None:
        return
    normalized = _path(ingesta_data_dir)
    if normalized != project_data_dir:
        raise ValueError(
            "ingesta.data_dir está deprecado y debe coincidir con "
            f"project.data_dir ({normalized!s} != {project_data_dir!s})"
        )
