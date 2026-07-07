#!/usr/bin/env python3
"""Punto de entrada del pipeline tono-politico.

Ejecuta el pipeline completo leyendo `config/config.yaml` y delega la
orquestación real a `PipelineRunner`, que es testeable con services fake.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from tono_politico.diarizacion import DiarizacionService
from tono_politico.filtrado import FiltradoService
from tono_politico.ingesta import IngestaService
from tono_politico.ingesta.playlist import obtener_info_playlist
from tono_politico.pipeline import PipelineRunner, ServiceFactories
from tono_politico.salida import SalidaService
from tono_politico.segmentacion import SegmentacionService
from tono_politico.temas import TemasService
from tono_politico.temas.models import ResultadoTemas
from tono_politico.tono import TonoService

CONFIG_PATH = Path("config/config.yaml")


# ──────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────


def cargar_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Carga config/config.yaml y devuelve el diccionario."""
    if not path.exists():
        raise FileNotFoundError(f"Config no encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config debe ser un mapping YAML: {path}")
    return cfg


# ──────────────────────────────────────────────────────────
# Factory de services desde config
# ──────────────────────────────────────────────────────────


def _build_ingesta(cfg: dict[str, Any]) -> IngestaService:
    c = cfg.get("ingesta", {})
    return IngestaService(
        data_dir=Path(c.get("data_dir", "data")),
        whisper_model=c.get("whisper_model", "large-v3-turbo"),
        idioma=c.get("idioma", "es"),
    )


def _build_diarizacion(cfg: dict[str, Any]) -> DiarizacionService:
    c = cfg.get("diarizacion", {})
    ref = c.get("referencia_voz", {})
    return DiarizacionService(
        actor=c.get("actor_objetivo", "Lilly Téllez"),
        video_ref_id=ref.get("video_id", "su9nURIj9XQ"),
        data_dir=Path(cfg.get("project", {}).get("data_dir", "data")),
        pipeline_name=c.get("pipeline", "pyannote-community/speaker-diarization-community-1"),
        umbral_match=c.get("umbral_match", 0.5),
        umbral_ambiguo=c.get("umbral_ambiguo", 0.7),
    )


def _build_segmentacion(cfg: dict[str, Any]) -> SegmentacionService:
    c = cfg.get("segmentacion", {})
    return SegmentacionService(
        spacy_model=c.get("spacy_model", "es_core_news_lg"),
        breakpoint_percentile=c.get("breakpoint_percentile", 95),
        min_oraciones=c.get("min_oraciones", 2),
        max_oraciones=c.get("max_oraciones", 8),
        max_palabras=c.get("max_palabras", 150),
    )


def _build_temas(cfg: dict[str, Any]) -> TemasService:
    c = cfg.get("temas", {})
    return TemasService(
        min_topic_size=c.get("min_topic_size", 3),
        n_neighbors=c.get("n_neighbors", 10),
        n_components=c.get("n_components", 5),
        embedding_model_name=c.get("embedding_model", "LiquidAI/LFM2.5-Embedding-350M"),
    )


def _build_filtrado(cfg: dict[str, Any], topico_id: int) -> FiltradoService:
    c = cfg.get("filtrado", {})
    return FiltradoService(
        topico_id=topico_id,
        min_relevancia=c.get("min_relevancia", 0.35),
        incluir_outliers=c.get("incluir_outliers", False),
    )


def _build_tono(cfg: dict[str, Any], actor: str, tema: str) -> TonoService:
    return TonoService(actor=actor, tema=tema)


def _build_salida(cfg: dict[str, Any], output_path: str | None) -> SalidaService:
    c = cfg.get("salida", {})
    fmt = c.get("formatos", ["json", "markdown"])

    if output_path:
        return SalidaService(output_path=output_path)
    if "json" in fmt and "markdown" in fmt:
        return SalidaService(output_path=Path("output"))
    return SalidaService(output_path=None)


def _service_factories() -> ServiceFactories:
    """Construye factories livianas para inyectar al runner."""
    return ServiceFactories(
        build_ingesta=_build_ingesta,
        build_diarizacion=_build_diarizacion,
        build_segmentacion=_build_segmentacion,
        build_temas=_build_temas,
        build_filtrado=_build_filtrado,
        build_tono=_build_tono,
        build_salida=_build_salida,
        get_playlist_info=obtener_info_playlist,
    )


# ──────────────────────────────────────────────────────────
# Presentación CLI
# ──────────────────────────────────────────────────────────


def _imprimir_topicos(resultado: ResultadoTemas) -> None:
    """Imprime los tópicos descubiertos en consola."""
    print("\n" + "=" * 60)
    print(f"Tópicos descubiertos: {resultado.num_topicos}")
    print("=" * 60)
    for topico in resultado.topicos:
        palabras = ", ".join(topico.palabras_clave[:8])
        print(
            f"  [{topico.id}] {topico.nombre} "
            f"({topico.num_segmentos} segmentos, "
            f"{topico.representatividad:.1%})\n"
            f"      Palabras: {palabras}"
        )
    print("=" * 60)
    print(
        "\nPara analizar un tópico, ejecuta:\n"
        "  uv run python main.py --playlist URL --topico N "
        '--tema "descripcion del tema"\n'
    )


def _imprimir_resumen_salida(informe_path: Path | None) -> None:
    print(f"\n✅ Pipeline completo. Informe: {informe_path or '(sin disco)'}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="tono-politico: análisis del tono político en YouTube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--playlist",
        required=True,
        help="URL de la playlist de YouTube",
    )
    parser.add_argument(
        "--topico",
        type=int,
        default=None,
        help="ID del tópico a analizar (Fase 2). Sin esto, solo descubre tópicos.",
    )
    parser.add_argument(
        "--tema",
        type=str,
        default=None,
        help="Descripción del tema a evaluar (ej. 'fracking'). Requerido con --topico.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida (archivo .json/.md o directorio). Default: output/",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help=f"Ruta del config YAML (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="No limpiar el cache de audios/transcripciones al finalizar.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Logging DEBUG en vez de INFO",
    )

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.topico is not None and not args.tema:
        parser.error("--tema es obligatorio cuando se usa --topico")

    cfg = cargar_config(Path(args.config))
    runner = PipelineRunner(
        cfg=cfg,
        factories=_service_factories(),
        keep_cache=args.keep_cache,
    )

    if args.topico is not None:
        tema = args.tema
        assert tema is not None
        result = runner.analyze(args.playlist, args.topico, tema, args.output)
        if result.exit_code == 0:
            _imprimir_resumen_salida(result.informe_path)
        return result.exit_code

    result = runner.discover(args.playlist)
    resultado_temas = getattr(runner, "last_resultado_temas", None)
    if resultado_temas is not None:
        _imprimir_topicos(resultado_temas)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
