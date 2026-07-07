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

from tono_politico.config import Config, load_config
from tono_politico.diarizacion import DiarizacionService
from tono_politico.filtrado import FiltradoService
from tono_politico.ingesta import IngestaService
from tono_politico.ingesta.playlist import obtener_info_playlist
from tono_politico.pipeline import PipelineRunner, ServiceFactories, resumen_final
from tono_politico.salida import SalidaService
from tono_politico.segmentacion import SegmentacionService
from tono_politico.temas import TemasService
from tono_politico.temas.models import ResultadoTemas
from tono_politico.tono import TonoService

CONFIG_PATH = Path("config/config.yaml")


# ──────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────


def cargar_config(path: Path = CONFIG_PATH) -> Config:
    """Carga config/config.yaml y devuelve config tipada."""
    return load_config(path)


# ──────────────────────────────────────────────────────────
# Factory de services desde config
# ──────────────────────────────────────────────────────────


def _build_ingesta(cfg: Config) -> IngestaService:
    return IngestaService(
        data_dir=cfg.project.data_dir,
        whisper_model=cfg.ingesta.whisper_model,
        idioma=cfg.ingesta.idioma,
    )


def _build_diarizacion(cfg: Config) -> DiarizacionService:
    return DiarizacionService(
        actor=cfg.diarizacion.actor_objetivo,
        video_ref_id=cfg.diarizacion.video_ref_id,
        data_dir=cfg.project.data_dir,
        pipeline_name=cfg.diarizacion.pipeline,
        fallback_pipeline=cfg.diarizacion.fallback_pipeline,
        device=cfg.diarizacion.device,
        umbral_match=cfg.diarizacion.umbral_match,
        umbral_ambiguo=cfg.diarizacion.umbral_ambiguo,
    )


def _build_segmentacion(cfg: Config) -> SegmentacionService:
    return SegmentacionService(
        spacy_model=cfg.segmentacion.spacy_model,
        breakpoint_percentile=cfg.segmentacion.breakpoint_percentile,
        min_oraciones=cfg.segmentacion.min_oraciones,
        max_oraciones=cfg.segmentacion.max_oraciones,
        max_palabras=cfg.segmentacion.max_palabras,
    )


def _build_temas(cfg: Config) -> TemasService:
    return TemasService(
        min_topic_size=cfg.temas.min_topic_size,
        n_neighbors=cfg.temas.n_neighbors,
        n_components=cfg.temas.n_components,
        embedding_model_name=cfg.temas.embedding_model,
    )


def _build_filtrado(cfg: Config, topico_id: int) -> FiltradoService:
    return FiltradoService(
        topico_id=topico_id,
        min_relevancia=cfg.filtrado.min_relevancia,
        incluir_outliers=cfg.filtrado.incluir_outliers,
    )


def _build_tono(cfg: Config, actor: str, tema: str) -> TonoService:
    return TonoService(actor=actor, tema=tema)


def _build_salida(cfg: Config, output_path: str | None) -> SalidaService:
    if output_path:
        return SalidaService(output_path=output_path)
    if "json" in cfg.salida.formatos and "markdown" in cfg.salida.formatos:
        return SalidaService(output_path=cfg.project.output_dir)
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


def _imprimir_resumen_salida(result) -> None:
    """Imprime el resumen final con datos del manifest."""
    print("\n" + "=" * 60)
    print(resumen_final(result))
    print("=" * 60)


def _imprimir_resumen_fallo(result) -> None:
    fase_fallida = next(
        (phase for phase in reversed(result.manifest.phases) if not phase.ok),
        None,
    )
    if fase_fallida is None:
        print("\n❌ Pipeline falló sin fase registrada.")
        return
    print(f"\n❌ Pipeline falló en fase {fase_fallida.phase}: {fase_fallida.message}")


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
        "--resume",
        type=str,
        default=None,
        help="Directorio de run anterior con fase1-topicos.json para reusar Fase 1.",
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

    if args.resume is not None and args.topico is None:
        parser.error("--resume requiere --topico y --tema")

    cfg = cargar_config(Path(args.config))
    runner = PipelineRunner(
        cfg=cfg,
        factories=_service_factories(),
        keep_cache=args.keep_cache,
    )

    # --resume: reusa Fase 1 desde disco, ejecuta solo Fase 2
    if args.resume is not None:
        assert args.topico is not None
        assert args.tema is not None
        result = runner.analyze_resume(args.resume, args.topico, args.tema, args.output)
        if result.exit_code == 0:
            _imprimir_resumen_salida(result)
        else:
            _imprimir_resumen_fallo(result)
        return result.exit_code

    if args.topico is not None:
        tema = args.tema
        assert tema is not None
        result = runner.analyze(args.playlist, args.topico, tema, args.output)
        if result.exit_code == 0:
            _imprimir_resumen_salida(result)
        else:
            _imprimir_resumen_fallo(result)
        return result.exit_code

    result = runner.discover(args.playlist)
    if result.exit_code != 0:
        _imprimir_resumen_fallo(result)
        return result.exit_code
    _imprimir_resumen_salida(result)
    resultado_temas = getattr(runner, "last_resultado_temas", None)
    if resultado_temas is not None:
        _imprimir_topicos(resultado_temas)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
