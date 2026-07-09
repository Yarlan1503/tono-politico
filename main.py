#!/usr/bin/env python3
"""Punto de entrada del pipeline tono-politico.

Ejecuta el pipeline completo leyendo `config/config.yaml` y delega la
orquestación real a `PipelineRunner`, que es testeable con services fake.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from tono_politico.config import Config, load_config
from tono_politico.execution import (
    ConfigValidationError,
    ExecutionFactories,
    ExecutionResult,
    ExecutionRunner,
    build_execution_plan,
    is_run_config_file,
    load_run_config,
    resolve_artifacts,
    validate_run_config,
)
from tono_politico.execution.models import RunConfig
from tono_politico.pipeline import PipelineRunner, ServiceFactories, resumen_final

if TYPE_CHECKING:
    from tono_politico.diarizacion import DiarizacionService
    from tono_politico.discursive_approach import DiscursiveApproachService
    from tono_politico.discursive_approach.argument_shape import ArgumentShapeService
    from tono_politico.discursive_approach.topics_approach import TopicsApproachService
    from tono_politico.discursive_approach.topics_cluster import TopicsClusterService
    from tono_politico.filtrado import FiltradoService
    from tono_politico.ingesta import IngestaService
    from tono_politico.salida import SalidaService
    from tono_politico.segmentacion import SegmentacionService
    from tono_politico.speech2text import SpeechToTextService
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
    from tono_politico.ingesta import IngestaService

    return IngestaService(
        data_dir=cfg.project.data_dir,
        whisper_model=cfg.ingesta.whisper_model,
        idioma=cfg.ingesta.idioma,
    )


def _build_diarizacion(cfg: Config) -> DiarizacionService:
    from tono_politico.diarizacion import DiarizacionService

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
    from tono_politico.segmentacion import SegmentacionService

    return SegmentacionService(
        spacy_model=cfg.segmentacion.spacy_model,
        breakpoint_percentile=cfg.segmentacion.breakpoint_percentile,
        min_oraciones=cfg.segmentacion.min_oraciones,
        max_oraciones=cfg.segmentacion.max_oraciones,
        max_palabras=cfg.segmentacion.max_palabras,
    )


def _build_temas(cfg: Config) -> TemasService:
    from tono_politico.temas import TemasService

    return TemasService(
        min_topic_size=cfg.temas.min_topic_size,
        n_neighbors=cfg.temas.n_neighbors,
        n_components=cfg.temas.n_components,
        embedding_model_name=cfg.temas.embedding_model,
    )


def _build_filtrado(cfg: Config, topico_id: int) -> FiltradoService:
    from tono_politico.filtrado import FiltradoService

    return FiltradoService(
        topico_id=topico_id,
        min_relevancia=cfg.filtrado.min_relevancia,
        incluir_outliers=cfg.filtrado.incluir_outliers,
    )


def _build_tono(cfg: Config, actor: str, tema: str) -> TonoService:
    from tono_politico.tono import TonoService

    return TonoService(actor=actor, tema=tema)


def _build_salida(cfg: Config, output_path: str | None) -> SalidaService:
    from tono_politico.salida import SalidaService

    if output_path:
        return SalidaService(output_path=output_path)
    if "json" in cfg.salida.formatos and "markdown" in cfg.salida.formatos:
        return SalidaService(output_path=cfg.project.output_dir)
    return SalidaService(output_path=None)


def _build_speech2text(cfg: Config) -> SpeechToTextService:
    from tono_politico.speech2text import SpeechToTextService

    return SpeechToTextService(
        data_dir=cfg.project.data_dir,
        actor=cfg.diarizacion.actor_objetivo,
        video_ref_id=cfg.diarizacion.video_ref_id,
        whisper_model=cfg.ingesta.whisper_model,
        idioma=cfg.ingesta.idioma,
        umbral_match=cfg.diarizacion.umbral_match,
        umbral_ambiguo=cfg.diarizacion.umbral_ambiguo,
        pipeline_name=cfg.diarizacion.pipeline,
        fallback_pipeline=cfg.diarizacion.fallback_pipeline,
        device=cfg.diarizacion.device,
    )


def _build_discursive(cfg: Config) -> DiscursiveApproachService:
    from tono_politico.discursive_approach import DiscursiveApproachService
    from tono_politico.discursive_approach.argument_shape import ArgumentShapeService
    from tono_politico.discursive_approach.topics_cluster import TopicsClusterService

    shape = ArgumentShapeService(
        spacy_model=cfg.segmentacion.spacy_model,
        breakpoint_percentile=cfg.segmentacion.breakpoint_percentile,
        min_oraciones=cfg.segmentacion.min_oraciones,
        max_oraciones=cfg.segmentacion.max_oraciones,
        max_palabras=cfg.segmentacion.max_palabras,
        embedding_model_name=cfg.segmentacion.embedding_model,
    )
    cluster = TopicsClusterService(
        min_topic_size=cfg.temas.min_topic_size,
        n_neighbors=cfg.temas.n_neighbors,
        n_components=cfg.temas.n_components,
        embedding_model_name=cfg.temas.embedding_model,
    )
    return DiscursiveApproachService(
        actor=cfg.diarizacion.actor_objetivo,
        shape_service=shape,
        cluster_service=cluster,
    )


def _service_factories() -> ServiceFactories:
    """Construye factories livianas para inyectar al runner."""
    from tono_politico.ingesta.playlist import obtener_info_playlist

    return ServiceFactories(
        build_ingesta=_build_ingesta,
        build_diarizacion=_build_diarizacion,
        build_segmentacion=_build_segmentacion,
        build_temas=_build_temas,
        build_filtrado=_build_filtrado,
        build_tono=_build_tono,
        build_salida=_build_salida,
        get_playlist_info=obtener_info_playlist,
        build_speech2text=_build_speech2text,
        build_discursive=_build_discursive,
    )


def _build_execution_speech2text(cfg: RunConfig) -> SpeechToTextService:
    from tono_politico.speech2text import SpeechToTextService

    s2t = cfg.speech2text
    speaker = s2t.speaker_timestamps
    transcribe = s2t.transcribe_speech
    return SpeechToTextService(
        data_dir=cfg.project.data_dir,
        actor=speaker.actor_objetivo,
        video_ref_id=speaker.referencia_voz.video_id,
        whisper_model=transcribe.whisper_model,
        idioma=transcribe.idioma,
        umbral_match=speaker.umbral_match,
        umbral_ambiguo=speaker.umbral_ambiguo,
        pipeline_name=speaker.pipeline,
        fallback_pipeline=speaker.fallback_pipeline,
        device=speaker.device,
    )


def _build_execution_argument_shape(cfg: RunConfig) -> ArgumentShapeService:
    from tono_politico.discursive_approach.argument_shape import ArgumentShapeService

    shape = cfg.discursive_approach.argument_shape
    return ArgumentShapeService(
        spacy_model=shape.spacy_model,
        embedding_model_name=shape.embedding_model,
        breakpoint_percentile=shape.breakpoint_percentile,
        min_oraciones=shape.min_oraciones,
        max_oraciones=shape.max_oraciones,
        max_palabras=shape.max_palabras,
    )


def _build_execution_topics_cluster(cfg: RunConfig) -> TopicsClusterService:
    from tono_politico.discursive_approach.topics_cluster import TopicsClusterService

    cluster = cfg.discursive_approach.topics_cluster
    return TopicsClusterService(
        min_topic_size=cluster.min_topic_size,
        n_neighbors=cluster.n_neighbors,
        n_components=cluster.n_components,
        embedding_model_name=cluster.embedding_model,
    )


def _build_execution_topics_approach(cfg: RunConfig) -> TopicsApproachService:
    from tono_politico.discursive_approach.topics_approach import TopicsApproachService

    actor = cfg.speech2text.speaker_timestamps.actor_objetivo
    return TopicsApproachService(actor=actor)


def _execution_factories() -> ExecutionFactories:
    """Construye factories del control plane nuevo."""
    return ExecutionFactories(
        build_speech2text=_build_execution_speech2text,
        build_argument_shape=_build_execution_argument_shape,
        build_topics_cluster=_build_execution_topics_cluster,
        build_topics_approach=_build_execution_topics_approach,
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


def _imprimir_enfoques(resultado) -> None:
    """Imprime temas y enfoques del path discursive_approach."""
    print("\n" + "=" * 60)
    print(f"Discursive: {resultado.num_temas} temas · {resultado.num_enfoques_total} enfoques")
    print("=" * 60)
    for tema in resultado.por_tema:
        t = tema.topico
        print(f"  [{t.id}] {t.nombre} ({t.num_argumentos} args)")
        for e in tema.enfoques:
            print(
                f"      · enfoque {e.id}: {e.nombre} "
                f"(n={e.num_argumentos}, {e.fecha_primera}→{e.fecha_ultima}, "
                f"stance={e.stance_dominante})"
            )
    print("=" * 60)
    print("Artefactos: discursive-temas.json · discursive-enfoques.json en el run dir")


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


def _imprimir_execution_plan(plan) -> None:
    """Imprime un dry-run legible del control plane nuevo."""
    print("\nExecution plan")
    print("=" * 60)
    print(f"Run ID: {plan.run_id}")
    print(f"Run dir: {plan.artifacts.run_dir}")
    for stage in plan.stages:
        estado = "run" if stage.should_run else "skip"
        razon = f" — {stage.skip_reason}" if stage.skip_reason else ""
        print(f"  [{estado}] {stage.name}{razon}")
    print("=" * 60)


def _imprimir_execution_result(result: ExecutionResult) -> None:
    """Imprime resumen corto del control plane nuevo."""
    print("\n" + "=" * 60)
    print(f"Execution result: exit_code={result.exit_code}")
    for stage in result.stage_results:
        detalle = f" — {stage.message}" if stage.message else ""
        print(f"  [{stage.status}] {stage.stage}{detalle}")
    if result.manifest_path is not None:
        print(f"Manifest: {result.manifest_path}")
    print("=" * 60)


def _execution_run_id() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d-%H%M%S")


def _run_execution_cli(config_path: Path, *, dry_run: bool, validate_only: bool) -> int:
    """Carga/valida/ejecuta el schema nuevo de config.yaml."""
    try:
        cfg = load_run_config(config_path)
        validate_run_config(cfg)
        run_id = cfg.run.id or _execution_run_id()
        artifacts = resolve_artifacts(cfg, run_id)
        plan = build_execution_plan(cfg, artifacts)
    except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
        print(f"Config inválido: {exc}", file=sys.stderr)
        return 2

    if validate_only:
        print(f"Config válido: {config_path}")
        return 0
    if dry_run:
        _imprimir_execution_plan(plan)
        return 0

    runner = ExecutionRunner(_execution_factories(), keep_cache=cfg.run.keep_cache)
    result = runner.execute(plan)
    _imprimir_execution_result(result)
    return result.exit_code


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
        default=None,
        help="URL de la playlist de YouTube (no necesario con --resume)",
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
        "--run-id",
        type=str,
        default=None,
        help="ID personalizado para la corrida (default: timestamp automático).",
    )
    parser.add_argument(
        "--discursive",
        action="store_true",
        help=(
            "Fase 1 path nuevo: speech2text → discursive_approach "
            "(shape/cluster/enfoques). Solo discover; no combina con --topico/--resume."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Logging DEBUG en vez de INFO",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida config nuevo e imprime el plan sin ejecutar stages.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Valida config nuevo y termina sin construir services.",
    )

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = Path(args.config)
    uses_legacy_cli = (
        args.playlist is not None
        or args.resume is not None
        or args.topico is not None
        or args.discursive
    )
    if is_run_config_file(config_path) and not uses_legacy_cli:
        return _run_execution_cli(
            config_path,
            dry_run=args.dry_run,
            validate_only=args.validate_config,
        )
    if args.dry_run or args.validate_config:
        parser.error("--dry-run/--validate-config requieren un config schema tono-politico.run.v1")

    if args.topico is not None and not args.tema:
        parser.error("--tema es obligatorio cuando se usa --topico")

    if args.resume is not None and args.topico is None:
        parser.error("--resume requiere --topico y --tema")

    if args.playlist is None and args.resume is None:
        parser.error("--playlist es obligatorio (salvo cuando se usa --resume)")

    if args.discursive and (args.topico is not None or args.resume is not None):
        parser.error("--discursive es solo para discover (no combinar con --topico/--resume)")

    cfg = cargar_config(config_path)
    runner = PipelineRunner(
        cfg=cfg,
        factories=_service_factories(),
        keep_cache=args.keep_cache,
        run_id=args.run_id,
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
        assert args.playlist is not None
        tema = args.tema
        assert tema is not None
        result = runner.analyze(args.playlist, args.topico, tema, args.output)
        if result.exit_code == 0:
            _imprimir_resumen_salida(result)
        else:
            _imprimir_resumen_fallo(result)
        return result.exit_code

    assert args.playlist is not None
    if args.discursive:
        result = runner.discover_discursive(args.playlist)
        if result.exit_code != 0:
            _imprimir_resumen_fallo(result)
            return result.exit_code
        _imprimir_resumen_salida(result)
        enfoques = getattr(runner, "last_resultado_enfoques", None)
        if enfoques is not None:
            _imprimir_enfoques(enfoques)
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
