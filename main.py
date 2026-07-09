#!/usr/bin/env python3
"""Punto de entrada del pipeline tono-politico.

Ejecuta el pipeline stage-based leyendo ``config/config.yaml`` (schema
``tono-politico.run.v1``) y delega la orquestación a ``ExecutionRunner``.

Uso::

    uv run python main.py --config config/config.yaml
    uv run python main.py --config config/config.yaml --dry-run
    uv run python main.py --config config/config.yaml --validate-config
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from tono_politico.execution import (
    ConfigValidationError,
    ExecutionFactories,
    ExecutionResult,
    ExecutionRunner,
    build_execution_plan,
    load_run_config,
    resolve_artifacts,
    validate_run_config,
)
from tono_politico.execution.models import RunConfig

if TYPE_CHECKING:
    from tono_politico.discursive_approach.argument_shape import ArgumentShapeService
    from tono_politico.discursive_approach.topics_approach import TopicsApproachService
    from tono_politico.discursive_approach.topics_cluster import TopicsClusterService
    from tono_politico.speech2text import SpeechToTextService

CONFIG_PATH = Path("config/config.yaml")


# ──────────────────────────────────────────────────────────
# Factory de services desde RunConfig
# ──────────────────────────────────────────────────────────


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
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help=f"Ruta del config YAML (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Logging DEBUG en vez de INFO",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida config e imprime el plan sin ejecutar stages.",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Valida config y termina sin construir services.",
    )

    args = parser.parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    return _run_execution_cli(
        Path(args.config),
        dry_run=args.dry_run,
        validate_only=args.validate_config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
