"""Runner stage-based para el path speech2text → discursive_approach."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from tono_politico.discursive_approach.argument_shape.models import Argumento
from tono_politico.discursive_approach.topics_approach.serializacion import (
    guardar_resultado_enfoques,
)
from tono_politico.discursive_approach.topics_cluster.models import ResultadoTemas
from tono_politico.discursive_approach.topics_cluster.serializacion import (
    cargar_resultado_temas,
    guardar_resultado_temas,
)
from tono_politico.speech2text.actor_transcript import (
    cargar_actor_transcript,
    guardar_actor_transcript,
)
from tono_politico.speech2text.audio_fetcher.cache import ruta_audio
from tono_politico.speech2text.models import ActorTranscript

from .artifacts import cargar_argumentos, guardar_argumentos
from .models import (
    ArtifactKey,
    ExecutionPlan,
    ExecutionResult,
    RunConfig,
    StageName,
    StageResult,
    StageSpec,
)


@dataclass(frozen=True)
class ExecutionFactories:
    """Factories inyectables para services del runner de ejecución."""

    build_speech2text: Callable[[RunConfig], Any]
    build_argument_shape: Callable[[RunConfig], Any]
    build_topics_cluster: Callable[[RunConfig], Any]
    build_topics_approach: Callable[[RunConfig], Any]


class ExecutionRunner:
    """Ejecuta un ``ExecutionPlan`` stage por stage."""

    def __init__(self, factories: ExecutionFactories, keep_cache: bool = False) -> None:
        self.factories = factories
        self.keep_cache = keep_cache

    def execute(self, plan: ExecutionPlan) -> ExecutionResult:
        plan.artifacts.run_dir.mkdir(parents=True, exist_ok=True)
        if plan.config.output.persist_resolved_config:
            self._persist_resolved_config(plan)

        context: dict[str, Any] = {}
        results: list[StageResult] = []
        exit_code = 0

        for stage in plan.stages:
            started = time.perf_counter()
            if not stage.should_run:
                try:
                    self._load_skipped_stage_context(plan, stage, context)
                    results.append(
                        StageResult(
                            stage=stage.name,
                            status="skipped",
                            message=stage.skip_reason or "stage saltado",
                            elapsed_seconds=time.perf_counter() - started,
                        )
                    )
                    continue
                except Exception as exc:
                    results.append(
                        StageResult(
                            stage=stage.name,
                            status="failed",
                            message=f"{type(exc).__name__}: {exc}",
                            elapsed_seconds=time.perf_counter() - started,
                        )
                    )
                    exit_code = 1
                    if plan.config.run.fail_fast:
                        break
                    continue

            missing = self._missing_dependencies(plan, stage, context)
            if missing:
                results.append(
                    StageResult(
                        stage=stage.name,
                        status="skipped",
                        message=f"dependencias no satisfechas: {', '.join(missing)}",
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
                exit_code = 1
                if plan.config.run.fail_fast:
                    break
                continue

            try:
                self._run_stage(plan, stage.name, context)
                results.append(
                    StageResult(
                        stage=stage.name,
                        status="ok",
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
            except Exception as exc:
                results.append(
                    StageResult(
                        stage=stage.name,
                        status="failed",
                        message=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=time.perf_counter() - started,
                    )
                )
                exit_code = 1
                if plan.config.run.fail_fast:
                    break
                continue

        manifest_path = None
        if plan.config.output.persist_manifest:
            manifest_path = self._persist_manifest(plan, results, exit_code)
        return ExecutionResult(
            exit_code=exit_code,
            plan=plan,
            stage_results=results,
            manifest_path=manifest_path,
        )

    def _run_stage(
        self,
        plan: ExecutionPlan,
        stage: StageName,
        context: dict[str, Any],
    ) -> None:
        if stage == "speech2text":
            transcripts = self._run_speech2text(plan)
            context["actor_transcripts"] = transcripts
            return
        if stage == "argument_shape":
            transcripts = self._get_actor_transcripts(plan, context)
            argumentos = self._run_argument_shape(plan.config, transcripts)
            guardar_argumentos(argumentos, plan.artifacts.argumentos_path)
            context["argumentos"] = argumentos
            return
        if stage == "topics_cluster":
            argumentos = self._get_argumentos(plan, context)
            resultado = self._run_topics_cluster(plan.config, argumentos)
            guardar_resultado_temas(resultado, plan.artifacts.temas_path)
            context["temas"] = resultado
            return
        temas = self._get_temas(plan, context)
        enfoques = self._run_topics_approach(plan.config, temas)
        guardar_resultado_enfoques(enfoques, plan.artifacts.enfoques_path)
        context["enfoques"] = enfoques

    def _run_speech2text(self, plan: ExecutionPlan) -> list[ActorTranscript]:
        if plan.config.input.playlist_url is None:
            raise RuntimeError("input.playlist_url requerido para speech2text")
        service = self.factories.build_speech2text(plan.config)
        if all(hasattr(service, attr) for attr in ("discover", "ensure_perfil", "procesar_one")):
            return self._run_speech2text_granular(plan, service)

        transcripts = list(service.procesar(plan.config.input.playlist_url))
        for transcript in transcripts:
            self._persist_actor_transcript(plan, transcript)
        return transcripts

    def _run_speech2text_granular(
        self,
        plan: ExecutionPlan,
        service: Any,
    ) -> list[ActorTranscript]:
        playlist, metas = service.discover(plan.config.input.playlist_url)
        metas_seleccionadas = self._filter_video_metas(plan.config, list(metas))
        transcripts: list[ActorTranscript] = []
        ref_video_id = plan.config.speech2text.speaker_timestamps.referencia_voz.video_id

        try:
            if not metas_seleccionadas:
                return []
            if not service.ensure_perfil(playlist.nombre, list(metas)):
                return []

            for meta in metas_seleccionadas:
                try:
                    transcript = service.procesar_one(meta, playlist.nombre)
                    if transcript is not None:
                        self._persist_actor_transcript(plan, transcript)
                        transcripts.append(transcript)
                finally:
                    self._cleanup_audio(plan, playlist.nombre, meta.video_id)
        finally:
            self._cleanup_audio(plan, playlist.nombre, ref_video_id)

        return transcripts

    def _filter_video_metas(self, cfg: RunConfig, metas: list[Any]) -> list[Any]:
        selected = metas
        if cfg.run.only_video_ids:
            allowed = set(cfg.run.only_video_ids)
            selected = [meta for meta in selected if meta.video_id in allowed]
        if cfg.run.max_videos is not None:
            selected = selected[: cfg.run.max_videos]
        return selected

    def _persist_actor_transcript(self, plan: ExecutionPlan, transcript: ActorTranscript) -> None:
        plan.artifacts.actor_transcripts_dir.mkdir(parents=True, exist_ok=True)
        guardar_actor_transcript(
            transcript,
            plan.artifacts.actor_transcripts_dir / f"{transcript.video_id}.json",
        )

    def _cleanup_audio(self, plan: ExecutionPlan, playlist_name: str, video_id: str) -> None:
        if self.keep_cache or plan.config.run.keep_cache:
            return
        path = ruta_audio(playlist_name, video_id, plan.config.project.data_dir)
        if path.exists() and path.is_file():
            path.unlink()

    def _run_argument_shape(
        self,
        cfg: RunConfig,
        transcripts: list[ActorTranscript],
    ) -> list[Argumento]:
        service = self.factories.build_argument_shape(cfg)
        if hasattr(service, "procesar_corpus"):
            return list(service.procesar_corpus(transcripts))
        return list(service.shape_corpus(transcripts))

    def _run_topics_cluster(self, cfg: RunConfig, argumentos: list[Argumento]) -> ResultadoTemas:
        service = self.factories.build_topics_cluster(cfg)
        if hasattr(service, "procesar"):
            return service.procesar(argumentos)
        return service.cluster(argumentos)

    def _run_topics_approach(self, cfg: RunConfig, temas: ResultadoTemas) -> Any:
        service = self.factories.build_topics_approach(cfg)
        if hasattr(service, "procesar"):
            return service.procesar(temas)
        return service.approaches(temas)

    def _missing_dependencies(
        self,
        plan: ExecutionPlan,
        stage: StageSpec,
        context: dict[str, Any],
    ) -> list[ArtifactKey]:
        return [
            dependency
            for dependency in stage.requires
            if not self._dependency_satisfied(plan, dependency, context)
        ]

    def _dependency_satisfied(
        self,
        plan: ExecutionPlan,
        dependency: ArtifactKey,
        context: dict[str, Any],
    ) -> bool:
        if dependency == "playlist_url":
            return plan.config.input.playlist_url is not None
        if dependency == "actor_transcripts_dir":
            return (
                "actor_transcripts" in context
                or self._dir_exists(plan.config.input.actor_transcripts_dir)
                or self._dir_exists(plan.config.discursive_approach.input.actor_transcripts_dir)
                or self._dir_exists(plan.artifacts.actor_transcripts_dir)
            )
        if dependency == "argumentos_path":
            return (
                "argumentos" in context
                or self._file_exists(plan.config.input.argumentos_path)
                or self._file_exists(plan.artifacts.argumentos_path)
            )
        if dependency == "temas_path":
            return (
                "temas" in context
                or self._file_exists(plan.config.input.temas_path)
                or self._file_exists(plan.artifacts.temas_path)
            )
        if dependency == "enfoques_path":
            return "enfoques" in context or self._file_exists(plan.config.input.enfoques_path)
        return False

    def _dir_exists(self, path: Path | None) -> bool:
        return path is not None and path.exists() and path.is_dir()

    def _file_exists(self, path: Path | None) -> bool:
        return path is not None and path.exists() and path.is_file()

    def _load_skipped_stage_context(
        self,
        plan: ExecutionPlan,
        stage: StageSpec,
        context: dict[str, Any],
    ) -> None:
        if stage.name == "speech2text":
            context["actor_transcripts"] = self._load_actor_transcripts(
                plan.artifacts.actor_transcripts_dir
            )
        elif stage.name == "argument_shape":
            context["argumentos"] = cargar_argumentos(plan.artifacts.argumentos_path)
        elif stage.name == "topics_cluster":
            context["temas"] = cargar_resultado_temas(plan.artifacts.temas_path)

    def _get_actor_transcripts(
        self,
        plan: ExecutionPlan,
        context: dict[str, Any],
    ) -> list[ActorTranscript]:
        if "actor_transcripts" in context:
            return context["actor_transcripts"]
        source = (
            plan.config.input.actor_transcripts_dir
            or plan.config.discursive_approach.input.actor_transcripts_dir
            or plan.artifacts.actor_transcripts_dir
        )
        return self._load_actor_transcripts(source)

    def _get_argumentos(self, plan: ExecutionPlan, context: dict[str, Any]) -> list[Argumento]:
        if "argumentos" in context:
            return context["argumentos"]
        path = plan.config.input.argumentos_path or plan.artifacts.argumentos_path
        return cargar_argumentos(path)

    def _get_temas(self, plan: ExecutionPlan, context: dict[str, Any]) -> ResultadoTemas:
        if "temas" in context:
            return context["temas"]
        path = plan.config.input.temas_path or plan.artifacts.temas_path
        return cargar_resultado_temas(path)

    def _load_actor_transcripts(self, source: Path) -> list[ActorTranscript]:
        return [cargar_actor_transcript(path) for path in sorted(source.glob("*.json"))]

    def _persist_resolved_config(self, plan: ExecutionPlan) -> None:
        plan.artifacts.resolved_config_path.parent.mkdir(parents=True, exist_ok=True)
        plan.artifacts.resolved_config_path.write_text(
            yaml.safe_dump(_jsonable(asdict(plan.config)), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def _persist_manifest(
        self,
        plan: ExecutionPlan,
        results: list[StageResult],
        exit_code: int,
    ) -> Path:
        plan.artifacts.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "tono-politico.execution_manifest.v1",
            "run_id": plan.run_id,
            "status": "ok" if exit_code == 0 else "failed",
            "stages": [_jsonable(asdict(result)) for result in results],
            "artifacts": _jsonable(asdict(plan.artifacts)),
        }
        plan.artifacts.manifest_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return plan.artifacts.manifest_path


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
