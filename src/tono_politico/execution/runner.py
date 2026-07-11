"""Runner stage-based para el path speech2text → discursive_approach."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
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
from tono_politico.speech2text.audio_fetcher.audio import ruta_audio
from tono_politico.speech2text.audio_fetcher.models import VideoMeta
from tono_politico.speech2text.models import ActorTranscript

from .artifacts import (
    cargar_actor_transcript,
    cargar_argumentos,
    guardar_actor_transcript,
    guardar_argumentos,
)
from .models import (
    ArtifactKey,
    ExecutionPlan,
    ExecutionResult,
    RunConfig,
    StageName,
    StageResult,
    StageSpec,
    UnitResult,
)
from .observability import build_quality_report, guardar_quality_report


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
            manifest_path = self._persist_manifest(
                plan,
                results,
                exit_code,
                context.get("unit_results", []),
                context.get("speech2text_provenance"),
            )
        return ExecutionResult(
            exit_code=exit_code,
            plan=plan,
            stage_results=results,
            manifest_path=manifest_path,
            unit_results=list(context.get("unit_results", [])),
        )

    def _run_stage(
        self,
        plan: ExecutionPlan,
        stage: StageName,
        context: dict[str, Any],
    ) -> None:
        if stage == "speech2text":
            transcripts, unit_results, provenance = self._run_speech2text(plan)
            context["unit_results"] = unit_results
            context["speech2text_provenance"] = provenance
            if any(unit.status == "failed" for unit in unit_results):
                raise RuntimeError("speech2text terminó con unidades fallidas")
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

    def _run_speech2text(
        self,
        plan: ExecutionPlan,
    ) -> tuple[list[ActorTranscript], list[UnitResult], dict[str, Any]]:
        if plan.config.input.playlist_url is None:
            raise RuntimeError("input.playlist_url requerido para speech2text")
        service = self.factories.build_speech2text(plan.config)
        required = ("discover", "ensure_perfil", "procesar_one")
        if not all(hasattr(service, attr) for attr in required):
            raise RuntimeError(
                "SpeechToTextService requiere la API granular: "
                "discover, ensure_perfil y procesar_one"
            )
        return self._run_speech2text_granular(plan, service)

    def _run_speech2text_granular(
        self,
        plan: ExecutionPlan,
        service: Any,
    ) -> tuple[list[ActorTranscript], list[UnitResult], dict[str, Any]]:
        playlist, metas = service.discover(plan.config.input.playlist_url)
        metas_seleccionadas = self._filter_video_metas(plan.config, list(metas))
        transcripts: list[ActorTranscript] = []
        unit_results: list[UnitResult] = []
        ref_video_id = plan.config.speech2text.speaker_timestamps.referencia_voz.video_id
        ref_meta = next((meta for meta in metas if meta.video_id == ref_video_id), None)
        reference_video = _video_meta_to_dict(ref_meta) or {
            "video_id": ref_video_id,
            "present_in_playlist": False,
        }
        reference_video["reason_code"] = "reference_profile"
        reference_video["present_in_playlist"] = ref_meta is not None
        provenance = {
            "playlist": {
                "name": playlist.nombre,
                "cache_name": playlist.cache_name,
                "playlist_id": playlist.playlist_id,
                "url": playlist.url or plan.config.input.playlist_url,
            },
            "discovered_videos": len(metas),
            "selected_videos": len(metas_seleccionadas),
            "reference_video": reference_video,
            "reference_profile_status": "pending",
        }

        try:
            if not metas_seleccionadas:
                provenance["reference_profile_status"] = "not_run"
                return transcripts, unit_results, provenance

            if not service.ensure_perfil(playlist, list(metas)):
                provenance["reference_profile_status"] = "failed"
                unit_results = [
                    UnitResult(
                        video_id=meta.video_id,
                        status="failed",
                        reason_code="reference_profile_missing",
                        video_title=meta.titulo,
                        fecha=meta.fecha,
                        fecha_fuente=meta.fecha_fuente,
                        duration=meta.duracion,
                    )
                    for meta in metas_seleccionadas
                ]
                raise RuntimeError("reference_profile_missing: no se pudo construir el perfil")

            provenance["reference_profile_status"] = "ready"
            for meta in metas_seleccionadas:
                if plan.config.run.resume and self._already_has_transcript(plan, meta.video_id):
                    existing = cargar_actor_transcript(
                        plan.artifacts.actor_transcripts_dir / f"{meta.video_id}.json"
                    )
                    transcripts.append(existing)
                    unit_results.append(
                        UnitResult(
                            video_id=meta.video_id,
                            status="ok",
                            reason_code="resumed_from_cache",
                            transcript=existing,
                            timings={"total": 0.0},
                            video_title=meta.titulo,
                            fecha=meta.fecha,
                            fecha_fuente=meta.fecha_fuente,
                            duration=meta.duracion,
                        )
                    )
                    self._persist_manifest_checkpoint(plan, unit_results, provenance)
                    continue

                started = time.perf_counter()
                try:
                    transcript = service.procesar_one(meta, playlist)
                    reason_code = getattr(service, "last_reason_code", None)
                    elapsed = time.perf_counter() - started
                    if transcript is None:
                        unit_results.append(
                            UnitResult(
                                video_id=meta.video_id,
                                status="skipped",
                                reason_code=reason_code or "asr_empty",
                                timings={"total": elapsed},
                                video_title=meta.titulo,
                                fecha=meta.fecha,
                                fecha_fuente=meta.fecha_fuente,
                                duration=meta.duracion,
                            )
                        )
                    else:
                        self._persist_actor_transcript(plan, transcript)
                        transcripts.append(transcript)
                        unit_results.append(
                            UnitResult(
                                video_id=meta.video_id,
                                status="ok",
                                reason_code="transcript_persisted",
                                transcript=transcript,
                                timings={"total": elapsed},
                                video_title=meta.titulo,
                                fecha=meta.fecha,
                                fecha_fuente=meta.fecha_fuente,
                                duration=meta.duracion,
                            )
                        )
                except Exception as exc:
                    unit_results.append(
                        UnitResult(
                            video_id=meta.video_id,
                            status="failed",
                            reason_code=_reason_code_for_exception(exc),
                            timings={"total": time.perf_counter() - started},
                            error=str(exc),
                            video_title=meta.titulo,
                            fecha=meta.fecha,
                            fecha_fuente=meta.fecha_fuente,
                            duration=meta.duracion,
                        )
                    )
                    if plan.config.run.fail_fast:
                        raise
                finally:
                    self._cleanup_audio(plan, playlist.cache_name, meta.video_id)
                    self._persist_manifest_checkpoint(plan, unit_results, provenance)
        finally:
            self._cleanup_audio(plan, playlist.cache_name, ref_video_id)
            self._persist_speech2text_quality(plan, unit_results, provenance)

        return transcripts, unit_results, provenance

    def _filter_video_metas(self, cfg: RunConfig, metas: list[Any]) -> list[Any]:
        return select_video_metas(
            metas,
            only_video_ids=cfg.run.only_video_ids,
            max_videos=cfg.run.max_videos,
        )

    def _persist_actor_transcript(self, plan: ExecutionPlan, transcript: ActorTranscript) -> None:
        plan.artifacts.actor_transcripts_dir.mkdir(parents=True, exist_ok=True)
        guardar_actor_transcript(
            transcript,
            plan.artifacts.actor_transcripts_dir / f"{transcript.video_id}.json",
        )

    def _persist_speech2text_quality(
        self,
        plan: ExecutionPlan,
        unit_results: list[UnitResult],
        provenance: dict[str, Any] | None = None,
    ) -> None:
        guardar_quality_report(
            build_quality_report(unit_results, provenance=provenance),
            plan.artifacts.speech2text_quality_path,
        )

    def _persist_manifest_checkpoint(
        self,
        plan: ExecutionPlan,
        unit_results: list[UnitResult],
        provenance: dict[str, Any],
    ) -> None:
        """Escribe checkpoint incremental con las unidades procesadas hasta ahora."""
        checkpoint_path = plan.artifacts.run_dir / "speech2text" / "checkpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "speech2text_checkpoint.v2",
            "speech2text": provenance,
            "units": [_unit_result_to_manifest(unit) for unit in unit_results],
        }
        checkpoint_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _already_has_transcript(self, plan: ExecutionPlan, video_id: str) -> bool:
        """Comprueba si un vídeo ya tiene transcript persistido válido."""
        path = plan.artifacts.actor_transcripts_dir / f"{video_id}.json"
        if not path.exists() or not path.is_file():
            return False
        from .artifacts import _valid_actor_transcript_file

        return _valid_actor_transcript_file(path)

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
            transcripts = self._load_actor_transcripts(plan.artifacts.actor_transcripts_dir)
            context["actor_transcripts"] = transcripts
            context["speech2text_provenance"] = self._load_speech2text_provenance(plan, transcripts)
            context["unit_results"] = [
                UnitResult(
                    video_id=transcript.video_id,
                    status="ok",
                    reason_code="resumed_from_artifact",
                    transcript=transcript,
                    video_title=transcript.source.video_title if transcript.source else None,
                    fecha=(
                        transcript.source.upload_date
                        if transcript.source and transcript.source.upload_date is not None
                        else transcript.fecha
                    ),
                    fecha_fuente=transcript.source.date_source if transcript.source else None,
                )
                for transcript in transcripts
            ]
            if not plan.artifacts.speech2text_quality_path.exists():
                self._persist_speech2text_quality(
                    plan,
                    context["unit_results"],
                    context["speech2text_provenance"],
                )
        elif stage.name == "argument_shape":
            context["argumentos"] = cargar_argumentos(plan.artifacts.argumentos_path)
        elif stage.name == "topics_cluster":
            context["temas"] = cargar_resultado_temas(plan.artifacts.temas_path)

    def _load_speech2text_provenance(
        self,
        plan: ExecutionPlan,
        transcripts: list[ActorTranscript],
    ) -> dict[str, Any] | None:
        """Recupera provenance existente sin inventar datos ausentes."""
        quality_path = plan.artifacts.speech2text_quality_path
        if quality_path.exists():
            try:
                quality = json.loads(quality_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                quality = None
            if isinstance(quality, dict) and isinstance(quality.get("provenance"), dict):
                return quality["provenance"]

        source = next((transcript.source for transcript in transcripts if transcript.source), None)
        if source is None:
            return None
        reference_id = plan.config.speech2text.speaker_timestamps.referencia_voz.video_id
        return {
            "playlist": {
                "name": source.playlist_name,
                "cache_name": source.playlist_name,
                "playlist_id": source.playlist_id,
                "url": source.playlist_url,
            },
            "discovered_videos": None,
            "selected_videos": len(transcripts),
            "reference_video": {
                "video_id": reference_id,
                "reason_code": "reference_profile",
                "present_in_playlist": None,
            },
            "reference_profile_status": "resumed_from_artifact",
        }

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
        unit_results: list[UnitResult],
        speech2text_provenance: dict[str, Any] | None,
    ) -> Path:
        plan.artifacts.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "tono-politico.execution_manifest.v2",
            "run_id": plan.run_id,
            "status": "ok" if exit_code == 0 else "failed",
            "stages": [_jsonable(asdict(result)) for result in results],
            "speech2text": speech2text_provenance,
            "units": [_unit_result_to_manifest(result) for result in unit_results],
            "artifacts": _jsonable(asdict(plan.artifacts)),
            "config_fingerprint": _config_fingerprint(plan.config),
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


def _unit_result_to_manifest(result: UnitResult) -> dict[str, Any]:
    """Serializa sólo procedencia de ejecución, nunca contenido textual."""
    return {
        "video_id": result.video_id,
        "status": result.status,
        "reason_code": result.reason_code,
        "timings": _jsonable(result.timings),
        "error": result.error,
        "video_title": result.video_title,
        "fecha": result.fecha,
        "fecha_fuente": result.fecha_fuente,
        "duration": result.duration,
    }


def _video_meta_to_dict(meta: VideoMeta | None) -> dict[str, Any] | None:
    if meta is None:
        return None
    return {
        "video_id": meta.video_id,
        "url": meta.url,
        "title": meta.titulo,
        "upload_date": meta.fecha,
        "date_source": meta.fecha_fuente,
        "duration": meta.duracion,
    }


def select_video_metas(
    metas: Sequence[VideoMeta],
    *,
    only_video_ids: Sequence[str] | None = None,
    max_videos: int | None = None,
) -> list[VideoMeta]:
    """Selecciona vídeos preservando el orden descubierto por la playlist."""
    if max_videos is not None and max_videos < 0:
        raise ValueError("max_videos no puede ser negativo")

    selected = list(metas)
    if only_video_ids:
        allowed = set(only_video_ids)
        selected = [meta for meta in selected if meta.video_id in allowed]
    if max_videos is not None:
        selected = selected[:max_videos]
    return selected


def _reason_code_for_exception(exc: Exception) -> str:
    if isinstance(exc, FileNotFoundError):
        return "audio_invalid"
    if isinstance(exc, TimeoutError):
        return "download_failed"
    return "transcript_invalid"


def _config_fingerprint(cfg: RunConfig) -> dict[str, Any]:
    """Registra parámetros operativos que afectan reproducibilidad."""
    speaker = cfg.speech2text.speaker_timestamps
    return {
        "speech2text_quality_schema": "speech2text_quality.v2",
        "actor_transcript_schema": "actor_transcript.v1",
        "whisper_model": cfg.speech2text.transcribe_speech.whisper_model,
        "whisper_idioma": cfg.speech2text.transcribe_speech.idioma,
        "pipeline": speaker.pipeline,
        "umbral_match": speaker.umbral_match,
        "umbral_ambiguo": speaker.umbral_ambiguo,
        "actor_objetivo": speaker.actor_objetivo,
        "only_video_ids": list(cfg.run.only_video_ids) if cfg.run.only_video_ids else [],
        "max_videos": cfg.run.max_videos,
    }
