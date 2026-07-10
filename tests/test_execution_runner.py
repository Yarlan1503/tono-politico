"""Tests del ExecutionRunner stage-based con services fake."""

from __future__ import annotations

import json
from pathlib import Path

from tono_politico.discursive_approach.argument_shape.models import Argumento, Oracion
from tono_politico.discursive_approach.topics_approach.models import ResultadoEnfoques
from tono_politico.discursive_approach.topics_cluster.models import (
    ArgumentoTematizado,
    ResultadoTemas,
    TopicoInfo,
)
from tono_politico.execution.artifacts import resolve_artifacts
from tono_politico.execution.config import load_run_config
from tono_politico.execution.plan import build_execution_plan
from tono_politico.execution.runner import ExecutionFactories, ExecutionRunner
from tono_politico.speech2text.audio_fetcher.audio import ruta_audio
from tono_politico.speech2text.audio_fetcher.models import PlaylistInfo, VideoMeta
from tono_politico.speech2text.models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
)


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "run-config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _load(tmp_path: Path, content: str):
    return load_run_config(_write_config(tmp_path, content))


def _actor_transcript(video_id: str = "vid-1") -> ActorTranscript:
    return ActorTranscript(
        schema_version="actor_transcript.v1",
        video_id=video_id,
        actor="Lilly Téllez",
        scope="actor_only",
        asr=AsrMetadata(provider="whisper", model="tiny", language="es"),
        segments=[
            ActorTranscriptSegment(
                text="Texto del actor",
                t_start=0.0,
                t_end=1.0,
                speaker="SPEAKER_00",
                source_turn_start=0.0,
                source_turn_end=1.0,
                word_count=3,
            )
        ],
        fecha="20260101",
    )


def _argumento() -> Argumento:
    return Argumento(
        texto="Texto del actor",
        t_start=0.0,
        t_end=1.0,
        oraciones=[Oracion(texto="Texto del actor", t_start=0.0, t_end=1.0)],
        word_count=3,
        video_id="vid-1",
        fecha="20260101",
    )


def _resultado_temas(argumento: Argumento | None = None) -> ResultadoTemas:
    argumento = argumento or _argumento()
    topico = TopicoInfo(id=0, nombre="seguridad", palabras_clave=["seguridad"], num_argumentos=1)
    return ResultadoTemas(
        argumentos=[ArgumentoTematizado(argumento=argumento, topico_id=0, probabilidad=1.0)],
        topicos=[topico],
        num_topicos=1,
    )


class Recorder:
    def __init__(self, name: str, calls: list[str]):
        self.name = name
        self.calls = calls


class LegacySpeechToText:
    def procesar(self, playlist_url: str) -> list[ActorTranscript]:
        return [_actor_transcript()]


class GranularSpeechToText:
    def __init__(self, calls: list[str], data_dir: Path) -> None:
        self.calls = calls
        self.data_dir = data_dir

    def discover(self, playlist_url: str) -> tuple[PlaylistInfo, list[VideoMeta]]:
        self.calls.append(f"discover:{playlist_url}")
        return PlaylistInfo(nombre="playlist"), [
            VideoMeta("vid-1", "https://youtu.be/vid-1", "Video 1", "20260101", 10.0),
            VideoMeta("vid-2", "https://youtu.be/vid-2", "Video 2", "20260102", 10.0),
            VideoMeta("vid-3", "https://youtu.be/vid-3", "Video 3", "20260103", 10.0),
        ]

    def ensure_perfil(self, nombre_playlist: str, metas: list[VideoMeta]) -> bool:
        self.calls.append(f"perfil:{nombre_playlist}:{len(metas)}")
        ref_audio = ruta_audio(nombre_playlist, "su9nURIj9XQ", self.data_dir)
        ref_audio.parent.mkdir(parents=True, exist_ok=True)
        ref_audio.write_text("ref wav", encoding="utf-8")
        return True

    def procesar_one(
        self,
        video: VideoMeta,
        nombre_playlist: str,
        *,
        archive_path: Path | None = None,
    ) -> ActorTranscript:
        self.calls.append(f"one:{video.video_id}")
        audio_path = ruta_audio(nombre_playlist, video.video_id, self.data_dir)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_text("wav", encoding="utf-8")
        return _actor_transcript(video.video_id)


class FailingGranularSpeechToText(GranularSpeechToText):
    def procesar_one(
        self,
        video: VideoMeta,
        nombre_playlist: str,
        *,
        archive_path: Path | None = None,
    ) -> ActorTranscript:
        self.calls.append(f"one:{video.video_id}")
        raise RuntimeError("speech roto")


class FakeArgumentShape(Recorder):
    def procesar_corpus(self, transcripts: list[ActorTranscript]) -> list[Argumento]:
        self.calls.append(f"{self.name}:{len(transcripts)}")
        return [_argumento()]


class FakeTopicsCluster(Recorder):
    def procesar(self, argumentos: list[Argumento]) -> ResultadoTemas:
        self.calls.append(f"{self.name}:{len(argumentos)}")
        return _resultado_temas(argumentos[0])


class FakeTopicsApproach(Recorder):
    def procesar(self, resultado: ResultadoTemas) -> ResultadoEnfoques:
        self.calls.append(f"{self.name}:{resultado.num_topicos}")
        return ResultadoEnfoques(num_temas=resultado.num_topicos, num_enfoques_total=0)


class FailingShape(Recorder):
    def procesar_corpus(self, transcripts: list[ActorTranscript]) -> list[Argumento]:
        self.calls.append(f"{self.name}:{len(transcripts)}")
        raise RuntimeError("shape roto")


def _factories(calls: list[str], *, failing_shape: bool = False) -> ExecutionFactories:
    return ExecutionFactories(
        build_speech2text=lambda cfg: GranularSpeechToText(calls, cfg.project.data_dir),
        build_argument_shape=lambda _cfg: (
            FailingShape("argument_shape", calls)
            if failing_shape
            else FakeArgumentShape("argument_shape", calls)
        ),
        build_topics_cluster=lambda _cfg: FakeTopicsCluster("topics_cluster", calls),
        build_topics_approach=lambda _cfg: FakeTopicsApproach("topics_approach", calls),
    )


def _factories_with_failing_speech(calls: list[str]) -> ExecutionFactories:
    return ExecutionFactories(
        build_speech2text=lambda cfg: FailingGranularSpeechToText(calls, cfg.project.data_dir),
        build_argument_shape=lambda _cfg: FakeArgumentShape("argument_shape", calls),
        build_topics_cluster=lambda _cfg: FakeTopicsCluster("topics_cluster", calls),
        build_topics_approach=lambda _cfg: FakeTopicsApproach("topics_approach", calls),
    )


def _factories_with_granular_speech(calls: list[str]) -> ExecutionFactories:
    return ExecutionFactories(
        build_speech2text=lambda cfg: GranularSpeechToText(calls, cfg.project.data_dir),
        build_argument_shape=lambda _cfg: FakeArgumentShape("argument_shape", calls),
        build_topics_cluster=lambda _cfg: FakeTopicsCluster("topics_cluster", calls),
        build_topics_approach=lambda _cfg: FakeTopicsApproach("topics_approach", calls),
    )


def test_execution_runner_ejecuta_stages_en_orden_y_persiste_artefactos(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text, argument_shape, topics_cluster, topics_approach]
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, "run-001")
    plan = build_execution_plan(cfg, artifacts)
    calls: list[str] = []

    result = ExecutionRunner(_factories(calls)).execute(plan)

    assert result.exit_code == 0
    assert calls == [
        "discover:playlist-url",
        "perfil:playlist:3",
        "one:vid-1",
        "one:vid-2",
        "one:vid-3",
        "argument_shape:3",
        "topics_cluster:1",
        "topics_approach:1",
    ]
    assert [stage.status for stage in result.stage_results] == ["ok", "ok", "ok", "ok"]
    assert (artifacts.actor_transcripts_dir / "vid-1.json").exists()
    assert artifacts.argumentos_path.exists()
    assert artifacts.temas_path.exists()
    assert artifacts.enfoques_path.exists()
    assert artifacts.manifest_path.exists()
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["units"]) == 3
    assert manifest["units"][0]["reason_code"] == "transcript_persisted"
    assert "Texto del actor" not in json.dumps(manifest, ensure_ascii=False)
    assert artifacts.resolved_config_path.exists()


def test_execution_runner_rechaza_service_speech2text_legacy(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    plan = build_execution_plan(cfg, resolve_artifacts(cfg, "run-001"))
    factories = ExecutionFactories(
        build_speech2text=lambda _cfg: LegacySpeechToText(),
        build_argument_shape=lambda _cfg: FakeArgumentShape("argument_shape", []),
        build_topics_cluster=lambda _cfg: FakeTopicsCluster("topics_cluster", []),
        build_topics_approach=lambda _cfg: FakeTopicsApproach("topics_approach", []),
    )

    result = ExecutionRunner(factories).execute(plan)

    assert result.exit_code == 1
    assert result.stage_results[0].status == "failed"
    assert "API granular" in (result.stage_results[0].message or "")


def test_execution_runner_salta_stage_y_carga_artefacto_para_dependientes(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text, argument_shape]
  resume: true
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, "run-001")
    artifacts.actor_transcripts_dir.mkdir(parents=True)
    from tono_politico.execution.artifacts import guardar_actor_transcript

    guardar_actor_transcript(_actor_transcript(), artifacts.actor_transcripts_dir / "vid-1.json")
    plan = build_execution_plan(cfg, artifacts)
    calls: list[str] = []

    result = ExecutionRunner(_factories(calls)).execute(plan)

    assert result.exit_code == 0
    assert [stage.status for stage in result.stage_results] == ["skipped", "ok"]
    assert calls == ["argument_shape:1"]
    assert artifacts.argumentos_path.exists()


def test_execution_runner_registra_fallo_y_respeta_fail_fast(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text, argument_shape, topics_cluster]
  fail_fast: true
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    plan = build_execution_plan(cfg, resolve_artifacts(cfg, "run-001"))
    calls: list[str] = []

    result = ExecutionRunner(_factories(calls, failing_shape=True)).execute(plan)

    assert result.exit_code == 1
    assert calls == [
        "discover:playlist-url",
        "perfil:playlist:3",
        "one:vid-1",
        "one:vid-2",
        "one:vid-3",
        "argument_shape:3",
    ]
    assert [stage.status for stage in result.stage_results] == ["ok", "failed"]
    assert "shape roto" in (result.stage_results[-1].message or "")


def test_execution_runner_fail_fast_false_continua_si_dependencia_externa_existe(
    tmp_path: Path,
):
    external_transcripts = tmp_path / "external-transcripts"
    external_transcripts.mkdir()
    from tono_politico.execution.artifacts import guardar_actor_transcript

    guardar_actor_transcript(_actor_transcript(), external_transcripts / "vid-1.json")
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text, argument_shape]
  fail_fast: false
input:
  playlist_url: playlist-url
  actor_transcripts_dir: {external_transcripts}
output:
  base_dir: {tmp_path / "output"}
""",
    )
    plan = build_execution_plan(cfg, resolve_artifacts(cfg, "run-001"))
    calls: list[str] = []

    result = ExecutionRunner(_factories_with_failing_speech(calls)).execute(plan)

    assert result.exit_code == 1
    assert calls == [
        "discover:playlist-url",
        "perfil:playlist:3",
        "one:vid-1",
        "one:vid-2",
        "one:vid-3",
        "argument_shape:1",
    ]
    assert [stage.status for stage in result.stage_results] == ["failed", "ok"]


def test_execution_runner_fail_fast_false_salta_dependiente_sin_artefacto(
    tmp_path: Path,
):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text, argument_shape]
  fail_fast: false
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    plan = build_execution_plan(cfg, resolve_artifacts(cfg, "run-001"))
    calls: list[str] = []

    result = ExecutionRunner(_factories_with_failing_speech(calls)).execute(plan)

    assert result.exit_code == 1
    assert calls == [
        "discover:playlist-url",
        "perfil:playlist:3",
        "one:vid-1",
        "one:vid-2",
        "one:vid-3",
    ]
    assert [stage.status for stage in result.stage_results] == ["failed", "skipped"]
    assert "dependencias no satisfechas" in result.stage_results[-1].message


def test_execution_runner_speech2text_respeta_filtros_y_limpia_wavs(tmp_path: Path):
    data_dir = tmp_path / "data"
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
  max_videos: 1
  only_video_ids: [vid-2, vid-3]
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
project:
  data_dir: {data_dir}
""",
    )
    artifacts = resolve_artifacts(cfg, "run-001")
    plan = build_execution_plan(cfg, artifacts)
    calls: list[str] = []

    result = ExecutionRunner(_factories_with_granular_speech(calls)).execute(plan)

    assert result.exit_code == 0
    assert calls == ["discover:playlist-url", "perfil:playlist:3", "one:vid-2"]
    assert (artifacts.actor_transcripts_dir / "vid-2.json").exists()
    quality = json.loads(artifacts.speech2text_quality_path.read_text(encoding="utf-8"))
    assert quality["schema_version"] == "speech2text_quality.v2"
    assert quality["total_selected_videos"] == 1
    assert quality["total_segments"] == 1
    assert not artifacts.speech2text_quality_path.is_dir()
    assert not (artifacts.actor_transcripts_dir / "vid-1.json").exists()
    assert not ruta_audio("playlist", "vid-2", data_dir).exists()
    assert not ruta_audio("playlist", "su9nURIj9XQ", data_dir).exists()


def test_execution_runner_keep_cache_true_conserva_wavs(tmp_path: Path):
    data_dir = tmp_path / "data"
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
  max_videos: 1
  only_video_ids: [vid-2]
  keep_cache: true
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
project:
  data_dir: {data_dir}
""",
    )
    artifacts = resolve_artifacts(cfg, "run-001")
    plan = build_execution_plan(cfg, artifacts)
    calls: list[str] = []

    result = ExecutionRunner(_factories_with_granular_speech(calls), keep_cache=True).execute(plan)

    assert result.exit_code == 0
    assert ruta_audio("playlist", "vid-2", data_dir).exists()
    assert ruta_audio("playlist", "su9nURIj9XQ", data_dir).exists()


def test_manifest_incluye_fingerprint_de_configuracion(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
  max_videos: 1
  only_video_ids: [vid-2]
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, "run-001")
    plan = build_execution_plan(cfg, artifacts)
    calls: list[str] = []

    ExecutionRunner(_factories_with_granular_speech(calls)).execute(plan)

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert "config_fingerprint" in manifest
    fp = manifest["config_fingerprint"]
    assert "speech2text_quality_schema" in fp
    assert "whisper_model" in fp
    assert "umbral_match" in fp
    assert "umbral_ambiguo" in fp
    assert "only_video_ids" in fp


def test_runner_escribe_checkpoint_despues_de_cada_video(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-ckpt
  stages: [speech2text]
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, "run-ckpt")
    plan = build_execution_plan(cfg, artifacts)
    calls: list[str] = []

    ExecutionRunner(_factories_with_granular_speech(calls)).execute(plan)

    checkpoint_path = artifacts.run_dir / "speech2text" / "checkpoint.json"
    assert checkpoint_path.exists()
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["schema_version"] == "speech2text_checkpoint.v1"
    assert len(checkpoint["units"]) == 3
    assert all(u["status"] == "ok" for u in checkpoint["units"])
    assert [u["video_id"] for u in checkpoint["units"]] == ["vid-1", "vid-2", "vid-3"]


def test_runner_resume_salta_videos_con_transcript_ya_persistido(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-resume
  stages: [speech2text]
  resume: true
  overwrite: true
input:
  playlist_url: playlist-url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, "run-resume")
    artifacts.actor_transcripts_dir.mkdir(parents=True)
    from tono_politico.execution.artifacts import guardar_actor_transcript

    guardar_actor_transcript(
        _actor_transcript("vid-1"),
        artifacts.actor_transcripts_dir / "vid-1.json",
    )
    plan = build_execution_plan(cfg, artifacts)
    calls: list[str] = []

    result = ExecutionRunner(_factories_with_granular_speech(calls)).execute(plan)

    assert result.exit_code == 0
    assert "one:vid-1" not in calls
    assert "one:vid-2" in calls
    assert "one:vid-3" in calls
