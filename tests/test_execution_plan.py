"""Tests de artifact paths y ExecutionPlan."""

from __future__ import annotations

from pathlib import Path

from tono_politico.execution.artifacts import artifact_exists, resolve_artifacts
from tono_politico.execution.config import load_run_config
from tono_politico.execution.plan import build_execution_plan


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "run-config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _load(tmp_path: Path, content: str):
    return load_run_config(_write_config(tmp_path, content))


def test_build_execution_plan_conserva_orden_y_resuelve_rutas(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text, argument_shape, topics_cluster, topics_approach]
input:
  playlist_url: url
output:
  base_dir: {tmp_path / "output"}
""",
    )

    artifacts = resolve_artifacts(cfg, run_id="run-001")
    plan = build_execution_plan(cfg, artifacts)

    assert artifacts.run_dir == tmp_path / "output" / "run-001"
    assert artifacts.manifest_path == artifacts.run_dir / "manifest.json"
    expected_transcripts_dir = artifacts.run_dir / "speech2text" / "actor_transcripts"
    assert artifacts.actor_transcripts_dir == expected_transcripts_dir
    assert artifacts.argumentos_path == artifacts.run_dir / "discursive" / "argumentos.json"
    assert [stage.name for stage in plan.stages] == [
        "speech2text",
        "argument_shape",
        "topics_cluster",
        "topics_approach",
    ]
    assert all(stage.should_run for stage in plan.stages)


def test_artifact_exists_rechaza_directorio_de_transcripts_vacio(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
input:
  playlist_url: url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, run_id="run-001")
    artifacts.actor_transcripts_dir.mkdir(parents=True)

    assert artifact_exists(artifacts, "actor_transcripts_dir") is False


def test_artifact_exists_detecta_directorio_con_transcript_valido(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
input:
  playlist_url: url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, run_id="run-001")
    artifacts.actor_transcripts_dir.mkdir(parents=True)
    (artifacts.actor_transcripts_dir / "vid-1.json").write_text(
        '{"schema_version":"actor_transcript.v1","video_id":"vid-1","actor":"Actor",'
        '"scope":"actor_only","asr":{"provider":"whisper","model":"tiny",'
        '"language":"es"},"segments":[]}',
        encoding="utf-8",
    )

    assert artifact_exists(artifacts, "actor_transcripts_dir") is True


def test_artifact_exists_rechaza_transcript_con_schema_incompatible(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
input:
  playlist_url: url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, run_id="run-001")
    artifacts.actor_transcripts_dir.mkdir(parents=True)
    (artifacts.actor_transcripts_dir / "vid-1.json").write_text(
        '{"schema_version":"actor_transcript.v0"}',
        encoding="utf-8",
    )

    assert artifact_exists(artifacts, "actor_transcripts_dir") is False


def test_build_execution_plan_salta_stage_si_resume_y_artefacto_existe(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
  resume: true
  overwrite: false
input:
  playlist_url: url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, run_id="run-001")
    artifacts.actor_transcripts_dir.mkdir(parents=True)
    (artifacts.actor_transcripts_dir / "vid-1.json").write_text(
        '{"schema_version":"actor_transcript.v1","video_id":"vid-1","actor":"Actor",'
        '"scope":"actor_only","asr":{"provider":"whisper","model":"tiny",'
        '"language":"es"},"segments":[]}',
        encoding="utf-8",
    )

    plan = build_execution_plan(cfg, artifacts)

    assert plan.stages[0].should_run is False
    assert "existe" in (plan.stages[0].skip_reason or "")


def test_build_execution_plan_force_recalcula_aunque_artefacto_exista(tmp_path: Path):
    transcripts_dir = tmp_path / "transcripts"
    transcripts_dir.mkdir()
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [argument_shape]
  resume: true
  overwrite: false
input:
  actor_transcripts_dir: {transcripts_dir}
output:
  base_dir: {tmp_path / "output"}
discursive_approach:
  argument_shape:
    force: true
""",
    )
    artifacts = resolve_artifacts(cfg, run_id="run-001")
    artifacts.argumentos_path.parent.mkdir(parents=True)
    artifacts.argumentos_path.write_text("[]", encoding="utf-8")

    plan = build_execution_plan(cfg, artifacts)

    assert plan.stages[0].name == "argument_shape"
    assert plan.stages[0].should_run is True
    assert plan.stages[0].skip_reason is None


def test_build_execution_plan_overwrite_recalcula_aunque_artefacto_exista(tmp_path: Path):
    cfg = _load(
        tmp_path,
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
  resume: true
  overwrite: true
input:
  playlist_url: url
output:
  base_dir: {tmp_path / "output"}
""",
    )
    artifacts = resolve_artifacts(cfg, run_id="run-001")
    artifacts.actor_transcripts_dir.mkdir(parents=True)
    (artifacts.actor_transcripts_dir / "vid-1.json").write_text(
        '{"schema_version":"actor_transcript.v1","video_id":"vid-1","actor":"Actor",'
        '"scope":"actor_only","asr":{"provider":"whisper","model":"tiny",'
        '"language":"es"},"segments":[]}',
        encoding="utf-8",
    )

    plan = build_execution_plan(cfg, artifacts)

    assert plan.stages[0].should_run is True
