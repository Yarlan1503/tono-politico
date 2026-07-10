"""Construcción de ExecutionPlan desde RunConfig."""

from __future__ import annotations

from .artifacts import artifact_exists
from .models import ArtifactKey, ArtifactPaths, ExecutionPlan, RunConfig, StageName, StageSpec

_STAGE_IO: dict[StageName, tuple[list[ArtifactKey], list[ArtifactKey]]] = {
    "speech2text": (["playlist_url"], ["actor_transcripts_dir"]),
    "argument_shape": (["actor_transcripts_dir"], ["argumentos_path"]),
    "topics_cluster": (["argumentos_path"], ["temas_path"]),
    "topics_approach": (["temas_path"], ["enfoques_path"]),
}


def build_execution_plan(cfg: RunConfig, artifacts: ArtifactPaths) -> ExecutionPlan:
    """Expande ``run.stages`` a specs ejecutables con política resume/force."""
    run_id = cfg.run.id or artifacts.run_dir.name
    stages = [_stage_spec(cfg, artifacts, stage) for stage in cfg.run.stages]
    return ExecutionPlan(run_id=run_id, config=cfg, artifacts=artifacts, stages=stages)


def _stage_spec(cfg: RunConfig, artifacts: ArtifactPaths, stage: StageName) -> StageSpec:
    requires, produces = _STAGE_IO[stage]
    enabled = _stage_enabled(cfg, stage)
    force = _stage_force(cfg, stage)
    output_exists = all(artifact_exists(artifacts, key) for key in produces)
    should_run = enabled
    skip_reason = None

    if enabled and cfg.run.resume and output_exists and not cfg.run.overwrite and not force:
        should_run = False
        skip_reason = f"artefacto de salida ya existe: {', '.join(produces)}"
    elif not enabled:
        should_run = False
        skip_reason = "stage deshabilitado"

    return StageSpec(
        name=stage,
        enabled=enabled,
        should_run=should_run,
        requires=list(requires),
        produces=list(produces),
        skip_reason=skip_reason,
    )


def _stage_enabled(cfg: RunConfig, stage: StageName) -> bool:
    if stage == "speech2text":
        return cfg.speech2text.enabled
    if stage == "argument_shape":
        return cfg.discursive_approach.enabled and cfg.discursive_approach.argument_shape.enabled
    if stage == "topics_cluster":
        return cfg.discursive_approach.enabled and cfg.discursive_approach.topics_cluster.enabled
    return cfg.discursive_approach.enabled and cfg.discursive_approach.topics_approach.enabled


def _stage_force(cfg: RunConfig, stage: StageName) -> bool:
    if stage == "speech2text":
        return False
    if stage == "argument_shape":
        return cfg.discursive_approach.argument_shape.force
    if stage == "topics_cluster":
        return cfg.discursive_approach.topics_cluster.force
    return cfg.discursive_approach.topics_approach.force
