"""Control de ejecución stage-based para tono-politico."""

from .artifacts import artifact_exists, resolve_artifacts
from .config import is_run_config_file, load_run_config
from .models import (
    ArtifactPaths,
    ExecutionPlan,
    ExecutionResult,
    RunConfig,
    StageResult,
    StageSpec,
)
from .plan import build_execution_plan
from .runner import ExecutionFactories, ExecutionRunner
from .validation import ConfigValidationError, validate_run_config

__all__ = [
    "ArtifactPaths",
    "ConfigValidationError",
    "ExecutionFactories",
    "ExecutionPlan",
    "ExecutionResult",
    "ExecutionRunner",
    "RunConfig",
    "StageResult",
    "StageSpec",
    "artifact_exists",
    "build_execution_plan",
    "is_run_config_file",
    "load_run_config",
    "resolve_artifacts",
    "validate_run_config",
]
