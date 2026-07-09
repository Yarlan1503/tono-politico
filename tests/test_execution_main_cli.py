"""Tests del nuevo path CLI --config basado en execution/."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from tono_politico.execution.models import ExecutionResult, StageResult


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("tono_politico_main_execution", Path("main.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_main_import_no_carga_modulos_pesados():
    code = """
import importlib.util
import json
import sys
from pathlib import Path

heavy_modules = [
    "bertopic",
    "numpy",
    "pyannote",
    "sentence_transformers",
    "spacy",
    "torch",
    "transformers",
    "whisper",
]
spec = importlib.util.spec_from_file_location("tono_politico_main_probe", Path("main.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps([module_name for module_name in heavy_modules if module_name in sys.modules]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    imported = json.loads(result.stdout)
    assert imported == []


def _execution_config(path: Path, *, playlist_url: str = "playlist-url") -> Path:
    config_path = path / "run-config.yaml"
    config_path.write_text(
        f"""
schema_version: tono-politico.run.v1
run:
  id: run-001
  stages: [speech2text]
input:
  playlist_url: {playlist_url}
output:
  base_dir: {path / "output"}
""",
        encoding="utf-8",
    )
    return config_path


class FakeExecutionRunner:
    instances: list[FakeExecutionRunner] = []

    def __init__(self, factories: Any, keep_cache: bool = False):
        self.factories = factories
        self.keep_cache = keep_cache
        self.executed_plans: list[Any] = []
        FakeExecutionRunner.instances.append(self)

    def execute(self, plan: Any) -> ExecutionResult:
        self.executed_plans.append(plan)
        return ExecutionResult(
            exit_code=0,
            plan=plan,
            stage_results=[StageResult(stage="speech2text", status="ok")],
            manifest_path=plan.artifacts.manifest_path,
        )


def test_main_config_nuevo_delega_en_execution_runner(monkeypatch, tmp_path: Path):
    cli = _load_cli_module()
    FakeExecutionRunner.instances = []
    monkeypatch.setattr(cli, "ExecutionRunner", FakeExecutionRunner, raising=False)

    exit_code = cli.main(["--config", str(_execution_config(tmp_path))])

    assert exit_code == 0
    assert len(FakeExecutionRunner.instances) == 1
    runner = FakeExecutionRunner.instances[0]
    assert runner.keep_cache is False
    assert len(runner.executed_plans) == 1
    assert runner.executed_plans[0].config.run.stages == ["speech2text"]


def test_main_validate_config_no_instancia_runner(monkeypatch, tmp_path: Path):
    cli = _load_cli_module()
    FakeExecutionRunner.instances = []
    monkeypatch.setattr(cli, "ExecutionRunner", FakeExecutionRunner, raising=False)

    exit_code = cli.main(["--config", str(_execution_config(tmp_path)), "--validate-config"])

    assert exit_code == 0
    assert FakeExecutionRunner.instances == []


def test_main_dry_run_imprime_plan_y_no_instancia_runner(monkeypatch, tmp_path: Path, capsys):
    cli = _load_cli_module()
    FakeExecutionRunner.instances = []
    monkeypatch.setattr(cli, "ExecutionRunner", FakeExecutionRunner, raising=False)

    exit_code = cli.main(["--config", str(_execution_config(tmp_path)), "--dry-run"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Execution plan" in captured.out
    assert "speech2text" in captured.out
    assert FakeExecutionRunner.instances == []


def test_main_config_invalido_retorna_2(monkeypatch, tmp_path: Path, capsys):
    cli = _load_cli_module()
    FakeExecutionRunner.instances = []
    monkeypatch.setattr(cli, "ExecutionRunner", FakeExecutionRunner, raising=False)
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
schema_version: tono-politico.run.v1
run:
  stages: [speech2text]
""",
        encoding="utf-8",
    )

    exit_code = cli.main(["--config", str(config_path), "--validate-config"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "input.playlist_url" in captured.err
    assert FakeExecutionRunner.instances == []
