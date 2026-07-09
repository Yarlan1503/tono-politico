# Requisitos — Control de ejecución `main.py` + `config.yaml`

> Estado: **R1-R8 implementados/verificados con TDD y gate local**.  
> Este documento define el contrato del control de ejecución stage-based.

---

## 1. Contexto y objetivo

### Contexto actual

- [x] El path preferido del proyecto ya es `speech2text → discursive_approach`.
- [x] `speech2text` produce `ActorTranscript` turn-level actor-only.
- [x] `discursive_approach` ejecuta `argument_shape → topics_cluster → topics_approach`.
- [x] El CLI actual permite `uv run python main.py --playlist URL --discursive`.
- [x] El CLI actual aún decide flujos con flags (`--discursive`, `--topico`, `--resume`).
- [x] `config/config.yaml` funciona como defaults de servicios, pero todavía no como contrato granular de ejecución.
- [x] El path nuevo reutiliza secciones legacy (`segmentacion`, `temas`, `diarizacion`) para configurar conceptos nuevos (`argument_shape`, `topics_cluster`, `topics_approach`).

### Problema a resolver

- [x] Hacer que `config/config.yaml` sea el **contrato canónico de cada corrida**.
- [x] Reducir `main.py` a un entrypoint delgado y testeable.
- [x] Permitir ejecutar granularmente:
  - [x] solo `speech2text`;
  - [x] solo `argument_shape` sobre transcripciones existentes;
  - [x] solo `topics_cluster` sobre argumentos existentes;
  - [x] solo `topics_approach` sobre temas existentes;
  - [x] el path completo `speech2text → argument_shape → topics_cluster → topics_approach`.
- [x] Permitir reusar artefactos caros sin repetir audio/pyannote/Whisper.
- [x] Mantener el path legacy vivo temporalmente, pero sin seguir inflándolo.

### Objetivo del rediseño

- [x] Introducir una capa `execution/` que convierta YAML tipado en un `ExecutionPlan`.
- [x] Ejecutar ese plan etapa por etapa con artefactos explícitos.
- [x] Persistir fronteras intermedias para auditoría y reanudación.
- [x] Mantener los services existentes intactos en la primera iteración.

---

## 2. Viabilidad y precedentes

### Investigación breve realizada

- [x] `argparse` oficial es suficiente para CLIs simples y delgados.
  - [x] Ventaja: estándar, sin dependencia nueva, fácil de testear.
  - [x] Limitación: no resuelve por sí solo configs YAML jerárquicos complejos.
  - [x] Conclusión: usarlo solo para `--config`, `--dry-run`, `--validate-config`, `--verbose`.
- [x] `jsonargparse` soporta YAML, dataclasses, nested config y `--print_config`.
  - [x] Ventaja: podría reducir boilerplate si el CLI crece mucho.
  - [x] Limitación: agrega dependencia y cambia la forma de declarar CLI/config.
  - [x] Conclusión: reservarlo como migración futura, no para R1.
- [x] Hydra/OmegaConf soporta composición jerárquica, overrides y multirun.
  - [x] Ventaja: fuerte para experimentación y matrices de configs.
  - [x] Limitación: introduce semánticas propias de working directory, overrides y composición.
  - [x] Conclusión: demasiado pesado para esta etapa.

### Decisión técnica

- [x] Usar **`argparse + PyYAML + dataclasses propias`**.
- [x] Validar YAML después del parse CLI, no mediante `argparse.type`.
- [x] Mantener contratos internos con dataclasses del proyecto, no `dict[str, Any]`.

---

## 3. Alcance y no-alcance

### Alcance R1-R6

- [x] Crear `src/tono_politico/execution/`.
- [x] Crear loader tipado para `RunConfig`.
- [x] Crear validaciones cruzadas del config.
- [x] Crear `ExecutionPlan` stage-based.
- [x] Crear runner de ejecución con services inyectables.
- [x] Refactorizar `main.py` para delegar en `execution/`.
- [x] Actualizar `config/config.yaml` al nuevo schema.
- [x] Actualizar documentación (`README.md`, `AGENTS.md`, `docs/configuracion.md`).
- [x] Mantener tests del path legacy verdes.

### Fuera de alcance inicial

- [x] No borrar `PipelineRunner` legacy.
- [x] No borrar flags legacy hasta decidir migración final.
- [x] No reescribir `SpeechToTextService`.
- [x] No reescribir `DiscursiveApproachService`.
- [x] No rediseñar taxonomía de tono.
- [x] No cambiar thresholds de diarización.
- [x] No cambiar BERTopic/HDBSCAN salvo exponer parámetros ya existentes.
- [x] No introducir Hydra/jsonargparse sin decisión explícita posterior.
- [x] No implementar R6 post-enfoques (`filtrado/salida`) dentro de este rediseño.

---

## 4. Arquitectura objetivo

### Estructura nueva

```text
src/tono_politico/
├── execution/
│   ├── __init__.py
│   ├── requisitos.md      # este documento
│   ├── config.py          # load_run_config(path) -> RunConfig
│   ├── validation.py      # validate_run_config(cfg) -> None
│   ├── models.py          # RunConfig/ExecutionPlan/StageSpec/ExecutionResult/...
│   ├── artifacts.py       # resolve_artifacts(cfg, run_id) -> ArtifactPaths
│   ├── plan.py            # build_execution_plan(cfg, artifacts) -> ExecutionPlan
│   └── runner.py          # ExecutionRunner.execute(plan) -> ExecutionResult
│
├── pipeline/
│   └── runner.py          # runner legacy, se mantiene temporalmente
│
main.py                    # entrypoint delgado
config/config.yaml          # contrato canónico de ejecución
```

### Flujo de alto nivel

```text
main.py
  parse_args(argv)
  load_run_config(config_path)
  validate_run_config(cfg)
  resolve_artifacts(cfg)
  build_execution_plan(cfg, artifacts)
  if dry_run: print plan and exit
  ExecutionRunner(...).execute(plan)
  print summary
  return exit_code
```

### Camino canónico nuevo

```text
speech2text
  audio_fetcher
  speaker_timestamps
  transcribe_speech
      → ActorTranscript[] persistidos

argument_shape
      ActorTranscript[] → Argumento[]

topics_cluster
      Argumento[] → ResultadoTemas

topics_approach
      ResultadoTemas → ResultadoEnfoques
```

---

## 5. Contrato de stages

### Stage IDs válidos

- [x] `speech2text`
- [x] `argument_shape`
- [x] `topics_cluster`
- [x] `topics_approach`

### Orden permitido

- [x] `speech2text` puede correr solo si hay `input.playlist_url`.
- [x] `argument_shape` puede correr si:
  - [x] `speech2text` corrió antes en el mismo plan; o
  - [x] existe `input.actor_transcripts_dir`; o
  - [x] existe `discursive_approach.input.actor_transcripts_dir`.
- [x] `topics_cluster` puede correr si:
  - [x] `argument_shape` corrió antes en el mismo plan; o
  - [x] existe `input.argumentos_path`.
- [x] `topics_approach` puede correr si:
  - [x] `topics_cluster` corrió antes en el mismo plan; o
  - [x] existe `input.temas_path`.

### Tabla de requires/produces

| Stage | Requires | Produces |
|---|---|---|
| `speech2text` | `playlist_url` | `actor_transcripts_dir` |
| `argument_shape` | `actor_transcripts_dir` | `argumentos_path` |
| `topics_cluster` | `argumentos_path` | `temas_path` |
| `topics_approach` | `temas_path` | `enfoques_path` |

### Reglas de salto/reanudación

- [x] Si `run.resume=true` y el artefacto de salida existe, la etapa se puede saltar.
- [x] Si `run.overwrite=true`, no se salta por existencia de artefacto.
- [x] Si `stage.force=true`, esa etapa se recomputa aunque exista su salida.
- [x] Si una etapa se salta, el manifest registra `skipped` y `skip_reason`.
- [x] Si una etapa falla, el manifest registra `failed` con tipo de error y mensaje.
- [x] Si `run.fail_fast=true`, el runner se detiene en el primer fallo.
- [x] Si `run.fail_fast=false`, el runner puede continuar solo si las dependencias de las etapas siguientes siguen satisfechas.

---

## 6. Contrato YAML v1

### Schema base

```yaml
schema_version: "tono-politico.run.v1"

run:
  id: null
  stages: [speech2text, argument_shape, topics_cluster, topics_approach]
  resume: true
  overwrite: false
  keep_cache: false
  fail_fast: false
  max_videos: null
  only_video_ids: []

input:
  playlist_url: null
  actor_transcripts_dir: null
  argumentos_path: null
  temas_path: null
  enfoques_path: null

output:
  base_dir: "output"
  run_dir: null
  persist_resolved_config: true
  persist_manifest: true

project:
  data_dir: "data"
  idioma: "es"
  random_state: 42

speech2text:
  enabled: true
  audio_fetcher:
    enabled: true
    force_download: false
    playlist_dir_template: "{playlist}"
    audio_dir_template: "videos-{playlist}"
  speaker_timestamps:
    enabled: true
    actor_objetivo: "Lilly Téllez"
    pipeline: "pyannote/speaker-diarization-community-1"
    fallback_pipeline: "pyannote-community/speaker-diarization-community-1"
    device: "auto"
    umbral_match: 0.5
    umbral_ambiguo: 0.7
    match_ambiguo: "descartar_como_otro_speaker"
    referencia_voz:
      origen: "misma_playlist"
      max_audios: 1
      video_id: "su9nURIj9XQ"
      url: "https://www.youtube.com/watch?v=su9nURIj9XQ&list=PLE9Zk7g9R__M&index=8"
      cache: "solo_ejecucion"
  transcribe_speech:
    enabled: true
    whisper_model: "large-v3-turbo"
    idioma: "es"
    word_timestamps: false
    force_retranscribe: false
    skip_existing_transcripts: true

discursive_approach:
  enabled: true
  input:
    source: "previous_stage" # previous_stage | actor_transcripts_dir
    actor_transcripts_dir: null
  argument_shape:
    enabled: true
    force: false
    spacy_model: "es_core_news_lg"
    embedding_model: "LiquidAI/LFM2.5-Embedding-350M"
    breakpoint_percentile: 95
    min_oraciones: 2
    max_oraciones: 8
    max_palabras: 150
  topics_cluster:
    enabled: true
    force: false
    embedding_model: "LiquidAI/LFM2.5-Embedding-350M"
    min_topic_size: 3
    n_neighbors: 10
    n_components: 5
    umap:
      metric: "cosine"
      random_state: 42
    hdbscan:
      min_samples: 1
      metric: "euclidean"
      cluster_selection_method: "eom"
    bertopic:
      language: "spanish"
      calculate_probabilities: false
      verbose: false
  topics_approach:
    enabled: true
    force: false
```

### Reglas de compatibilidad

- [x] El nuevo schema usa `schema_version: "tono-politico.run.v1"`.
- [x] El loader debe fallar si `schema_version` es desconocido.
- [x] La versión legacy de `Config` puede coexistir temporalmente en `src/tono_politico/config.py`.
- [x] Las secciones legacy (`ingesta`, `diarizacion`, `segmentacion`, `temas`, `filtrado`, `tono`, `salida`) no son la fuente canónica del path nuevo.
- [x] Si se mantienen flags legacy, deben mapear a la ruta antigua o a un bridge explícito, no mezclar silently configs nuevos/viejos.

---

## 7. Contratos de datos internos

### `RunConfig`

| Campo | Tipo | Por qué existe |
|---|---|---|
| `schema_version` | `str` | Detectar contrato YAML incompatible |
| `run` | `RunSettings` | Política de stages, reanudación y cache |
| `input` | `InputConfig` | Origen externo de datos iniciales |
| `output` | `OutputConfig` | Ubicación de artefactos durables |
| `project` | `ProjectExecutionConfig` | Defaults transversales: idioma, data dir, seed |
| `speech2text` | `SpeechToTextExecutionConfig` | Config granular del umbrella speech2text |
| `discursive_approach` | `DiscursiveApproachExecutionConfig` | Config granular del umbrella discursivo |

### `RunSettings`

| Campo | Tipo | Regla |
|---|---|---|
| `id` | `str | None` | `None` genera timestamp/run id automático |
| `stages` | `list[StageName]` | No vacío, orden validado |
| `resume` | `bool` | Permite saltar por artefactos existentes |
| `overwrite` | `bool` | Recalcula salidas aunque existan |
| `keep_cache` | `bool` | Conserva `.wav` y cache pesado si `true` |
| `fail_fast` | `bool` | Detiene en primer fallo si `true` |
| `max_videos` | `int | None` | Limita videos para smoke/debug |
| `only_video_ids` | `list[str]` | Subset explícito de videos |

### `StageSpec`

| Campo | Tipo | Regla |
|---|---|---|
| `name` | `StageName` | Uno de los IDs válidos |
| `enabled` | `bool` | Derivado de config de stage |
| `should_run` | `bool` | Derivado de resume/overwrite/force/artifacts |
| `requires` | `list[ArtifactKey]` | Inputs requeridos |
| `produces` | `list[ArtifactKey]` | Outputs esperados |
| `skip_reason` | `str | None` | Obligatorio si `should_run=false` |

### `ArtifactPaths`

| Campo | Tipo | Ruta esperada |
|---|---|---|
| `run_dir` | `Path` | `output/<run_id>/` salvo override |
| `manifest_path` | `Path` | `output/<run_id>/manifest.json` |
| `resolved_config_path` | `Path` | `output/<run_id>/resolved-config.yaml` |
| `actor_transcripts_dir` | `Path` | `output/<run_id>/speech2text/actor_transcripts/` |
| `argumentos_path` | `Path` | `output/<run_id>/discursive/argumentos.json` |
| `temas_path` | `Path` | `output/<run_id>/discursive/discursive-temas.json` |
| `enfoques_path` | `Path` | `output/<run_id>/discursive/discursive-enfoques.json` |

### `ExecutionResult`

| Campo | Tipo | Regla |
|---|---|---|
| `exit_code` | `int` | `0` éxito, `1` fallo ejecución, `2` config inválido |
| `plan` | `ExecutionPlan` | Plan resuelto usado por la corrida |
| `stage_results` | `list[StageResult]` | Resultado por etapa |
| `manifest_path` | `Path | None` | Ruta final si se persistió |

---

## 8. Política de artefactos y cache

### Artefactos durables

- [x] `output/<run_id>/manifest.json`
- [x] `output/<run_id>/resolved-config.yaml`
- [x] `output/<run_id>/speech2text/actor_transcripts/<video_id>.json`
- [x] `output/<run_id>/discursive/argumentos.json`
- [x] `output/<run_id>/discursive/discursive-temas.json`
- [x] `output/<run_id>/discursive/discursive-enfoques.json`

### Cache runtime

- [x] `data/<playlist>/videos-<playlist>/<video_id>.wav`
- [x] `.wav` se considera cache pesado, no artefacto durable.
- [x] Si `run.keep_cache=false`, `.wav` se limpia al terminar la unidad/corrida cuando sea seguro.
- [x] Si `run.keep_cache=true`, `.wav` se conserva para debug.

### Reglas de persistencia

- [x] `ActorTranscript` se persiste siempre que `speech2text` produzca salida válida.
- [x] `Argumento[]` se persiste siempre que `argument_shape` complete.
- [x] `ResultadoTemas` se persiste siempre que `topics_cluster` complete.
- [x] `ResultadoEnfoques` se persiste siempre que `topics_approach` complete.
- [x] El manifest registra también artefactos saltados por resume.

---

## 9. Comportamiento esperado del CLI

### CLI nuevo

```bash
uv run python main.py --config config/config.yaml
uv run python main.py --config config/config.yaml --dry-run
uv run python main.py --config config/config.yaml --validate-config
uv run python main.py --config config/config.yaml --verbose
```

### Requisitos de `main.py`

- [x] `main(argv) -> int`.
- [x] No ejecutar `sys.exit()` dentro de `main()`.
- [x] Único `sys.exit` permitido: `raise SystemExit(main())` al final del archivo.
- [x] No importar modelos pesados al parsear CLI.
- [x] No construir services si `--validate-config`.
- [x] No ejecutar stages si `--dry-run`.
- [x] Imprimir un resumen corto de plan/resultados.
- [x] Delegar toda lógica de ejecución a `ExecutionRunner`.

### Compatibilidad temporal con flags actuales

- [x] Mantener `--playlist`, `--discursive`, `--topico`, `--tema`, `--resume` solo si se decide migración suave.
- [x] Si se mantienen, documentarlos como legacy/deprecated.
- [x] No mezclar flags legacy con `run.stages` sin una regla explícita.
- [x] Preferir que el path canónico documentado sea `--config`.

---

## 10. Validaciones obligatorias

### Validaciones de estructura YAML

- [x] El YAML raíz debe ser mapping.
- [x] Cada sección conocida debe ser mapping.
- [x] `schema_version` debe existir.
- [x] `run.stages` debe ser lista no vacía.
- [x] Todo stage en `run.stages` debe ser válido.
- [x] `only_video_ids` debe ser lista de strings.
- [x] `max_videos` debe ser `null` o entero positivo.

### Validaciones de dependencias

- [x] `speech2text` requiere `input.playlist_url`.
- [x] `argument_shape` requiere transcripts por etapa previa o directorio configurado.
- [x] `topics_cluster` requiere argumentos por etapa previa o path configurado.
- [x] `topics_approach` requiere temas por etapa previa o path configurado.
- [x] Si una etapa está en `run.stages` pero su sección `enabled=false`, el config es inválido salvo que se defina explícitamente como skip.

### Validaciones de path

- [x] `output.base_dir` se normaliza a `Path`.
- [x] `project.data_dir` se normaliza a `Path`.
- [x] Paths relativos se interpretan desde la raíz del repo/proceso actual.
- [x] `output.run_dir` si existe reemplaza `output.base_dir / run_id`.

### Validaciones de parámetros

- [x] `umbral_match < umbral_ambiguo`.
- [x] `breakpoint_percentile` entre `0` y `100`.
- [x] `min_oraciones >= 1`.
- [x] `max_oraciones >= min_oraciones`.
- [x] `max_palabras >= 1`.
- [x] `min_topic_size >= 1`.
- [x] `n_neighbors >= 2`.
- [x] `n_components >= 2`.

---

## 11. Mapa de módulos y APIs públicas

### `execution/config.py`

- [x] `load_run_config(path: Path) -> RunConfig`
- [x] `RunConfig.from_mapping(data: Mapping[str, Any]) -> RunConfig`
- [x] No construir services aquí.
- [x] No tocar filesystem salvo leer YAML.

### `execution/validation.py`

- [x] `validate_run_config(cfg: RunConfig) -> None`
- [x] `ConfigValidationError(ValueError)` para errores de usuario.
- [x] Mensajes de error deben nombrar la ruta YAML (`run.stages`, `input.playlist_url`, etc.).

### `execution/artifacts.py`

- [x] `resolve_artifacts(cfg: RunConfig, run_id: str) -> ArtifactPaths`
- [x] `artifact_exists(paths: ArtifactPaths, key: ArtifactKey) -> bool`
- [x] Centralizar convenciones de rutas de artefactos durables.

### `execution/plan.py`

- [x] `build_execution_plan(cfg: RunConfig, artifacts: ArtifactPaths) -> ExecutionPlan`
- [x] Expandir `run.stages` a `StageSpec[]`.
- [x] Calcular `should_run` y `skip_reason`.
- [x] No construir services.
- [x] No ejecutar modelos.

### `execution/runner.py`

- [x] `ExecutionRunner(factories: ExecutionFactories, keep_cache: bool = False)`.
- [x] `execute(plan: ExecutionPlan) -> ExecutionResult`.
- [x] Usar services inyectables/fakes en tests.
- [x] Registrar `StageResult` por etapa.
- [x] Persistir manifest/resolved config si el plan lo pide.

### `execution/models.py`

- [x] Definir dataclasses tipadas.
- [x] Definir `StageName = Literal[...]`.
- [x] Definir `ArtifactKey = Literal[...]`.
- [x] Evitar `dict[str, Any]` en fronteras públicas salvo carga YAML inicial.

---

## 12. Plan de desarrollo TDD

### R0 — Requisitos

- [x] Crear `src/tono_politico/execution/requisitos.md`.
- [x] Revisar este documento con el usuario.
- [x] Cerrar decisiones pendientes antes de código.

### R1 — Config tipado

- [x] Crear `tests/test_run_config.py`.
- [x] RED: carga YAML mínimo y falla porque `load_run_config` no existe.
- [x] Implementar `execution/config.py` mínimo.
- [x] GREEN: tests de carga/defaults pasan.
- [x] Tests obligatorios:
  - [x] YAML mínimo válido.
  - [x] `schema_version` inválido falla.
  - [x] `run.stages` vacío falla.
  - [x] secciones no mapping fallan.
  - [x] paths se normalizan como `Path`.

### R2 — Validación cruzada

- [x] Crear `tests/test_run_config_validation.py`.
- [x] RED: config inválido no falla todavía.
- [x] Implementar `execution/validation.py`.
- [x] GREEN: errores claros por dependencia faltante.
- [x] Tests obligatorios:
  - [x] `speech2text` sin `playlist_url` falla.
  - [x] `argument_shape` sin transcripts falla.
  - [x] `topics_cluster` sin argumentos falla.
  - [x] `topics_approach` sin temas falla.
  - [x] thresholds inválidos fallan.

### R3 — Artifacts + ExecutionPlan

- [x] Crear `tests/test_execution_plan.py`.
- [x] RED: `build_execution_plan` no existe.
- [x] Implementar `execution/artifacts.py`, `execution/models.py`, `execution/plan.py`.
- [x] GREEN: plan resuelve etapas y rutas.
- [x] Tests obligatorios:
  - [x] orden de stages se conserva.
  - [x] rutas bajo `output/<run_id>/`.
  - [x] `resume=true` salta si artefacto existe.
  - [x] `force=true` recomputa aunque exista.
  - [x] `overwrite=true` recomputa aunque exista.

### R4 — Runner con fakes

- [x] Crear `tests/test_execution_runner.py`.
- [x] RED: runner no existe.
- [x] Implementar `execution/runner.py` con factories fakeables.
- [x] GREEN: orquestación sin modelos reales.
- [x] Tests obligatorios:
  - [x] ejecuta etapas en orden.
  - [x] pasa output de una etapa a la siguiente.
  - [x] salta etapas con `should_run=false`.
  - [x] registra fallo de etapa.
  - [x] respeta `fail_fast`.

### R5 — `main.py` delgado

- [x] Crear/actualizar `tests/test_main_cli.py`.
- [x] RED: `main(["--config", path])` no delega en ejecución nueva.
- [x] Refactorizar `main.py`.
- [x] GREEN: CLI usa runner fake sin cargar modelos.
- [x] Tests obligatorios:
  - [x] `--validate-config` no ejecuta stages.
  - [x] `--dry-run` imprime plan y no ejecuta.
  - [x] config inválido retorna código controlado.
  - [x] `main(argv)` retorna `int`.

### R6 — Integración real de services

- [x] Conectar `speech2text` a `SpeechToTextService` existente.
- [x] Conectar `argument_shape` a `ArgumentShapeService` existente.
- [x] Conectar `topics_cluster` a `TopicsClusterService` existente.
- [x] Conectar `topics_approach` a `TopicsApproachService`/`DiscursiveApproachService` existente.
- [x] Persistir artefactos en las rutas nuevas.
- [x] Tests de integración con fakes/adapters, sin modelos pesados.

### R7 — Config y docs

- [x] Actualizar `config/config.yaml` al schema v1.
- [x] Actualizar `docs/configuracion.md`.
- [x] Actualizar `README.md`.
- [x] Actualizar `AGENTS.md`.
- [x] Actualizar docs de `speech2text` y `discursive_approach` si cambian comandos.
- [x] Buscar referencias stale a `--discursive` como path canónico.

### R8 — Gate

- [x] `uv run ruff check src/ tests/ main.py`.
- [x] `uv run ruff format --check src/ tests/ main.py`.
- [x] `uv run ty check`.
- [x] `uv run pytest tests/ -v -m "not slow"`.
- [x] Alternativamente: `bash check.sh`.

Verificación local de esta revisión: `uv run ruff check src/ tests/ main.py`, `uv run ruff format --check src/ tests/ main.py`, `uv run ty check`, `uv run pytest tests/ -q -m "not slow"` → **461 passed, 5 deselected**.

---

## 13. Decisiones cerradas

| # | Decisión | Elegido | Rationale |
|---|---|---|---|
| D1 | Framework CLI/config | `argparse + PyYAML + dataclasses` | Suficiente, sin dependencia nueva, testeable |
| D2 | Ubicación de lógica nueva | `src/tono_politico/execution/` | Evita inflar `pipeline/runner.py` legacy |
| D3 | Selección de ejecución | `run.stages` explícito | Más granular que `mode` |
| D4 | Reanudación | `resume` + `overwrite` + `stage.force` | Permite iterar etapas caras sin repetir todo |
| D5 | Fronteras persistidas | Todas las fronteras principales | Auditoría, debugging y reuse |
| D6 | Legacy path | Mantener temporalmente | No bloquear R6 filtrado/salida ni romper tests |
| D7 | Nombres de config | Nuevos nombres canónicos | Evita mapear `segmentacion/temas` a conceptos nuevos |
| D8 | Tests | TDD estricto | Regla del proyecto y reduce riesgo del refactor |

---

## 14. Checklist de aceptación

- [x] Se puede ejecutar el path completo con `uv run python main.py --config config/config.yaml`.
- [x] Se puede validar config sin correr modelos.
- [x] Se puede hacer dry-run e imprimir stages/rutas.
- [x] Se puede ejecutar solo `speech2text`.
- [x] Se puede ejecutar solo `argument_shape` desde transcripts persistidos.
- [x] Se puede ejecutar solo `topics_cluster` desde `argumentos.json`.
- [x] Se puede ejecutar solo `topics_approach` desde `discursive-temas.json`.
- [x] `ActorTranscript` se persiste como artefacto durable.
- [x] `argumentos.json` se persiste como artefacto durable.
- [x] `discursive-temas.json` se persiste como artefacto durable.
- [x] `discursive-enfoques.json` se persiste como artefacto durable.
- [x] El manifest registra etapas ejecutadas, saltadas y fallidas.
- [x] Tests unitarios no cargan modelos pesados ni llaman red.
- [x] Path legacy sigue verde.
- [x] README/AGENTS/docs/config están sincronizados.

---

## 15. Decisiones aplicadas

Estas preguntas quedaron cerradas durante R1-R8 y ya están reflejadas en código/config/docs:

- [x] Reemplazamos `config/config.yaml` directamente; no se creó `config/run-config.yaml`.
- [x] Mantenemos flags actuales como compatibilidad legacy/deprecated durante una iteración.
- [x] No se agregan aliases tipo `run.mode`; el contrato se queda en `run.stages` explícito.
- [x] `run_dir` queda como `output/<run_id>/` por defecto.
- [x] `topics_approach` consume `temas_path`; no requiere `argumentos.json` explícito en esta iteración.
