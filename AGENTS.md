# AGENTS.md — Tono Político

## Proyecto

NLP para análisis de tono político de actores políticos mexicanos desde transcripciones de YouTube.

**Repo:** `~/Documentos/Proyectos/tono-politico`
**Python:** 3.11 · **Gestor:** uv · **Linter:** ruff · **Type checker:** ty

## Reglas de trabajo

- **TDD estricto:** RED (test fallando) → GREEN (mínima implementación) → REFACTOR.
- **Build method-by-method:** diseñar → aprobar → implementar → testear → preguntar antes de avanzar al siguiente método/componente.
- **Justificar antes de implementar:** thresholds, modelos, algoritmos y frameworks deben tener rationale técnico explícito.
- **uv obligatorio:** `uv venv`, `uv pip install`, `uv run`, `uv lock`. Nunca `pip`/`venv` nativo.
- **ruff + ty + tests limpios antes de cerrar.**
- **Sin `PYTHONPATH=src`:** el paquete debe estar instalado editable con `uv pip install -e ".[dev]"`.
- **No scaffold masivo:** no implementar componentes futuros sin diseño/aprobación.

## Comandos

```bash
# Tests del módulo speech2text
uv run pytest tests/speech2text/ -q

# Suite general
uv run pytest tests/ -v -m "not slow"

# Lint
uv run ruff check src/ tests/ main.py

# Format check
uv run ruff format --check src/ tests/ main.py

# Type check
uv run ty check

# Verificación completa
uv run ruff check src/ tests/ main.py && uv run ty check && uv run pytest tests/ -v -m "not slow"

# Pipeline CLI
uv run python main.py --config config/config.yaml --dry-run        # plan stage-based
uv run python main.py --config config/config.yaml --validate-config
uv run python main.py --config config/config.yaml                  # speech2text → enfoques
```

Verificación local: **`191 passed, 1 skipped`** con las dos suites legacy de `discursive_approach` excluidas por imports retirados (`-m "not slow"`). Ruff y formato pasan; `ty` conserva seis diagnósticos de esos imports legacy.

Gate canónico: `bash check.sh` (ruff + ty + pytest). Con slow: `RUN_SLOW=1 bash check.sh`.

Limpieza: `bash clean.sh` (output/ + data/ + caches Python, con confirmación). Filtros: `--output`, `--data`, `--caches`. Dry-run: `--dry-run`. Sin confirmación: `-y`.

## Configuración

- Config canónica: `config/config.yaml` (`schema_version: tono-politico.run.v1`).
- `main.py --config config/config.yaml` carga `RunConfig`, valida dependencias, construye `ExecutionPlan` y ejecuta stages granulares con artefactos durables.
- Stages canónicos: `speech2text`, `argument_shape`, `topics_cluster`, `topics_approach`.
- Artefactos: `output/<run_id>/speech2text/actor_transcripts/`, `speech2text/checkpoint.json`, `speech2text/quality.json`, `discursive/argumentos.json`, `discursive/discursive-temas.json`, `discursive/discursive-enfoques.json`, `manifest.json`, `resolved-config.yaml`.
- `run.max_videos` y `run.only_video_ids` filtran la etapa `speech2text` después del discover; `run.keep_cache=false` borra `.wav` runtime al terminar cada unidad/ref, `true` los conserva para debug.
- **speech2text:** `audio_fetcher` + `speaker_timestamps` + `transcribe_speech` → `ActorTranscript` turn-level con `TranscriptSource` opcional y provenance en control plane. `speaker_timestamps` fusiona unidades consecutivas del mismo speaker y usa `[0, duración]` cuando el único speaker validado es el actor. Tests de dominio: `tests/speech2text/` (`123 passed`); tests cross-cutting de ejecución en `tests/` (`18 passed`). Documentación: `docs/module-speech2text.md` y `docs/component_*.md`. Smoke Play-PoliTest: 3/3 vídeos cortos, 34 segmentos, provenance verificada.
- **discursive_approach:** `argument_shape` → `topics_cluster` → `topics_approach` está temporalmente bloqueado; sus dependencias legacy fueron retiradas y se reconstruirá después.
- ASR en speech2text: Whisper `large-v3-turbo` **por unidad temporal del actor** con `word_timestamps=False`; un speaker único validado usa el audio completo.
- Diarización (stack interno): Community-1, `device=auto`, thresholds 0.5 / 0.7; ambiguo → descartar.
- Si cambia un default en código, actualizar también `config/config.yaml`, `README.md`, `AGENTS.md` y `docs/`.

## Estructura

```text
src/tono_politico/
├── execution/             # Control plane stage-based ✅
│   ├── config.py          # load_run_config(path) -> RunConfig
│   ├── validation.py      # validate_run_config(cfg)
│   ├── artifacts.py       # resolve_artifacts + serialización Argumento[]
│   ├── plan.py            # build_execution_plan
│   ├── runner.py          # ExecutionRunner con factories fakeables
│   └── models.py          # RunConfig, StageSpec, ArtifactPaths, ExecutionResult
├── speech2text/           # autocontenido: playlist/audio → ActorTranscript actor-only
│   ├── service.py         # SpeechToTextService — discover / ensure_perfil / procesar_one
│   ├── models.py          # ActorTranscript, TranscriptSource y compatibilidad de DTOs
│   ├── requisitos.md      # checklist + viabilidad + decisiones
│   ├── audio_fetcher/     # playlist + descarga .wav (sin Whisper)
│   │   ├── models.py      # VideoMeta, AudioVideo, DownloadResult, PlaylistInfo
│   │   ├── cache.py       # ruta_dir_videos, ruta_audio (solo .wav)
│   │   ├── playlist.py    # obtener_info_playlist → (PlaylistInfo, list[VideoMeta])
│   │   ├── audio.py       # verificar_cache_videos, descargar_audio_result
│   │   └── service.py     # AudioFetcherService
│   ├── speaker_timestamps/ # pyannote + match + fusión → TurnoOrador[]
│   │   ├── models.py      # DTOs canónicos de diarización/matching
│   │   ├── service.py     # SpeakerTimestampsService
│   │   ├── matching.py    # identificar_actor, clasificar_speaker
│   │   └── perfil_voz.py  # construir_perfil_desde_output
│   └── transcribe_speech/ # Whisper por unidad → ActorTranscript (+ fecha)
│       ├── models.py      # DTOs del transcriptor
│       ├── actor_clip.py  # recorte y persistencia de unidades actor-only
│       ├── transcription_clip.py # transcripción Whisper por unidad/clip
│       └── service.py     # TranscribeSpeechService
├── discursive_approach/   # ActorTranscript → temas + enfoques ⚠️ bloqueado
│   ├── service.py         # DiscursiveApproachService
│   ├── requisitos.md      # decisiones 1–9 + checklist
│   ├── argument_shape/    # Oracion/Argumento; spaCy + breakpoints LFM
│   ├── topics_cluster/    # BERTopic sobre Argumento[]
│   └── topics_approach/   # pendiente de reconstrucción
main.py                    # CLI entry point — delega a ExecutionRunner
```

## Arquitectura

### ExecutionRunner (control plane)

```python
class ExecutionRunner:
    def __init__(self, factories: ExecutionFactories, keep_cache: bool = False): ...
    def execute(self, plan: ExecutionPlan) -> ExecutionResult: ...
```

- Ejecuta stages en orden, pasando artefactos vía contexto en memoria.
- `fail_fast=true` detiene en el primer fallo; `fail_fast=false` continúa si las dependencias siguientes están satisfechas.
- `resume=true` salta etapas con artefacto existente; `overwrite=true` recomputa. `force` queda reservado a stages discursivos que lo declaran.
- `keep_cache=false` borra `.wav` por video/ref al terminar.

### Services principales

```python
class SpeechToTextService:
    def discover(self, url) -> tuple[PlaylistInfo, list[VideoMeta]]: ...
    def ensure_perfil(self, playlist: PlaylistInfo | str, metas) -> bool: ...
    def procesar_one(self, video: VideoMeta, playlist: PlaylistInfo | str) -> ActorTranscript | None: ...

class DiscursiveApproachService:
    def shape_corpus(self, transcripts) -> list[Argumento]: ...
    def cluster(self, argumentos) -> ResultadoTemas: ...
    def approaches(self, resultado) -> ResultadoEnfoques: ...
    def procesar(self, transcripts) -> ResultadoEnfoques: ...  # shape → cluster → approaches

```

- **Config encapsulada:** todos los hiperparámetros viven en el constructor del service.
- **Lazy loading:** modelos pesados se cargan en el primer `.procesar()`, no al importar. El CLI no importa módulos pesados al parsear args.

## Estado por componente

### speech2text — ✅ autocontenido

Las suites específicas y de control plane están en verde. Smoke real Play-PoliTest: **3/3** vídeos cortos, **34** segmentos, 0 errores y provenance presente en los cuatro artefactos.

Doc: `docs/module-speech2text.md` · Componentes: `docs/component_*.md` · Requisitos: `src/tono_politico/speech2text/requisitos.md`

| Clase / API | Módulo | Responsabilidad |
|---|---|---|
| `SpeechToTextService` | `speech2text/service.py` | Orquesta discover + perfil + procesar_one |
| `AudioFetcherService` | `audio_fetcher/service.py` | `discover` / `fetch_one` → `AudioVideo` |
| `SpeakerTimestampsService` | `speaker_timestamps/service.py` | pyannote exclusive + fusión + match → unidades actor |
| `TranscribeSpeechService` | `transcribe_speech/service.py` | Whisper por unidad → `ActorTranscript` |

### discursive_approach — ⚠️ bloqueado temporalmente

Sus módulos propios permanecen en el árbol, pero la ruta no importa mientras se reconstruyen los contratos retirados.

Doc: `docs/componente-discursive-approach.md` · Requisitos: `src/tono_politico/discursive_approach/requisitos.md`

| Clase / API | Módulo | Responsabilidad |
|---|---|---|
| `DiscursiveApproachService` | `discursive_approach/service.py` | Orquesta shape → cluster → approaches (bloqueado) |
| `ArgumentShapeService` | `argument_shape/service.py` | `ActorTranscript` → `Argumento[]` (1 audio) |
| `TopicsClusterService` | `topics_cluster/service.py` | `Argumento[]` → `ResultadoTemas` (BERTopic) |
| `TopicsApproachService` | `topics_approach/service.py` | Tono por tema + firmas → `ResultadoEnfoques` |

### speaker_timestamps — diarización del actor (dentro de speech2text)

Diarización pyannote + identificación del actor. **No es un subpaquete separado** — la lógica vive dentro de `speaker_timestamps/`:

| Función/Clase | Módulo | Responsabilidad |
|---|---|---|
| `SpeakerTimestampsService` | `speaker_timestamps/service.py` | Orquesta pyannote, fusión y match → unidades actor |
| `load_pyannote_pipeline` | `speaker_timestamps/service.py` | Carga primary/fallback + device + ProgressHook |
| `fusionar_turnos_consecutivos` | `speaker_timestamps/service.py` | Une tramos adyacentes del mismo speaker sin cruzar videos |
| `identificar_actor` | `speaker_timestamps/matching.py` | Compara speakers contra el perfil → `list[SpeakerMatch]` |
| `construir_perfil_desde_output` | `speaker_timestamps/perfil_voz.py` | Speaker dominante desde `speaker_embeddings` |

### execution — ✅ control plane

**30 tests** en verde.

| Clase / API | Módulo | Responsabilidad |
|---|---|---|
| `ExecutionRunner` | `runner.py` | Ejecuta stages con factories fakeables |
| `load_run_config` | `config.py` | Carga YAML → `RunConfig` tipado |
| `validate_run_config` | `validation.py` | Valida dependencias entre stages |
| `build_execution_plan` | `plan.py` | Resuelve etapas, skips y artefactos |
| `resolve_artifacts` | `artifacts.py` | Rutas de artefactos durables |

### main.py — CLI entry point

| Función | Responsabilidad |
|---|---|
| `_run_execution_cli` | Carga/valida/ejecuta `RunConfig` |
| `_execution_factories` | Factories de cada service desde `RunConfig` |
| `main(argv)` | Parser CLI (solo `--config`, `--dry-run`, `--validate-config`, `--verbose`); retorna `int` |

- `main.py` es un wrapper ligero que retorna `int`; `raise SystemExit(main())` al final.
- Cada corrida deja `output/<run_id>/manifest.json` con status, stages y timings.
