# Tono Político

Herramienta NLP para analizar el tono de actores políticos mexicanos a partir de transcripciones de YouTube.

## Pipeline

```text
speech2text
  audio_fetcher         playlist + .wav (yt-dlp)
  speaker_timestamps    pyannote exclusive + match actor
  transcribe_speech     Whisper large-v3-turbo por clip del actor
        → ActorTranscript (actor_transcript.v1, turn-level, + source metadata)
discursive_approach
  argument_shape        1 audio → Argumento[] (spaCy + LFM2.5)
  topics_cluster        corpus → ResultadoTemas (BERTopic)
  topics_approach       bloqueado hasta reconstruir sus contratos
        → output/<run_id>/manifest.json + artefactos durables
```

## Estado actual

| Componente | Estado | Tests (aprox.) | Salida |
|---|---|---:|---|
| **speech2text** | ✅ + smoke real | — | `ActorTranscript` + provenance + `quality.json` |
| · audio_fetcher | ✅ | (suite speech2text) | `VideoMeta`, `.wav` cache |
| · speaker_timestamps | ✅ | (suite speech2text) | `TurnoOrador[]` actor |
| · transcribe_speech | ✅ | (suite speech2text) | `ActorTranscript` |
| **discursive_approach** | ⚠️ bloqueado | — | Pendiente de desacoplar/reconstruir |
| · argument_shape | ⚠️ bloqueado | — | Ruta dependiente temporalmente inactiva |
| · topics_cluster | ⚠️ bloqueado | — | Ruta dependiente temporalmente inactiva |
| · topics_approach | ⚠️ bloqueado | — | Depende de lógica retirada |
| **execution** (control plane) | ✅ | 30 | `RunConfig`, `ExecutionPlan`, artefactos |

Verificación local: **`191 passed, 1 skipped`** con las dos suites legacy de `discursive_approach` excluidas por imports retirados. Ruff y formato pasan; `ty` conserva seis errores de esos mismos imports legacy. Gate: `bash check.sh`.

**Smoke real de metadata (Play-PoliTest):** 3/3 vídeos cortos seleccionados, stage `speech2text=ok`, 34 segmentos, provenance verificada en transcript, manifest, checkpoint y quality. El quick original de tres vídeos incluía un vídeo largo y excedió el timeout de 600 s durante ASR.

Limpieza: `bash clean.sh` (output/ + data/ + caches Python). Filtros: `--output`, `--data`, `--caches`. Dry-run: `--dry-run`. Sin confirmación: `-y`.

## Uso del pipeline

```bash
# Validar config sin cargar modelos
uv run python main.py --config config/config.yaml --validate-config

# Previsualizar plan de ejecución
uv run python main.py --config config/config.yaml --dry-run

# Ejecutar pipeline completo
uv run python main.py --config config/config.yaml
# → output/<run_id>/speech2text/actor_transcripts/ + checkpoint.json + quality.json + manifest.json
```

`run.stages` conserva la secuencia declarativa `speech2text` → `argument_shape` → `topics_cluster` → `topics_approach`, pero las etapas discursivas quedan bloqueadas hasta reconstruir sus contratos. La ruta activa es `speech2text`; permite `--dry-run`, `--validate-config`, reanudar por artefactos, recomputar con `run.overwrite`, limitar smoke runs con `run.max_videos` / `run.only_video_ids`, y controlar cache `.wav` con `run.keep_cache`.

## speech2text — audio → texto del actor

```text
discover(playlist) → VideoMeta[]
ensure_perfil(video_ref)
por video: fetch_one → speaker_timestamps → transcribe_speech
         → ActorTranscript (actor_transcript.v1)
```

- **audio_fetcher:** yt-dlp playlist + `.wav` (cache en `data/<playlist>/videos-…/`), con nombre visible/cache y fuente explícita de fecha.
- **speaker_timestamps:** Community-1 `exclusive_speaker_diarization` + match coseno al perfil (0.5 / 0.7).
- **transcribe_speech:** Whisper `large-v3-turbo` **solo en clips del actor**, `word_timestamps=False`.
- Doc: [`docs/module-speech2text.md`](docs/module-speech2text.md).

### `speaker_timestamps` / actor (detalle técnico)

- **Modelo:** primary `pyannote/speaker-diarization-community-1`; fallback `pyannote-community/speaker-diarization-community-1`.
- **Device:** `auto` (CUDA si hay, si no CPU) + `ProgressHook` cuando existe.
- **Embeddings:** `output.speaker_embeddings` del pipeline (sin modelo aparte).
- **Match:** distancia coseno; ambiguo 0.5–0.7 → descartar.

## Estado de `discursive_approach`

La ruta `argument_shape → topics_cluster → topics_approach` queda temporalmente bloqueada porque sus adaptadores todavía dependían de `filtrado`, `segmentacion`, `temas` y `tono`. Se reconstruirá en una fase posterior con contratos propios.

## Decisiones de arquitectura

- **Stage-based execution:** `ExecutionRunner` orquesta etapas con `RunConfig` tipado, artefactos durables, y dependencias declarativas.
- **Config como contrato:** `config/config.yaml` (`schema_version: tono-politico.run.v1`) define stages, parámetros y dependencias.
- **Lazy loading:** Whisper, spaCy, BERTopic, pyannote y modelos LFM2.5 se cargan solo cuando se usan. El CLI no importa módulos pesados al parsear args.
- **ASR:** `large-v3-turbo` en **clips del actor** con `word_timestamps=False` (timestamps de turno desde pyannote).
- **Diarización:** primary `pyannote/speaker-diarization-community-1`, fallback `pyannote-community/speaker-diarization-community-1`, `device=auto`. Actor identificado con perfil de voz cacheado durante la ejecución.
- **discursive_approach:** bloqueado temporalmente tras retirar sus dependencias legacy; no se ejecuta como parte del pipeline válido actual.

## Estructura del código

```text
src/tono_politico/
├── execution/             # Control plane stage-based ✅
│   ├── runner.py          # ExecutionRunner
│   ├── config.py          # load_run_config
│   ├── plan.py            # build_execution_plan
│   ├── artifacts.py       # resolve_artifacts + serialización ActorTranscript
│   ├── validation.py      # validate_run_config
│   ├── observability.py   # build_quality_report (speech2text_quality.v2)
│   └── models.py          # RunConfig, StageResult, ExecutionPlan, UnitResult
├── speech2text/           # autocontenido: audio → ActorTranscript ✅
│   ├── service.py         # SpeechToTextService
│   ├── models.py          # ActorTranscript, TranscriptSource y compatibilidad de DTOs
│   ├── requisitos.md      # checklist + viabilidad
│   ├── audio_fetcher/     # playlist + descarga .wav
│   │   ├── models.py      # VideoMeta, AudioVideo, DownloadResult, PlaylistInfo
│   │   ├── playlist.py    # obtener_info_playlist
│   │   ├── audio.py       # descarga + cache
│   │   └── service.py     # AudioFetcherService
│   ├── speaker_timestamps/# pyannote + match actor (diarización del actor)
│   │   ├── models.py      # DTOs canónicos de diarización/matching
│   │   ├── service.py     # SpeakerTimestampsService + load_pyannote_pipeline
│   │   ├── matching.py    # identificar_actor, clasificar_speaker, distancia_coseno
│   │   └── perfil_voz.py  # construir_perfil_desde_output
│   └── transcribe_speech/ # Whisper clips actor-only → ActorTranscript
│       ├── service.py     # TranscribeSpeechService
│       ├── actor_clip.py  # padding y mapeo temporal de clips
│       ├── transcription_clip.py # WhisperFfmpegClipTranscriber
│       └── models.py      # ClipTranscriptSegment, ClipTranscriber
├── discursive_approach/   # ActorTranscript → temas + enfoques ⚠️ bloqueado
│   ├── service.py         # DiscursiveApproachService
│   ├── requisitos.md      # decisiones 1–9 + checklist
│   ├── argument_shape/    # Oracion/Argumento (spaCy + LFM2.5)
│   ├── topics_cluster/    # BERTopic sobre Argumento[]
│   └── topics_approach/   # pendiente de reconstrucción
main.py                    # CLI entry point — delega a ExecutionRunner
```

> **Nota:** `discursive_approach` conserva código histórico incompleto, pero sus dependencias `filtrado`, `segmentacion`, `temas` y `tono` fueron eliminadas deliberadamente. Su reparación queda fuera de esta fase.

## Configuración

Defaults de proyecto: [`config/config.yaml`](config/config.yaml).

`main.py` carga automáticamente `config/config.yaml` y construye cada service con los valores de su sección. Ver [`docs/configuracion.md`](docs/configuracion.md) para el detalle de cada sección.

## Entorno local

El proyecto usa el stack Astral: `uv`, `ruff`, `ty`.

```bash
cd ~/Documentos/Proyectos/tono-politico
uv venv --python 3.11
uv pip install -e ".[dev]"
uv lock
```

Modelo spaCy:

```bash
uv run python -m spacy download es_core_news_lg
```

## Calidad

```bash
# Tests (excluye los que cargan modelos pesados)
uv run pytest tests/ -v -m "not slow"

# Tests de integración (cargan modelos reales)
uv run pytest tests/ -v -m slow

# Lint
uv run ruff check src/ tests/ main.py

# Format check
uv run ruff format --check src/ tests/ main.py

# Type check
uv run ty check

# Todo antes de cerrar un cambio
uv run ruff check src/ tests/ main.py && uv run ty check && uv run pytest tests/ -v -m "not slow"
```

## Uso programático

### speech2text

```python
from pathlib import Path
from tono_politico.speech2text import SpeechToTextService

svc = SpeechToTextService(
    data_dir=Path("data"),
    actor="Lilly Téllez",
    video_ref_id="su9nURIj9XQ",
    whisper_model="large-v3-turbo",
    idioma="es",
)

playlist, metas = svc.discover("https://youtube.com/playlist?list=...")
svc.ensure_perfil(playlist.nombre, metas)

transcripts = []
for meta in metas:
    tx = svc.procesar_one(meta, playlist.nombre)
    if tx is not None:
        transcripts.append(tx)
# transcripts: list[ActorTranscript]  (actor_transcript.v1)
```

### discursive_approach

```python
from tono_politico.discursive_approach import DiscursiveApproachService

svc = DiscursiveApproachService(actor="Lilly Téllez")
enfoques = svc.procesar(actor_transcripts)  # shape → cluster → approaches
```

O por capas: `shape_corpus` → `cluster` → `approaches`.
Doc: [`docs/componente-discursive-approach.md`](docs/componente-discursive-approach.md).

## Documentación técnica

- [**speech2text**](docs/module-speech2text.md)
- [audio_fetcher](docs/component_audio_fetcher.md)
- [speaker_timestamps](docs/component_speaker_timestamps.md)
- [transcribe_speech](docs/component_transcribe_speech.md)
- [**discursive_approach**](docs/componente-discursive-approach.md)
- [Configuración](docs/configuracion.md)
