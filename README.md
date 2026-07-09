# Tono Político

Herramienta NLP para analizar el tono de actores políticos mexicanos a partir de transcripciones de YouTube.

## Pipeline

```text
speech2text
  audio_fetcher         playlist + .wav (yt-dlp)
  speaker_timestamps    pyannote exclusive + match actor
  transcribe_speech     Whisper large-v3-turbo por clip del actor
        → ActorTranscript (actor_transcript.v1, turn-level, + fecha)
discursive_approach
  argument_shape        1 audio → Argumento[] (spaCy + LFM2.5)
  topics_cluster        corpus → ResultadoTemas (BERTopic)
  topics_approach       temas → ResultadoEnfoques (base = Tono + firmas)
        → output/<run_id>/manifest.json + artefactos durables
```

## Estado actual

| Componente | Estado | Tests (aprox.) | Salida |
|---|---|---:|---|
| **speech2text** | ✅ + smoke real | 42 | `ActorTranscript` turn-level (+ fecha) |
| · audio_fetcher | ✅ | (suite speech2text) | `VideoMeta`, `.wav` cache |
| · speaker_timestamps | ✅ | (suite speech2text) | `TurnoOrador[]` actor |
| · transcribe_speech | ✅ | (suite speech2text) | `ActorTranscript` |
| **discursive_approach** | ✅ | ~31 | `ResultadoEnfoques` |
| · argument_shape | ✅ | (suite discursive) | `Argumento[]` |
| · topics_cluster | ✅ | (suite discursive) | `ResultadoTemas` |
| · topics_approach | ✅ | (suite discursive) | `ResultadoEnfoques` (Tono + firmas) |
| **execution** (control plane) | ✅ | 30 | `RunConfig`, `ExecutionPlan`, artefactos |
| **tono** (stack de inferencia) | ✅ | ~20 | `ResultadoTono` (base de approaches) |
| **diarizacion** (DTOs + utils) | ✅ | ~20 | DTOs reusados por speech2text |

Verificación local: **`218 passed`** (`-m "not slow"`, 4 slow deselected). Gate: `bash check.sh`.

**Smoke real (Play-PoliTest):** 7/7 videos, 195 turnos del actor, 1961 palabras, 4 tópicos, 19 enfoques. ~66 min en CPU.

Limpieza: `bash clean.sh` (output/ + data/ + caches Python). Filtros: `--output`, `--data`, `--caches`. Dry-run: `--dry-run`. Sin confirmación: `-y`.

## Uso del pipeline

```bash
# Validar config sin cargar modelos
uv run python main.py --config config/config.yaml --validate-config

# Previsualizar plan de ejecución
uv run python main.py --config config/config.yaml --dry-run

# Ejecutar pipeline completo
uv run python main.py --config config/config.yaml
# → output/<run_id>/speech2text/actor_transcripts/ + discursive/*.json + manifest.json
```

`run.stages` define `speech2text` → `argument_shape` → `topics_cluster` → `topics_approach`, con artefactos durables en `output/<run_id>/`. Permite `--dry-run`, `--validate-config`, reanudar por artefactos, recomputar etapas con `force`/`overwrite`, limitar smoke runs con `run.max_videos` / `run.only_video_ids`, y controlar cache `.wav` con `run.keep_cache`.

## speech2text — audio → texto del actor

```text
discover(playlist) → VideoMeta[]
ensure_perfil(video_ref)
por video: fetch_one → speaker_timestamps → transcribe_speech
         → ActorTranscript (actor_transcript.v1)
```

- **audio_fetcher:** yt-dlp playlist + `.wav` (cache en `data/<playlist>/videos-…/`).
- **speaker_timestamps:** Community-1 `exclusive_speaker_diarization` + match coseno al perfil (0.5 / 0.7).
- **transcribe_speech:** Whisper `large-v3-turbo` **solo en clips del actor**, `word_timestamps=False`.
- Doc: [`docs/componente-speech2text.md`](docs/componente-speech2text.md).

### Diarización / actor (detalle técnico)

- **Modelo:** primary `pyannote/speaker-diarization-community-1`; fallback `pyannote-community/speaker-diarization-community-1`.
- **Device:** `auto` (CUDA si hay, si no CPU) + `ProgressHook` cuando existe.
- **Embeddings:** `output.speaker_embeddings` del pipeline (sin modelo aparte).
- **Match:** distancia coseno; ambiguo 0.5–0.7 → descartar.

## Tono — arquitectura híbrida

El análisis de tono usa dos enfoques complementarios de la familia Liquid AI:

**Embeddings** (`LFM2.5-Embedding-350M` con mean pooling manual):

| Dimensión | Labels |Qué mide |
|---|---|---|
| Lógica política | 6 | nacionalista, globalista, populista, tecnócrata, corporativista, estatista |
| Sentimiento | 5 | esperanza, angustia, indignación, orgullo, empatía |
| Estilo discursivo | 6 | directo, académico, confrontativo, conciliador, catastrofista, testimonial |
| Función discursiva | 3 | crítica, propuesta, narrativa personal |
| Intensidad antagónica | 5 niveles | escala 1 (conciliador) a 5 (beligerante) |

**LLM** (`LFM2.5-1.2B-Instruct`):

| Dimensión | Qué mide |
|---|---|
| Stance | apoyo o rechazo respecto al tema evaluado, con contexto del actor |

Cada label de embeddings se evalúa independientemente mediante similitud coseno contra
prototipos textuales en español. El LLM razona stance con actor + tema + few-shot balanceado.

## Decisiones de arquitectura

- **Stage-based execution:** `ExecutionRunner` orquesta etapas con `RunConfig` tipado, artefactos durables, y dependencias declarativas.
- **Config como contrato:** `config/config.yaml` (`schema_version: tono-politico.run.v1`) define stages, parámetros y dependencias.
- **Lazy loading:** Whisper, spaCy, BERTopic, pyannote y modelos LFM2.5 se cargan solo cuando se usan. El CLI no importa módulos pesados al parsear args.
- **ASR:** `large-v3-turbo` en **clips del actor** con `word_timestamps=False` (timestamps de turno desde pyannote).
- **Diarización:** primary `pyannote/speaker-diarization-community-1`, fallback `pyannote-community/speaker-diarization-community-1`, `device=auto`. Actor identificado con perfil de voz cacheado durante la ejecución.
- **Embeddings compartidos:** `discursive_approach` y `tono` usan `LiquidAI/LFM2.5-Embedding-350M`.
- **Mean pooling manual:** `sentence-transformers` produce embeddings degenerados con LFM2.5; se usa `AutoModel` directo con mean pooling.

## Estructura del código

```text
src/tono_politico/
├── execution/             # Control plane stage-based ✅
│   ├── runner.py          # ExecutionRunner
│   ├── config.py          # load_run_config
│   ├── plan.py            # build_execution_plan
│   ├── artifacts.py       # resolve_artifacts
│   ├── validation.py      # validate_run_config
│   └── models.py          # RunConfig, StageResult, ExecutionPlan
├── speech2text/           # autocontenido: audio → ActorTranscript ✅
│   ├── service.py         # SpeechToTextService
│   ├── models.py          # ActorTranscript, TurnoOrador, PerfilVozActor, SpeakerMatch
│   ├── actor_transcript.py # Serialización JSON actor_transcript.v1
│   ├── requisitos.md      # checklist + viabilidad
│   ├── audio_fetcher/     # playlist + descarga .wav
│   │   ├── models.py      # VideoMeta, AudioVideo, DownloadResult, PlaylistInfo, VideoInfo
│   │   ├── cache.py       # rutas .wav
│   │   ├── playlist.py    # obtener_info_playlist
│   │   ├── audio.py       # descarga + cache
│   │   └── service.py     # AudioFetcherService
│   ├── speaker_timestamps/# pyannote + match actor (diarización del actor)
│   │   ├── service.py     # SpeakerTimestampsService
│   │   ├── adapter.py     # load_pyannote_pipeline
│   │   ├── matching.py    # identificar_actor, clasificar_speaker
│   │   └── perfil_voz.py  # construir_perfil_desde_output
│   └── transcribe_speech/ # Whisper clips actor-only → ActorTranscript
│       ├── service.py     # TranscribeSpeechService
│       ├── whisper_clip.py    # WhisperFfmpegClipTranscriber
│       └── transcripcion_actor.py
├── discursive_approach/   # ActorTranscript → temas + enfoques ✅
│   ├── service.py         # DiscursiveApproachService
│   ├── requisitos.md      # decisiones 1–9 + checklist
│   ├── argument_shape/    # Oracion/Argumento (spaCy + LFM2.5)
│   ├── topics_cluster/    # BERTopic sobre Argumento[]
│   └── topics_approach/   # Tono + firmas → ResultadoEnfoques
├── tono/                  # Stack de inferencia de tono (reusado por topics_approach)
│   ├── service.py         # TonoService (orquestador híbrido)
│   ├── embeddings.py      # EmbeddorTono + similitud coseno
│   ├── zero_shot.py       # ClasificadorLLM para stance
│   ├── taxonomia.py       # 25 prototipos en 5 dimensiones
│   └── models.py          # ResultadoTono, SegmentoConTono
├── segmentacion/models.py # DTOs: Segmento, Oracion, WordTimestamp
├── temas/models.py        # DTOs: ResultadoTemas, TopicoInfo
└── filtrado/models.py     # DTOs: ResultadoFiltrado, SegmentoFiltrado
main.py                    # CLI entry point — delega a ExecutionRunner
```

> **Nota:** `segmentacion/models.py`, `temas/models.py` y `filtrado/models.py` contienen solo DTOs que `tono/` y `topics_approach/adapter.py` referencian. Se consolidarán en la Fase 2 del desacoplamiento de `discursive_approach`.

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

- [**speech2text**](docs/componente-speech2text.md)
- [**discursive_approach**](docs/componente-discursive-approach.md)
- [Diarización (stack interno)](docs/componente-diarizacion.md)
- [Configuración](docs/configuracion.md)
