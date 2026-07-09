# Tono Político

Herramienta NLP para analizar el tono de actores políticos mexicanos a partir de transcripciones de YouTube.

## Pipeline

```text
PATH PREFERIDO (--config config/config.yaml)
  speech2text
    audio_fetcher         playlist + .wav (yt-dlp)
    speaker_timestamps    pyannote exclusive + match actor
    transcribe_speech     Whisper large-v3-turbo por clip del actor
          → ActorTranscript (actor_transcript.v1, turn-level, + fecha)
  discursive_approach
    argument_shape        1 audio → Argumento[] (spaCy + LFM2.5)
    topics_cluster        corpus → ResultadoTemas (BERTopic)
    topics_approach       temas → ResultadoEnfoques (base = Tono + firmas)
          → [filtrado / salida posteriores — R6 pendiente]

PATH LEGACY (default CLI sin --discursive)
  Ingesta → DiarizacionService → Segmentación → Temas
  → Filtrado → Tono → Salida
```

## Estado actual

| Componente | Estado | Tests (aprox.) | Salida |
|---|---|---:|---|
| **speech2text** | ✅ + smoke real | 42 | `ActorTranscript` turn-level (+ fecha) |
| **discursive_approach** | ✅ R1–R5 | **~31** | `ResultadoEnfoques` |
| · argument_shape | ✅ | (suite discursive) | `Argumento[]` |
| · topics_cluster | ✅ | (suite discursive) | `ResultadoTemas` |
| · topics_approach | ✅ | (suite discursive) | `ResultadoEnfoques` (Tono + firmas) |
| 1. Ingesta (legacy) | ✅ | 56 | `list[VideoTranscript]` |
| 1.5 Diarización (legacy) | ✅ | 88 | `list[VideoTranscript]` filtrado |
| 2. Segmentación (legacy path) | ✅ | 35 | `list[Segmento]` |
| 3. Temas (legacy path) | ✅ | 21 | `ResultadoTemas` |
| 4. Filtrado | ✅ | 5 | `ResultadoFiltrado` |
| 5. Tono | ✅ | 65 | `ResultadoTono` (también base de approaches) |
| 6. Salida | ✅ | 35 | `InformeTono` |
| execution/main/config | ✅ stage-based config | 30 | `RunConfig`, `ExecutionPlan`, artefactos |

Verificación local: **`461 passed`** (`-m "not slow"`, 5 slow deselected; **466** collected). Gate: `bash check.sh`.

**Smoke speech2text (Play-PoliTest):** 7/7 videos, 195 turnos del actor, 0 errores (~38 min). Artefactos en `output/speech2text-smoke/`.

Limpieza: `bash clean.sh` (output/ + data/ + caches Python). Filtros: `--output`, `--data`, `--caches`. Dry-run: `--dry-run`. Sin confirmación: `-y`.

## Uso del pipeline completo

El pipeline se ejecuta con `main.py`, que lee `config/config.yaml` automáticamente.

```bash
# Preferido: config granular stage-based (speech2text → discursive_approach)
uv run python main.py --config config/config.yaml --dry-run
uv run python main.py --config config/config.yaml --validate-config
uv run python main.py --config config/config.yaml
# → output/<run_id>/speech2text/actor_transcripts/ + discursive/*.json

# Legacy discover: speech2text → discursive_approach vía flag anterior
uv run python main.py --playlist "https://youtube.com/playlist?list=..." --discursive --keep-cache
# → output/<run_id>/discursive-temas.json + discursive-enfoques.json

# Legacy Fase 1: descubrir tópicos (Ingesta → Diarización → Segmentación → Temas)
uv run python main.py --playlist "https://youtube.com/playlist?list=..."

# Debug: conservar audios/transcripciones runtime en data/<playlist>/
uv run python main.py --playlist "https://youtube.com/playlist?list=..." --keep-cache

# Legacy Fase 2: analizar un tópico específico
uv run python main.py \
    --playlist "https://youtube.com/playlist?list=..." \
    --topico 0 \
    --tema "fracking" \
    --output output/

# Reusar Fase 1 legacy para otro tópico
uv run python main.py \
    --resume output/<run_id> \
    --topico 1 \
    --tema "seguridad" \
    --output output/
```

**Path preferido (`--config config/config.yaml`):** `run.stages` define `speech2text` → `argument_shape` → `topics_cluster` → `topics_approach`, con artefactos durables en `output/<run_id>/`. Permite `--dry-run`, `--validate-config`, reanudar por artefactos, recomputar etapas con `force`/`overwrite`, limitar smoke runs con `run.max_videos` / `run.only_video_ids`, y controlar cache `.wav` con `run.keep_cache`.

**Path legacy (`--discursive`):** ejecuta el discover nuevo desde el runner anterior. Se mantiene temporalmente para compatibilidad.

**Path legacy clásico:** Fase 1 Ingesta → Diarización → Segmentación → Temas; Fase 2 Filtrado → Tono → Salida con `--topico`/`--tema`. `--resume` reutiliza `fase1-topicos.json`.

## speech2text — audio → texto del actor

Camino preferido para pasar de playlist a texto del actor (sin Whisper full-video):

```text
discover(playlist) → VideoMeta[]
ensure_perfil(video_ref)
por video: fetch_one → speaker_timestamps → transcribe_speech
         → ActorTranscript (actor_transcript.v1)
```

- **audio_fetcher:** yt-dlp playlist + `.wav` (cache en `data/<playlist>/videos-…/`).
- **speaker_timestamps:** Community-1 `exclusive_speaker_diarization` + match coseno al perfil (0.5 / 0.7).
- **transcribe_speech:** Whisper `large-v3-turbo` **solo en clips del actor**, `word_timestamps=False`.
- **Smoke real:** Play-PoliTest **7/7**, **195** turnos, ~1959 palabras, 0 errores (`scripts_smoke_speech2text.py`).
- Doc: [`docs/componente-speech2text.md`](docs/componente-speech2text.md).

> **Nota:** con `--discursive`, `PipelineRunner.discover_discursive` orquesta speech2text + discursive_approach. El default CLI sin flag sigue el path legacy (Ingesta + DiarizacionService).

### Diarización / actor (detalle técnico, compartido)

- **Modelo:** primary `pyannote/speaker-diarization-community-1`; fallback `pyannote-community/speaker-diarization-community-1`.
- **Device:** `auto` (CUDA si hay, si no CPU) + `ProgressHook` cuando existe.
- **Embeddings:** `output.speaker_embeddings` del pipeline (sin modelo aparte).
- **Match:** distancia coseno; ambiguo 0.5–0.7 → descartar.

## Componente 5: Tono — arquitectura híbrida

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

- **Services OOP por componente:** cada componente implementa `ComponenteProtocol` mediante `.procesar(input) -> output`.
- **Config encapsulada:** los hiperparámetros viven en el constructor del service. `main.py` los carga desde `config/config.yaml`.
- **Helpers puros:** la lógica interna se mantiene en funciones testeables.
- **Lazy loading:** Whisper, spaCy, BERTopic, pyannote y modelos LFM2.5 se cargan solo cuando se usan.
- **ASR default (speech2text):** `large-v3-turbo` en **clips del actor** con `word_timestamps=False` (timestamps de turno desde pyannote). El camino legacy de Ingesta aún usa Whisper full-video con word timestamps.
- **Diarización implementada:** primary `pyannote/speaker-diarization-community-1`, fallback `pyannote-community/speaker-diarization-community-1`, `device=auto` y `ProgressHook` si está disponible; los embeddings por speaker salen de `output.speaker_embeddings` del pipeline. El actor se identifica con un perfil de voz cacheado solo durante la ejecución. Si el match es ambiguo, se descarta como otro speaker y el pipeline continúa.
- **DTOs compartidos vs locales:** `src/tono_politico/models.py` contiene DTOs compartidos por más de un componente. Los DTOs específicos viven dentro de su componente.
- **Embeddings compartidos:** Segmentación, Temas, Tono y `discursive_approach` usan `LiquidAI/LFM2.5-Embedding-350M`.
- **Mean pooling manual en Tono:** `sentence-transformers` produce embeddings degenerados con LFM2.5; el Componente 5 usa `AutoModel` directo con mean pooling.

## Estructura del código

```text
src/tono_politico/
├── models.py              # DTOs compartidos (legacy + slim pendientes)
├── protocol.py            # ComponenteProtocol
├── config.py              # Config tipada (dataclasses) + load_config
├── speech2text/           # Preferido: audio → ActorTranscript ✅
│   ├── service.py         # SpeechToTextService (orquestador)
│   ├── requisitos.md      # checklist + viabilidad
│   ├── audio_fetcher/     # playlist + descarga .wav
│   ├── speaker_timestamps/# pyannote + match actor
│   └── transcribe_speech/ # Whisper clips actor-only
├── discursive_approach/   # Preferido: ActorTranscript → temas + enfoques ✅
│   ├── service.py         # DiscursiveApproachService
│   ├── requisitos.md      # decisiones 1–9 + checklist
│   ├── argument_shape/    # Oracion/Argumento (sin word-level)
│   ├── topics_cluster/    # BERTopic sobre Argumento[]
│   └── topics_approach/   # Tono + firmas → ResultadoEnfoques
├── ingesta/               # Componente 1 legacy ✅ (Whisper full-video)
│   ├── service.py         # IngestaService
│   ├── playlist.py        # metadata de playlists vía yt-dlp
│   ├── audio.py           # descarga/cache de audios .wav (yt-dlp + --download-archive)
│   ├── transcripcion.py   # Whisper + JSON
│   ├── models.py          # DownloadResult (errores estructurados)
│   └── cache.py           # rutas centralizadas
├── diarizacion/           # Componente 1.5 legacy ✅ (stack reusado por speech2text)
│   ├── service.py         # DiarizacionService (orquestador + lazy-load via adapter)
│   ├── adapter.py         # load_pyannote_pipeline (primary/fallback/device/ProgressHook)
│   ├── diarizacion.py     # diarizar() — pyannote → TurnoOrador[]
│   ├── perfil_voz.py      # construir_perfil_desde_output() — speaker dominante desde speaker_embeddings
│   ├── matching.py        # distancia_coseno(), clasificar_speaker(), identificar_actor()
│   ├── alineacion.py      # filtrar_por_actor() — midpoint → segmentos del actor (bisect)
│   ├── transcripcion_actor.py  # transcribir_turnos_actor — turnos → ActorTranscript
│   ├── whisper_clip.py    # WhisperFfmpegClipTranscriber — ffmpeg + Whisper por clip
│   ├── actor_transcript.py     # Serialización JSON actor_transcript.v1
│   └── models.py          # TurnoOrador, PerfilVozActor, SpeakerMatch, ActorTranscript, AsrMetadata
├── segmentacion/          # Componente 2 ✅
│   ├── service.py         # SegmentacionService
│   ├── sentencias.py      # spaCy nlp.pipe → Oracion[]
│   ├── breakpoints.py     # distancia coseno + percentil 95
│   ├── agrupacion.py      # guardrails min/max
│   └── models.py          # Oracion, Segmento
├── temas/                 # Componente 3 ✅
│   ├── service.py         # TemasService
│   ├── descubrimiento.py  # BERTopic + UMAP (random_state) + HDBSCAN
│   ├── serializacion.py   # guardar_fase1 / cargar_fase1 (JSON para --resume)
│   └── models.py          # SegmentoTematizado, TopicoInfo, ResultadoTemas
├── filtrado/              # Componente 4 ✅
│   ├── service.py         # FiltradoService
│   ├── filtro.py          # filtrar determinista por tópico/relevancia
│   └── models.py          # CriterioFiltrado, SegmentoFiltrado, ResultadoFiltrado
├── tono/                  # Componente 5 ✅
│   ├── service.py         # TonoService (orquestador híbrido)
│   ├── embeddings.py      # EmbeddorTono (mean pooling, batch real) + similitud coseno
│   ├── zero_shot.py       # ClasificadorLLM para stance (do_sample=False)
│   ├── taxonomia.py       # 25 prototipos en 5 dimensiones
│   └── models.py          # EtiquetaScore, Resultado*, SegmentoConTono, ResultadoTono
├── salida/                # Componente 6 ✅
│   ├── service.py         # SalidaService
│   ├── agregacion.py      # colapsar ResultadoTono → PerfilActor
│   ├── serializacion.py   # JSON + Markdown
│   └── models.py          # Provenance, PerfilActor, InformeTono
├── pipeline/              # Orquestación ✅
│   ├── runner.py          # PipelineRunner (discover/discover_discursive/analyze/…)
│   ├── manifest.py        # guardar_manifest + resumen_final (CLI summary)
│   ├── models.py          # RunManifest, RunResult, PhaseRunStatus, VideoRunStatus
│   └── __init__.py        # exports públicos
main.py                    # CLI entry point — wrapper ligero que delega a PipelineRunner
```

## Configuración

Defaults de proyecto: [`config/config.yaml`](config/config.yaml).

`main.py` carga automáticamente `config/config.yaml` y construye cada service con los valores de su sección. Los services siguen siendo configurables por constructor para uso programático.

## Entorno local

El proyecto usa el stack Astral: `uv`, `ruff`, `ty`.

```bash
cd ~/Documentos/Proyectos/tono-politico
uv venv --python 3.11
uv pip install -e ".[dev]"
uv lock
```

Modelo spaCy para ejecución real del Componente 2:

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

## Uso programático por componente

### speech2text (preferido)

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

Smoke end-to-end sobre Play-PoliTest:

```bash
uv run python scripts_smoke_speech2text.py
# → output/speech2text-smoke/summary.json + actor_transcripts/
```

### discursive_approach (preferido post-speech2text)

```python
from tono_politico.discursive_approach import DiscursiveApproachService

svc = DiscursiveApproachService(actor="Lilly Téllez")
enfoques = svc.procesar(actor_transcripts)  # shape → cluster → approaches
```

O por capas: `shape_corpus` → `cluster` → `approaches`.  
Doc: [`docs/componente-discursive-approach.md`](docs/componente-discursive-approach.md).

### Componente 2: Segmentación (legacy path)

> Entrada `list[VideoTranscript]`. El path preferido usa `argument_shape` sobre `ActorTranscript`.

```python
from tono_politico.segmentacion import SegmentacionService

svc = SegmentacionService(
    spacy_model="es_core_news_lg",
    breakpoint_percentile=95,
    min_oraciones=2,
    max_oraciones=8,
    max_palabras=150,
)

segmentos = svc.procesar(transcripciones_actor)  # legacy VideoTranscript[]
```

### Componente 3: Temas

```python
from tono_politico.temas import TemasService

svc = TemasService(
    min_topic_size=3,
    n_neighbors=10,
    n_components=5,
)

resultado = svc.procesar(segmentos)
```

### Componente 4: Filtrado

```python
from tono_politico.filtrado import FiltradoService

svc = FiltradoService(
    topico_id=0,
    min_relevancia=0.35,
    incluir_outliers=False,
)

resultado_filtrado = svc.procesar(resultado)
```

### Componente 5: Tono

```python
from tono_politico.tono import TonoService

svc = TonoService(
    actor="Lilly Téllez",
    tema="fracking",
)

resultado_tono = svc.procesar(resultado_filtrado)
```

### Componente 6: Salida

```python
from tono_politico.salida import SalidaService

svc = SalidaService(output_path="output/")  # directorio → genera informe.json + informe.md

informe = svc.procesar(resultado_tono)
```

## Documentación técnica

- [**speech2text** (preferido)](docs/componente-speech2text.md)
- [**discursive_approach** (preferido)](docs/componente-discursive-approach.md)
- [Componente 1: Ingesta (legacy)](docs/componente-ingesta.md)
- [Componente 1.5: Diarización (legacy / stack interno)](docs/componente-diarizacion.md)
- [Componente 2: Segmentación (legacy path)](docs/componente-2-segmentacion.md)
- [Componente 3: Temas (legacy path)](docs/componente-3-temas.md)
- [Componente 4: Filtrado](docs/componente-4-filtrado.md)
- [Componente 5: Tono](docs/componente-5-tono.md)
- [Componente 6: Salida](docs/componente-6-salida.md)
- [Configuración](docs/configuracion.md)
