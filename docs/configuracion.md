# Configuración

> **Archivo canónico:** `config/config.yaml`

## Propósito

`config/config.yaml` es el contrato canónico de ejecución (`schema_version: tono-politico.run.v1`) para el pipeline `speech2text → argument_shape → topics_cluster → topics_approach`.

```bash
uv run python main.py --config config/config.yaml
uv run python main.py --config config/config.yaml --dry-run
uv run python main.py --config config/config.yaml --validate-config
```

## Secciones del schema v1

| Sección YAML | Service / módulo | Estado | Notas |
|---|---|---:|---|
| `schema_version` | loader `execution.config` | ✅ | Debe ser `tono-politico.run.v1`. |
| `run` | `ExecutionPlan` | ✅ | `stages`, `resume`, `overwrite`, `keep_cache`, `fail_fast`, `max_videos`, `only_video_ids`. |
| `input` | dependencias iniciales | ✅ | `playlist_url` o artefactos previos (`actor_transcripts_dir`, `argumentos_path`, `temas_path`). |
| `output` | `ArtifactPaths` | ✅ | `output/<run_id>/manifest.json`, `resolved-config.yaml`, artefactos por stage, `speech2text/quality.json`. |
| `project` | defaults globales | ✅ | `data_dir`, `idioma`, `random_state`. |
| `speech2text` | `SpeechToTextService` + `ExecutionRunner` | ✅ | Descarga/cache, diarización, ASR actor-only y quality report. |
| `speech2text.speaker_timestamps` | `SpeakerTimestampsService` | ✅ | pyannote + actor match. |
| `speech2text.transcribe_speech` | `TranscribeSpeechService` | ✅ | Whisper actor-only por turno. |
| `discursive_approach.argument_shape` | `ArgumentShapeService` | ✅ | `ActorTranscript[] → Argumento[]`. |
| `discursive_approach.topics_cluster` | `TopicsClusterService` | ✅ | `Argumento[] → ResultadoTemas`. |
| `discursive_approach.topics_approach` | `TopicsApproachService` | ✅ | `ResultadoTemas → ResultadoEnfoques`. |

## Política de ejecución stage-based

- `run.stages` es el contrato explícito de qué etapas correr y en qué orden.
- `run.resume=true` salta etapas cuyo artefacto de salida ya existe; `run.overwrite=true` recomputa aunque exista salida. Los stages discursivos conservan sus opciones `force`; speech2text no duplica ese control.
- `run.fail_fast=false` permite continuar después de una falla solo si la siguiente etapa todavía tiene dependencias satisfechas por contexto o artefactos externos.
- `run.max_videos` y `run.only_video_ids` aplican a `speech2text` después del `discover`; el perfil de voz se resuelve con la metadata completa de la playlist.
- Los `.wav` viven en `data/<playlist>/videos-<playlist>/` como cache runtime: `run.keep_cache=false` los borra al cerrar cada video/ref, `run.keep_cache=true` los conserva para debug.

## speech2text

```yaml
speech2text:
  enabled: true

  speaker_timestamps:
    actor_objetivo: "Lilly Téllez"
    pipeline: "pyannote/speaker-diarization-community-1"
    fallback_pipeline: "pyannote-community/speaker-diarization-community-1"
    device: "auto"
    umbral_match: 0.5
    umbral_ambiguo: 0.7
    referencia_voz:
      video_id: "su9nURIj9XQ"

  transcribe_speech:
    whisper_model: "large-v3-turbo"
    idioma: "es"
```

### Diarización / actor

- **Implementado:** tests cubren models, adapter, perfil de voz, matching, service, transcripción actor-only, clip transcriber Whisper+ffmpeg y serialización actor_transcript.v1.
- El pipeline primary es `pyannote/speaker-diarization-community-1`; si falla, el adapter intenta el fallback `pyannote-community/speaker-diarization-community-1`.
- `device: "auto"` usa CUDA si está disponible y CPU si no.
- El perfil de voz se construye desde `output.speaker_embeddings` seleccionando el speaker dominante.
- Para el perfil de voz se toma un solo audio de referencia de la misma playlist: `su9nURIj9XQ`.
- Si el match de speaker contra el perfil es ambiguo, ese speaker se trata como no-actor y el pipeline continúa.
- La salida contiene únicamente texto atribuido al actor objetivo.
- **Transcripción por clips:** `WhisperFfmpegClipTranscriber` recorta cada turno pyannote a un WAV temporal normalizado (mono 16 kHz PCM) con ffmpeg, lo transcribe con Whisper (`word_timestamps=False`), y reubica los timestamps al timeline absoluto.
- **Thresholds:** `umbral_match=0.5` y `umbral_ambiguo=0.7` basados en pyannote 3.1 (0.7046), distribuciones VoxCeleb/SpeechBrain y smoke Play-PoliTest.

## discursive_approach

```yaml
discursive_approach:
  enabled: true

  input:
    source: "previous_stage"
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

## Tono (stack de inferencia compartido)

El análisis de tono usa una arquitectura híbrida de la familia Liquid AI:

**Embeddings** (`LFM2.5-Embedding-350M` con mean pooling manual):

| Dimensión | Labels |
|---|---|
| Lógica política | nacionalista, globalista, populista, tecnócrata, corporativista, estatista |
| Sentimiento | esperanza, angustia, indignación, orgullo, empatía |
| Estilo discursivo | directo, académico, confrontativo, conciliador, catastrofista, testimonial |
| Función discursiva | crítica, propuesta, narrativa personal |
| Intensidad antagónica | 5 niveles (1 = conciliador, 5 = beligerante) |

**LLM** (`LFM2.5-1.2B-Instruct`):

| Dimensión | Qué mide |
|---|---|
| Stance | apoyo o rechazo respecto al tema evaluado, con contexto del actor |

- Mean pooling manual (no `sentence-transformers`) para evitar embeddings degenerados con LFM2.5.
- Prototipos textuales en español; cada label se evalúa independientemente mediante similitud coseno.
- El LLM razona stance con actor + tema + few-shot balanceado.
