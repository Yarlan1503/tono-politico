# Configuración

> **Archivo canónico:** `config/config.yaml`

## Propósito

`config/config.yaml` es ahora el contrato canónico de ejecución (`schema_version: tono-politico.run.v1`) para el path preferido `speech2text → argument_shape → topics_cluster → topics_approach`.

El CLI nuevo se invoca así:

```bash
uv run python main.py --config config/config.yaml
uv run python main.py --config config/config.yaml --dry-run
uv run python main.py --config config/config.yaml --validate-config
```

El CLI legacy con `--playlist`, `--discursive`, `--topico` y `--resume` sigue vivo temporalmente, pero el control granular del path nuevo vive en `run.stages` y en las secciones `speech2text` / `discursive_approach`.

## Secciones canónicas del schema v1

| Sección YAML | Service / módulo | Estado | Notas |
|---|---|---:|---|
| `schema_version` | loader `execution.config` | ✅ | Debe ser `tono-politico.run.v1`. |
| `run` | `ExecutionPlan` | ✅ | `stages`, `resume`, `overwrite`, `keep_cache`, `fail_fast`, `max_videos`, `only_video_ids`. |
| `input` | dependencias iniciales | ✅ | `playlist_url` o artefactos previos (`actor_transcripts_dir`, `argumentos_path`, `temas_path`). |
| `output` | `ArtifactPaths` | ✅ | `output/<run_id>/manifest.json`, `resolved-config.yaml`, artefactos por stage. |
| `project` | defaults globales | ✅ | `data_dir`, `idioma`, `random_state`. |
| `speech2text.audio_fetcher` | `AudioFetcherService` | ✅ | Descarga/cache de `.wav`. |
| `speech2text.speaker_timestamps` | `SpeakerTimestampsService` | ✅ | pyannote + actor match. |
| `speech2text.transcribe_speech` | `TranscribeSpeechService` | ✅ | Whisper actor-only por turno. |
| `discursive_approach.argument_shape` | `ArgumentShapeService` | ✅ | `ActorTranscript[] → Argumento[]`. |
| `discursive_approach.topics_cluster` | `TopicsClusterService` | ✅ | `Argumento[] → ResultadoTemas`. |
| `discursive_approach.topics_approach` | `TopicsApproachService` | ✅ | `ResultadoTemas → ResultadoEnfoques`. |

## Secciones legacy

Las secciones legacy (`ingesta`, `diarizacion`, `segmentacion`, `temas`, `filtrado`, `tono`, `salida`) ya no son la fuente canónica del path nuevo. Permanecen documentadas abajo para entender el path legacy y los services originales.

## Política de ejecución stage-based

- `run.stages` es el contrato explícito de qué etapas correr y en qué orden.
- `run.resume=true` salta etapas cuyo artefacto de salida ya existe; `run.overwrite=true` y los `force` por stage recomputan aunque exista salida.
- `run.fail_fast=false` permite continuar después de una falla solo si la siguiente etapa todavía tiene dependencias satisfechas por contexto o artefactos externos.
- `run.max_videos` y `run.only_video_ids` aplican a `speech2text` después del `discover`; el perfil de voz se resuelve con la metadata completa de la playlist.
- Los `.wav` viven en `data/<playlist>/videos-<playlist>/` como cache runtime: `run.keep_cache=false` los borra al cerrar cada video/ref, `run.keep_cache=true` los conserva para debug.

## Componente 1: `ingesta`

```yaml
project:
  data_dir: "data"
  idioma: "es"
  random_state: 42

ingesta:
  whisper_model: "large-v3-turbo"
  idioma: "es"
```

Equivale a:

```python
IngestaService(
    data_dir=Path("data"),
    whisper_model="large-v3-turbo",
    idioma="es",
)
```

No incluir `whisper_device` mientras `transcribir()` no lo acepte explícitamente. Hoy Whisper se ejecuta con `fp16=False`, amigable con CPU.

## Componente 1.5: `diarizacion`

```yaml
diarizacion:
  estado: "implementado"
  pipeline: "pyannote/speaker-diarization-community-1"
  fallback_pipeline: "pyannote-community/speaker-diarization-community-1"
  device: "auto"
  actor_objetivo: "Lilly Téllez"
  umbral_match: 0.5
  umbral_ambiguo: 0.7
  referencia_voz:
    origen: "misma_playlist"
    max_audios: 1
    video_id: "su9nURIj9XQ"
    url: "https://www.youtube.com/watch?v=su9nURIj9XQ&list=PLE9Zk7g9R__M&index=8"
    cache: "solo_ejecucion"
  match_ambiguo: "descartar_como_otro_speaker"
  salida: "solo_texto_actor"
```

Equivale a:

```python
DiarizacionService(
    actor="Lilly Téllez",
    video_ref_id="su9nURIj9XQ",
    data_dir=Path("data"),
    pipeline_name="pyannote/speaker-diarization-community-1",
    fallback_pipeline="pyannote-community/speaker-diarization-community-1",
    device="auto",
    umbral_match=0.5,
    umbral_ambiguo=0.7,
)
```

Notas:

- **Implementado:** 88 tests en verde (10 archivos) cubren models, adapter, perfil de voz, diarización, alineación, matching, service, transcripción actor-only, clip transcriber Whisper+ffmpeg y serialización actor_transcript.v1.
- El pipeline primary es `pyannote/speaker-diarization-community-1`; si falla por acceso/gating/model-not-found, el adapter intenta el fallback local validado `pyannote-community/speaker-diarization-community-1`.
- `device: "auto"` usa CUDA si está disponible y CPU si no; las llamadas largas usan `ProgressHook` cuando pyannote lo expone. El adapter usa `torch.device` (no `str` — fix para pyannote 4.x).
- Los embeddings por speaker salen de `output.speaker_embeddings` del propio pipeline; no se carga un modelo separado de embeddings.
- El perfil de voz se construye desde `output.speaker_embeddings` seleccionando el **speaker dominante** (mayor duración total en el audio de referencia).
- La playlist de pruebas actual es `Play-PoliTest` (`PLE9Zk7g9R__M`) y contiene solo intervenciones de Lilly Téllez.
- Para el perfil de voz se toma **un solo audio de referencia** de la misma playlist: `su9nURIj9XQ`.
- El perfil de voz se cachea únicamente durante la ejecución del pipeline; no se persiste como artefacto estable.
- Si el match de speaker contra el perfil es ambiguo, ese speaker se trata como no-actor y el pipeline continúa.
- La salida hacia Segmentación debe contener únicamente texto atribuido al actor objetivo.
- **Transcripción por clips:** `WhisperFfmpegClipTranscriber` recorta cada turno pyannote a un WAV temporal normalizado (mono 16 kHz PCM) con ffmpeg, lo transcribe con Whisper (`word_timestamps=False`), y reubica los timestamps al timeline absoluto. Ver `docs/componente-diarizacion.md` para detalles.
- **Thresholds calibrados con research + smoke real:** `umbral_match=0.5` y `umbral_ambiguo=0.7` basados en el clustering threshold de pyannote 3.1 (0.7046), distribuciones VoxCeleb/SpeechBrain y smoke Play-PoliTest con distancias 0.075–0.131.
- **Smoke test Fase 1 completado:** 7/7 videos de Play-PoliTest procesados con éxito (139 segmentos del actor, 2 tópicos descubiertos, 0 videos omitidos).

## Componente 2: `segmentacion`

```yaml
segmentacion:
  spacy_model: "es_core_news_lg"
  embedding_model: "LiquidAI/LFM2.5-Embedding-350M"
  breakpoint_percentile: 95
  min_oraciones: 2
  max_oraciones: 8
  max_palabras: 150
```

Equivale a:

```python
SegmentacionService(
    spacy_model="es_core_news_lg",
    breakpoint_percentile=95,
    min_oraciones=2,
    max_oraciones=8,
    max_palabras=150,
)
```

Notas:

- `embedding_model` está documentado aunque hoy el service lo carga internamente como `LiquidAI/LFM2.5-Embedding-350M`.
- No usar `pausa_minima` ni `similitud_umbral`: la segmentación actual es 100% semántica, usa **distancia coseno** y corta por **percentil 95**.

## Componente 3: `temas`

```yaml
temas:
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
    calculate_probabilities: true
    verbose: false
```

Equivale a:

```python
TemasService(
    min_topic_size=3,
    n_neighbors=10,
    n_components=5,
    embedding_model_name="LiquidAI/LFM2.5-Embedding-350M",
)
```

Notas:

- `umap`, `hdbscan` y `bertopic` reflejan defaults internos de `descubrir_temas()`.
- Si `len(segmentos) < min_topic_size`, todos los segmentos se asignan a un único tópico controlado `id=0` (no outlier `-1`), para que el pipeline pueda filtrarlos y analizarlos.

## Componente 4: `filtrado`

```yaml
filtrado:
  min_relevancia: 0.35
  incluir_outliers: false
```

Equivale a:

```python
FiltradoService(
    topico_id=<id_elegido>,
    min_relevancia=0.35,
    incluir_outliers=False,
)
```

Notas:

- `topico_id` no tiene default global porque se elige después de inspeccionar `ResultadoTemas.topicos`.
- `min_relevancia` filtra por la probabilidad de asignación del segmento al tópico.
- `incluir_outliers=False` evita analizar accidentalmente el tópico `-1`, que BERTopic usa para ruido/outliers.

## Componente 5: `tono`

```yaml
tono:
  embedding_model: "LiquidAI/LFM2.5-Embedding-350M"
  llm_model: "LiquidAI/LFM2.5-1.2B-Instruct"
  device: "cpu"

  # Stance (vía LLM con actor + tema + few-shot balanceado)
  stance_labels: ["apoyo", "rechazo"]

  # Intensidad antagónica (vía embeddings, 5 prototipos escalonados 1-5)
  intensidad_niveles: 5

  # Lógica política (vía embeddings, multi-label)
  logica_politica_labels:
    ["nacionalista", "globalista", "populista", "tecnocrata", "corporativista", "estatista"]

  # Sentimiento (vía embeddings, multi-label)
  sentimiento_labels:
    ["esperanza", "angustia", "indignacion", "orgullo", "empatia"]

  # Estilo discursivo (vía embeddings, multi-label)
  estilo_discursivo_labels:
    ["directo", "academico", "confrontativo", "conciliador", "catastrofista", "testimonial"]

  # Función discursiva (vía embeddings, multi-label)
  funcion_discursiva_labels:
    ["critica", "propuesta", "narrativa_personal"]
```

Equivale a:

```python
TonoService(
    actor="Lilly Téllez",  # de config diarizacion.actor_objetivo o CLI --tema
    tema="fracking",       # del CLI --tema
)
```

Notas:

- Arquitectura híbrida: embeddings para dimensiones multi-label + LLM para stance.
- Mean pooling manual (no `sentence-transformers`) para evitar embeddings degenerados con LFM2.5.
- `actor` y `tema` se pasan al constructor, no desde config: dependen de la ejecución.
- Tests marcados `@pytest.mark.slow` cargan modelos reales (LFM2.5-Embedding + LFM2.5-1.2B-Instruct).

## Componente 6: `salida`

```yaml
salida:
  formatos: ["json", "markdown"]
  incluir_provenance: true
  redondear_scores: 4
```

Equivale a:

```python
SalidaService(
    output_path="output/",  # directorio → genera informe.json + informe.md
)
```

Notas:

- Provenance obligatorio: todo informe declara modelos usados y advertencia de confianza.
- `output_path` flexible: archivo `.json`, archivo `.md`, directorio (ambos), o `None` (solo memoria).
- `redondear_scores` controla la precisión decimal en la serialización.
