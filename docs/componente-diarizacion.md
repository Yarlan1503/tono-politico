# Componente 1.5: Diarización e identificación de actor

> **Estado:** ✅ Completo · **Tests:** 88 (10 archivos)

> ⚠️ **Camino legacy.** El camino preferido es [**speech2text**](componente-speech2text.md) (`audio_fetcher` + `speaker_timestamps` + `transcribe_speech` → `ActorTranscript`). Este documento describe Diarización como aún usa `PipelineRunner` / `main.py`.


## Propósito

Inserta una etapa entre Ingesta y Segmentación para asegurar que el análisis de tono se aplique solo a las intervenciones del actor político objetivo, aunque el audio contenga varios oradores. Toma `VideoTranscript[]` + audio WAV, ejecuta diarización de pyannote, identifica al actor por embedding de voz, y devuelve `VideoTranscript[]` filtrado — mismo contrato, solo segmentos del actor.

```text
VideoTranscript[] + audio WAV
  │
  ├── Whisper large-v3-turbo → texto + WordTimestamp[]  (Ingesta, ya hecho)
  └── pyannote Community-1   → TurnoOrador[] + speaker_embeddings
                                 │
                                 ▼ construir_perfil_desde_output()  — perfil de referencia cacheado
                                 ▼ identificar_actor()              — distancia coseno por speaker
                                 ▼ filtrar_por_actor()              — midpoint dentro de turnos del actor
                                 │
                                 VideoTranscript[] (solo segmentos del actor)
                                 │
                                 ▼ Segmentación
```

## API

### `DiarizacionService`

```python
from pathlib import Path
from tono_politico.diarizacion import DiarizacionService

svc = DiarizacionService(
    actor="Lilly Téllez",
    video_ref_id="su9nURIj9XQ",
    data_dir=Path("data"),
    pipeline_name="pyannote/speaker-diarization-community-1",
    fallback_pipeline="pyannote-community/speaker-diarization-community-1",
    device="auto",
    umbral_match=0.5,
    umbral_ambiguo=0.7,
)

filtrados: list[VideoTranscript] = svc.procesar(transcripts, nombre_playlist="Play-PoliTest")
```

### Configuración

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `actor` | `str` | `"Lilly Téllez"` | Nombre del actor político objetivo |
| `video_ref_id` | `str` | `"su9nURIj9XQ"` | ID del video de referencia de voz |
| `data_dir` | `Path` | `Path("data")` | Directorio raíz de datos (mismo que IngestaService) |
| `pipeline_name` | `str` | `"pyannote/speaker-diarization-community-1"` | Pipeline primary oficial |
| `fallback_pipeline` | `str \| None` | `"pyannote-community/speaker-diarization-community-1"` | Fallback local validado si el primary falla |
| `device` | `str` | `"auto"` | `auto` usa CUDA si está disponible; si no, CPU |
| `umbral_match` | `float` | `0.5` | Distancia coseno por debajo de la cual se acepta |
| `umbral_ambiguo` | `float` | `0.7` | Distancia coseno por encima de la cual se rechaza |

### Lazy loading

pyannote `Pipeline`, `Audio` helper y el perfil de voz se cargan perezosamente en el primer `.procesar()`, no al importar el módulo. El adapter intenta primero `pyannote/speaker-diarization-community-1`, cae a `pyannote-community/speaker-diarization-community-1` si el primary falla, aplica `device=auto` y usa `ProgressHook` cuando está disponible. Community-1 entrega `output.speaker_embeddings`, por lo que no se carga un modelo separado de embeddings. Esto permite:

- Tests rápidos sin modelos pesados (mockeando el pipeline y el extractor)
- Import del paquete sin dependencias pesadas instaladas

## Módulos internos

### `service.py` — Orquestador OOP

`DiarizacionService` encapsula toda la configuración y orquesta el flujo completo. Implementa `ComponenteProtocol`.

**Flujo de `procesar()`:**

Pipeline **video-por-video** (no batch):

1. Construye el `PerfilVozActor` una sola vez (cache en memoria).
2. Resuelve el pipeline de diarización (lazy-load con fallback).
3. Por cada `VideoTranscript`:
   - **Diarizar**: `run_pyannote_pipeline(pipeline, audio_path)` → `exclusive_speaker_diarization` + `speaker_embeddings`
   - Si no hay turnos → transcript vacío, siguiente video.
   - **Embeddings por speaker**: `_extraer_embeddings(output)` lee `output.speaker_embeddings` alineado con `output.speaker_diarization.labels()` → `{speaker_id: embedding}`.
   - Si no hay embeddings → transcript vacío, siguiente video.
   - **Identificar actor**: `identificar_actor(speaker_embs, perfil)` → `SpeakerMatch[]`; los aceptados son el actor.
   - Si no hay speakers aceptados → transcript vacío, siguiente video.
   - **Filtrar**: `filtrar_por_actor(transcript, turnos_actor)` → `VideoTranscript` con solo segmentos del actor.

### `adapter.py` — Carga y ejecución del pipeline pyannote

**`load_pyannote_pipeline(primary, fallback, token, device) → LoadedPyannotePipeline`**

Carga el pipeline primary (`Pipeline.from_pretrained`); si falla, intenta el fallback. Aplica `device` con `torch.device` (no `str` — fix para pyannote 4.x). Devuelve `LoadedPyannotePipeline(pipeline, pipeline_name, used_fallback)`.

Si ambos fallan, lanza `PyannotePipelineLoadError` con mensaje accionable.

**`run_pyannote_pipeline(pipeline, audio_path, progress_hook_cls) → output`**

Ejecuta el pipeline sobre el audio. Si `ProgressHook` está disponible (auto-detectado), lo usa para reportar progreso. Si no, ejecuta directo: `pipeline(str(audio_path))`.

### `diarizacion.py` — Extracción de turnos

**`diarizar(audio_path, pipeline, video_id) → list[TurnoOrador]`**

Función pura que recibe un pipeline ya cargado (no lo instancia) y ejecuta la diarización. Usa `exclusive_speaker_diarization` —no `itertracks` estándar— para obtener turnos **sin traslapes**, ideal para alinear limpiamente con los timestamps de Whisper.

### `perfil_voz.py` — Perfil de voz del actor

**`construir_perfil_desde_output(output, actor, video_ref_id, pipeline_name) → PerfilVozActor`**

Función pura que construye el perfil desde `output.speaker_embeddings` público del pipeline. Selecciona el embedding del **speaker dominante** (mayor duración total en el audio de referencia). El resultado se aplana a `list[float]` para mantener el DTO libre de numpy. No se accede a `pipeline._inferences` ni a ningún API privado.

Lanza `ValueError` si el output no contiene embeddings o speakers.

**`construir_perfil(audio_ref, actor, video_id_ref, modelo_embedding, embedding_pipeline, audio_helper) → PerfilVozActor`**

Implementación legacy con extractor y audio_helper inyectados por separado. Mantiene compatibilidad con tests que mockean dependencias individuales.

### `matching.py` — Matching de speakers

**`distancia_coseno(a, b) → float`**

Distancia coseno (`1 - similitud`) en Python puro con `math`:

- `0.0` = vectores idénticos (mismo speaker)
- `1.0` = ortogonales
- `2.0` = opuestos (o norma cero → retorno defensivo)

No depende de numpy — usa `sum(x * y for x, y in zip(a, b, strict=True))` y `math.sqrt`.

**`clasificar_speaker(speaker_id, distancia, umbral_match=0.5, umbral_ambiguo=0.7) → SpeakerMatch`**

Clasifica con **fronteras exclusivas**:

```text
distancia < umbral_match              → aceptado
umbral_match ≤ distancia < umbral_ambiguo → ambiguo (descartar)
distancia ≥ umbral_ambiguo            → rechazado
```

**`identificar_actor(speaker_embeddings, perfil, umbral_match, umbral_ambiguo) → list[SpeakerMatch]`**

Compara cada `{speaker_id: embedding_promedio}` contra `perfil.embedding`. Devuelve la lista ordenada por distancia ascendente.

### `alineacion.py` — Alineación con Whisper

**`filtrar_por_actor(transcript, turnos_actor) → VideoTranscript`**

Conserva solo los `SegmentoRaw` cuyo **midpoint temporal** cae dentro de algún turno del actor. Usa `bisect` sobre un índice ordenado de turnos para verificación en O(log M) por segmento.

Criterio de pertenencia:

```python
midpoint = (seg.t_start + seg.t_end) / 2.0
r_start <= midpoint < r_end   # inclusivo en inicio, exclusivo en fin
```

- Turnos de otro `video_id` se ignoran.
- Devuelve un `VideoTranscript` nuevo con metadata preservada y `raw_segments` filtrado.
- Si no hay turnos del actor para ese video, devuelve transcript vacío (metadata igual, 0 segmentos).

### `transcripcion_actor.py` — Transcripción actor-only por clips

**`transcribir_turnos_actor(audio_path, turnos_actor, *, video_id, actor, transcriptor, modelo, idioma, padding, duracion_audio) → ActorTranscript`**

Transcribe los turnos pyannote atribuidos al actor objetivo, turno por turno, usando un `ClipTranscriber` inyectado por dependencias. El padding solo modifica el clip enviado al transcriptor; el contrato persistible conserva los límites originales del turno pyannote como `source_turn_*` y reubica los timestamps del ASR en el timeline absoluto con `_clamp`.

Valores constantes:
- `SCHEMA_VERSION = "actor_transcript.v1"`
- `SCOPE_ACTOR_ONLY = "actor_only"`
- `ASR_PROVIDER = "whisper"`

**`ClipTranscriber` (Protocol):**

Contrato mínimo para transcribir un rango temporal de un audio:

```python
class ClipTranscriber(Protocol):
    def transcribir_clip(
        self, audio_path: Path, *, t_start: float, t_end: float,
        modelo: str, idioma: str,
    ) -> list[ClipTranscriptSegment]: ...
```

**`ClipTranscriptSegment`:** DTO interno con `text`, `t_start`, `t_end` relativos al inicio del clip.

### `whisper_clip.py` — Adaptador Whisper + ffmpeg

**`WhisperFfmpegClipTranscriber`**

Implementa `ClipTranscriber` usando ffmpeg para recortar el clip temporal y Whisper para transcribir. El archivo temporal se normaliza a PCM WAV mono 16 kHz (`-ac 1 -ar 16000 -c:a pcm_s16le`), el formato base de Whisper. El temporal se limpia en `finally`.

- Carga modelos Whisper perezosamente y los cachea por nombre (`self._models`).
- `word_timestamps=False` (turn-level, no word-level).
- `fp16=False` (CPU-friendly).
- `model_loader` y `runner` inyectables para tests.

### `actor_transcript.py` — Serialización actor_transcript.v1

Funciones de serialización para el contrato JSON `actor_transcript.v1`:

- `actor_transcript_to_dict(transcript) → dict`
- `actor_transcript_to_json(transcript) → str` (compacto, UTF-8 friendly)
- `actor_transcript_from_json(json_str) → ActorTranscript`
- `guardar_actor_transcript(transcript, path) → Path`
- `cargar_actor_transcript(path) → ActorTranscript`

Formato de segmento en JSON:

```json
{
  "text": "...",
  "t_start": 12.5,
  "t_end": 18.3,
  "speaker": "SPEAKER_00",
  "source_turn": {"t_start": 12.2, "t_end": 18.5},
  "word_count": 23
}
```

## DTOs

Definidos en `src/tono_politico/diarizacion/models.py`:

### `TurnoOrador`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `video_id` | `str` | ID del video de YouTube |
| `speaker_id` | `str` | Etiqueta de pyannote (`SPEAKER_00`, ...) |
| `t_start` | `float` | Inicio del turno (segundos) |
| `t_end` | `float` | Fin del turno (segundos) |

### `PerfilVozActor`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `actor` | `str` | Nombre del actor político objetivo |
| `video_id_referencia` | `str` | ID del video usado como referencia |
| `embedding` | `list[float]` | Embedding promedio del audio de referencia (aplanado, sin numpy) |
| `modelo_embedding` | `str` | Identificador del origen (`speaker_embeddings:<pipeline_name>`) |
| `duracion_segundos` | `float` | Duración del audio de referencia procesado |

### `SpeakerMatch`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `speaker_id` | `str` | Etiqueta del speaker evaluado |
| `distancia` | `float` | Distancia coseno al perfil del actor |
| `aceptado` | `bool` | `True` si se acepta como el actor objetivo |
| `es_ambiguo` | `bool` | `True` si el match cae en zona ambigua (descartar) |

### `AsrMetadata`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `provider` | `str` | Proveedor ASR (`whisper`) |
| `model` | `str` | Modelo ASR (`large-v3-turbo`) |
| `language` | `str` | Idioma configurado (`es`) |

### `ActorTranscriptSegment`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `text` | `str` | Texto transcrito del turno |
| `t_start` | `float` | Inicio absoluto en el timeline del video (segundos) |
| `t_end` | `float` | Fin absoluto en el timeline del video (segundos) |
| `speaker` | `str` | Etiqueta pyannote del speaker |
| `source_turn_start` | `float` | Inicio original del turno pyannote |
| `source_turn_end` | `float` | Fin original del turno pyannote |
| `word_count` | `int` | Número de palabras transcritas |

### `ActorTranscript`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `schema_version` | `str` | Versión del contrato (`actor_transcript.v1`) |
| `video_id` | `str` | ID del video de YouTube |
| `actor` | `str` | Nombre del actor objetivo |
| `scope` | `str` | Alcance (`actor_only`) |
| `asr` | `AsrMetadata` | Metadatos del motor ASR |
| `segments` | `list[ActorTranscriptSegment]` | Turnos transcritos del actor |

## Decisiones de diseño

### `exclusive_speaker_diarization` para alineación limpia

Se usa `output.exclusive_speaker_diarization` en vez del `itertracks` estándar. Esto produce turnos **sin traslapes**: cada instante del audio pertenece a un solo speaker. Sin traslapes, el criterio de midpoint en `filtrar_por_actor` no tiene ambigüedad — un segmento de Whisper siempre cae dentro de un único turno.

### Pipeline cargado una vez y embeddings nativos del output

`DiarizacionService` hace lazy-loading de pyannote (`_get_pipeline`) y construye el perfil de referencia desde el mismo `output.speaker_embeddings` público del pipeline. Para cada video usa una sola llamada al pipeline y extrae turnos desde `exclusive_speaker_diarization` y embeddings desde `output.speaker_embeddings`; esto evita cargar un modelo separado o recurrir a APIs privadas, y mantiene los contratos serializables.

### `torch.device` en vez de `str` (fix pyannote 4.x)

pyannote 4.x rompe al recibir `pipeline.to("cuda")` como string — requiere `torch.device`. El adapter `_resolve_device()` siempre devuelve `torch.device(...)`, nunca `str`.

### Speaker dominante para perfil de voz

`construir_perfil_desde_output` selecciona el speaker con mayor duración total en el audio de referencia como perfil del actor. En un video de referencia bien escogido (donde el actor habla la mayor parte del tiempo), esto es robusto sin necesidad de etiquetado manual.

### Criterio midpoint con bisect O(log M) en alineación

Para decidir si un `SegmentoRaw` pertenece al actor, se calcula `midpoint = (t_start + t_end) / 2` y se usa `bisect_right` sobre los starts ordenados de los turnos, luego se verifica `midpoint < r_end`. La frontera es semiabierta `[inicio, fin)`: inclusiva en el `t_start` del turno, exclusiva en `t_end`.

### `distancia_coseno` sin numpy

Implementada en Python puro con `math` (`zip(strict=True)` + `math.sqrt`). Ninguna función pura del componente importa numpy — solo el adapter/extractor interno convierte arrays del pipeline a `list[float]` antes de salir del service.

### Embedding aplanado a `list[float]`

`PerfilVozActor.embedding` y los embeddings por speaker son `list[float]`, no `np.ndarray`. El service convierte la salida del pipeline con `.tolist()` antes de entregarla a helpers/DTOs. Esto mantiene los contratos serializables y libres de numpy entre capas.

### Thresholds calibrados con research (0.5 aceptar · 0.7 ambiguo)

Los thresholds por defecto (`umbral_match=0.5`, `umbral_ambiguo=0.7`) están calibrados con base en documentación/literatura de diarización y un smoke real del pipeline:

- El clustering threshold de **pyannote 3.1** es **0.7046** (distancia coseno); `umbral_ambiguo=0.7` queda alineado con ese punto de operación.
- **SpeechBrain ECAPA-TDNN** usa threshold de similitud ~0.25 (distancia ~0.75); la zona ambigua absorbe ese rango.
- En el smoke inicial de Play-PoliTest, 3 videos reales produjeron distancias `0.075–0.131`, muy por debajo de `umbral_match=0.5`.
- En el smoke Fase 1 `politest-smoke-device-fix`, 7/7 videos de Play-PoliTest fueron procesados con éxito: 139 segmentos del actor, 2 tópicos descubiertos, 0 videos omitidos.

Fuentes consultadas:

- pyannote.audio 3.1 — clustering threshold 0.7046
- SpeechBrain ECAPA-TDNN — threshold de similitud 0.25
- Community discussions (Hugging Face forums, pyannote-audio issues)
- Smoke local Play-PoliTest — distancias 0.075–0.131 en 3 videos reales y validación Fase 1 completa sobre 7 videos

### Match ambiguo: descartar y continuar

Si el mejor match cae en zona ambigua, se descarta como no-actor. No se pide selección manual. Si ningún speaker es aceptado, el video produce un `VideoTranscript` vacío (metadata preservada, 0 segmentos) y el pipeline continúa con el siguiente video.

### Transcripción por clips con ffmpeg

`WhisperFfmpegClipTranscriber` recorta cada turno pyannote a un WAV temporal normalizado (mono 16 kHz PCM) con ffmpeg, lo transcribe con Whisper, y limpia el temporal. Los timestamps del ASR se reubican al timeline absoluto del video con `_clamp` a los límites del turno original. Los límites pyannote se conservan como `source_turn_*`.

### `word_timestamps=False` en transcripción por clips

La transcripción actor-only es turn-level: no persiste timestamps por palabra, probabilidades por palabra ni datos verbose de Whisper. La segmentación temática ocurre en componentes posteriores. Esto alinea con la decisión `actor_transcript.v1` de no incluir words/probability/pausa_antes/verbose.

### Salida hacia Segmentación

El contrato de salida de `DiarizacionService.procesar()` es `list[VideoTranscript]` —mismo tipo que la entrada— con `raw_segments` filtrado a solo los del actor. No se introducen DTOs nuevos en la frontera con Segmentación; el `VideoTranscript` ya existe y se reutiliza.

El contrato `ActorTranscript` (`actor_transcript.v1`) es una representación alternativa turn-level para persistencia y análisis futuro, pero la frontera operacional del pipeline usa `VideoTranscript`.

## Dependencias externas

| Herramienta | Uso |
|-------------|-----|
| `pyannote.audio` + `pyannote/speaker-diarization-community-1` | Primary oficial de diarización + `speaker_embeddings` por speaker |
| `pyannote-community/speaker-diarization-community-1` | Fallback local validado si el primary no está disponible |
| `torch` | Resolución de device (`torch.device`) para pyannote 4.x |
| `numpy` | Manipulación interna de embeddings (dentro del service, no en DTOs) |
| `openai-whisper` | Transcripción de clips del actor (`WhisperFfmpegClipTranscriber`) |
| `ffmpeg` | Recorte de clips temporales normalizados a mono 16 kHz PCM |
| GPU recomendada | pyannote es lento en CPU; GPU acelera diarización |

## Notas de implementación

- El `PerfilVozActor` **no se persiste** en disco — el cache es solo en memoria durante la ejecución del pipeline.
- El `DiarizacionService` resuelve rutas de audio con `ruta_audio()` del Componente 1 (Ingesta), asumiendo que los `.wav` ya fueron descargados.
- El token de Hugging Face se lee desde `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env var o `~/.cache/huggingface/token`.
- **Smoke Fase 1 real completado:** perfil + matching + segmentación + temas fueron validados en `politest-smoke-device-fix` sobre los 7 videos de `Play-PoliTest`; 7/7 procesados, 139 segmentos del actor, 2 tópicos descubiertos y 0 videos omitidos. El fallo previo de pyannote 4.x (`pipeline.to(str)`) se corrigió pasando `torch.device` al pipeline.
