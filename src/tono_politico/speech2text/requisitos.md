# AudioFetcher — Requisitos

## Contexto

El Componente 1 actual (Ingesta) hace dos cosas en un solo service:

1. **Descarga de audio + metadata** — playlist info con yt-dlp, descarga de `.wav`, cache de dos niveles (audios + transcripciones).
2. **Transcripción completa con Whisper** — `word_timestamps=True`, `fp16=False`, produce `VideoTranscript` con `SegmentoRaw[]` (texto + timestamps + words + pausa_antes + probability por palabra).

El problema: la arquitectura actor-first del pipeline demuestra que **la transcripción del video completo es desperdicio**. El pipeline diariza con pyannote, identifica al actor, filtra `SegmentoRaw[]` por midpoint contra turnos del actor, y todo lo demás se descarta. La transcripción por clips (`transcribir_turnos_actor` + `WhisperFfmpegClipTranscriber`) ya existe en el módulo de diarización y transcribe **solo los turnos del actor**, que es lo único que importa.

### Qué se desperdicia hoy

- **Compute de Whisper sobre audio completo** cuando solo se necesita el ~30-50% que es del actor.
- **JSON inflado**: `VideoTranscript.raw_segments` incluye `words: list[WordTimestamp]` con probability por palabra para TODOS los segmentos, incluyendo los de otros speakers que se descartan en Diarización.
- **Acoplamiento innecesario**: Ingesta necesita saber de Whisper (`import whisper`, `load_model`, `fp16`, `word_timestamps`) cuando su rol natural es solo traer el audio.

### Decisión de diseño (ya acordada en sesiones previas)

> *"Turn-level is ideal, because only the target actor's participations matter; diarize/identify actor first, transcribe only actor turns."*

AudioFetcher remplaza a Ingesta asumiendo **solo la descarga de audio y metadata**. La transcripción se mueve al componente de Diarización/Actor (donde ya existe `transcribir_turnos_actor`).

## Objetivo

Separar la responsabilidad de **obtención de audio** de la de **transcripción**, eliminando la transcripción full-video de Ingesta y el JSON inflado de `VideoTranscript`.

## Frontera del componente

### AudioFetcher hace

- [ ] Obtener metadata de la playlist (nombre, videos) vía `yt-dlp --flat-playlist`
- [ ] Descargar audio de cada video como `.wav` vía `yt-dlp -x --audio-format wav`
- [ ] Cachear audios `.wav` con validación (`--download-archive`, `--retries 10`)
- [ ] Devolver metadata + rutas de audio como `list[AudioVideo]`

### AudioFetcher NO hace

- [ ] ~~`transcribir()` con Whisper~~ → se elimina; la transcripción pasa a Diarización/Actor (`transcribir_turnos_actor`)
- [ ] ~~`guardar_transcripcion()` / `cargar_transcripcion()`~~ → se elimina; la transcripción es turn-level, no full-video
- [ ] ~~`verificar_cache_transcripciones()`~~ → se elimina
- [ ] ~~`VideoTranscript.raw_segments` con `WordTimestamp[]`~~ → se reemplaza por `ActorTranscript` turn-level

### Cambio en el contrato del pipeline

```text
ANTES (batch por fase):
  Ingesta(playlist) → [VideoTranscript×N]
  Diarizacion([VT×N]) → [VT filtrado×N]
  Segmentacion([VT×N]) → [Segmento…]
  Temas([Segmento…]) → ResultadoTemas

AHORA (Fase 1 = loop video-por-video + temas al final):
  discover(playlist) → PlaylistInfo + [VideoMeta×N]
  para cada video:
      fetch_one(VideoMeta) → AudioVideo | omitir
      diarizar_one(AudioVideo) → ActorTranscript | omitir
      segmentar_one(ActorTranscript) → list[Segmento]
      (opcional) borrar .wav si no --keep-cache
  Temas(todos los Segmento) → ResultadoTemas   # fuera del loop
```

### Orquestación Fase 1 (decisión)

La **Fase 1** agrupa la lógica de **descarga → diarización → segmentación** y se ejecuta **video por video**, no en tres batches globales.

| Subpaso | Dueño | Unidad | Salida |
|---|---|---|---|
| 1. Mapear playlist | `AudioFetcher` (`playlist.py`) | playlist | `PlaylistInfo` + `list[VideoMeta]` |
| 2. Descargar audio | `AudioFetcher` (`audio.py` / `fetch_one`) | 1 video | `AudioVideo` o skip |
| 3. Diarizar + ASR actor | `DiarizacionService` | 1 video | `ActorTranscript` o skip |
| 4. Segmentar | `SegmentacionService` | 1 video | `list[Segmento]` |
| 5. Descubrir temas | `TemasService` | **todos** los segmentos | `ResultadoTemas` |

**Por qué el loop para en segmentación y no en temas:**

- Descarga, diarización y segmentación son **por video** (audio local, speakers locales, oraciones locales).
- Temas (BERTopic) necesita el **corpus completo** de segmentos de la playlist para clusterizar; no tiene sentido correrlo N veces dentro del loop.

**Quién orquesta el loop:** `PipelineRunner` (Fase 1), **no** `AudioFetcherService`.
AudioFetcher se queda delgado: `discover` + `fetch_one`. Diarización y Segmentación exponen procesamiento **por unidad**.

**Cache delgado en el loop:**

```text
data/<playlist>/videos-<playlist>/<video_id>.wav   # temporal por video
# tras diarizar+segmentar con éxito, borrar .wav salvo --keep-cache
# durable opcional: ActorTranscript / segmentos en output/runs/<run_id>/
```

Beneficios: pico de disco ≈ 1 WAV; fallos aislados por `video_id`; manifest por video natural; alineado con actor-first.

## Contrato de entrada/salida

### API de `AudioFetcherService` (orientada al loop)

```python
AudioFetcherService(data_dir=Path("data"))

# Mapa de la playlist (sin descargar audio)
discover(url_playlist: str) -> tuple[PlaylistInfo, list[VideoMeta]]

# Una unidad del loop Fase 1
fetch_one(
    video: VideoMeta,
    nombre_playlist: str,
    *,
    archive_path: Path | None = None,
) -> AudioVideo | None
# None si la descarga falla (el runner marca skip y sigue)

# Wrapper opcional de conveniencia (tests / uso ad-hoc):
# procesar(url) = discover + fetch_one para cada meta → list[AudioVideo]
# No es el camino del PipelineRunner en producción.
```

### Salida — `AudioVideo` (nuevo DTO)

| Campo | Tipo | Descripción |
|---|---|---|
| `video_id` | `str` | ID del video de YouTube |
| `url` | `str` | URL del video |
| `titulo` | `str` | Título del video |
| `fecha` | `str \| None` | Fecha YYYYMMDD |
| `audio_path` | `Path` | Ruta al `.wav` descargado y cacheado |
| `duracion` | `float` | Duración en segundos (metadata de yt-dlp) |

### `PlaylistInfo` y `VideoInfo` (slim-down en `models.py` compartido)

**`PlaylistInfo`** — solo se queda con:

| Campo | Tipo | Descripción |
|---|---|---|
| `nombre` | `str` | Nombre sanitizado de la playlist |

(`url` y `videos` se eliminan — la URL ya se pasó al service, y la lista de videos vive en `list[AudioVideo]`.)

**`VideoInfo`** — solo se queda con:

| Campo | Tipo | Descripción |
|---|---|---|
| `titulo` | `str` | Título del video |
| `fecha` | `str \| None` | Fecha YYYYMMDD |

(`id`, `url`, `duracion` migran a `AudioVideo`.)

## Módulos del componente `audio_fetcher`

> Ruta canónica: `src/tono_politico/speech2text/audio_fetcher/`.
> Requisitos del umbrella: `src/tono_politico/speech2text/requisitos.md`.

```text
src/tono_politico/speech2text/audio_fetcher/
├── __init__.py            # API pública del subpaquete
├── models.py              # DTOs del componente
├── cache.py               # Rutas del cache de audios
├── playlist.py            # Metadata de playlist (yt-dlp --flat-playlist)
├── audio.py               # Cache + descarga de .wav
└── service.py             # AudioFetcherService
```

**No existe** `transcripcion.py` aquí (ASR actor-only vive en `speech2text/transcribe_speech/`).

### Por qué un DTO intermedio (`VideoMeta`)

Con el slim-down:

- `PlaylistInfo` solo tiene `nombre`
- `VideoInfo` solo tiene `titulo` + `fecha`
- `AudioVideo` es la **salida final** (incluye `audio_path`)

La descarga necesita `video_id` / `url` / `duracion` **antes** de tener `.wav`.
Eso no cabe en `VideoInfo` slim ni en `AudioVideo` (que ya exige `audio_path`).

Por tanto el componente usa un DTO interno de descubrimiento:

| DTO | Rol |
|---|---|
| `VideoMeta` | Metadata pre-descarga (interno del componente) |
| `AudioVideo` | Metadata + `audio_path` (salida pública de `procesar`) |
| `DownloadResult` | Resultado estructurado de una descarga |
| `PlaylistInfo` | Solo `nombre` (compartido, en `tono_politico.models`) |

---

### 1. `models.py` — DTOs del componente

```python
@dataclass(frozen=True)
class VideoMeta:
    video_id: str
    url: str
    titulo: str
    fecha: str | None
    duracion: float  # segundos; 0.0 si yt-dlp no la reporta

@dataclass(frozen=True)
class AudioVideo:
    video_id: str
    url: str
    titulo: str
    fecha: str | None
    audio_path: Path
    duracion: float

@dataclass(frozen=True)
class DownloadResult:
    video_id: str
    path: Path | None
    ok: bool
    error: str | None = None
```

| DTO | Público | Notas |
|---|---|---|
| `AudioVideo` | sí | Contrato de salida del service |
| `DownloadResult` | sí (tests/debug) | Migrado desde `ingesta/models.py` |
| `VideoMeta` | interno / tests | Alimenta cache + descarga; no es salida del pipeline |

Helper opcional (puede vivir en `models.py` o en `service.py`):

```python
def audio_video_from(meta: VideoMeta, audio_path: Path) -> AudioVideo: ...
```

---

### 2. `cache.py` — rutas de cache (solo audios)

Migrado desde `ingesta/cache.py`, **sin** rutas de transcripción.

| Función / constante | Firma | Responsabilidad |
|---|---|---|
| `DATA_DIR` | `Path("data")` | Default del cache runtime |
| `ruta_dir_videos` | `(nombre_playlist, base_dir=None) -> Path` | `data/<playlist>/videos-<playlist>/` |
| `ruta_audio` | `(nombre_playlist, video_id, base_dir=None) -> Path` | `.../<video_id>.wav` |

**Eliminado de este módulo:**

- `ruta_dir_transcripciones`
- `ruta_transcripcion`

Estructura en disco:

```text
data/
└── <playlist>/
    └── videos-<playlist>/
        └── <video_id>.wav
```

---

### 3. `playlist.py` — metadata de playlist

Migrado desde `ingesta/playlist.py`, adaptado al slim-down.

| Función | Firma | Responsabilidad |
|---|---|---|
| `sanitizar_nombre_directorio` | `(nombre: str) -> str` | Nombre seguro para filesystem |
| `obtener_info_playlist` | `(url: str) -> tuple[PlaylistInfo, list[VideoMeta]]` | yt-dlp `--flat-playlist` + JSON lines |

Comportamiento:

1. Ejecuta `yt-dlp --flat-playlist --extractor-args youtubetab:approximate_date -j`.
2. Extrae nombre de playlist → `sanitizar_nombre_directorio` → `PlaylistInfo(nombre=...)`.
3. Por cada video accesible construye `VideoMeta(video_id, url, titulo, fecha, duracion)`.
4. Omite entradas sin `id` (privados/eliminados ya filtrados por yt-dlp).
5. Si yt-dlp falla → `RuntimeError`.

**Cambio vs Ingesta:** ya no devuelve `PlaylistInfo(nombre, url, videos=[VideoInfo...])`.
La URL de playlist no se guarda en el DTO (ya entró al service).

---

### 4. `audio.py` — cache y descarga de `.wav`

Migrado desde `ingesta/audio.py`, firmas sobre `VideoMeta` (no `VideoInfo`).

| Función | Firma | Responsabilidad |
|---|---|---|
| `verificar_cache_videos` | `(nombre_playlist, videos: list[VideoMeta], base_dir=None) -> dict[str, list[VideoMeta]]` | `{"existentes", "faltantes"}` según existencia del `.wav` |
| `descargar_audio_result` | `(video: VideoMeta, nombre_playlist, base_dir=None, archive_path=None) -> DownloadResult` | yt-dlp `-x --audio-format wav`, no crashea |
| `descargar_audio` | misma firma → `Path \| None` | Wrapper legacy opcional; preferir `descargar_audio_result` |

Parámetros yt-dlp a conservar:

- `-x --audio-format wav -f bestaudio/best`
- `--retries 10`
- timeout 600s
- `--download-archive` opcional vía `archive_path`

---

### 5. `service.py` — `AudioFetcherService`

Orquestador OOP delgado; **no** diariza ni segmenta.

| Miembro | Firma | Responsabilidad |
|---|---|---|
| `__init__` | `(data_dir: Path = Path("data"))` | Solo config de cache; **sin** Whisper |
| `discover` | `(url_playlist: str) -> tuple[PlaylistInfo, list[VideoMeta]]` | Mapa de la playlist (delega en `playlist.py`) |
| `fetch_one` | `(video: VideoMeta, nombre_playlist, *, archive_path=None) -> AudioVideo \| None` | Cache hit o descarga de un video; `None` si falla |
| `procesar` | `(url_playlist: str) -> list[AudioVideo]` | Wrapper opcional: discover + fetch_one×N (tests/ad-hoc) |

Flujo de `fetch_one` (unidad del loop Fase 1):

```text
1. path = ruta_audio(nombre_playlist, video.video_id, data_dir)
2. si path existe → AudioVideo(..., audio_path=path)
3. si no → descargar_audio_result(video, ...)
4. si ok → AudioVideo(..., audio_path=result.path)
5. si no → None (runner marca skip)
```

Flujo de `procesar` (wrapper, no es el camino del runner):

```text
1. playlist, videos = discover(url)
2. para cada meta: fetch_one → acumular si no None
3. devolver list[AudioVideo]
```

**No hace:** Whisper, diarización, segmentación, JSON de transcripción, loop de Fase 1.

---

### 6. `__init__.py` — exports públicos

```python
from .models import AudioVideo, DownloadResult
from .service import AudioFetcherService

__all__ = ["AudioFetcherService", "AudioVideo", "DownloadResult"]
```

`VideoMeta` puede exportarse solo si hace falta en tests; no es contrato del pipeline.

---

### Dependencia entre módulos

```text
service.py
  ├── playlist.py  → models.VideoMeta, tono_politico.models.PlaylistInfo
  ├── audio.py     → models.VideoMeta, models.DownloadResult, cache.*
  ├── cache.py
  └── models.py    → AudioVideo, VideoMeta, DownloadResult
```

Helpers puros en `playlist` / `audio` / `cache`; el service solo compone.

---

### Qué NO forma parte de `audio_fetcher`

| Archivo / concepto | Destino |
|---|---|
| `transcripcion.py` (Whisper full-video) | Eliminado del componente 1 |
| `guardar/cargar_transcripcion` | Eliminado |
| `verificar_cache_transcripciones` | Eliminado |
| `SegmentoRaw` / `WordTimestamp` / `VideoTranscript` | Eliminados de `models.py` compartido (fase posterior del checklist) |
| Transcripción actor-only | Se queda en `diarizacion/` |

## Checklist de implementación

### DTOs del componente

- [x] Crear `VideoMeta` en `audio_fetcher/models.py` (video_id, url, titulo, fecha, duracion)
- [x] Crear `AudioVideo` en `audio_fetcher/models.py` (video_id, url, titulo, fecha, audio_path, duracion)
- [x] Crear `DownloadResult` en `audio_fetcher/models.py` (migración desde ingesta pendiente de cableado)
- [ ] Slim-down `PlaylistInfo` en `models.py` compartido: solo `nombre`
- [ ] Slim-down `VideoInfo` en `models.py` compartido: solo `titulo` y `fecha`
- [ ] Eliminar `SegmentoRaw` de `models.py` compartido
- [ ] Eliminar `WordTimestamp` de `models.py` compartido
- [ ] Eliminar `VideoTranscript` de `models.py` compartido

### Umbrella `speech2text/`

- [x] Crear paquete `src/tono_politico/speech2text/`
- [x] `SpeechToTextService`: `discover` + `procesar_one(VideoMeta) -> ActorTranscript | None`
- [x] Tres subpaquetes: `audio_fetcher/`, `speaker_timestamps/`, `transcribe_speech/`
- [x] Exports: `SpeechToTextService`, `AudioVideo`, `ActorTranscript`, …
- [x] Segmentación y Temas **no** viven en `speech2text`

### Módulos `speech2text/audio_fetcher/`

- [x] `cache.py`: `DATA_DIR`, `ruta_dir_videos`, `ruta_audio` (sin rutas de transcripción)
- [x] `playlist.py`: `sanitizar_nombre_directorio` + `obtener_info_playlist(url) -> tuple[PlaylistInfo, list[VideoMeta]]`
- [x] `audio.py`: `verificar_cache_videos`, `descargar_audio_result` (sobre `VideoMeta`)
- [x] `service.py`: `discover`, `fetch_one` (API principal); `procesar` opcional como wrapper
- [x] `__init__.py`: exports `AudioFetcherService`, `AudioVideo`, `DownloadResult`, `VideoMeta`
- [ ] Eliminar `ingesta/transcripcion.py` completo
- [ ] Eliminar `ingesta/service.py` completo
- [ ] Eliminar paquete `ingesta/` cuando todo esté migrado

### Módulos `speech2text/speaker_timestamps/`

- [ ] Migrar desde `diarizacion/`: `adapter`, `diarizacion`, `perfil_voz`, `matching`, DTOs de speakers
- [x] API por unidad: `AudioVideo` → `list[TurnoOrador]` (solo turnos del actor) o DTO equivalente
- [x] Perfil de voz: construir una vez por corrida; reutilizar en el loop
- [ ] Eliminar `filtrar_por_actor` y `alineacion.py` (no hay `SegmentoRaw` que filtrar)
- [x] Resolver audio desde `AudioVideo.audio_path`

### Módulos `speech2text/transcribe_speech/`

- [ ] Migrar: `whisper_clip`, `transcripcion_actor`, `actor_transcript`, DTOs `ActorTranscript*`
- [x] API por unidad: `(AudioVideo, turnos_actor) → ActorTranscript | None`
- [x] Whisper `large-v3-turbo`, turn-level, **sin** word-level / probability / `pausa_antes`
- [ ] Persistencia `actor_transcript.v1` (texto, t_start/t_end, source_turn/speaker, word_count opcional)

### Segmentación — adaptar entrada (por video)

- [ ] API por unidad: entrada `ActorTranscript` → salida `list[Segmento]`
- [ ] Construir `Oracion` desde `ActorTranscriptSegment.text` (sin `WordTimestamp` / `pausa_antes`)
- [ ] Eliminar dependencia de `VideoTranscript` / `SegmentoRaw`

### PipelineRunner — Fase 1 video-por-video

- [ ] `discover` playlist → `PlaylistInfo` + `list[VideoMeta]`
- [ ] Loop por video: `speech2text.procesar_one` (fetch → speakers → ASR) → `segmentar_one`
- [ ] Acumular `Segmento[]` de todos los videos exitosos
- [ ] Tras cada video exitoso: borrar `.wav` salvo `--keep-cache`
- [ ] Tras el loop: `TemasService.procesar(todos_los_segmentos)` → `ResultadoTemas`
- [ ] Manifest por video: ok/skip/failed + fase en la que falló
- [ ] Mantener `fase1-topicos.json` + `--resume` sobre el agregado de temas

### main.py / factories / config

- [ ] Renombrar `_build_ingesta` → `_build_speech2text` (compone los 3 subservicios)
- [ ] Config YAML: `audio_fetcher` + `speaker_timestamps` + `transcribe_speech` (Whisper solo en este último)
- [ ] Actualizar `config.py` dataclasses
- [ ] Actualizar `_service_factories()` en `main.py`

### Documentación

- [ ] Actualizar `AGENTS.md`: estructura, estado por componente, arquitectura OOP
- [ ] Actualizar `README.md`: pipeline, tabla de estado, estructura, uso programático
- [ ] Actualizar `docs/configuracion.md`: secciones `speech2text` / subpaquetes
- [ ] Reemplazar `docs/componente-ingesta.md` → docs de `speech2text` / `audio_fetcher`
- [ ] Reemplazar `docs/componente-diarizacion.md` → `speaker_timestamps` + `transcribe_speech`

## Empaquetado: `speech2text` (esqueleto)

Umbrella **audio → `ActorTranscript` turn-level**. Tres subpaquetes con fronteras claras.
**No** incluye segmentación ni temas.

```text
src/tono_politico/speech2text/
├── __init__.py                      # API pública del umbrella
├── service.py                       # SpeechToTextService — compone los 3
├── requisitos.md                    # (mover desde audio_fetcher/requisitos.md)
│
├── audio_fetcher/                   # I/O: playlist + descarga .wav
│   ├── __init__.py
│   ├── models.py                    # VideoMeta, AudioVideo, DownloadResult
│   ├── cache.py                     # DATA_DIR, ruta_dir_videos, ruta_audio
│   ├── playlist.py                  # sanitizar + obtener_info_playlist
│   ├── audio.py                     # verificar_cache + descargar_audio_result
│   └── service.py                   # AudioFetcherService (discover, fetch_one)
│
├── speaker_timestamps/              # Quién habla cuándo + match del actor
│   ├── __init__.py
│   ├── models.py                    # TurnoOrador, PerfilVozActor, SpeakerMatch
│   ├── adapter.py                   # load/run pyannote (primary/fallback, device)
│   ├── diarizacion.py               # exclusive_speaker_diarization → TurnoOrador[]
│   ├── perfil_voz.py                # construir_perfil_desde_output (speaker dominante)
│   ├── matching.py                  # distancia_coseno, clasificar, identificar_actor
│   └── service.py                   # AudioVideo → turnos del actor (sin texto)
│
└── transcribe_speech/               # ASR actor-only (Whisper large-v3-turbo)
    ├── __init__.py
    ├── models.py                    # ActorTranscript, ActorTranscriptSegment, AsrMetadata
    ├── whisper_clip.py              # WhisperFfmpegClipTranscriber (ffmpeg + Whisper)
    ├── transcripcion_actor.py       # ClipTranscriber + transcribir_turnos_actor
    ├── actor_transcript.py          # serialización actor_transcript.v1
    └── service.py                   # (AudioVideo, turnos_actor) → ActorTranscript
```

### Responsabilidades por subpaquete

| Subpaquete | Hace | No hace |
|---|---|---|
| `audio_fetcher` | Mapear playlist, descargar/cachear `.wav` | Whisper, pyannote, texto |
| `speaker_timestamps` | pyannote, perfil de voz, match actor, turnos | Descarga YouTube, ASR, segmentación semántica |
| `transcribe_speech` | Whisper sobre clips del actor, `ActorTranscript` | Match de actor, yt-dlp, spaCy/BERTopic |

### Contratos entre los 3

```text
audio_fetcher:
  discover(url) → (PlaylistInfo, list[VideoMeta])
  fetch_one(VideoMeta, nombre_playlist) → AudioVideo | None

speaker_timestamps:
  procesar_one(AudioVideo) → list[TurnoOrador]  # solo turnos del actor
  # perfil de voz: lazy, 1× por corrida (video_ref)

transcribe_speech:
  procesar_one(AudioVideo, turnos_actor) → ActorTranscript | None
  # Whisper large-v3-turbo, turn-level, sin word-level
```

### Flujo de `SpeechToTextService.procesar_one`

```text
1. audio  = audio_fetcher.fetch_one(meta, nombre_playlist)
   └─ None → return None
2. turnos = speaker_timestamps.procesar_one(audio)
   └─ vacío / error → return None
3. actor_tx = transcribe_speech.procesar_one(audio, turnos)
   └─ return ActorTranscript | None
```

### API del orquestador

```python
SpeechToTextService(...)

discover(url) -> tuple[PlaylistInfo, list[VideoMeta]]

# Unidad del loop (camino principal)
procesar_one(video: VideoMeta, nombre_playlist: str) -> ActorTranscript | None
# por dentro: fetch_one → speaker_timestamps → transcribe_speech
# None = skip (fallo de descarga / sin actor / error ASR)

# Wrapper opcional
procesar(url) -> list[ActorTranscript]  # tests / ad-hoc; no es el runner de prod
```

### Relación con Fase 1

```text
Fase 1 (PipelineRunner):
  playlist, metas = speech2text.discover(url)
  for meta in metas:
      actor_tx = speech2text.procesar_one(meta, playlist.nombre)
      if actor_tx is None: skip; continue
      segs = segmentacion.procesar_one(actor_tx)  # FUERA de speech2text
      # borrar .wav salvo --keep-cache
  Temas(all segs)                                 # FUERA de speech2text
```

### Persistencia (`actor_transcript.v1`)

- **Sí:** texto, t_start/t_end, source_turn/speaker, word_count opcional, metadata ASR.
- **No:** word-level timestamps, probability por palabra, `pausa_antes`, captions de YouTube, verbose Whisper, full-video `VideoTranscript`.

### Qué se elimina al migrar

| Hoy | Destino |
|---|---|
| `ingesta/transcripcion.py` | ❌ |
| `ingesta/*` (resto) | → `speech2text/audio_fetcher/` |
| `diarizacion/alineacion.py` | ❌ |
| `diarizacion/` (speakers/match) | → `speech2text/speaker_timestamps/` |
| `diarizacion/` (Whisper clips + ActorTranscript) | → `speech2text/transcribe_speech/` |

---


## Viabilidad y precedentes (investigación)

Evaluación contra documentación oficial y proyectos similares (2026-07).

### Documentación oficial

| Pieza | Evidencia | Implicación |
|---|---|---|
| **pyannote Community-1** | Card HF + blog: `exclusive_speaker_diarization` pensado para reconciliar diarización con STT (Whisper). Entrada mono 16 kHz. | `speaker_timestamps` como capa **sin texto** es el diseño oficial. Preferir exclusive turns. |
| **Whisper** | `transcribe(..., word_timestamps=False)` produce segmentos turn/utterance. `word_timestamps=True` activa alineación extra y puede cambiar el texto. | Clips actor-only con `word_timestamps=False` y schema turn-level (`actor_transcript.v1`) son correctos. |
| **yt-dlp** | Pipeline extractor → downloader → postprocessors (audio). Sin acoplamiento a ASR. | `audio_fetcher` aislado es el diseño natural. |

### Dos arquitecturas en la comunidad

| Enfoque | Orden | Ejemplos | ¿Para tono-politico? |
|---|---|---|---|
| **A. ASR full → diarize → assign** | Whisper(X) todo el audio, luego pyannote, merge word↔speaker | WhisperX | No ideal: ASR de todos los speakers + empuja word-level |
| **B. Diarize → slice → ASR por turno** | pyannote “quién cuándo”, luego Whisper en cada clip | speechlib, faster-whisper + pyannote | **Sí** — patrón adoptado |

Community consensus (faster-whisper discussions): *pyannote timestamps slice audio; Whisper transcribes segments — don't let Whisper segment independently for speaker-attributed text.*

### Qué validamos de nuestra propuesta

- Split `audio_fetcher` / `speaker_timestamps` / `transcribe_speech`: **viable y alineado a docs**.
- Orden B (speakers primero, ASR después): **mejor que WhisperX monobloque** para actor-only + identidad.
- Match de embeddings (enrollment / video_ref + umbrales): stack open-source estándar (identification ≠ diarization).
- Loop video-por-video + cache delgado: viable; reduce pico de disco.
- Segmentación/Temas fuera de `speech2text`: correcto (NLP ≠ S2T).

### Qué no adoptar como núcleo

- **WhisperX como arquitectura core** del refactor: resuelve “quién dijo cada palabra en toda la reunión”, no “solo el actor X con perfil de voz”. Opcional después solo como backend de `transcribe_speech` si duele latencia.

### Riesgos no bloqueantes

| Riesgo | Mitigación |
|---|---|
| Overlap / desalinear STT | `exclusive_speaker_diarization` (Community-1) |
| Compute desperdiciado | ASR solo clips del actor aceptado |
| Match actor frágil | video_ref + umbrales 0.5/0.7 + skip si ambiguo |
| Pico disco playlist | 1 video a la vez; borrar `.wav` salvo `--keep-cache` |
| GPU/RAM pyannote+Whisper | lazy-load; CPU ya contemplado |

### Veredicto

**Viable.** El núcleo de B ya existe en el repo (`transcribir_turnos_actor`, `WhisperFfmpegClipTranscriber`, `identificar_actor`). El refactor a 3 subpaquetes es empaquetado + contratos limpios, no un experimento de ML sin base.

---
## Decisiones resueltas

1. **Service de descarga:** `AudioFetcherService` (`speech2text/audio_fetcher`).
2. **`PlaylistInfo` / `VideoInfo` slim-down:** solo `nombre` / solo `titulo`+`fecha`. `id`/`url`/`duracion` en `VideoMeta`/`AudioVideo`.
3. **Speakers y ASR se separan:** `speaker_timestamps` (quién habla) + `transcribe_speech` (qué dijo el actor). Se eliminan `filtrar_por_actor` / `alineacion.py`.
4. **`SegmentoRaw` y `WordTimestamp` se eliminan de `models.py`.** Segmentación lee `ActorTranscriptSegment.text`.
5. **Fase 1 = loop video-por-video** en `PipelineRunner`: `speech2text.procesar_one` → `segmentacion.procesar_one` → **Temas una vez** al final. Cache delgado (borrar `.wav` salvo `--keep-cache`).
6. **Umbrella `speech2text`:** tres subpaquetes `audio_fetcher` + `speaker_timestamps` + `transcribe_speech`. Orquestador: `SpeechToTextService`. **No** incluye segmentación ni temas.
