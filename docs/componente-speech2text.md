# speech2text — audio → ActorTranscript (actor-only)

> **Estado:** ✅ Implementado (API nueva) · **Tests:** 42  
> **Smoke real:** Play-PoliTest 7/7 · 195 turnos · 0 errores · ~38 min  
> **Requisitos de diseño:** [`src/tono_politico/speech2text/requisitos.md`](../src/tono_politico/speech2text/requisitos.md)

## Propósito

Convertir una playlist de YouTube en **transcripciones turn-level solo del actor objetivo**, sin transcribir el video completo ni persistir word-level timestamps.

Reemplaza conceptualmente el acoplamiento **Ingesta (Whisper full) + Diarización (filtrar SegmentoRaw)** por tres fronteras claras:

```text
audio_fetcher        → traer .wav + metadata
speaker_timestamps   → quién habla cuándo + match del actor
transcribe_speech    → qué dijo el actor (Whisper por clip)
```

Segmentación semántica, Temas, Filtrado, Tono y Salida **quedan fuera** de este umbrella.

## Arquitectura

```text
URL playlist
    │
    ▼ SpeechToTextService.discover()
    │     audio_fetcher.playlist (yt-dlp --flat-playlist)
    │     → (PlaylistInfo, list[VideoMeta])
    │
    ▼ ensure_perfil(nombre, metas)
    │     fetch video_ref → speaker_timestamps.build_perfil()
    │
    ▼ por cada VideoMeta (loop video-por-video):
    │     1. audio_fetcher.fetch_one → AudioVideo | None
    │     2. speaker_timestamps.procesar_one → list[TurnoOrador] (solo actor)
    │     3. transcribe_speech.procesar_one → ActorTranscript | None
    │
    list[ActorTranscript]   # schema actor_transcript.v1
```

### Por qué este orden (diarize → slice → ASR)

Patrón B de la comunidad open-source (pyannote + Whisper por segmento), no WhisperX monobloque:

| Enfoque | Orden | Problema para tono-politico |
|---|---|---|
| A. WhisperX-like | ASR full → diarize → assign words | Transcribe a todos; empuja word-level |
| **B. El nuestro** | **Diarize → clip → ASR actor** | Solo paga ASR del actor; turn-level limpio |

Community-1 expone `exclusive_speaker_diarization` precisamente para reconciliar diarización con STT sin overlaps.

## API pública

### `SpeechToTextService` (orquestador)

```python
from pathlib import Path
from tono_politico.speech2text import SpeechToTextService

svc = SpeechToTextService(
    data_dir=Path("data"),
    actor="Lilly Téllez",
    video_ref_id="su9nURIj9XQ",
    whisper_model="large-v3-turbo",
    idioma="es",
    umbral_match=0.5,
    umbral_ambiguo=0.7,
    device="auto",
)

playlist, metas = svc.discover("https://youtube.com/playlist?list=...")
assert svc.ensure_perfil(playlist.nombre, metas)

for meta in metas:
    tx = svc.procesar_one(meta, playlist.nombre)
    if tx is None:
        continue  # skip: fallo descarga / sin actor / sin texto
    # tx: ActorTranscript (actor_transcript.v1)
```

| Método | Entrada | Salida |
|---|---|---|
| `discover(url)` | URL playlist | `(PlaylistInfo, list[VideoMeta])` |
| `ensure_perfil(nombre, metas)` | playlist + metas | `bool` (perfil listo) |
| `procesar_one(meta, nombre)` | un video | `ActorTranscript \| None` |
| `procesar(url)` | URL playlist | `list[ActorTranscript]` (wrapper ad-hoc/tests) |

### Subservicios

```python
from tono_politico.speech2text import (
    AudioFetcherService,
    SpeakerTimestampsService,
    TranscribeSpeechService,
)

# 1) I/O
fetcher = AudioFetcherService(data_dir=Path("data"))
playlist, metas = fetcher.discover(url)
audio = fetcher.fetch_one(metas[0], playlist.nombre)  # AudioVideo | None

# 2) Speakers (requiere perfil previo)
speakers = SpeakerTimestampsService(actor="Lilly Téllez", video_ref_id="su9nURIj9XQ")
speakers.build_perfil(ref_audio)  # o set_perfil(perfil)
turnos = speakers.procesar_one(audio)  # solo turnos del actor

# 3) ASR actor-only
asr = TranscribeSpeechService(actor="Lilly Téllez", whisper_model="large-v3-turbo")
tx = asr.procesar_one(audio, turnos)  # ActorTranscript | None
```

## DTOs

### `VideoMeta` (pre-descarga)

| Campo | Tipo | Descripción |
|---|---|---|
| `video_id` | `str` | ID YouTube |
| `url` | `str` | URL del video |
| `titulo` | `str` | Título |
| `fecha` | `str \| None` | YYYYMMDD |
| `duracion` | `float` | segundos (yt-dlp) |

### `AudioVideo` (post-descarga)

Igual que `VideoMeta` + `audio_path: Path`. Factory: `AudioVideo.from_meta(meta, audio_path=...)`.

### `DownloadResult`

| Campo | Tipo | Descripción |
|---|---|---|
| `video_id` | `str` | |
| `path` | `Path \| None` | `.wav` si ok |
| `ok` | `bool` | |
| `error` | `str \| None` | mensaje truncado si falló |

### `ActorTranscript` (`actor_transcript.v1`)

| Campo | Descripción |
|---|---|
| `schema_version` | `"actor_transcript.v1"` |
| `video_id`, `actor`, `scope` | `scope="actor_only"` |
| `asr` | `AsrMetadata(provider, model, language)` |
| `segments[]` | turnos con `text`, `t_start`, `t_end`, `speaker`, `source_turn_*`, `word_count` opcional |

**No se persiste:** word-level timestamps, probability por palabra, `pausa_antes`, captions YouTube, verbose Whisper, full-video `VideoTranscript`.

## Módulos

```text
src/tono_politico/speech2text/
├── service.py                   # SpeechToTextService
├── requisitos.md                # checklist + viabilidad
├── audio_fetcher/
│   ├── models.py                # VideoMeta, AudioVideo, DownloadResult
│   ├── cache.py                 # DATA_DIR, ruta_dir_videos, ruta_audio
│   ├── playlist.py              # sanitizar + obtener_info_playlist
│   ├── audio.py                 # verificar_cache + descargar_audio_result
│   └── service.py               # AudioFetcherService
├── speaker_timestamps/
│   └── service.py               # SpeakerTimestampsService
│                                # (reusa adapter/matching/perfil de diarizacion/)
└── transcribe_speech/
    └── service.py               # TranscribeSpeechService
                                 # (reusa whisper_clip + transcripcion_actor)
```

### Estado de migración

| Capa | Código propio | Deuda |
|---|---|---|
| `audio_fetcher` | ✅ completo e independiente | `ingesta/` legacy sigue en el repo |
| `speaker_timestamps` | ✅ API nueva | importa stack desde `diarizacion/` |
| `transcribe_speech` | ✅ API nueva | importa `whisper_clip` / `transcripcion_actor` |
| `PipelineRunner` | ✅ `discover_discursive` (speech2text → discursive_approach) | path legacy `discover`/`analyze` sigue con Ingesta+Diarización |

## Configuración relevante

Desde `config/config.yaml` (hasta renombrar secciones):

| Clave | Uso en speech2text |
|---|---|
| `project.data_dir` | Cache de `.wav` |
| `project.idioma` | Whisper |
| `ingesta.whisper_model` | Modelo ASR (hoy `large-v3-turbo`) |
| `diarizacion.pipeline` / `fallback_pipeline` | Community-1 |
| `diarizacion.device` | `auto` |
| `diarizacion.actor_objetivo` | Nombre del actor |
| `diarizacion.umbral_match` / `umbral_ambiguo` | 0.5 / 0.7 |
| `diarizacion.referencia_voz.video_id` | p.ej. `su9nURIj9XQ` |

## Cache en disco

```text
data/<playlist>/
└── videos-<playlist>/
    └── <video_id>.wav
```

- **No** hay `transcripciones-*.json` de Whisper full-video en este componente.
- Persistencia recomendada de salida: `ActorTranscript` JSON (`actor_transcript.v1`).
- En producción, el runner debería borrar `.wav` tras procesar cada video salvo `--keep-cache`.

## Smoke real — Play-PoliTest

Script: `scripts_smoke_speech2text.py`

```bash
uv run python scripts_smoke_speech2text.py
# → output/speech2text-smoke/summary.json
# → output/speech2text-smoke/actor_transcripts/<video_id>.json
```

| Métrica | Resultado (2026-07-08) |
|---|---|
| Playlist | `Play-PoliTest` (`PLE9Zk7g9R__M`) |
| Videos | **7/7 ok**, 0 skip, 0 error |
| Segmentos turn-level | **195** |
| Palabras ≈ | **1 959** |
| Actor match | 1 aceptado en todos los videos |
| Tiempo total | ~38.4 min (CPU, clip-a-clip Whisper) |
| Schema | `actor_transcript.v1` limpio (sin word-level) |

Comparación con smoke viejo (Ingesta full + filtro): 139 segmentos Whisper-window vs 195 turnos exclusive pyannote — más granular, esperado.

### Cuellos de botella observados

1. **Whisper por clip** (~7–8 s fijos incluso en turnos cortos) domina el tiempo en videos con muchos turnos (62–69).
2. Turnos de cortesía residuales (`Gracias.`, `Gracias, Senadora.`) pueden colarse si el speaker aceptado incluye cierre de sesión — filtrable por longitud mínima.
3. Sin borrado de wav en el smoke: ~201 MB de audio en `data/speech2text-smoke/`.

## Decisiones de diseño

1. **Tres subpaquetes**, no monobloque: I/O ≠ diarización ≠ ASR.
2. **ASR solo del actor** — no Whisper full-video en el camino feliz.
3. **`word_timestamps=False`** en clips: evita alineación extra y cambios de texto; timestamps de turno vienen de pyannote.
4. **`exclusive_speaker_diarization`** para turnos sin overlap.
5. **Perfil de voz 1× por corrida** desde `video_ref_id` de la misma playlist.
6. **Match ambiguo (0.5–0.7) se descarta** — no se fuerza identidad.
7. **Segmentación y Temas fuera** — siguiente fase discursiva (`discursive_approach`) consume `ActorTranscript`, no vive dentro de speech2text.
8. **Loop video-por-video** — cache delgado, fallos aislados por `video_id`.

## Tests

| Suite | Archivo | Cobertura |
|---|---|---|
| DTOs | `tests/test_audio_fetcher_models.py` | VideoMeta, AudioVideo, DownloadResult |
| Cache | `tests/test_audio_fetcher_cache.py` | rutas, sin transcripciones |
| Playlist | `tests/test_audio_fetcher_playlist.py` | sanitizar + parse yt-dlp → VideoMeta |
| Audio | `tests/test_audio_fetcher_audio.py` | cache + descarga mock |
| AudioFetcherService | `tests/test_audio_fetcher_service.py` | discover / fetch_one / procesar |
| Speakers | `tests/test_speaker_timestamps_service.py` | perfil + filtrado actor (mocks) |
| ASR | `tests/test_transcribe_speech_service.py` | ClipTranscriber fake |
| Orquestador | `tests/test_speech2text_service.py` | composición + ensure_perfil |

```bash
uv run pytest tests/test_audio_fetcher_*.py \
  tests/test_speaker_timestamps_service.py \
  tests/test_transcribe_speech_service.py \
  tests/test_speech2text_service.py -q
# 42 passed
```

## Relación con el resto del pipeline

```text
Fase 1 (objetivo):
  speech2text.procesar_one × N  →  ActorTranscript[]
  segmentacion (adaptar a ActorTranscript) → Segmento[]
  temas → ResultadoTemas

Fase 2 (sin cambio de idea):
  filtrado → tono → salida
```

Hoy `PipelineRunner` aún orquesta **Ingesta + DiarizacionService** legacy. El smoke de speech2text valida la API nueva de forma independiente vía `scripts_smoke_speech2text.py`.

## Véase también

- Requisitos y checklist: `src/tono_politico/speech2text/requisitos.md`
- Legacy (coexistencia hasta cablear runner): `docs/componente-ingesta.md`, `docs/componente-diarizacion.md`
- Config: `docs/configuracion.md`
