# Requisitos de `speech2text`

> **Ruta:** `src/tono_politico/speech2text/` · **Estado:** implementado y auditado (2026-07-10).
>
> Documentación: [`module-speech2text.md`](../../../docs/module-speech2text.md) · [`audio_fetcher`](../../../docs/component_audio_fetcher.md) · [`speaker_timestamps`](../../../docs/component_speaker_timestamps.md) · [`transcribe_speech`](../../../docs/component_transcribe_speech.md)

## 1. Objetivo y frontera

`speech2text` convierte una playlist en transcripciones **actor-only** y **turn-level**.

```text
playlist → discover → audio WAV → diarización/match → clips Whisper → ActorTranscript[]
```

- [x] Descubre metadata sin descargar audio.
- [x] Descarga/reutiliza `.wav` con validación de archivo regular.
- [x] Construye perfil de voz y selecciona turnos del actor.
- [x] Ejecuta ASR solo sobre clips del actor.
- [x] Persiste `ActorTranscript`; la observabilidad y el manifest pertenecen a `execution/`.
- [x] No hace segmentación semántica, temas, tono ni filtrado temático.
- [x] No usa Whisper full-video ni persiste word-level timestamps/probabilities.
- [x] Conserva segmentos cortos; no aplica filtros sin datos etiquetados.

## 2. Flujo y ownership

```text
ExecutionRunner
  ├─ SpeechToTextService.discover(url) → PlaylistInfo + VideoMeta[]
  ├─ SpeechToTextService.ensure_perfil(playlist, metas)
  │    └─ video_ref_id → AudioVideo → pyannote → PerfilVozActor
  └─ por cada VideoMeta seleccionado
       ├─ si resume + transcript válido → saltar (resumed_from_cache)
       ├─ AudioFetcherService.fetch_one() → AudioVideo | None
       ├─ SpeakerTimestampsService.procesar_one() → TurnoOrador[]
       ├─ TranscribeSpeechService.procesar_one() → ActorTranscript | None
       ├─ UnitResult(status, reason_code, timings)
       └─ checkpoint incremental → speech2text/checkpoint.json
  └─ observability.py → speech2text/quality.json (v2)
```

- [x] El loop por video pertenece al runner.
- [x] El perfil se resuelve con la metadata completa antes de `max_videos`.
- [x] La pérdida del perfil marca el stage como `failed`, no `ok` vacío.
- [x] Los fallos por vídeo generan `UnitResult(status="failed")` con `reason_code`.
- [x] Resume a nivel de unidad: vídeos con transcript válido se saltan.

## 3. Contratos DTO

### `PlaylistInfo`

- [x] Conserva `nombre` visible y `nombre_cache` sanitizado para filesystem.
- [x] Conserva `playlist_id` y `url` para provenance.
- [x] No contiene `videos`; la lista vive en `list[VideoMeta]`.

### `VideoMeta` — pre-descarga

| Campo | Tipo |
|---|---|
| `video_id` | `str` |
| `url` | `str` |
| `titulo` | `str` |
| `fecha` | `str \\| None` (`YYYYMMDD`) |
| `fecha_fuente` | `str \\| None` (`upload_date`, `release_date`, `timestamp`, `missing`, `invalid`) |
| `duracion` | `float` |

### `AudioVideo` — post-descarga

- [x] Conserva `VideoMeta` y añade `audio_path: Path` obligatorio.
- [x] Se construye con `AudioVideo.from_meta(meta, audio_path=...)`.

### `DownloadResult`

`video_id: str`, `path: Path | None`, `ok: bool`, `error: str | None`.

- [x] Los fallos de yt-dlp son estructurados y no crashean la unidad.

### Diarización

- [x] `TurnoOrador`: `video_id`, `speaker_id`, `t_start`, `t_end`.
- [x] `PerfilVozActor`: actor, video de referencia, embedding, modelo y duración.
- [x] `SpeakerMatch`: speaker, distancia, aceptación y ambigüedad.
- [x] Un match ambiguo se descarta.

### `ActorTranscript` (`actor_transcript.v1`)

`schema_version`, `video_id`, `actor`, `scope="actor_only"`, `AsrMetadata`, `segments`, `fecha` y `source` opcional.

Cada segmento contiene `text`, timestamps absolutos, `speaker`, `source_turn_start`, `source_turn_end` y `word_count`.

- [x] Solo contiene texto atribuido al actor.
- [x] Conserva los límites del turno pyannote y propaga `fecha`.
- [x] Persiste título, playlist y procedencia de fecha en `source` cuando están disponibles.
- [x] No contiene words, probabilities, captions ni verbose Whisper.

### Estados de ejecución (`UnitResult` en `execution/`)

- [x] `UnitResult`: `video_id`, `status` (`ok`/`skipped`/`failed`), `reason_code`, `timings`, `transcript`, `error`, título, fecha y fuente.
- [x] `UNIT_REASON_CODES`: `transcript_persisted`, `resumed_from_cache`, `skipped`, `audio_fetch_failed`, `transcription_failed`, `diarization_failed`, `no_segments`, `error`, `reference_profile_missing`, `audio_invalid`, `download_failed`, `transcript_invalid`, `asr_empty`.
- [x] El manifest serializa `UnitResult` sin texto de transcripciones.

## 4. Estructura de módulos

```text
speech2text/
├── __init__.py
├── service.py                    # orquesta los tres submódulos
├── models.py                     # DTOs del dominio de speech2text
│
├── audio_fetcher/
│   ├── __init__.py
│   ├── playlist.py               # playlist_name + metadata de vídeos
│   ├── audio.py                  # descarga y cache de audio
│   ├── service.py                # orquesta audio_fetcher
│   └── models.py                 # DTOs del dominio de audio_fetcher
│
├── speaker_timestamps/
│   ├── __init__.py
│   ├── models.py                 # DTOs canónicos de diarización/matching
│   ├── perfil_voz.py             # construye el perfil de voz
│   ├── matching.py               # identifica turnos y valida resultados
│   └── service.py                # carga modelo y orquesta diarización
│
└── transcribe_speech/
    ├── __init__.py
    ├── actor_clip.py             # padding y mapeo temporal de clips
    ├── transcription_clip.py     # Whisper sobre clips normalizados
    ├── service.py                # orquesta transcribe_speech
    └── models.py                 # DTOs del dominio de transcribe_speech
```

**Fuera del módulo:** `errors.py`, `validation.py`, `cache.py`, `adapter.py`, `output.py`, `quality.py`, `results.py`, `requisitos.md`, `actor_transcript.py`, `whisper_clip.py`, `transcripcion_actor.py`. La persistencia y observabilidad pertenecen a `execution/`.

APIs canónicas:

- [x] `SpeechToTextService.discover`, `ensure_perfil`, `procesar_one`.
- [x] `AudioFetcherService.discover`, `fetch_one`.
- [x] `SpeakerTimestampsService.build_perfil`, `set_perfil`, `procesar_one`.
- [x] `TranscribeSpeechService.procesar_one`.
- [x] `construir_perfil_desde_output`, `transcribir_turnos_actor`.
- [x] `WhisperFfmpegClipTranscriber.transcribir_clip`.
- [x] Serialización: `guardar_actor_transcript`, `cargar_actor_transcript` (en `execution/artifacts.py`).
- [x] Observabilidad: `build_quality_report`, `guardar_quality_report` (en `execution/observability.py`).

## 5. Reglas por componente

### `audio_fetcher`

- [x] `obtener_info_playlist()` usa yt-dlp `--flat-playlist`.
- [x] Fecha: `upload_date` → `release_date` → `timestamp`, con estado explícito cuando falta o es inválida.
- [x] Sanitiza el nombre y usa `data/<playlist>/videos-<playlist>/<video_id>.wav`.
- [x] Reutiliza cache (valida archivo regular + tamaño > 0) y no importa Whisper/pyannote.
- [x] `audio.py` captura `FileNotFoundError` (binario ausente) además de `TimeoutExpired`.

### `speaker_timestamps`

- [x] Carga pipeline primary/fallback con device auto en `service.py` (sin `adapter.py`).
- [x] Usa Community-1 y `exclusive_speaker_diarization`.
- [x] `service.py` valida cantidad de embeddings vs labels, NaN/Inf, turnos con rangos inválidos.
- [x] `matching.py` valida dimensiones compatibles y vectores finitos en `distancia_coseno`.
- [x] `perfil_voz.py` valida consistencia embeddings↔labels y segmentos inválidos.
- [x] Defaults de matching: `umbral_match=0.5`, `umbral_ambiguo=0.7`.

### `transcribe_speech`

- [x] `actor_clip.py`: recibe turnos del actor, aplica padding acotado y mapea timestamps.
- [x] `transcription_clip.py`: ffmpeg mono 16 kHz PCM + Whisper con `word_timestamps=False`.
- [x] Cachea modelos, elimina temporales en `finally` y omite texto vacío.
- [x] `models.py`: `ClipTranscriptSegment`, `ClipTranscriber` (Protocol), `ClipWindow`.
- [x] Reubica timestamps al timeline y los limita al turno fuente.

## 6. Calidad y artefactos

Informe `speech2text_quality.v2` en `execution/observability.py`:

```text
output/<run_id>/speech2text/quality.json
```

- [x] DTO separado, sin texto de transcripciones.
- [x] Métricas desde `UnitResult`: cuenta ok/skipped/failed, segmentos, palabras, vacíos.
- [x] No filtra segmentos.
- [ ] Etiquetar una muestra real antes de proponer filtros o métricas de cortesía.

Checkpoint incremental:

```text
output/<run_id>/speech2text/checkpoint.json
```

- [x] Se escribe después de cada vídeo procesado.
- [x] Permite reanudar sin duplicar transcripts.

Manifest con fingerprint de configuración:

```text
output/<run_id>/manifest.json
```

- [x] `config_fingerprint`: whisper_model, idioma, pipeline, thresholds, only_video_ids, max_videos.
- [x] `units`: una entrada por vídeo sin texto.
- [x] `stages`: estado y duración por etapa.

Artefactos y cache:

```text
output/<run_id>/speech2text/actor_transcripts/<video_id>.json
data/<playlist>/videos-<playlist>/<video_id>.wav
```

- [x] `resume` + `overwrite=true` permite reanudar a nivel de unidad saltando vídeos con transcript válido.
- [x] `keep_cache=false` elimina WAV cuando corresponde; `true` los conserva.

## 7. Configuración

Fuente: `config/config.yaml`, schema `tono-politico.run.v1`.

- [x] `speech2text.enabled` activa la etapa.
- [x] `project.data_dir` controla el cache.
- [x] `speaker_timestamps` contiene actor, pipelines, device, thresholds y `referencia_voz.video_id`.
- [x] `transcribe_speech` contiene `whisper_model` e `idioma`.
- [x] `run.max_videos`, `only_video_ids`, `keep_cache`, `resume`, `overwrite`, `fail_fast` viven en `run`.
- [x] Coerción de booleanos type-safe: `bool("false")` rechazado en vez de aceptado como `True`.
- [x] `--validate-config` y `--dry-run` pasan.

## 8. Documentación y validación

- [x] `docs/module-speech2text.md` y los tres `docs/component_*.md` existen.
- [x] Los documentos históricos de Ingesta/Diarización fueron eliminados.

Suite focalizada:

```bash
uv run pytest tests/speech2text/ \
  tests/test_metadata_propagation.py \
  tests/test_execution_speech2text_contracts.py \
  tests/test_execution_speech2text_provenance.py \
  tests/test_execution_observability.py \
  tests/test_execution_actor_transcript_artifacts.py -q
```

Gates actuales:

```bash
uv run ruff check src/ tests/ main.py          # ✅ All checks passed
uv run ruff format --check src/ tests/ main.py  # ✅ 76 files already formatted
uv run pytest tests/ -m "not slow" \
  --ignore=tests/test_discursive_approach_service.py \
  --ignore=tests/test_topics_approach.py -q     # ✅ 191 passed, 1 skipped
uv run python main.py --config config/config.yaml --validate-config  # ✅
uv run python main.py --config config/config.yaml --dry-run          # ✅
```

- [x] `ty check`: 6 errores en `discursive_approach/topics_approach` (imports legacy de `tono`, `filtrado`, `segmentacion`, `temas`). Bloqueo intencional aceptado; no afecta a `speech2text`.
- [x] `tests/test_discursive_approach_service.py` y `tests/test_topics_approach.py` excluidos por `ModuleNotFoundError: tono_politico.tono`.
- [ ] Ejecutar smoke real progresivo y comparar contra baseline actual.

## 9. Definición de terminado

- [x] Contratos, APIs, configuración y documentación coinciden con el código.
- [x] Umbrella de tres subpaquetes; diarizar → identificar → ASR actor-only.
- [x] `ActorTranscript` turn-level y segmentos cortos sin filtros arbitrarios.
- [x] `PlaylistInfo` mínimo y DTOs pre/post-descarga separados.
- [x] Wrappers legacy y módulos eliminados (`quality.py`, `actor_transcript.py`, `adapter.py`, `transcripcion_actor.py`, `whisper_clip.py`).
- [x] Estados de ejecución, persistencia y observabilidad fuera del núcleo (`execution/`).
- [x] `resume` no confunde directorios vacíos con artefactos completos.
- [x] Cache y output se validan como archivos/contratos, no sólo por `exists()`.
- [x] `speaker_timestamps/service.py` carga el modelo; `matching.py` valida resultados.
- [x] `actor_clip.py` preserva el mapeo entre audio editado y timestamps originales.
- [x] Checkpoint incremental por vídeo + fingerprint de configuración en manifest.
- [x] Coerción de booleanos type-safe en `execution/models.py`.
- [ ] Smoke real y comparación contra datos actuales actualizados.
- [ ] Desacoplar `discursive_approach/topics_approach` para que el gate global pase.
