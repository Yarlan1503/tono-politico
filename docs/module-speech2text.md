# Módulo `speech2text`

> **Estado:** activo y autocontenido.
>
> **Entrada:** playlist de YouTube + configuración de actor/modelos.
>
> **Salida:** `ActorTranscript[]` actor-only, turn-level, más el informe separado `speech2text_quality.v1`.

## Propósito

`speech2text` convierte audio de una playlist en transcripciones únicamente de las participaciones del actor objetivo. El módulo separa tres responsabilidades:

1. `audio_fetcher`: descubre la playlist y descarga/cachea `.wav`.
2. `speaker_timestamps`: diariza el audio, construye el perfil de referencia e identifica los turnos del actor.
3. `transcribe_speech`: recorta cada turno del actor y lo transcribe con Whisper.

El módulo no realiza segmentación semántica, descubrimiento de temas, clasificación de tono ni filtrado temático. Esas tareas pertenecen a `discursive_approach` y etapas posteriores.

## Flujo canónico

```text
ExecutionRunner
    │
    ├── SpeechToTextService.discover(playlist_url)
    │       └── PlaylistInfo(nombre), VideoMeta[]
    │
    ├── SpeechToTextService.ensure_perfil(nombre, metas)
    │       └── descarga video_ref_id → pyannote → PerfilVozActor
    │
    └── por cada VideoMeta seleccionado
            │
            ├── AudioFetcherService.fetch_one()
            │       └── AudioVideo | None
            │
            ├── SpeakerTimestampsService.procesar_one()
            │       └── TurnoOrador[] del actor
            │
            └── TranscribeSpeechService.procesar_one()
                    └── ActorTranscript | None

    └── Speech2TextQualityReport → speech2text/quality.json
```

El loop por video es responsabilidad del `ExecutionRunner`, no de los servicios. El perfil se construye una vez por ejecución usando la metadata completa de la playlist, incluso cuando `run.max_videos` limita los videos que finalmente se procesan.

## Componentes

| Documento | Responsabilidad | Salida principal |
|---|---|---|
| [`component_audio_fetcher.md`](component_audio_fetcher.md) | discover de playlist, cache y descarga `.wav` | `PlaylistInfo`, `VideoMeta`, `AudioVideo`, `DownloadResult` |
| [`component_speaker_timestamps.md`](component_speaker_timestamps.md) | pyannote, perfil de voz y match del actor | `PerfilVozActor`, `SpeakerMatch`, `TurnoOrador[]` |
| [`component_transcribe_speech.md`](component_transcribe_speech.md) | clips ffmpeg + Whisper actor-only | `ActorTranscript` |

La observabilidad de segmentos cortos vive en `src/tono_politico/speech2text/quality.py` y no modifica el contrato de transcript.

## Orquestador público

Módulo: `src/tono_politico/speech2text/service.py`

```python
class SpeechToTextService:
    def discover(self, url_playlist: str) -> tuple[PlaylistInfo, list[VideoMeta]]: ...

    def ensure_perfil(
        self,
        nombre_playlist: str,
        metas: list[VideoMeta],
    ) -> bool: ...

    def procesar_one(
        self,
        video: VideoMeta,
        nombre_playlist: str,
        *,
        archive_path: Path | None = None,
    ) -> ActorTranscript | None: ...
```

### `discover`

Obtiene la identidad de la playlist y la metadata pre-descarga. No descarga audio ni carga modelos pesados.

### `ensure_perfil`

Busca `video_ref_id` dentro de `metas`, descarga ese audio una sola vez y delega la construcción del perfil a `SpeakerTimestampsService`. Devuelve `False` si el video de referencia no está disponible o no se pudo descargar.

### `procesar_one`

Ejecuta la unidad `fetch → diarización → ASR`. Devuelve `None` si falla la descarga, no se identifica al actor, no hay turnos o el ASR no produce texto.

El orquestador no expone wrappers batch legacy como `procesar()` ni atajos como `fetch_one()`.

## Contratos de datos

### Metadata y audio

| DTO | Campos | Etapa |
|---|---|---|
| `PlaylistInfo` | `nombre` | identidad de playlist/cache |
| `VideoMeta` | `video_id`, `url`, `titulo`, `fecha`, `duracion` | discover, antes de efectos secundarios |
| `AudioVideo` | metadata de `VideoMeta` + `audio_path: Path` | audio local disponible |
| `DownloadResult` | `video_id`, `path`, `ok`, `error` | resultado estructurado de descarga |

`PlaylistInfo` no contiene URL ni lista de videos. La lista pre-descarga vive en `list[VideoMeta]`.

### Diarización

- `TurnoOrador`: `video_id`, `speaker_id`, `t_start`, `t_end`.
- `PerfilVozActor`: actor, video de referencia, embedding, modelo y duración.
- `SpeakerMatch`: speaker, distancia coseno, aceptación y estado ambiguo.

### `ActorTranscript` (`actor_transcript.v1`)

El transcript persistido es actor-only y turn-level:

```json
{
  "schema_version": "actor_transcript.v1",
  "video_id": "...",
  "actor": "...",
  "scope": "actor_only",
  "asr": {"provider": "whisper", "model": "large-v3-turbo", "language": "es"},
  "segments": [
    {
      "text": "...",
      "t_start": 12.5,
      "t_end": 18.3,
      "speaker": "SPEAKER_00",
      "source_turn": {"t_start": 12.2, "t_end": 18.5},
      "word_count": 23
    }
  ],
  "fecha": "20260101"
}
```

No se persisten timestamps por palabra, probabilidades, captions de YouTube, datos verbose de Whisper ni segmentación temática.

## Informe de calidad

`Speech2TextQualityReport` se persiste en:

```text
output/<run_id>/speech2text/quality.json
```

Schema: `speech2text_quality.v1`.

Mide, por video y de forma agregada:

- videos procesados;
- transcripts sin segmentos;
- segmentos totales;
- palabras totales;
- segmentos de una palabra;
- segmentos de hasta dos palabras.

El informe permite estudiar salidas como `Gracias.` sin eliminarlas. No aplica filtros por número de palabras o duración.

## Configuración operativa

La fuente canónica es `config/config.yaml` con `schema_version: tono-politico.run.v1`.

| Configuración | Consumidor |
|---|---|
| `project.data_dir` | cache de audio |
| `speech2text.enabled` | activación de la etapa |
| `speech2text.speaker_timestamps.actor_objetivo` | identidad objetivo |
| `speech2text.speaker_timestamps.pipeline` | pipeline pyannote principal |
| `speech2text.speaker_timestamps.fallback_pipeline` | fallback pyannote |
| `speech2text.speaker_timestamps.device` | device de inferencia |
| `speech2text.speaker_timestamps.umbral_match` | aceptación del speaker |
| `speech2text.speaker_timestamps.umbral_ambiguo` | rechazo/ambigüedad |
| `speech2text.speaker_timestamps.referencia_voz.video_id` | audio de referencia |
| `speech2text.transcribe_speech.whisper_model` | modelo Whisper |
| `speech2text.transcribe_speech.idioma` | idioma ASR |
| `run.max_videos` / `run.only_video_ids` | selección de videos |
| `run.keep_cache` | conservación de `.wav` |
| `run.resume` / `run.overwrite` | reanudación y recomputación |

No existen opciones anidadas `force_download`, `force_retranscribe`, `skip_existing_transcripts`, `word_timestamps` ni templates de rutas. El ASR usa siempre clips actor-only con `word_timestamps=False`.

## Artefactos y cache

```text
output/<run_id>/
├── speech2text/
│   ├── actor_transcripts/<video_id>.json
│   └── quality.json
├── manifest.json
└── resolved-config.yaml

data/<playlist>/
└── videos-<playlist>/<video_id>.wav
```

Los `.wav` son cache runtime. Con `run.keep_cache=false`, el runner los elimina después de cada unidad/ref cuando corresponde; con `true`, los conserva para depuración.

## API pública de paquetes

- `audio_fetcher`: `AudioFetcherService`, `VideoMeta`, `AudioVideo`, `DownloadResult`, `PlaylistInfo`.
- `speaker_timestamps`: `SpeakerTimestampsService`, `TurnoOrador`, `PerfilVozActor`, `SpeakerMatch`.
- `transcribe_speech`: `TranscribeSpeechService`, `ActorTranscript`, `ActorTranscriptSegment`, `AsrMetadata`.
- umbrella `speech2text`: `SpeechToTextService`, servicios y DTOs canónicos, serialización de `ActorTranscript`.

## Validación

```bash
uv run python main.py --config config/config.yaml --validate-config
uv run python main.py --config config/config.yaml --dry-run
uv run pytest tests/test_audio_fetcher_*.py \
  tests/test_speaker_timestamps_service.py \
  tests/test_transcribe_speech_service.py \
  tests/test_speech2text_service.py \
  tests/test_transcripcion_actor.py \
  tests/test_whisper_clip_transcriber.py \
  tests/test_actor_transcript_serializacion.py \
  tests/test_speech2text_quality.py -q
```
