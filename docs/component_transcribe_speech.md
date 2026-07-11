# Componente `transcribe_speech`

> **Ruta:** `src/tono_politico/speech2text/transcribe_speech/`
>
> **Responsabilidad:** convertir los turnos del actor en texto mediante clips temporales de Whisper.
>
> **No hace:** descubrir playlists, diarizar audio, identificar speakers ni realizar análisis temático.

## Flujo

```text
AudioVideo + TurnoOrador[] del actor
    │
    ▼ transcribir_turnos_actor()
por cada turno
    ├── calcular límites del clip
    ├── ffmpeg → WAV temporal mono, 16 kHz, PCM
    ├── Whisper → segmentos relativos al clip
    ├── reubicar timestamps al timeline del video
    └── conservar solo texto no vacío
            │
            ▼
ActorTranscript (actor_transcript.v1)
```

## API pública

### `TranscribeSpeechService`

Módulo: `service.py`.

```python
class TranscribeSpeechService:
    def __init__(
        self,
        actor: str = "Lilly Téllez",
        whisper_model: str = "large-v3-turbo",
        idioma: str = "es",
        padding_segundos: float = 0.0,
        transcriptor: Any | None = None,
    ) -> None: ...

    def procesar_one(
        self,
        audio: AudioVideo,
        turnos_actor: list[TurnoOrador],
    ) -> ActorTranscript | None: ...
```

`procesar_one()` devuelve `None` si no hay turnos o si ningún clip produce texto. El transcriptor se inyecta en tests; en runtime se crea perezosamente `WhisperFfmpegClipTranscriber`.

### `transcribir_turnos_actor`

Módulo: `transcripcion_actor.py`.

```python
def transcribir_turnos_actor(
    audio_path: Path | str,
    turnos_actor: list[TurnoOrador],
    *,
    video_id: str,
    actor: str,
    transcriptor: ClipTranscriber,
    modelo: str = "large-v3-turbo",
    idioma: str = "es",
    padding_segundos: float = 0.0,
    duracion_audio: float | None = None,
    fecha: str | None = None,
) -> ActorTranscript: ...
```

El protocolo `ClipTranscriber` solo requiere:

```python
class ClipTranscriber(Protocol):
    def transcribir_clip(
        self,
        audio_path: Path,
        *,
        t_start: float,
        t_end: float,
        modelo: str,
        idioma: str,
    ) -> list[ClipTranscriptSegment]: ...
```

## Clips y Whisper

`WhisperFfmpegClipTranscriber`:

- valida que el audio existe y que `t_end > t_start`;
- calcula un WAV temporal por clip;
- usa ffmpeg con `-ac 1 -ar 16000 -c:a pcm_s16le`;
- ejecuta Whisper con `word_timestamps=False`, `fp16=False` y `verbose=False`;
- cachea el modelo por nombre durante la instancia;
- elimina el WAV temporal en `finally`, incluso cuando ffmpeg o Whisper fallan;
- convierte errores de ffmpeg en `RuntimeError` accionable.

El modelo default es `large-v3-turbo` y el idioma default es `es`.

## Límites temporales

El padding es `0.0` por default. Si se configura:

- el clip enviado a Whisper se amplía por ambos lados;
- el inicio no puede ser menor que `0.0`;
- el final se limita a `duracion_audio` cuando está disponible;
- los límites originales del turno pyannote se mantienen como fuente contractual.

Los segmentos de Whisper relativos al clip se reubican al timeline absoluto y se clamped dentro del turno original del actor.

## Contrato `ActorTranscript`

Cada transcript contiene:

- `schema_version="actor_transcript.v1"`;
- `video_id`, `actor` y `scope="actor_only"`;
- `AsrMetadata(provider="whisper", model, language)`;
- segmentos con `text`, timestamps absolutos, `speaker`, límites del turno fuente y `word_count`;
- `fecha` propagada desde `AudioVideo.fecha` cuando existe.

No se persisten:

- timestamps por palabra;
- probabilidades por palabra;
- captions de YouTube;
- salida verbose de Whisper;
- `pausa_antes`;
- segmentos de otros speakers.

Los segmentos cortos y fórmulas como `Gracias.` se conservan. Su medición pertenece al informe separado `speech2text_quality.v2`, no a este componente.

## Validaciones y casos vacíos

- audio inexistente → `FileNotFoundError`;
- padding negativo → `ValueError`;
- `video_id` de un turno distinto al audio → `ValueError`;
- turno con `t_end <= t_start` → `ValueError`;
- clip sin texto → no se añade segmento;
- ningún segmento con texto → `TranscribeSpeechService.procesar_one()` devuelve `None`.

## Módulos

| Archivo | Responsabilidad |
|---|---|
| `transcripcion_actor.py` | protocolo de clip, composición y timestamps |
| `whisper_clip.py` | ffmpeg temporal + adaptador Whisper |
| `service.py` | fachada de configuración e inyección |
| `__init__.py` | exports públicos del subpaquete |

## Tests

```bash
uv run pytest \
  tests/speech2text/test_transcribe_speech_service.py \
  tests/speech2text/test_actor_clip.py \
  tests/speech2text/test_transcription_clip.py -q
```
