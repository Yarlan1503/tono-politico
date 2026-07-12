# Componente `speaker_timestamps`

> **Ruta:** `src/tono_politico/speech2text/speaker_timestamps/`
>
> **Responsabilidad:** diarizar el audio, construir el perfil de voz de referencia, identificar al actor y normalizar sus unidades temporales de transcripción.
>
> **No hace:** descargar playlists, ejecutar Whisper ni producir texto.

## Flujo

```text
AudioVideo (WAV)
    │
    ├── build_perfil(ref_audio)
    │       └── pyannote → speaker_embeddings → PerfilVozActor
    │
    └── procesar_one(audio)
            ├── pyannote exclusive_speaker_diarization
            ├── fusionar_turnos_consecutivos() sobre la secuencia completa
            ├── embeddings por speaker
            ├── distancia coseno contra PerfilVozActor
            └── TurnoOrador[] actor normalizados
                    ├── bloques consecutivos del mismo speaker
                    └── [0, duración] si sólo existe un speaker actor
```

El perfil se construye una vez por corrida desde el audio de referencia. `SpeechToTextService.ensure_perfil()` es quien obtiene ese audio desde `audio_fetcher`.

## API pública

### `SpeakerTimestampsService`

Módulo: `service.py`.

```python
class SpeakerTimestampsService:
    def __init__(
        self,
        actor: str = "Lilly Téllez",
        video_ref_id: str = "su9nURIj9XQ",
        pipeline_name: str = "pyannote/speaker-diarization-community-1",
        fallback_pipeline: str | None = "pyannote-community/speaker-diarization-community-1",
        device: str = "auto",
        umbral_match: float = 0.5,
        umbral_ambiguo: float = 0.7,
    ) -> None: ...

    def build_perfil(self, ref_audio: AudioVideo) -> PerfilVozActor: ...
    def set_perfil(self, perfil: PerfilVozActor) -> None: ...
    def procesar_one(self, audio: AudioVideo) -> list[TurnoOrador]: ...
```

### `build_perfil`

- reutiliza el perfil si ya fue construido;
- ejecuta pyannote sobre el audio de referencia;
- selecciona el speaker dominante por duración total;
- guarda el embedding como `list[float]` en `PerfilVozActor`;
- mantiene el perfil en memoria durante la ejecución.

### `set_perfil`

Permite inyectar un perfil ya construido, principalmente para tests o reuso controlado.

### `procesar_one`

Requiere que exista un perfil. Primero fusiona participaciones adyacentes con la misma etiqueta en la secuencia diarizada completa; un speaker intermedio impide la fusión. Después aplica el matching del actor. Si queda un único speaker y coincide con el perfil, devuelve una unidad `[0, audio.duracion]` para que Whisper transcriba el audio completo sin cortes. Devuelve una lista vacía cuando no hay turnos, embeddings o un speaker aceptado como el actor. Nunca produce texto.

## Pipeline pyannote

`service.py` concentra los detalles runtime:

- pipeline principal: `pyannote/speaker-diarization-community-1`;
- fallback configurable: `pyannote-community/speaker-diarization-community-1`;
- `device="auto"` resuelve CUDA si está disponible y CPU en caso contrario;
- usa `ProgressHook` cuando la instalación lo ofrece;
- el token de Hugging Face se lee desde `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` o el token local de Hugging Face;
- si no puede cargar ningún pipeline, lanza `PyannotePipelineLoadError` sin exponer credenciales.

La ejecución usa `exclusive_speaker_diarization`, evitando solapamientos para facilitar el recorte posterior de Whisper.

## Matching de speakers

Módulo: `matching.py`.

### `distancia_coseno`

Calcula distancia coseno en Python puro:

- `0.0`: vectores idénticos;
- `1.0`: vectores ortogonales;
- `2.0`: vectores opuestos o norma cero.

### `clasificar_speaker`

Las fronteras son exclusivas:

```text
distancia < 0.5                 → aceptado
distancia >= 0.5 y < 0.7        → ambiguo, descartar
distancia >= 0.7                → rechazado
```

Los valores se reciben por constructor y configuración; los defaults vigentes son `0.5` y `0.7`.

### `identificar_actor`

Recibe `{speaker_id: embedding}` y un `PerfilVozActor`, devuelve `SpeakerMatch[]` ordenado por distancia ascendente.

## Perfil de voz

La única función vigente para construirlo desde la salida pública de pyannote es:

```python
construir_perfil_desde_output(
    output,
    actor: str,
    video_ref_id: str,
    pipeline_name: str,
) -> PerfilVozActor
```

Usa `output.speaker_embeddings`, `output.speaker_diarization.labels()` y `output.exclusive_speaker_diarization`. El constructor legacy `construir_perfil()` fue retirado.

## DTOs

Los DTOs canónicos del componente viven en `speaker_timestamps/models.py`. El módulo umbrella `speech2text.models` los reexporta sólo para compatibilidad con imports existentes.

| DTO | Campos | Propósito |
|---|---|---|
| `TurnoOrador` | `video_id`, `speaker_id`, `t_start`, `t_end` | unidad temporal diarizada, fusionada o de audio completo entregada al ASR |
| `PerfilVozActor` | `actor`, `video_id_referencia`, `embedding`, `modelo_embedding`, `duracion_segundos` | referencia en memoria |
| `SpeakerMatch` | `speaker_id`, `distancia`, `aceptado`, `es_ambiguo` | decisión de identidad |

## Módulos

| Archivo | Responsabilidad |
|---|---|
| `matching.py` | distancia y clasificación de speakers |
| `perfil_voz.py` | perfil desde `speaker_embeddings` público |
| `service.py` | carga/fallback/device/ProgressHook, diarización, fusión y match por audio |
| `models.py` | DTOs canónicos de diarización y matching |
| `__init__.py` | exports públicos del subpaquete |

## Integración con ASR

La salida `TurnoOrador[]` se entrega sin texto a `TranscribeSpeechService`. Cada elemento representa una unidad de ASR: los bloques del mismo speaker ya vienen fusionados, mientras que el caso de speaker único validado usa `[0, audio.duracion]` para transcribir el WAV completo. Sus límites se persisten en `source_turn: {t_start, t_end}`; el loader también acepta la forma plana legacy `source_turn_start`/`source_turn_end`.

## Tests

```bash
uv run pytest \
  tests/speech2text/test_speaker_timestamps_service.py \
  tests/speech2text/test_speaker_timestamps_matching.py \
  tests/speech2text/test_speaker_timestamps_models.py \
  tests/speech2text/test_speaker_timestamps_profile.py -q
```
