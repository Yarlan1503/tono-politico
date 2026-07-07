# Componente 1.5: Diarización e identificación de actor

> **Estado:** ✅ Implementado · **Tests:** 62 (6 archivos)

## Propósito

Inserta una etapa entre Ingesta y Segmentación para asegurar que el análisis de tono se aplique solo a las intervenciones del actor político objetivo, aunque el audio contenga varios oradores. Toma `VideoTranscript[]` + audio WAV, ejecuta diarización de pyannote, identifica al actor por embedding de voz, y devuelve `VideoTranscript[]` filtrado — mismo contrato, solo segmentos del actor.

```text
VideoTranscript[] + audio WAV
  │
  ├── Whisper large-v3-turbo → texto + WordTimestamp[]  (Ingesta, ya hecho)
  └── pyannote Community-1   → TurnoOrador[] + speaker_embeddings
                                 │
                                 ▼ construir_perfil()        — perfil de referencia cacheado
                                 ▼ identificar_actor()       — distancia coseno por speaker_embeddings
                                 ▼ filtrar_por_actor()       — midpoint dentro de turnos del actor
                                 │
                                 VideoTranscript[] (solo segmentos del actor)
                                 │
                                 ▼ Segmentación
```

## Playlist de pruebas

Playlist: `Play-PoliTest`
URL: `https://youtube.com/playlist?list=PLE9Zk7g9R__M&si=n7wQu_7VnRQDow_V`

Verificación con `yt-dlp --flat-playlist -J` mostró 7 videos, todos centrados en intervenciones de Lilly Téllez.

## Audio de referencia para perfil de voz

Usar el video:

```text
https://www.youtube.com/watch?v=su9nURIj9XQ&list=PLE9Zk7g9R__M&index=8
```

Metadata verificada con `yt-dlp --no-playlist -J`:

| Campo | Valor |
|---|---|
| `video_id` | `su9nURIj9XQ` |
| Título | `Pregunta de la senadora Lilly Téllez al diputado Gibrán Ramírez, en el apartado de Agenda política` |
| Duración | 30 s |
| Canal | `SenadoresPANTV` |
| Fecha | `20260610` |

## API

### `DiarizacionService`

```python
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

| Función/Clase | Módulo | Responsabilidad |
|---|---|---|
| `TurnoOrador` | `models.py` | DTO: turno individual de un orador (video_id, speaker_id, t_start, t_end) |
| `PerfilVozActor` | `models.py` | DTO: perfil de voz cacheado en memoria (actor, embedding, modelo, duración) |
| `SpeakerMatch` | `models.py` | DTO: resultado de matching (speaker_id, distancia, aceptado, es_ambiguo) |
| `DiarizacionService` | `service.py` | Orquestador OOP (lazy-load pyannote + speaker embeddings + perfil cache) |
| `diarizar()` | `diarizacion.py` | Ejecuta pyannote con `exclusive_speaker_diarization` → `TurnoOrador[]` |
| `construir_perfil()` | `perfil_voz.py` | Extrae embedding del audio de referencia → `PerfilVozActor` |
| `distancia_coseno()` | `matching.py` | Distancia coseno (Python puro con `math`) |
| `clasificar_speaker()` | `matching.py` | Clasifica un speaker: aceptado / ambiguo / rechazado |
| `identificar_actor()` | `matching.py` | Compara todos los speakers contra el perfil → `SpeakerMatch[]` |
| `filtrar_por_actor()` | `alineacion.py` | Conserva segmentos cuyo midpoint cae en un turno del actor |

### `diarizacion.py` — Extracción de turnos

**`diarizar(audio_path, pipeline, video_id) → list[TurnoOrador]`**

Función pura que recibe un pipeline ya cargado (no lo instancia) y ejecuta la diarización. Usa `exclusive_speaker_diarization` —no `itertracks` estándar— para obtener turnos **sin traslapes**, ideal para alinear limpiamente con los timestamps de Whisper.

### Perfil de voz del actor

**Helper puro:** `construir_perfil_desde_output(output, actor, video_ref_id, pipeline_name) → PerfilVozActor`.

**Producción actual:** `DiarizacionService._get_perfil(nombre_playlist)` ejecuta el pipeline sobre el audio de referencia y construye el perfil desde `output.speaker_embeddings` público, eligiendo el speaker con mayor duración total. El resultado se aplana a `list[float]` para mantener el DTO libre de numpy. No se accede a `pipeline._inferences` ni a ningún API privado.

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

**`identificar_actor(speaker_embeddings, perfil, umbral_match=0.5, umbral_ambiguo=0.7) → list[SpeakerMatch]`**

Compara cada `{speaker_id: embedding_promedio}` contra `perfil.embedding`. Devuelve la lista ordenada por distancia ascendente.

### `alineacion.py` — Alineación con Whisper

**`filtrar_por_actor(transcript, turnos_actor) → VideoTranscript`**

Conserva solo los `SegmentoRaw` cuyo **midpoint temporal** cae dentro de algún turno del actor. Criterio de pertenencia:

```python
midpoint = (seg.t_start + seg.t_end) / 2.0
r_start <= midpoint < r_end   # inclusivo en inicio, exclusivo en fin
```

- Turnos de otro `video_id` se ignoran.
- Devuelve un `VideoTranscript` nuevo con metadata preservada y `raw_segments` filtrado.
- Si no hay turnos del actor para ese video, devuelve transcript vacío (metadata igual, 0 segmentos).

### `service.py` — Orquestación

**`DiarizacionService.procesar(transcripts, nombre_playlist) → list[VideoTranscript]`**

Pipeline **video-por-video** (no batch):

1. Construye el `PerfilVozActor` una sola vez (cache en memoria).
2. Resuelve el pipeline de diarización.
3. Por cada `VideoTranscript`:
   - **Diarizar**: `pipeline(audio_path)` → `exclusive_speaker_diarization` + `speaker_embeddings`
   - Si no hay turnos → transcript vacío, siguiente video.
   - **Embeddings por speaker**: `_extraer_embeddings(output)` lee `output.speaker_embeddings` alineado con `output.speaker_diarization.labels()` → `{speaker_id: embedding}`.
   - **Identificar actor**: `identificar_actor(speaker_embs, perfil)` → `SpeakerMatch[]`; los aceptados son el actor.
   - Si no hay speakers aceptados → transcript vacío, siguiente video.
   - **Filtrar**: `filtrar_por_actor(transcript, turnos_actor)` → `VideoTranscript` con solo segmentos del actor.

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
| `modelo_embedding` | `str` | Identificador del origen del embedding (`speaker_embeddings:<pipeline_name>`) |
| `duracion_segundos` | `float` | Duración del audio de referencia procesado |

### `SpeakerMatch`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `speaker_id` | `str` | Etiqueta del speaker evaluado |
| `distancia` | `float` | Distancia coseno al perfil del actor |
| `aceptado` | `bool` | `True` si se acepta como el actor objetivo |
| `es_ambiguo` | `bool` | `True` si el match cae en zona ambigua (descartar) |

## Decisiones de implementación

### `exclusive_speaker_diarization` para alineación limpia

Se usa `output.exclusive_speaker_diarization` en vez del `itertracks` estándar. Esto produce turnos **sin traslapes**: cada instante del audio perteneve a un solo speaker. Sin traslapes, el criterio de midpoint en `filtrar_por_actor` no tiene ambigüedad — un segmento de Whisper siempre cae dentro de un único turno.

### Pipeline cargado una vez y embeddings nativos del output

`DiarizacionService` hace lazy-loading de pyannote (`_get_pipeline`) y construye el perfil de referencia desde el mismo `output.speaker_embeddings` público del pipeline. Para cada video usa una sola llamada al pipeline y extrae turnos desde `exclusive_speaker_diarization` y embeddings desde `output.speaker_embeddings`; esto evita cargar un modelo separado o recurrir a APIs privadas, y mantiene los contratos serializables.

### Criterio midpoint en alineación

Para decidir si un `SegmentoRaw` pertenece al actor, se calcula `midpoint = (t_start + t_end) / 2` y se verifica si cae en algún turno. La frontera es semiabierta `[inicio, fin)`: inclusiva en el `t_start` del turno, exclusiva en `t_end`. Esto evita que dos turnos adyacentes se adjudiquen el mismo segmento.

### `distancia_coseno` sin numpy

Implementada en Python puro con `math` (`zip(strict=True)` + `math.sqrt`). Ninguna función pura del componente importa numpy — solo el adapter/extractor interno convierte arrays del pipeline a `list[float]` antes de salir del service.

### Fronteras exclusivas en `clasificar_speaker`

```text
distancia < umbral_match              → aceptado
umbral_match ≤ distancia < umbral_ambiguo → ambiguo
distancia ≥ umbral_ambiguo            → rechazado
```

`<` en el umbral de aceptación; `>=` en el de ambigüedad. La zona intermedia `[umbral_match, umbral_ambiguo)` se descarta como ambigua — no se pide selección manual, se trata como no-actor y el pipeline continúa.

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

### Salida hacia Segmentación

El contrato de salida es `list[VideoTranscript]` —mismo tipo que la entrada— con `raw_segments` filtrado a solo los del actor. No se introducen DTOs nuevos en la frontera con Segmentación; el `VideoTranscript` ya existe y se reutiliza.

## Dependencias externas

| Herramienta | Uso |
|-------------|-----|
| `pyannote.audio` + `pyannote/speaker-diarization-community-1` | Primary oficial de diarización + `speaker_embeddings` por speaker |
| `pyannote-community/speaker-diarization-community-1` | Fallback local validado si el primary no está disponible |
| `pyannote.core` + `Audio` | Helper para recortar waveforms por turno y medir duración |
| GPU recomendada | pyannote es lento en CPU; GPU acelera diarización |

## Notas de implementación

- El `PerfilVozActor` **no se persiste** en disco — el cache es solo en memoria durante la ejecución del pipeline.
- El `DiarizacionService` resuelve rutas de audio con `ruta_audio()` del Componente 1 (Ingesta), asumiendo que los `.wav` ya fueron descargados.
- **Smoke Fase 1 real completado:** perfil + matching + segmentación + temas fueron validados en `politest-smoke-device-fix` sobre los 7 videos de `Play-PoliTest`; 7/7 procesados, 139 segmentos del actor, 2 tópicos descubiertos y 0 videos omitidos. El fallo previo de pyannote 4.x (`pipeline.to(str)`) se corrigió pasando `torch.device` al pipeline.
