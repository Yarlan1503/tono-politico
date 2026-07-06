# Componente 1: Ingesta

> **Estado:** ✅ Completo · **Tests:** 47

## Propósito

Recibe la URL de una playlist de YouTube, descarga el audio de cada video, lo transcribe con Whisper y devuelve transcripciones estructuradas con timestamps por palabra y pausas entre segmentos.

## Arquitectura

```
URL Playlist
    │
    ▼ obtener_info_playlist()    — yt-dlp --flat-playlist
    │
    ▼ verificar_cache_transcripciones()
    │
    ├─ Cache hit → cargar_transcripcion()
    │
    └─ Cache miss
        │
        ▼ verificar_cache_videos()
        │
        ├─ Cache hit → usar .wav existente
        └─ Cache miss → descargar_audio()  — yt-dlp -x
        │
        ▼ transcribir()            — Whisper large-v3-turbo
        │
        ▼ guardar_transcripcion()  — persistir a JSON
        │
        ▼ cargar_transcripcion()   — reconstruir VideoTranscript
        │
    list[VideoTranscript]
```

## API

### `IngestaService`

```python
from pathlib import Path
from tono_politico.ingesta import IngestaService

svc = IngestaService(
    data_dir=Path("data"),     # raíz del cache local
    whisper_model="large-v3-turbo",  # modelo de Whisper
    idioma="es",               # idioma forzado para transcripción
)

transcripciones: list[VideoTranscript] = svc.procesar(
    "https://youtube.com/playlist?list=..."
)
```

### Configuración

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `data_dir` | `Path` | `Path("data")` | Directorio raíz para cache de audios y transcripciones |
| `whisper_model` | `str` | `"large-v3-turbo"` | Modelo de Whisper a cargar |
| `idioma` | `str` | `"es"` | Código de idioma para forzar la transcripción |

## Módulos internos

### `service.py` — Orquestador OOP

`IngestaService` encapsula toda la configuración y orquesta el flujo completo. Implementa `ComponenteProtocol`.

**Flujo de `procesar()`:**

1. `obtener_info_playlist(url)` → metadata de la playlist
2. `verificar_cache_transcripciones(nombre, videos, data_dir)` → qué ya está listo
3. Para los faltantes:
   - `verificar_cache_videos(nombre, faltantes, data_dir)` → qué audios ya existen
   - `descargar_audio(video, nombre, data_dir)` → yt-dlp para los que faltan
   - `transcribir(audio, modelo, idioma)` → Whisper
   - `guardar_transcripcion(transcript, nombre, data_dir)` → persistir
4. `cargar_transcripcion(ruta)` para todos → `list[VideoTranscript]`

### `playlist.py` — Metadata

**`obtener_info_playlist(url) → PlaylistInfo`**

Usa `yt-dlp --flat-playlist --extractor-args youtubetab:approximate_date -j` para obtener:
- Nombre de la playlist (sanitizado para directorios)
- Lista de videos con: `id`, `titulo`, `url`, `duracion`, `fecha` (YYYYMMDD)

Los videos privados, eliminados o inaccesibles se filtran automáticamente.

**`sanitizar_nombre_directorio(nombre) → str`**

Reemplaza caracteres problemáticos, colapsa espacios, elimina guiones bajos extremos. Fallback: `"playlist_sin_nombre"`.

### `audio.py` — Descarga y cache

**`verificar_cache_videos(nombre, videos, base_dir) → {existentes, faltantes}`**

Revisa `videos-<playlist>/` y compara archivos `.wav` contra los video_ids.

**`descargar_audio(video, nombre, base_dir) → Path`**

Ejecuta `yt-dlp -x --audio-format wav -f bestaudio/best` y guarda como `<video_id>.wav`.

### `transcripcion.py` — Whisper + persistencia

**`transcribir(audio_path, modelo, idioma) → list[SegmentoRaw]`**

Carga Whisper, transcribe con `word_timestamps=True` y `fp16=False` (CPU-friendly). Normaliza la salida a `SegmentoRaw` con:
- `texto`: texto limpio sin espacios extremos
- `t_start` / `t_end`: timestamps en segundos
- `pausa_antes`: gap acústico `t_start[n] - t_end[n-1]` (primer segmento = 0.0)
- `words`: `list[WordTimestamp]` con palabra, timestamps y probabilidad

**`guardar_transcripcion(transcript, nombre, base_dir) → Path`**

Serializa `VideoTranscript` a JSON con el formato documentado abajo.

**`cargar_transcripcion(ruta) → VideoTranscript`**

Reconstruye un `VideoTranscript` desde JSON. Lanza `FileNotFoundError` si no existe, `ValueError` si el JSON es inválido.

**`verificar_cache_transcripciones(nombre, videos, base_dir)`**

Considera válida solo una transcripción cuyo JSON existe, parsea correctamente, y su campo `video_id` coincide. Un JSON corrupto o con ID distinto se trata como faltante.

### `cache.py` — Convención de rutas

Todas las funciones reciben `base_dir` opcional (default = `DATA_DIR = Path("data")`):

```python
ruta_dir_videos(nombre, base_dir)           # data/<playlist>/videos-<playlist>/
ruta_dir_transcripciones(nombre, base_dir)  # data/<playlist>/transcripciones-<playlist>/
ruta_audio(nombre, video_id, base_dir)      # .../videos-<playlist>/<video_id>.wav
ruta_transcripcion(nombre, video_id, base_dir)  # .../transcripciones-<playlist>/<video_id>.json
```

## DTOs

Definidos en `src/tono_politico/models.py` (nivel paquete, compartidos):

### `WordTimestamp`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `word` | `str` | Palabra transcrita |
| `start` | `float` | Tiempo de inicio (segundos) |
| `end` | `float` | Tiempo de fin (segundos) |
| `probability` | `float \| None` | Confianza de Whisper |

### `SegmentoRaw`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `texto` | `str` | Texto transcrito |
| `t_start` | `float` | Inicio (segundos) |
| `t_end` | `float` | Fin (segundos) |
| `pausa_antes` | `float` | Gap respecto al segmento anterior |
| `words` | `list[WordTimestamp]` | Timestamps por palabra |

### `VideoTranscript`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `video_id` | `str` | ID del video de YouTube |
| `url` | `str` | URL del video |
| `titulo` | `str` | Título del video |
| `fecha` | `str \| None` | Fecha YYYYMMDD |
| `raw_segments` | `list[SegmentoRaw]` | Segmentos de Whisper |

### `PlaylistInfo` / `VideoInfo`

Metadata de la playlist y videos individuales (id, título, url, duración, fecha).

## Formato JSON de transcripción

```json
{
  "video_id": "abc123",
  "url": "https://www.youtube.com/watch?v=abc123",
  "titulo": "Discurso en conferencia",
  "fecha": "20260315",
  "raw_segments": [
    {
      "texto": "Hoy quiero hablar de seguridad pública.",
      "t_start": 0.0,
      "t_end": 3.5,
      "pausa_antes": 0.0,
      "words": [
        {
          "word": "Hoy",
          "start": 0.0,
          "end": 0.4,
          "probability": 0.95
        }
      ]
    }
  ]
}
```

## Dependencias externas

| Herramienta | Uso |
|-------------|-----|
| `yt-dlp` | Descarga de metadata de playlists y extracción de audio |
| `openai-whisper` | Transcripción con word timestamps |
| CPU | `fp16=False` — no requiere GPU |

## Decisiones de diseño

- **Cache de dos niveles:** audios `.wav` (caros de descargar) y transcripciones `.json` (caras de generar con Whisper). Ambos son idempotentes y validados.
- **`base_dir` threaded:** todas las funciones de cache reciben `base_dir` opcional. El service pasa `self.data_dir`. Tests usan `tmp_path` sin mutar globales.
- **Import diferido de Whisper:** `import whisper` dentro de `transcribir()`, no a nivel módulo. Permite testear sin instalar Whisper.
- **`pausa_antes` calculada en `transcribir`:** se calcula como `t_start[n] - t_end[n-1]` durante la normalización, no en un paso separado.
