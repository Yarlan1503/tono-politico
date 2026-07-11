# Componente `audio_fetcher`

> **Ruta:** `src/tono_politico/speech2text/audio_fetcher/`
>
> **Responsabilidad:** descubrir metadata de playlists y entregar audio `.wav` local para las etapas posteriores.
>
> **No hace:** diarización, Whisper, identificación de speakers ni transcripción.

## Flujo

```text
URL playlist
    │
    ▼ obtener_info_playlist()
PlaylistInfo(nombre, nombre_cache, playlist_id, url) + VideoMeta[]
    │
    ▼ AudioFetcherService.fetch_one(meta, playlist)
AudioVideo | None
```

## API pública

### `AudioFetcherService`

Módulo: `service.py`.

```python
class AudioFetcherService:
    def __init__(self, data_dir: Path = Path("data")) -> None: ...

    def discover(
        self,
        url_playlist: str,
    ) -> tuple[PlaylistInfo, list[VideoMeta]]: ...

    def fetch_one(
        self,
        video: VideoMeta,
        playlist: PlaylistInfo | str,
        *,
        archive_path: Path | None = None,
    ) -> AudioVideo | None: ...
```

#### `discover`

Delega en `obtener_info_playlist()` y no descarga audio. Conserva el nombre visible original y calcula `nombre_cache` como identidad segura para filesystem.

#### `fetch_one`

- reutiliza el `.wav` si ya existe;
- si no existe, llama a `descargar_audio_result()`;
- devuelve `AudioVideo` solo cuando el archivo está disponible;
- devuelve `None` ante un fallo de descarga, dejando el detalle en logs y en el `DownloadResult` interno.

El loop sobre varios videos pertenece al `ExecutionRunner`.

## DTOs

### `PlaylistInfo`

```python
@dataclass
class PlaylistInfo:
    nombre: str
    nombre_cache: str | None = None
    playlist_id: str | None = None
    url: str | None = None
```

`nombre` es el nombre visible; `nombre_cache` es la versión sanitizada. `playlist_id` y `url` permiten reconstruir provenance. No contiene la lista de vídeos.

### `VideoMeta`

DTO de discover, antes de descargar:

| Campo | Tipo | Uso |
|---|---|---|
| `video_id` | `str` | identidad estable del video |
| `url` | `str` | URL resuelta por yt-dlp |
| `titulo` | `str` | título de presentación/log |
| `fecha` | `str \\| None` | fecha `YYYYMMDD` si existe |
| `fecha_fuente` | `str \\| None` | `upload_date`, `release_date`, `timestamp`, `missing` o `invalid` |
| `duracion` | `float` | duración en segundos |

### `AudioVideo`

DTO posterior a la descarga. Repite la metadata de `VideoMeta`, conserva `playlist` y añade `audio_path: Path` obligatorio.

```python
AudioVideo.from_meta(meta, audio_path=path)
```

### `DownloadResult`

Resultado estructurado de una unidad de descarga:

| Campo | Tipo | Significado |
|---|---|---|
| `video_id` | `str` | video procesado |
| `path` | `Path \| None` | ruta si la descarga fue válida |
| `ok` | `bool` | éxito de la operación |
| `error` | `str \| None` | motivo resumido del fallo |

## Módulos

| Archivo | Responsabilidad |
|---|---|
| `models.py` | DTOs de metadata, audio y descarga |
| `playlist.py` | `sanitizar_nombre_directorio`, `obtener_info_playlist` |
| `cache.py` | `DATA_DIR`, `ruta_dir_videos`, `ruta_audio` |
| `audio.py` | `verificar_cache_videos`, `descargar_audio_result` |
| `service.py` | fachada `discover` + `fetch_one` |
| `__init__.py` | exports públicos del subpaquete |

## Discover con yt-dlp

`obtener_info_playlist()` ejecuta:

```text
yt-dlp --flat-playlist --extractor-args youtubetab:approximate_date -j --no-warnings <url>
```

Por cada línea JSON válida construye `VideoMeta`. La fecha canónica es `upload_date`, con fallback a `release_date` y `timestamp`. Cuando no es confiable se conserva `fecha=None` y `fecha_fuente` queda en `missing` o `invalid`; la duración ausente se normaliza a `0.0`; si falta la URL individual se construye una URL de YouTube por `video_id`.

Los errores de yt-dlp producen `RuntimeError` en discover. Las líneas JSON inválidas se omiten con warning.

## Cache

```text
data/<playlist_sanitizada>/
└── videos-<playlist_sanitizada>/
    └── <video_id>.wav
```

Las convenciones están centralizadas en `cache.py`. No se almacenan aquí transcripciones JSON ni rutas de Whisper.

## Descarga

`descargar_audio_result()` usa yt-dlp con extracción de audio y conversión WAV:

- formato `wav`;
- formato preferido `bestaudio/best`;
- reintentos `10`;
- timeout de proceso de `600s`;
- `--download-archive` opcional cuando el runner entrega `archive_path`.

Un timeout, código de salida no cero, archivo ausente o descarga incompleta devuelven `DownloadResult(ok=False)`.

## Integración

`SpeechToTextService` utiliza este componente en dos momentos:

1. `ensure_perfil()` descarga el `video_ref_id`.
2. `procesar_one()` descarga/reutiliza el audio del video actual.

`audio_fetcher` no conoce la identidad del actor ni el modelo ASR.

## Tests

```bash
uv run pytest tests/speech2text/test_audio_fetcher_*.py -q
```

La cobertura incluye sanitización, parseo de playlist, DTOs, cache, descargas exitosas/fallidas y ausencia del wrapper batch legacy `AudioFetcherService.procesar`.
