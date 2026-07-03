# Tono Político

Herramienta de análisis NLP para determinar el tono de un actor político configurable respecto a un tema, usando transcripciones de videos de YouTube como fuente.

## Tres lecturas de tono

1. **Sentimiento** — positivo / negativo / neutral
2. **Stance** — favor / contra / neutro respecto al tema
3. **Tono retórico** — populista / técnico / institucional (eje primario) + confrontativo / conciliador / emocional / nacionalista / victimizante (eje secundario multi-label)

## Arquitectura

El proyecto sigue una arquitectura orientada a servicios. Cada componente es un service que implementa `ComponenteProtocol` (contrato `.procesar(input) → output`), con configuración encapsulada en su constructor.

```
1. Ingesta        →  YouTube playlist → Whisper → transcripciones con timestamps + pausas
2. Segmentación   →  Segmentos crudos → segmentos semánticos coherentes (pausas + embeddings)
3. Temas          →  BERTopic → descubrimiento automático de temas predominantes
4. Filtrado       →  Tema seleccionado → subset de segmentos relevantes
5. Tono           →  3 modelos zero-shot sobre los segmentos filtrados
6. Salida         →  Agregación → JSON
```

### Estructura del código

```
src/tono_politico/
├── models.py              # DTOs compartidos (WordTimestamp, SegmentoRaw, VideoTranscript, ...)
├── protocol.py            # ComponenteProtocol
├── ingesta/               # Componente 1 ✅
│   ├── service.py         # IngestaService (orquestador OOP)
│   ├── playlist.py        # Metadata de playlists (yt-dlp)
│   ├── audio.py           # Descarga y cache de audios .wav
│   ├── transcripcion.py   # Whisper + persistencia JSON
│   └── cache.py           # Convención de rutas (base_dir threaded)
├── segmentacion/          # Componente 2 (pendiente)
├── temas/                 # Componente 3 (pendiente)
├── filtrado/              # Componente 4 (pendiente)
├── tono/                  # Componente 5 (pendiente)
└── salida/                # Componente 6 (pendiente)
```

## Configuración

```bash
# Crear entorno virtual con uv
uv venv --python 3.11

# Instalar dependencias (incluye el paquete en modo editable)
uv pip install -e ".[dev]"
```

## Herramientas

El proyecto usa el ecosistema de [Astral](https://astral.sh):

| Herramienta | Uso | Comando |
|-------------|-----|---------|
| **uv** | Gestión de dependencias y entorno | `uv run pytest tests/` |
| **ruff** | Linter + formatter | `ruff check src/ tests/` |
| **ty** | Type checker | `ty check src/` |

## Componente 1: Ingesta

Recibe la URL de una playlist de YouTube, descarga el audio de cada video, lo transcribe con Whisper y devuelve transcripciones estructuradas con timestamps y pausas entre segmentos.

### Uso

```python
from pathlib import Path
from tono_politico.ingesta import IngestaService

svc = IngestaService(
    data_dir=Path("data"),
    whisper_model="large-v3",
    idioma="es",
)

transcripciones = svc.procesar("https://youtube.com/playlist?list=...")
```

### Cache de dos niveles

La ingesta mantiene cache inteligente para evitar trabajo repetido:

```
data/
└── <playlist>/
    ├── videos-<playlist>/              # audios .wav
    │   ├── <video_id>.wav
    │   └── ...
    └── transcripciones-<playlist>/     # transcripciones .json
        ├── <video_id>.json
        └── ...
```

- **Audios:** si el `.wav` ya existe, no se descarga de nuevo.
- **Transcripciones:** si el `.json` existe y su `video_id` coincide, no se retranscribe. Un JSON corrupto o con ID distinto se trata como faltante.

### Formato de transcripción

Cada `VideoTranscript` serializado a JSON contiene:

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
        {"word": "Hoy", "start": 0.0, "end": 0.4, "probability": 0.95}
      ]
    }
  ]
}
```

El campo `pausa_antes` es el gap acústico en segundos entre el segmento actual y el anterior (`t_start[n] - t_end[n-1]`). El primer segmento siempre tiene `pausa_antes = 0.0`.

### Módulos internos

| Módulo | Responsabilidad |
|--------|----------------|
| `service.py` | `IngestaService` — orquesta todo el flujo con config encapsulada |
| `playlist.py` | `obtener_info_playlist(url)` — metadata vía `yt-dlp --flat-playlist` |
| `audio.py` | `verificar_cache_videos` + `descargar_audio` — descarga y cache `.wav` |
| `transcripcion.py` | `transcribir` (Whisper) + `guardar`/`cargar`/`verificar` JSON |
| `cache.py` | Rutas centralizadas con `base_dir` inyectable |

## Estado del proyecto

| Componente | Estado | Tests |
|------------|--------|-------|
| 1. Ingesta | ✅ Completo | 47 |
| 2. Segmentación | Pendiente | — |
| 3. Temas | Pendiente | — |
| 4. Filtrado | Pendiente | — |
| 5. Tono | Pendiente | — |
| 6. Salida | Pendiente | — |

## Uso completo (próximamente)

```bash
# Descubrir temas predominantes
uv run python -m tono_politico --playlist "URL" --actor "Sheinbaum" --descubrir-temas

# Analizar tono sobre un tema específico
uv run python -m tono_politico --playlist "URL" --actor "Sheinbaum" --tema "seguridad pública"
```
