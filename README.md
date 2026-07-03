# Tono Político

Herramienta de análisis NLP para determinar el tono de un actor político configurable respecto a un tema, usando transcripciones de videos de YouTube como fuente.

## Tres lecturas de tono

1. **Sentimiento** — positivo / negativo / neutral
2. **Stance** — favor / contra / neutro respecto al tema
3. **Tono retórico** — populista / técnico / institucional (eje primario) + confrontativo / conciliador / emocional / nacionalista / victimizante (eje secundario multi-label)

## Arquitectura

El proyecto sigue una arquitectura orientada a servicios. Cada componente es un service que implementa `ComponenteProtocol` (contrato `.procesar(input) → output`), con configuración encapsulada en su constructor.

```
1. Ingesta        →  YouTube playlist → Whisper → transcripciones con timestamps
2. Segmentación   →  Transcripciones → spaCy + embeddings → segmentos semánticos
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
├── segmentacion/          # Componente 2 ✅
│   ├── service.py         # SegmentacionService (orquestador OOP)
│   ├── sentencias.py      # spaCy → oraciones con timestamps
│   ├── breakpoints.py     # Distancia coseno + percentil 95 (estándar LangChain)
│   ├── agrupacion.py      # Guardrails (min/max oraciones, max palabras)
│   └── models.py          # DTOs (Oracion, Segmento)
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
- **Transcripciones:** si el `.json` existe y su `video_id` coincide, no se retranscribe.

### Módulos internos

| Módulo | Responsabilidad |
|--------|----------------|
| `service.py` | `IngestaService` — orquesta todo el flujo con config encapsulada |
| `playlist.py` | `obtener_info_playlist(url)` — metadata vía `yt-dlp --flat-playlist` |
| `audio.py` | `verificar_cache_videos` + `descargar_audio` — descarga y cache `.wav` |
| `transcripcion.py` | `transcribir` (Whisper) + `guardar`/`cargar`/`verificar` JSON |
| `cache.py` | Rutas centralizadas con `base_dir` inyectable |

## Componente 2: Segmentación

Toma las transcripciones crudas de Whisper y las reagrupa en segmentos semánticamente coherentes — bloques de discurso que tratan un mismo tema.

### Pipeline

```
VideoTranscript[] (Whisper)
    │
    ▼ extraer_oraciones()     — spaCy divide en Oracion[] con words asignadas
    │
    ▼ detectar_breakpoints()  — embeddings detectan cambios de tópico
    │
    ▼ agrupar_segmentos()     — guardrails → Segmento[]
```

### Detección de breakpoints semánticos

Sigue el estándar de **LangChain SemanticChunker** / **LlamaIndex**:

1. Codifica todas las oraciones con sentence-transformers
2. Calcula **distancia coseno** entre oraciones consecutivas
3. Marca breakpoint donde la distancia supera el **percentil 95**

La segmentación acústica la realiza Whisper internamente — este componente se enfoca exclusivamente en la señal semántica.

### Guardrails

| Parámetro | Default | Función |
|-----------|---------|---------|
| `min_oraciones` | 2 | Fusiona segmentos demasiado pequeños con el anterior |
| `max_oraciones` | 8 | Subdivide segmentos demasiado largos |
| `max_palabras` | 150 | Límite de palabras por segmento |

### Uso

```python
from tono_politico.segmentacion import SegmentacionService

svc = SegmentacionService(
    spacy_model="es_core_news_lg",
    breakpoint_percentile=95,
)

segmentos = svc.procesar(transcripciones)
```

## Estado del proyecto

| Componente | Estado | Tests |
|------------|--------|-------|
| 1. Ingesta | ✅ Completo | 47 |
| 2. Segmentación | ✅ Completo | 35 |
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
