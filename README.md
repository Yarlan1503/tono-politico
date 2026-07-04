# Tono Político

Herramienta NLP para analizar el tono de actores políticos mexicanos a partir de transcripciones de YouTube.

El objetivo es pasar de una playlist a evidencia segmentada por tema y, después, a tres lecturas de tono:

1. **Sentimiento** — positivo / negativo / neutral.
2. **Stance** — a favor / en contra / neutral respecto a un tema.
3. **Tono retórico** — eje primario: populista / técnico / institucional; eje secundario multi-label: confrontativo, conciliador, emocional, nacionalista, victimizante.

## Pipeline

```text
1. Ingesta       YouTube playlist -> Whisper -> transcripciones con timestamps
2. Segmentación  transcripciones -> spaCy + embeddings -> segmentos semánticos
3. Temas         segmentos -> BERTopic -> tópicos y asignaciones
4. Filtrado      tópico/tema seleccionado -> subset relevante
5. Tono          modelos zero-shot -> sentimiento, stance y tono retórico
6. Salida        agregación + provenance -> JSON/reportes
```

## Estado actual

| Componente | Estado | Tests | Salida |
|---|---:|---:|---|
| 1. Ingesta | ✅ Completo | 47 | `list[VideoTranscript]` |
| 2. Segmentación | ✅ Completo | 35 | `list[Segmento]` |
| 3. Temas | ✅ MVP implementado | 11 | `ResultadoTemas` |
| 4. Filtrado | ✅ MVP implementado | 5 | `ResultadoFiltrado` |
| 5. Tono | Pendiente | — | lecturas de tono |
| 6. Salida | Pendiente | — | JSON/reportes |

Verificación local actual: `98 passed`, `ruff check` limpio y `ty check` limpio.

## Decisiones de arquitectura

- **Services OOP por componente:** cada componente implementa `ComponenteProtocol` mediante `.procesar(input) -> output`.
- **Config encapsulada:** los hiperparámetros viven en el constructor del service.
- **Helpers puros:** la lógica interna se mantiene en funciones testeables.
- **Lazy loading:** Whisper, spaCy, sentence-transformers y BERTopic se cargan solo cuando se usan.
- **DTOs compartidos vs locales:** `src/tono_politico/models.py` contiene DTOs compartidos por más de un componente (`VideoTranscript`, `SegmentoRaw`, `WordTimestamp`, etc.). Los DTOs específicos viven dentro de su componente, por ejemplo `segmentacion/models.py` y `temas/models.py`.
- **Embeddings compartidos:** Segmentación y Temas usan `LiquidAI/LFM2.5-Embedding-350M` para mantener consistencia semántica entre detección de cortes y clustering temático.

## Estructura del código

```text
src/tono_politico/
├── models.py              # DTOs compartidos: WordTimestamp, SegmentoRaw, VideoTranscript, VideoInfo, PlaylistInfo
├── protocol.py            # ComponenteProtocol
├── ingesta/               # Componente 1 ✅
│   ├── service.py         # IngestaService
│   ├── playlist.py        # metadata de playlists vía yt-dlp
│   ├── audio.py           # descarga/cache de audios .wav
│   ├── transcripcion.py   # Whisper + JSON
│   └── cache.py           # rutas centralizadas
├── segmentacion/          # Componente 2 ✅
│   ├── service.py         # SegmentacionService
│   ├── sentencias.py      # spaCy -> Oracion[]
│   ├── breakpoints.py     # distancia coseno + percentil 95
│   ├── agrupacion.py      # guardrails min/max
│   └── models.py          # Oracion, Segmento
├── temas/                 # Componente 3 ✅ MVP
│   ├── service.py         # TemasService
│   ├── descubrimiento.py  # BERTopic + UMAP + HDBSCAN
│   └── models.py          # SegmentoTematizado, TopicoInfo, ResultadoTemas
├── filtrado/              # Componente 4 ✅ MVP
│   ├── service.py         # FiltradoService
│   ├── filtro.py          # filtrado determinista por tópico/relevancia
│   └── models.py          # CriterioFiltrado, SegmentoFiltrado, ResultadoFiltrado
├── tono/                  # Componente 5 (pendiente)
└── salida/                # Componente 6 (pendiente)
```

## Configuración

Defaults de proyecto: [`config/config.yaml`](config/config.yaml).

Importante: el YAML documenta la configuración canónica; por ahora los services reciben esos valores por constructor. Todavía no hay loader global/CLI que lea automáticamente ese archivo.

## Entorno local

El proyecto usa el stack Astral: `uv`, `ruff`, `ty`.

```bash
cd ~/Documentos/Proyectos/tono-politico
uv venv --python 3.11
uv pip install -e ".[dev]"
uv lock
```

Modelo spaCy para ejecución real del Componente 2:

```bash
uv run python -m spacy download es_core_news_lg
```

## Calidad

```bash
# Tests
uv run pytest tests/ -v

# Lint
ruff check src/ tests/

# Type check
ty check src/

# Todo antes de cerrar un cambio
ruff check src/ tests/ && ty check src/ && uv run pytest tests/ -v
```

## Uso por componente

### Componente 1: Ingesta

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

### Componente 2: Segmentación

```python
from tono_politico.segmentacion import SegmentacionService

svc = SegmentacionService(
    spacy_model="es_core_news_lg",
    breakpoint_percentile=95,
    min_oraciones=2,
    max_oraciones=8,
    max_palabras=150,
)

segmentos = svc.procesar(transcripciones)
```

### Componente 3: Temas

```python
from tono_politico.temas import TemasService

svc = TemasService(
    min_topic_size=3,
    n_neighbors=10,
    n_components=5,
)

resultado = svc.procesar(segmentos)
```

`resultado` contiene:

- `segmentos`: cada `Segmento` con `topico_id` y `probabilidad`.
- `topicos`: metadata de tópicos (`palabras_clave`, conteo, representatividad).
- `num_topicos`: número de tópicos sin contar outliers (`-1`).

### Componente 4: Filtrado

```python
from tono_politico.filtrado import FiltradoService

svc = FiltradoService(
    topico_id=0,
    min_relevancia=0.35,
    incluir_outliers=False,
)

resultado_filtrado = svc.procesar(resultado)
```

`resultado_filtrado` contiene:

- `criterio`: tópico elegido, umbral y política de outliers.
- `topico`: metadata del tópico elegido si existe.
- `segmentos`: subset de `SegmentoFiltrado` para pasar a análisis de tono.
- `total_segmentos_entrada` / `total_segmentos_filtrados`: conteos de provenance.

## Documentación técnica

- [Componente 1: Ingesta](docs/componente-1-ingesta.md)
- [Componente 2: Segmentación](docs/componente-2-segmentacion.md)
- [Componente 3: Temas](docs/componente-3-temas.md)
- [Componente 4: Filtrado](docs/componente-4-filtrado.md)
- [Configuración](docs/configuracion.md)

## Próximos componentes

5. **Tono** — aplicar zero-shot `xlm-roberta-large-xnli` en tres lecturas.
6. **Salida** — agregar resultados con provenance y exportar JSON/reportes.
