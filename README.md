# Tono Político

Herramienta NLP para analizar el tono de actores políticos mexicanos a partir de transcripciones de YouTube.

## Pipeline

```text
1. Ingesta       YouTube playlist → Whisper large-v3-turbo → transcripciones con timestamps
1.5 Diarización  pyannote.audio + perfil de voz → texto del actor objetivo
2. Segmentación  texto del actor → spaCy + embeddings → segmentos semánticos
3. Temas         segmentos → BERTopic → tópicos y asignaciones
4. Filtrado      tópico/tema seleccionado → subset relevante
5. Tono          embeddings + LLM → stance, intensidad, lógica política, sentimiento, estilo, función
6. Salida        agregación + provenance → JSON + Markdown
```

## Estado actual

| Componente | Estado | Tests | Salida |
|---|---:|---:|---|
| 1. Ingesta | ✅ Completo | 47 | `list[VideoTranscript]` |
| 1.5 Diarización / actor | ✅ Completo | 62 | `list[VideoTranscript]` (filtrado) |
| 2. Segmentación | ✅ Completo | 35 | `list[Segmento]` |
| 3. Temas | ✅ Completo | 11 | `ResultadoTemas` |
| 4. Filtrado | ✅ Completo | 5 | `ResultadoFiltrado` |
| 5. Tono | ✅ Completo | 61 | `ResultadoTono` |
| 6. Salida | ✅ Completo | 35 | `InformeTono` |

Verificación local: `256 passed` (+ 5 slow), `ruff check` limpio.

## Uso del pipeline completo

El pipeline se ejecuta con `main.py`, que lee `config/config.yaml` automáticamente.

```bash
# Fase 1: descubrir tópicos
uv run python main.py --playlist "https://youtube.com/playlist?list=..."

# Fase 2: analizar un tópico específico
uv run python main.py \
    --playlist "https://youtube.com/playlist?list=..." \
    --topico 0 \
    --tema "fracking" \
    --output output/
```

**Flujo de dos fases:** la Fase 1 (Ingesta → Diarización → Segmentación → Temas) siempre se ejecuta y muestra los tópicos descubiertos. La Fase 2 (Filtrado → Tono → Salida) requiere `--topico N` y `--tema "descripción"`, y genera el informe final en `output/`.

## Componente 1.5: Diarización — detección del actor objetivo

El pipeline no asume que todo el audio pertenece al actor. Un componente de diarización identifica quién habla cuándo usando `pyannote/speaker-diarization-community-1`, compara cada speaker contra un perfil de voz del actor objetivo, y filtra las transcripciones para conservar solo las intervenciones del actor.

**Modelo de diarización:** `pyannote/speaker-diarization-community-1` (VBxClustering, threshold 0.6)  
**Embeddings de voz:** `pyannote/embedding` (x-vector TDNN, 2.8% EER en VoxCeleb1)  
**Criterio de matching:** distancia coseno contra perfil de voz, thresholds 0.5 (aceptar) / 0.7 (rechazar)  
**Criterio de alineación:** midpoint temporal del segmento dentro del turno del actor  
**Política de ambigüedad:** si el match cae en zona ambigua (0.5–0.7), se descarta como otro speaker  

## Componente 5: Tono — arquitectura híbrida

El análisis de tono usa dos enfoques complementarios de la familia Liquid AI:

**Embeddings** (`LFM2.5-Embedding-350M` con mean pooling manual):

| Dimensión | Labels |Qué mide |
|---|---|---|
| Lógica política | 6 | nacionalista, globalista, populista, tecnócrata, corporativista, estatista |
| Sentimiento | 5 | esperanza, angustia, indignación, orgullo, empatía |
| Estilo discursivo | 6 | directo, académico, confrontativo, conciliador, catastrofista, testimonial |
| Función discursiva | 3 | crítica, propuesta, narrativa personal |
| Intensidad antagónica | 5 niveles | escala 1 (conciliador) a 5 (beligerante) |

**LLM** (`LFM2.5-1.2B-Instruct`):

| Dimensión | Qué mide |
|---|---|
| Stance | apoyo o rechazo respecto al tema evaluado, con contexto del actor |

Cada label de embeddings se evalúa independientemente mediante similitud coseno contra
prototipos textuales en español. El LLM razona stance con actor + tema + few-shot balanceado.

## Decisiones de arquitectura

- **Services OOP por componente:** cada componente implementa `ComponenteProtocol` mediante `.procesar(input) -> output`.
- **Config encapsulada:** los hiperparámetros viven en el constructor del service. `main.py` los carga desde `config/config.yaml`.
- **Helpers puros:** la lógica interna se mantiene en funciones testeables.
- **Lazy loading:** Whisper, spaCy, BERTopic, pyannote y modelos LFM2.5 se cargan solo cuando se usan.
- **ASR default:** `large-v3-turbo` por balance calidad/velocidad; mantiene `word_timestamps=True` para alinear con speaker turns.
- **Diarización implementada:** `pyannote/speaker-diarization-community-1`; el actor se identifica con un perfil de voz cacheado solo durante la ejecución. Si el match es ambiguo, se descarta como otro speaker y el pipeline continúa.
- **DTOs compartidos vs locales:** `src/tono_politico/models.py` contiene DTOs compartidos por más de un componente. Los DTOs específicos viven dentro de su componente.
- **Embeddings compartidos:** Segmentación, Temas y Tono usan `LiquidAI/LFM2.5-Embedding-350M`.
- **Mean pooling manual en Tono:** `sentence-transformers` produce embeddings degenerados con LFM2.5; el Componente 5 usa `AutoModel` directo con mean pooling.

## Estructura del código

```text
src/tono_politico/
├── models.py              # DTOs compartidos
├── protocol.py            # ComponenteProtocol
├── ingesta/               # Componente 1 ✅
│   ├── service.py         # IngestaService
│   ├── playlist.py        # metadata de playlists vía yt-dlp
│   ├── audio.py           # descarga/cache de audios .wav
│   ├── transcripcion.py   # Whisper + JSON
│   └── cache.py           # rutas centralizadas
├── diarizacion/           # Componente 1.5 ✅
│   ├── service.py         # DiarizacionService (orquestador + lazy-load)
│   ├── diarizacion.py     # diarizar() — pyannote → TurnoOrador[]
│   ├── perfil_voz.py      # construir_perfil() — embedding de referencia
│   ├── matching.py        # distancia_coseno(), clasificar_speaker(), identificar_actor()
│   ├── alineacion.py      # filtrar_por_actor() — midpoint → segmentos del actor
│   └── models.py          # TurnoOrador, PerfilVozActor, SpeakerMatch
├── segmentacion/          # Componente 2 ✅
│   ├── service.py         # SegmentacionService
│   ├── sentencias.py      # spaCy → Oracion[]
│   ├── breakpoints.py     # distancia coseno + percentil 95
│   ├── agrupacion.py      # guardrails min/max
│   └── models.py          # Oracion, Segmento
├── temas/                 # Componente 3 ✅
│   ├── service.py         # TemasService
│   ├── descubrimiento.py  # BERTopic + UMAP + HDBSCAN
│   └── models.py          # SegmentoTematizado, TopicoInfo, ResultadoTemas
├── filtrado/              # Componente 4 ✅
│   ├── service.py         # FiltradoService
│   ├── filtro.py          # filtrado determinista por tópico/relevancia
│   └── models.py          # CriterioFiltrado, SegmentoFiltrado, ResultadoFiltrado
├── tono/                  # Componente 5 ✅
│   ├── service.py         # TonoService (orquestador híbrido)
│   ├── embeddings.py      # EmbeddorTono (mean pooling) + similitud coseno
│   ├── zero_shot.py       # ClasificadorLLM para stance
│   ├── taxonomia.py       # 25 prototipos en 5 dimensiones
│   └── models.py          # EtiquetaScore, Resultado*, SegmentoConTono, ResultadoTono
├── salida/                # Componente 6 ✅
│   ├── service.py         # SalidaService
│   ├── agregacion.py      # colapsar ResultadoTono → PerfilActor
│   ├── serializacion.py   # JSON + Markdown
│   └── models.py          # Provenance, PerfilActor, InformeTono
main.py                    # CLI entry point — lee config/config.yaml
```

## Configuración

Defaults de proyecto: [`config/config.yaml`](config/config.yaml).

`main.py` carga automáticamente `config/config.yaml` y construye cada service con los valores de su sección. Los services siguen siendo configurables por constructor para uso programático.

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
# Tests (excluye los que cargan modelos pesados)
uv run pytest tests/ -v -m "not slow"

# Tests de integración (cargan modelos reales)
uv run pytest tests/ -v -m slow

# Lint
ruff check src/ tests/

# Type check
ty check src/

# Todo antes de cerrar un cambio
ruff check src/ tests/ && ty check src/ && uv run pytest tests/ -v -m "not slow"
```

## Uso programático por componente

### Componente 1: Ingesta

```python
from pathlib import Path

from tono_politico.ingesta import IngestaService

svc = IngestaService(
    data_dir=Path("data"),
    whisper_model="large-v3-turbo",
    idioma="es",
)

transcripciones = svc.procesar("https://youtube.com/playlist?list=...")
```

### Componente 1.5: Diarización

```python
from tono_politico.diarizacion import DiarizacionService

svc = DiarizacionService(
    actor="Lilly Téllez",
    video_ref_id="su9nURIj9XQ",
    umbral_match=0.5,
    umbral_ambiguo=0.7,
)

# Filtra transcripciones conservando solo intervenciones del actor
transcripciones_actor = svc.procesar(transcripciones, nombre_playlist="Play-PoliTest")
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

segmentos = svc.procesar(transcripciones_actor)
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

### Componente 5: Tono

```python
from tono_politico.tono import TonoService

svc = TonoService(
    actor="Lilly Téllez",
    tema="fracking",
)

resultado_tono = svc.procesar(resultado_filtrado)
```

### Componente 6: Salida

```python
from tono_politico.salida import SalidaService

svc = SalidaService(output_path="output/")  # directorio → genera informe.json + informe.md

informe = svc.procesar(resultado_tono)
```

## Documentación técnica

- [Componente 1: Ingesta](docs/componente-1-ingesta.md)
- [Componente 1.5: Diarización y actor](docs/componente-1-5-diarizacion.md)
- [Componente 2: Segmentación](docs/componente-2-segmentacion.md)
- [Componente 3: Temas](docs/componente-3-temas.md)
- [Componente 4: Filtrado](docs/componente-4-filtrado.md)
- [Componente 5: Tono](docs/componente-5-tono.md)
- [Componente 6: Salida](docs/componente-6-salida.md)
- [Configuración](docs/configuracion.md)
