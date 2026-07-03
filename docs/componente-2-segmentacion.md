# Componente 2: Segmentación

> **Estado:** ✅ Completo · **Tests:** 35

## Propósito

Toma las transcripciones crudas de Whisper (ventanas acústicas arbitrarias) y las reagrupa en segmentos semánticamente coherentes — bloques de discurso que tratan un mismo tema, listos para análisis de tono.

## Arquitectura

```
list[VideoTranscript] (Componente 1)
    │
    ▼ extraer_oraciones()     — spaCy divide SegmentoRaw en Oracion[]
    │                           Mapea WordTimestamp por offsets de caracteres
    │
    ▼ detectar_breakpoints()  — Embeddings → distancia coseno → percentil 95
    │                           (estándar LangChain SemanticChunker)
    │
    ▼ agrupar_segmentos()     — Corta por breakpoints + aplica guardrails
    │                           (min/max oraciones, max palabras)
    │
    list[Segmento]
```

## API

### `SegmentacionService`

```python
from tono_politico.segmentacion import SegmentacionService

svc = SegmentacionService(
    spacy_model="es_core_news_lg",
    breakpoint_percentile=95,
    min_oraciones=2,
    max_oraciones=8,
    max_palabras=150,
)

segmentos: list[Segmento] = svc.procesar(transcripciones)
```

### Configuración

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `spacy_model` | `str` | `"es_core_news_lg"` | Modelo de spaCy para división en oraciones |
| `breakpoint_percentile` | `int` | `95` | Percentil de distancia coseno (estándar LangChain) |
| `min_oraciones` | `int` | `2` | Mínimo de oraciones por segmento (fusiona si es menor) |
| `max_oraciones` | `int` | `8` | Máximo de oraciones por segmento (subdivide si excede) |
| `max_palabras` | `int` | `150` | Máximo de palabras por segmento (subdivide si excede) |

### Lazy loading

spaCy y sentence-transformers se cargan en el primer `.procesar()`, no al importar el módulo. Esto permite:
- Tests rápidos sin modelos pesados (mockeando `_get_nlp` / `_get_embedder`)
- Import del paquete sin dependencias pesadas instaladas

## Módulos internos

### `sentencias.py` — Extracción de oraciones

**`extraer_oraciones(segmentos, nlp) → list[Oracion]`**

Toma `list[SegmentoRaw]` y un modelo spaCy, divide cada segmento en oraciones preservando los timestamps de Whisper.

**Algoritmo de mapeo words→oración:**

1. spaCy divide el texto del `SegmentoRaw` en oraciones con offsets de caracteres (`sent.start_char` / `sent.end_char`)
2. Para cada `WordTimestamp`, se calcula su posición `[char_start, char_end)` en el texto original mediante búsqueda secuencial
3. Cada word se asigna a la oración cuyo rango de caracteres la contiene
4. `t_start` y `t_end` de cada oración se derivan de su primera y última word

**Edge cases:**
- Segmento sin words → se omite con warning
- Palabra no encontrada en texto → fallback a posición acumulada
- Búsqueda case-insensitive para robustez ante errores de Whisper

### `breakpoints.py` — Detección semántica

**`detectar_breakpoints(oraciones, model, breakpoint_percentile) → list[Breakpoint]`**

Sigue el estándar de **LangChain SemanticChunker** / **LlamaIndex**:

1. Codifica todas las oraciones con sentence-transformers (`model.encode(textos)`)
2. Calcula **distancia coseno** (`1 - similitud`) entre embeddings de oraciones consecutivas
3. El **percentil 95** de las distancias observadas es el umbral
4. Marca breakpoint donde `distancia >= umbral`

**`Breakpoint`:**

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `indice` | `int` | Cortar ANTES de esta oración (0-indexed) |
| `intensidad` | `float` | Distancia coseno entre las dos oraciones |

**Por qué no hay señal acústica:**

Whisper ya segmenta el audio acústicamente — sus ventanas ya incorporan pausas y silencios. Repetir esa detección sobre los gaps temporales sería redundante. La innovación de este componente es la **señal semántica** sobre las ventanas de Whisper.

**Umbral mínimo (`EPSILON = 1e-4`):**

Evita falsos positivos cuando todas las distancias son esencialmente cero (oraciones muy similares). Sin esto, el percentil de valores ≈0 puede marcar breakpoints por ruido de punto flotante.

**Mínimo de oraciones:**

Con `< 3` oraciones no hay suficiente información estadística para que el percentil sea significativo. Se omite la detección semántica.

### `agrupacion.py` — Guardrails

**`agrupar_segmentos(oraciones, breakpoints, ...) → list[Segmento]`**

Pipeline de 4 pasos:

#### Paso 1: Cortar por breakpoints

Los breakpoints dividen la secuencia en bloques. Un breakpoint en `indice=3` significa que las oraciones `[0,1,2]` forman un bloque y `[3,...]` inicia otro.

#### Paso 2: Subdividir bloques grandes

Si un bloque excede `max_oraciones` o `max_palabras`, se divide en sub-bloques que respeten los límites. La división es greedy: acumula oraciones hasta tocar un límite, corta, y continúa.

#### Paso 3: Fusionar bloques pequeños

Si un bloque tiene menos de `min_oraciones`, se fusiona con el bloque anterior. Esto evita segmentos triviales de 1 oración (a menos que el usuario configure `min_oraciones=1`).

#### Paso 4: Construir Segmento

Cada bloque se convierte en un `Segmento` con:
- `texto`: oraciones concatenadas con espacio
- `t_start` / `t_end`: de la primera y última oración
- `word_count`: suma de words de todas las oraciones
- `video_id`: propagado del transcript de origen

## DTOs

Definidos en `src/tono_politico/segmentacion/models.py`:

### `Oracion`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `texto` | `str` | Texto de la oración |
| `t_start` | `float` | Inicio (de la primera word) |
| `t_end` | `float` | Fin (de la última word) |
| `words` | `list[WordTimestamp]` | Words de Whisper asignadas a esta oración |

### `Segmento`

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `texto` | `str` | Oraciones concatenadas |
| `t_start` | `float` | Inicio del segmento |
| `t_end` | `float` | Fin del segmento |
| `oraciones` | `list[Oracion]` | Oraciones que componen el segmento |
| `word_count` | `int` | Total de palabras |
| `video_id` | `str` | Video de origen |

## Dependencias externas

| Herramienta | Uso |
|-------------|-----|
| `spacy` + `es_core_news_lg` | División en oraciones con offsets de caracteres |
| `sentence-transformers` + `all-MiniLM-L6-v2` | Embeddings para distancia semántica |
| CPU | No requiere GPU (MiniLM es ligero) |

## Decisiones de diseño

### Breakpoints 100% semánticos

La señal acústica fue eliminada porque Whisper ya hace VAD y segmentación acústica internamente. Los gaps temporales entre segmentos de Whisper ya reflejan silencios reales. Este componente aporta la señal que Whisper no puede dar: coherencia semántica entre oraciones.

### Estándar LangChain (distancia + percentil 95)

Alineado con LangChain SemanticChunker y LlamaIndex:
- Métrica: **distancia coseno** (no similitud) — distancia alta = cambio de tópico
- Umbral: **percentil 95** de las distancias observadas — adapta el corte al contenido específico del video
- Referencia: [LangChain SemanticChunker](https://python.langchain.com/docs/modules/data_connection/document_transformers/semantic_chunker)

### Mapeo words→oración por offsets

En vez de heurísticas de puntuación, usamos los offsets de caracteres que spaCy provee (`sent.start_char`, `sent.end_char`) para mapear cada `WordTimestamp` a su oración. Esto garantiza que los timestamps de Whisper se preserven con precisión en la salida.

### Guardrails como capas independientes

Los 3 guardrails (`min_oraciones`, `max_oraciones`, `max_palabras`) se aplican en pipeline secuencial, no simultáneamente. Esto permite razonar sobre cada uno de forma aislada y testearlos independientemente.
