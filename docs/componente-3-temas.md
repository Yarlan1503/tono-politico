# Componente 3: Temas

> **Estado:** ✅ MVP implementado · **Tests:** 11

## Propósito

Toma los `Segmento` producidos por el Componente 2 y descubre temas predominantes con BERTopic. La salida conserva cada segmento original, agrega una asignación de tópico y resume los tópicos con palabras clave, conteos y representatividad.

Este componente responde: **“¿de qué temas se habló en el corpus ya segmentado semánticamente?”**

## Arquitectura

```text
list[Segmento] (Componente 2)
    │
    ▼ TemasService.procesar()
    │
    ▼ _get_embedder()                 — lazy-load SentenceTransformer
    │                                   LiquidAI/LFM2.5-Embedding-350M
    │
    ▼ descubrir_temas()
        ├─ UMAP(metric="cosine")      — reducción dimensional
        ├─ HDBSCAN(min_samples=1)      — clustering denso + outliers
        └─ BERTopic(language="spanish")
            │
            ▼ ResultadoTemas
                ├─ segmentos: list[SegmentoTematizado]
                ├─ topicos: list[TopicoInfo]
                └─ num_topicos: int
```

## API

### `TemasService`

```python
from tono_politico.temas import TemasService

svc = TemasService(
    min_topic_size=3,
    n_neighbors=10,
    n_components=5,
)

resultado = svc.procesar(segmentos)
```

### Configuración

| Parámetro | Tipo | Default | Descripción |
|---|---:|---:|---|
| `min_topic_size` | `int` | `3` | Mínimo de segmentos para formar un tópico. También alimenta `HDBSCAN(min_cluster_size=...)`. |
| `n_neighbors` | `int` | `10` | Vecinos de UMAP; se clampa a `len(segmentos) - 1`. |
| `n_components` | `int` | `5` | Dimensiones UMAP; se clampa a `len(segmentos) - 1`. |
| `embedding_model_name` | `str` | `LiquidAI/LFM2.5-Embedding-350M` | Modelo sentence-transformers cargado perezosamente. |

Los defaults también están documentados en `config/config.yaml`.

## Módulos internos

### `service.py` — Orquestador OOP

`TemasService` encapsula los hiperparámetros y compone la función pura `descubrir_temas()`.

**Flujo de `procesar()`:**

1. Si no hay segmentos, devuelve `ResultadoTemas()` vacío.
2. Carga perezosamente el modelo de embeddings con `_get_embedder()`.
3. Llama a `descubrir_temas(segmentos, embedding_model, min_topic_size, n_neighbors, n_components)`.
4. Devuelve un `ResultadoTemas` estructurado.

### `descubrimiento.py` — BERTopic y metadata

**`descubrir_temas(segmentos, embedding_model, ...) -> ResultadoTemas`**

1. Extrae `texto` de cada `Segmento`.
2. Si `len(segmentos) < min_topic_size`, evita inventar clusters y devuelve todos los segmentos como outlier `-1`.
3. Configura UMAP:
   - `metric="cosine"`
   - `random_state=42`
   - `n_neighbors=min(n_neighbors, len(textos) - 1)`
   - `n_components=min(n_components, len(textos) - 1)`
4. Configura HDBSCAN:
   - `min_cluster_size=min(min_topic_size, len(textos))`
   - `min_samples=1`
   - `metric="euclidean"`
   - `cluster_selection_method="eom"`
5. Ejecuta `BERTopic.fit_transform(textos)`.
6. Extrae metadata con `get_topic_info()` y `get_topic(tid)`.
7. Construye `SegmentoTematizado`, `TopicoInfo` y `ResultadoTemas`.

### `models.py` — DTOs locales

#### `SegmentoTematizado`

| Campo | Tipo | Descripción |
|---|---|---|
| `segmento` | `Segmento` | Segmento original del Componente 2. |
| `topico_id` | `int` | ID BERTopic; `-1` significa outlier/ruido. |
| `probabilidad` | `float` | Confianza de asignación, 0.0 a 1.0. |

#### `TopicoInfo`

| Campo | Tipo | Descripción |
|---|---|---|
| `id` | `int` | ID del tópico; `-1` para outliers. |
| `nombre` | `str` | Etiqueta auto-generada por BERTopic. |
| `palabras_clave` | `list[str]` | Términos top vía c-TF-IDF. |
| `num_segmentos` | `int` | Número de segmentos asignados. |
| `representatividad` | `float` | Proporción del corpus (`num_segmentos / total`). |

#### `ResultadoTemas`

| Campo | Tipo | Descripción |
|---|---|---|
| `segmentos` | `list[SegmentoTematizado]` | Segmentos enriquecidos con tópico. |
| `topicos` | `list[TopicoInfo]` | Metadata por tópico. |
| `num_topicos` | `int` | Número de tópicos, excluyendo outliers `-1`. |

## Decisiones de diseño

### BERTopic sobre segmentos semánticos

BERTopic se aplica después de Segmentación porque los `Segmento` ya son unidades discursivas coherentes. Eso evita que el clustering opere sobre ventanas acústicas arbitrarias de Whisper.

### Por qué BERTopic

BERTopic es mejor fit para este componente que LDA/Top2Vec en esta arquitectura porque:

- usa embeddings contextuales, útiles para discurso político con sinónimos y paráfrasis;
- separa embedding, reducción dimensional, clustering y representación c-TF-IDF, lo que permite ajustar cada etapa;
- maneja outliers explícitos (`-1`), importante para segmentos marginales;
- produce palabras clave interpretables por tópico, necesarias para inspección humana antes de Filtrado.

### Modelo de embeddings compartido

`LiquidAI/LFM2.5-Embedding-350M` se usa tanto en Segmentación como en Temas para evitar que los cortes semánticos y el clustering vivan en espacios semánticos distintos.

### Corpus chico: no inventar tópicos

Si hay menos segmentos que `min_topic_size`, el componente devuelve un único tópico outlier `-1`. Esto prefiere admitir “no hay evidencia suficiente” en vez de fabricar clusters espurios.

### Lazy loading

`SentenceTransformer` se importa y carga dentro de `_get_embedder()`, no al importar el paquete. Esto mantiene rápidos los tests unitarios y evita dependencia obligatoria de modelos pesados durante import.

## Dependencias externas

| Herramienta | Uso |
|---|---|
| `sentence-transformers` | Carga de `LiquidAI/LFM2.5-Embedding-350M`. |
| `bertopic` | Pipeline de topic modeling. |
| `umap-learn` | Reducción dimensional antes de HDBSCAN. |
| `hdbscan` | Clustering y detección de outliers. |
| `numpy` | Manejo de probabilidades. |

## Tests

`tests/test_temas.py` cubre:

- defaults de DTOs;
- configuración de `TemasService`;
- cumplimiento de `ComponenteProtocol`;
- input vacío;
- corpus menor que `min_topic_size` → outlier `-1`;
- camino con dos tópicos mockeando BERTopic;
- preservación de `video_id`;
- suma de representatividad ≈ 1.0.

## Pendientes antes del Componente 4

- Validar BERTopic con transcripciones reales del corpus objetivo.
- Decidir cómo se seleccionará el tópico/tema para Filtrado: `topico_id`, keywords, query textual o combinación.
- Integrar un loader de `config/config.yaml` si se necesita ejecución end-to-end desde CLI.
