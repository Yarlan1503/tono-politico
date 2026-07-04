# Configuración

> **Archivo canónico:** `config/config.yaml`

## Propósito

`config/config.yaml` documenta los defaults acordados para cada componente del pipeline. Es la referencia que debe mantenerse sincronizada con los constructores de los services y con la documentación.

Estado actual: **no hay loader global/CLI** que lea automáticamente este YAML. Los services se configuran por constructor. Si más adelante se agrega CLI o pipeline end-to-end, este archivo será la fuente natural de defaults.

## Convención general

- Cada componente tiene su propia sección (`ingesta`, `segmentacion`, `temas`, etc.).
- Los valores implementados deben coincidir con los defaults reales del service correspondiente.
- Las secciones de componentes pendientes pueden existir, pero deben marcarse con `estado: "pendiente"`.
- Cuando cambie un default en código, actualizar también:
  - `config/config.yaml`
  - `README.md`
  - `AGENTS.md`
  - `docs/`

## Mapeo de secciones a services

| Sección YAML | Service / módulo | Estado | Notas |
|---|---|---:|---|
| `project` | defaults globales | referencia | `data_dir`, `idioma`, `random_state`. |
| `ingesta` | `IngestaService` | ✅ | `data_dir`, `whisper_model`, `idioma`. |
| `segmentacion` | `SegmentacionService` | ✅ | spaCy, percentil de breakpoints y guardrails. |
| `temas` | `TemasService` + `descubrir_temas()` | ✅ MVP | BERTopic, UMAP, HDBSCAN y modelo de embeddings. |
| `filtrado` | pendiente | — | Placeholder para Componente 4. |
| `tono` | pendiente | — | Labels y modelo zero-shot planeado. |
| `salida` | pendiente | — | Formato de exportación/provenance planeado. |

## Componente 1: `ingesta`

```yaml
ingesta:
  data_dir: "data"
  whisper_model: "large-v3"
  idioma: "es"
```

Equivale a:

```python
IngestaService(
    data_dir=Path("data"),
    whisper_model="large-v3",
    idioma="es",
)
```

No incluir `whisper_device` mientras `transcribir()` no lo acepte explícitamente. Hoy Whisper se ejecuta con `fp16=False`, amigable con CPU.

## Componente 2: `segmentacion`

```yaml
segmentacion:
  spacy_model: "es_core_news_lg"
  embedding_model: "LiquidAI/LFM2.5-Embedding-350M"
  breakpoint_percentile: 95
  min_oraciones: 2
  max_oraciones: 8
  max_palabras: 150
```

Equivale a:

```python
SegmentacionService(
    spacy_model="es_core_news_lg",
    breakpoint_percentile=95,
    min_oraciones=2,
    max_oraciones=8,
    max_palabras=150,
)
```

Notas:

- `embedding_model` está documentado aunque hoy el service lo carga internamente como `LiquidAI/LFM2.5-Embedding-350M`.
- No usar `pausa_minima` ni `similitud_umbral`: la segmentación actual es 100% semántica, usa **distancia coseno** y corta por **percentil 95**.

## Componente 3: `temas`

```yaml
temas:
  embedding_model: "LiquidAI/LFM2.5-Embedding-350M"
  min_topic_size: 3
  n_neighbors: 10
  n_components: 5
  umap:
    metric: "cosine"
    random_state: 42
  hdbscan:
    min_samples: 1
    metric: "euclidean"
    cluster_selection_method: "eom"
  bertopic:
    language: "spanish"
    calculate_probabilities: true
    verbose: false
```

Equivale a:

```python
TemasService(
    min_topic_size=3,
    n_neighbors=10,
    n_components=5,
    embedding_model_name="LiquidAI/LFM2.5-Embedding-350M",
)
```

Notas:

- `umap`, `hdbscan` y `bertopic` reflejan defaults internos de `descubrir_temas()`.
- Si `len(segmentos) < min_topic_size`, no se inventan clusters: todos los segmentos quedan como outlier `-1`.

## Componentes pendientes

`filtrado`, `tono` y `salida` son placeholders de diseño. Sus valores deben tratarse como intención documentada, no como API implementada, hasta que existan services y tests.
