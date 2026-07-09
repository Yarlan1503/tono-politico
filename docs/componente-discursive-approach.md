# discursive_approach — argumentos → temas → enfoques de tono

Umbrella **actor-only** que convierte `ActorTranscript[]` (salida de `speech2text`) en:

1. **Argumentos** semánticos (por audio)
2. **Temas** del corpus (BERTopic)
3. **Enfoques** de tono por tema × tiempo (taxonomía de Tono + firmas)

> **Requisitos de diseño (fuente de verdad):**  
> [`src/tono_politico/discursive_approach/requisitos.md`](../src/tono_politico/discursive_approach/requisitos.md)

---

## Pipeline

```text
speech2text
    → ActorTranscript[]  (+ fecha YYYYMMDD desde VideoMeta)
         │
         ▼
discursive_approach
    1. argument_shape     # un audio → Argumento[]
       · spaCy + LFM2.5 (cortes semánticos, sin word-level)
    2. topics_cluster     # corpus → ResultadoTemas
       · BERTopic (LFM2.5 + UMAP + HDBSCAN + c-TF-IDF)
    3. topics_approach    # ResultadoTemas → ResultadoEnfoques
       · BASE = Tono (taxonomía v3 híbrida)
       · TonoService por argumento × tema=label del tópico
       · firmas deterministas + orden temporal
         │
         ▼
[filtrado / salida — posteriores, fuera del umbrella]
```

### Tres preguntas

| Capa | Pregunta | Alcance | Señal |
|---|---|---|---|
| `argument_shape` | ¿Qué oraciones forman un mismo argumento? | 1 audio | Similitud semántica entre oraciones consecutivas |
| `topics_cluster` | ¿De qué temas habla el actor en el corpus? | multi-video | BERTopic sobre texto de argumentos |
| `topics_approach` | ¿Cómo (con qué tono) aborda cada tema en el tiempo? | por tema × tiempo | **Taxonomía de Tono** (no 2.º topic model) |

### Fuera del umbrella

| Componente | Rol |
|---|---|
| `speech2text` | Upstream: audio → `ActorTranscript` |
| Filtrado | Posterior: elegir tema y/o enfoque para el informe |
| Salida | Posterior: JSON/Markdown del informe |

**Legacy:** `segmentacion/` + `temas/` + filtrado→tono del runner clásico siguen en el repo y en `discover`/`analyze` sin `--discursive`.

---

## DTOs clave

| DTO | Dónde | Rol |
|---|---|---|
| `Oracion` | `argument_shape/models.py` | texto + t_start/t_end (sin words) |
| `Argumento` | `argument_shape/models.py` | unidad semántica; + `video_id`, `fecha` |
| `TopicoInfo` / `ArgumentoTematizado` / `ResultadoTemas` | `topics_cluster/models.py` | temas del corpus |
| `PerfilTonoArgumento` | `topics_approach/models.py` | dominantes de taxonomía + stance + intensidad |
| `EnfoqueInfo` / `ArgumentoConEnfoque` / `ResultadoEnfoques` | `topics_approach/models.py` | enfoques por tema |

### Fecha (orden temporal)

```text
VideoMeta.fecha → AudioVideo.fecha → ActorTranscript.fecha → Argumento.fecha
```

Propagada en `transcribir_turnos_actor(..., fecha=)` y `TranscribeSpeechService`.

---

## Subpaquetes y API

```text
src/tono_politico/discursive_approach/
├── service.py                 # DiscursiveApproachService
├── requisitos.md
├── argument_shape/
│   ├── models.py              # Oracion, Argumento
│   ├── sentencias.py          # ActorTranscript → Oracion[] (spaCy)
│   ├── breakpoints.py         # coseno + percentil
│   ├── agrupacion.py          # guardrails → Argumento[]
│   └── service.py             # ArgumentShapeService
├── topics_cluster/
│   ├── models.py              # ArgumentoTematizado, TopicoInfo, ResultadoTemas
│   ├── descubrimiento.py      # BERTopic
│   ├── serializacion.py       # discursive_resultado_temas.v1
│   └── service.py             # TopicsClusterService
└── topics_approach/
    ├── models.py              # PerfilTono, EnfoqueInfo, ResultadoEnfoques…
    ├── adapter.py             # Argumento → Segmento / ResultadoFiltrado (Tono)
    ├── enfoques.py            # firmas + bin intensidad + orden temporal
    ├── serializacion.py       # discursive_resultado_enfoques.v1
    └── service.py             # TopicsApproachService
```

### Orquestador

```python
from tono_politico.discursive_approach import DiscursiveApproachService
from tono_politico.discursive_approach.argument_shape import ArgumentShapeService
from tono_politico.discursive_approach.topics_cluster import TopicsClusterService

shape = ArgumentShapeService(
    spacy_model="es_core_news_lg",
    breakpoint_percentile=95,
    min_oraciones=2,
    max_oraciones=8,
    max_palabras=150,
    embedding_model_name="LiquidAI/LFM2.5-Embedding-350M",
)
cluster = TopicsClusterService(
    min_topic_size=3,
    n_neighbors=10,
    n_components=5,
    embedding_model_name="LiquidAI/LFM2.5-Embedding-350M",
)
svc = DiscursiveApproachService(
    actor="Lilly Téllez",
    shape_service=shape,
    cluster_service=cluster,
)

enfoques = svc.procesar(actor_transcripts)  # shape → cluster → approaches
# o por capas:
args = svc.shape_corpus(actor_transcripts)
temas = svc.cluster(args)
enfoques = svc.approaches(temas)
```

### Firmas de enfoque (v1)

```text
firma = (
  stance,
  logica_dominante,
  sentimiento_dominante,
  estilo_dominante,
  funcion_dominante,
  intensidad_bin,   # 1–2 | 3 | 4–5
)
```

- Misma firma → mismo `enfoque_id` **dentro de un tema**
- Asignación dura (`probabilidad_enfoque=1.0`)
- **Sin HDBSCAN** en approach (HDBSCAN solo en `topics_cluster`/BERTopic)
- Outliers de tópico (`topico_id=-1`) se omiten del análisis de enfoques

---

## CLI y PipelineRunner

### Discover (path nuevo)

```bash
uv run python main.py --playlist "URL" --discursive --keep-cache
```

Fases del manifest:

```text
speech2text → argument_shape → topics_cluster → topics_approach
```

Artefactos en `output/<run_id>/` (vía `guardar_manifest`):

- `manifest.json`
- `discursive-temas.json`
- `discursive-enfoques.json`

### API runner

```python
from tono_politico.pipeline import PipelineRunner, ServiceFactories

# factories con build_speech2text + build_discursive (ver main.py)
result = runner.discover_discursive(playlist_url)
# runner.last_resultado_enfoques
# runner.last_resultado_temas_discursive
```

### Path legacy (sin `--discursive`)

```bash
uv run python main.py --playlist "URL"                    # Ingesta → Diarización → Segmentación → Temas
uv run python main.py --playlist "URL" --topico 0 --tema "X"
uv run python main.py --resume output/<run_id> --topico 0 --tema "X"
```

`--discursive` **no** se combina con `--topico` / `--resume` (aún no hay Fase 2 sobre enfoques).

---

## Decisiones de diseño (resumen)

| # | Decisión |
|---|---|
| 1 | Tres fronteras: shape / cluster / approach |
| 2 | Enfoques = firmas de tono, no sub-clustering semántico |
| 3 | Argumentos no cruzan videos |
| 4 | `Argumento` (no `Segmento`) en el path nuevo |
| 5–6 | Filtrado y Salida **fuera** y **después** del umbrella |
| 7 | Embeddings LFM2.5 compartidos con el resto del stack |
| 8 | spaCy sin words en oraciones actor-only |
| 9 | Tono = base analítica de `topics_approach` |

---

## Tests

**~31 tests** dedicados al umbrella + cableado:

| Archivo | Qué cubre |
|---|---|
| `tests/test_argument_shape.py` | DTOs, breakpoints, agrupación, service (fakes) |
| `tests/test_topics_cluster.py` | BERTopic fake / dataset chico |
| `tests/test_topics_approach.py` | firmas, bins, outliers, adapter |
| `tests/test_discursive_approach_service.py` | orquestador con fakes |
| `tests/test_fecha_propagacion.py` | VideoMeta → ActorTranscript → Argumento |
| `tests/test_argument_shape_smoke.py` | smoke ligero sobre `output/speech2text-smoke/` |
| `tests/test_pipeline_discursive.py` | `discover_discursive` + persistencia |

```bash
uv run pytest \
  tests/test_argument_shape.py \
  tests/test_topics_cluster.py \
  tests/test_topics_approach.py \
  tests/test_discursive_approach_service.py \
  tests/test_fecha_propagacion.py \
  tests/test_argument_shape_smoke.py \
  tests/test_pipeline_discursive.py -q
```

---

## Estado

| Fase | Estado |
|---|---|
| R0 diseño (decisiones 1–9) | ✅ |
| R1 `argument_shape` | ✅ |
| R2 `topics_cluster` | ✅ |
| R3 `topics_approach` | ✅ |
| R4 orquestador | ✅ |
| R5 fecha + runner + CLI + smoke ligero | ✅ |
| R6 filtrado/salida post-enfoques | ⏳ pendiente |
| Smoke e2e modelos pesados (BERTopic + Tono reales) | ⏳ pendiente |

---

## Relación con componentes legacy

| Legacy | Path nuevo |
|---|---|
| `segmentacion/` | `argument_shape` (misma idea; DTOs `Argumento`, entrada `ActorTranscript`) |
| `temas/` | `topics_cluster` (misma idea BERTopic; DTOs sobre `Argumento`) |
| `filtrado` → `tono` | `topics_approach` aplica Tono **por tema** y agrupa en enfoques; filtrado queda post |
| `salida` | sin cambio de contrato aún (espera `ResultadoTono`; adaptar a enfoques es R6) |
