# discursive_approach — requisitos y arquitectura

> Estado: **implementado v1 (R1–R5)** — decisiones 1–9 (2026-07-08).  
> Código en `src/tono_politico/discursive_approach/`.  
> Doc: `docs/componente-discursive-approach.md`.  
> CLI: `main.py --discursive`. Runner: `PipelineRunner.discover_discursive`.  
> Umbrella paralelo a `speech2text`: tres fronteras claras, TDD method-by-method.  
> Formato: checklist de implementación al final; decisiones y contratos arriba.

---

## Propósito

Del discurso del actor (turn-level, ya diarizado), producir:

1. **Argumentos** — unidades coherentes **dentro de cada audio**.
2. **Temas** — clusters semánticos **entre todos los argumentos del corpus**.
3. **Enfoques** — perfiles de **tono** con que el actor trata cada tema **a lo largo del tiempo**.

```text
speech2text
    → ActorTranscript[]  (+ fecha desde VideoMeta)
         │
         ▼
discursive_approach
    1. argument_shape     # un audio → Argumento[]
       · spaCy + LFM2.5 (cortes semánticos)
    2. topics_cluster     # corpus → ResultadoTemas
       · BERTopic (LFM2.5 + UMAP + HDBSCAN + c-TF-IDF)
    3. topics_approach    # ResultadoTemas → ResultadoEnfoques
       · BASE = Tono (taxonomía v3)
       · TonoService por argumento × tema
       · firmas de tono + orden temporal
         │
         ▼
[filtrado — etapa POSTERIOR, fuera del umbrella]
    selección de tema y/o enfoque para el informe
         │
         ▼
salida                    # informe JSON/MD (fuera del umbrella)
```

### Mapa mental de las tres preguntas

| Capa | Pregunta | Alcance | Señal principal |
|---|---|---|---|
| **argument_shape** | ¿Qué fragmentos forman un mismo argumento? | 1 audio | Similitud semántica entre oraciones consecutivas |
| **topics_cluster** | ¿De qué temas habla el actor en el corpus? | multi-video | BERTopic sobre texto de argumentos |
| **topics_approach** | ¿Cómo (con qué tono) aborda cada tema en el tiempo? | por tema × tiempo | **Taxonomía de Tono** (no 2.º topic model) |

### Qué queda fuera de este umbrella

| Componente | Relación |
|---|---|
| `speech2text` | Upstream: produce `ActorTranscript` |
| **Filtrado** (C4 legacy) | **Después** de `discursive_approach`: elige tema/enfoque para el informe. **No** corre antes de Tono en este diseño (Tono ya se invoca dentro de `topics_approach`). |
| **Tono** (`tono/`) | **No** es un paso suelto del runner aquí: es **dependencia/base** de `topics_approach`. No se copia taxonomía. |
| **Salida** | Downstream: materializa el informe; fuera del umbrella |

---

## Decisiones cerradas (fuente de verdad)

| # | Decisión | Valor |
|---|---|---|
| 1 | Unidad semántica | **`Segmento` → `Argumento`** en el diseño nuevo. Alias de compatibilidad mientras existan `temas/` / `filtrado` / `tono` legacy sobre `Segmento`. |
| 2 | Agrupación de enfoques | **Firmas de tono** (stance + dominantes taxonomía v3 + `intensidad_bin`) + orden temporal. **Sin HDBSCAN** en el camino feliz de approach. |
| 3 | Alcance de approach | **Todos** los temas no-outlier. Sin `topico_id` obligatorio en v1. |
| 4 | Tiempo | **`VideoMeta.fecha`** (YYYYMMDD) → `ActorTranscript` → `Argumento.fecha`. |
| 5 | Filtrado | **Fuera** del umbrella; etapa **posterior** (selección para informe), no previa a Tono. |
| 6 | Tono vs Salida | **Tono = base de `topics_approach`**. **Salida** fuera (informe). |
| 7 | Orden | `shape` → `cluster` → `topics_approach`(=Tono+firmas) → \[filtrado\] → `salida`. |
| 8 | Embedder canónico | **`LiquidAI/LFM2.5-Embedding-350M`** en shape, cluster y dimensiones embedding de Tono. No ColBERT. |
| 9 | Definición de enfoque | Perfil de tono recurrente del actor sobre un tema en el tiempo, expresado como **firma** de la taxonomía v3. Stance LLM usa `tema` = `TopicoInfo.nombre`. |

---

## Esqueleto de paquete

```text
src/tono_politico/discursive_approach/
├── __init__.py
├── service.py                 # DiscursiveApproachService
├── requisitos.md              # este archivo
├── argument_shape/
│   ├── models.py              # Oracion, Argumento
│   ├── service.py             # ArgumentShapeService
│   ├── sentencias.py          # spaCy sobre turnos (sin word-level)
│   ├── breakpoints.py         # distancia coseno + percentil (LFM2.5)
│   └── agrupacion.py          # guardrails → Argumento[]
├── topics_cluster/
│   ├── models.py              # ArgumentoTematizado, TopicoInfo, ResultadoTemas
│   ├── service.py             # TopicsClusterService
│   ├── descubrimiento.py      # BERTopic (aquí SÍ hay HDBSCAN, vía BERTopic)
│   └── serializacion.py
└── topics_approach/
    ├── models.py              # EnfoqueInfo, ArgumentoConEnfoque, ResultadoEnfoques*
    ├── service.py             # TopicsApproachService (inyecta TonoService)
    └── enfoques.py            # firma de tono + orden temporal (sin HDBSCAN)
```

Dependencias externas:

| Paquete | Uso |
|---|---|
| `tono/` | Base de approach: `TonoService`, `taxonomia.py` v3, DTOs de scores |
| `speech2text` / diarización | Entrada `ActorTranscript` (+ fecha a extender) |
| `segmentacion/`, `temas/` | Referencia legacy a migrar; no importar a largo plazo |
| `filtrado/` | No se mueve aquí |

---

## Contratos DTO (borrador)

### Propagación de fecha (decisión 4)

```text
VideoMeta.fecha
    → ActorTranscript.fecha   (extender contrato speech2text / puente)
    → Argumento.fecha         (copia en argument_shape)
    → topics_approach ordena por (fecha, t_start)
```

- [ ] Extender `ActorTranscript` con `fecha: str | None` (o DTO de contexto).
- [ ] `Argumento.fecha` siempre rellenada cuando exista metadata; si falta → `None` + warning.

### `argument_shape`

```python
@dataclass
class Oracion:
    texto: str
    t_start: float
    t_end: float
    # sin words obligatorias en path actor-only

@dataclass
class Argumento:  # ex-Segmento
    texto: str
    t_start: float
    t_end: float
    oraciones: list[Oracion]
    word_count: int          # contar tokens de texto, NO len(words) vacío
    video_id: str
    fecha: str | None        # YYYYMMDD
```

```python
class ArgumentShapeService:
    def procesar_one(self, transcript: ActorTranscript) -> list[Argumento]: ...
    def procesar_corpus(self, transcripts: list[ActorTranscript]) -> list[Argumento]: ...
```

Pipeline interno (**solo un audio**):

1. Turnos del actor → spaCy `sents` (sin `WordTimestamp`).
2. Tiempos: del turno; si un turno se parte en N oraciones → **reparto proporcional por caracteres** en `[t_start, t_end]`.
3. Breakpoints: distancia coseno entre embeddings LFM2.5 de oraciones consecutivas; corte si ≥ percentil **95** y > `EPSILON`.
4. Guardrails: `min_oraciones=2`, `max_oraciones=8`, `max_palabras=150`.
5. **No** fusionar oraciones de distintos `video_id`.

Fiabilidad (resumen):

| Garantía dura (tests) | Garantía blanda (validación) |
|---|---|
| Un video, solo actor, contigüidad, bounds, trazabilidad `oraciones[]` | Coherencia percibida (spot-check), percentil relativo al doc |
| No es Argument Mining (claim/premise) supervisado | Unidades de análisis estables para cluster/tono |

### `topics_cluster`

```python
@dataclass
class ArgumentoTematizado:
    argumento: Argumento
    topico_id: int
    probabilidad: float

@dataclass
class TopicoInfo:
    id: int
    nombre: str
    palabras_clave: list[str]
    num_argumentos: int
    representatividad: float

@dataclass
class ResultadoTemas:
    argumentos: list[ArgumentoTematizado]
    topicos: list[TopicoInfo]
    num_topicos: int
```

```python
class TopicsClusterService:
    def procesar(self, argumentos: list[Argumento]) -> ResultadoTemas: ...
```

- BERTopic: LFM2.5 + UMAP + **HDBSCAN** + c-TF-IDF, `language="spanish"`.
- `len(argumentos) < min_topic_size` → tópico controlado `id=0`.
- **Una sola corrida** sobre el corpus completo (nunca dentro del loop de video).
- Aquí **sí** pertenece HDBSCAN (componente de BERTopic). No confundir con approach.

### `topics_approach` (base = Tono)

```python
@dataclass
class EnfoqueInfo:
    id: int
    topico_id: int
    nombre: str                       # legible desde la firma de tono
    palabras_clave: list[str]         # apoyo léxico opcional (no define el enfoque)
    num_argumentos: int
    fecha_primera: str | None
    fecha_ultima: str | None
    stance_dominante: str | None      # apoyo | rechazo
    intensidad_media: float | None
    logica_dominante: str | None
    sentimiento_dominante: str | None
    estilo_dominante: str | None
    funcion_dominante: str | None

@dataclass
class ArgumentoConEnfoque:
    argumento: Argumento
    topico_id: int
    enfoque_id: int
    probabilidad_topico: float
    probabilidad_enfoque: float       # v1 firmas: 1.0 (asignación dura)
    tono: object | None               # snapshot ligero de scores Tono

@dataclass
class ResultadoEnfoquesTema:
    topico: TopicoInfo
    enfoques: list[EnfoqueInfo]
    argumentos: list[ArgumentoConEnfoque]  # orden (fecha, t_start)

@dataclass
class ResultadoEnfoques:
    por_tema: list[ResultadoEnfoquesTema]
    num_temas: int
    num_enfoques_total: int
```

```python
class TopicsApproachService:
    def __init__(self, tono_service: "TonoService", actor: str, ...): ...
    def procesar(self, resultado: ResultadoTemas) -> ResultadoEnfoques:
        """Todos los temas no-outlier.
        Por tema: Tono(argumentos, tema=topico.nombre) → firmas → orden temporal.
        """
```

#### Piezas de Tono reutilizadas

| Pieza | Rol |
|---|---|
| `taxonomia.py` v3 | Espacio de labels (no se duplica) |
| LFM2.5-Embedding + prototipos | lógica, sentimiento, estilo, función, intensidad |
| LFM2.5-1.2B-Instruct | stance apoyo/rechazo vs `tema` |
| `TonoService` | Ejecución; requiere **adaptador** `Argumento` → unidad que Tono entienda (hoy espera `Segmento` vía `ResultadoFiltrado` — deuda de integración) |

#### Algoritmo v1 (por cada `topico_id != -1`)

1. Tomar argumentos del tema.
2. `tema` para Tono = `TopicoInfo.nombre` (label del cluster); `actor` = actor del pipeline.
3. Ejecutar Tono → perfil por argumento (stance, intensidad, dominantes de cada dimensión).
4. Construir **firma**:
   ```text
   (stance, logica_dom, sentimiento_dom, estilo_dom, funcion_dom, intensidad_bin)
   intensidad_bin ∈ {1–2, 3, 4–5}
   ```
5. Cada firma distinta → un `enfoque_id` (asignación dura; `probabilidad_enfoque=1.0`).
6. Si no hay argumentos → tema omitido / lista vacía.
7. Si hay argumentos pero una sola firma → un enfoque (caso normal en corpus chico).
8. `EnfoqueInfo` agrega perfil de la firma; keywords de texto **opcionales**.
9. Ordenar argumentos por `(fecha, t_start)`; set `fecha_primera` / `fecha_ultima`.

**Nota sobre `min_approach_size`:** con firmas **no** colapsamos todo a un enfoque artificial solo por n pequeño. El número de enfoques = número de firmas distintas. Un default `min_approach_size` solo aplica si en v1.1 se filtra ruido (p.ej. no publicar enfoques con `n=1` salvo que sea el único).

#### Por qué no HDBSCAN en approach

| | |
|---|---|
| Ejes ya discretos | Firmas = labels de Tono; HDBSCAN redescubre peor |
| n chico | Colapsa a 1 cluster o outliers |
| Opacidad | Firma legible vs `enfoque_3` |
| Doble trabajo | Tono ya hizo el ML pesado |
| HDBSCAN sí | Solo en **`topics_cluster`** (BERTopic) |

#### Qué no hace `topics_approach`

- No inventa taxonomía paralela a v3.
- No define enfoques por 2.º BERTopic / HDBSCAN de texto.
- No emite informe de `salida`.
- No aplica el marco de “polarización”; labels de lógica (incl. populista) son descriptivos.

### Orquestador

```python
class DiscursiveApproachService:
    def shape_one(self, transcript: ActorTranscript) -> list[Argumento]: ...
    def shape_corpus(self, transcripts: list[ActorTranscript]) -> list[Argumento]: ...
    def cluster(self, argumentos: list[Argumento]) -> ResultadoTemas: ...
    def approaches(self, resultado: ResultadoTemas) -> ResultadoEnfoques: ...
    def procesar(self, transcripts: list[ActorTranscript]) -> ResultadoEnfoques:
        """shape_corpus → cluster → approaches."""
```

`procesar` **no** llama a filtrado ni a salida.

---

## Investigación (contexto de la decisión 2)

### Qué se evaluó para *enfoques* (no para temas)

| Opción | Veredicto |
|---|---|
| **Firmas de tono** | **Elegida v1** — interpretable, estable, n chico OK |
| HDBSCAN sobre scores de tono | Rechazada — redundante y opaca |
| 2.ª BERTopic / sub-topics de texto | Rechazada como definición de enfoque (mezcla “qué” con “cómo”) |
| Hierarchical topics BERTopic | Responde similitud *entre temas*, no enfoques *dentro* de un tema |
| Frames LLM ad hoc | Rechazada v1 — taxonomía v3 ya cubre el espacio |

### Embedder

**Congelado:** `LiquidAI/LFM2.5-Embedding-350M` para shape, cluster y dimensiones embedding de Tono (decisión 8).

### Dónde sí hay HDBSCAN

Solo en **`topics_cluster`**, como parte de BERTopic sobre embeddings de **texto de argumentos**.

---

## Defaults tentativos

| Parámetro | Capa | Default | Notas |
|---|---|---|---|
| `breakpoint_percentile` | shape | 95 | estándar SemanticChunker |
| `min_oraciones` / `max_oraciones` / `max_palabras` | shape | 2 / 8 / 150 | del segmentador actual |
| `min_topic_size` | cluster | 3 | BERTopic / HDBSCAN |
| `n_neighbors` / `n_components` | cluster | 10 / 5 | UMAP |
| `embedding_model` | shape, cluster, **tono** | LFM2.5-Embedding-350M | no “approach” directo |
| `intensidad_bin` | approach | 1–2 / 3 / 4–5 | evita atomizar firmas |
| outliers `topico_id=-1` | approach | omitidos | sin enfoques de ruido |
| `probabilidad_enfoque` (firmas) | approach | 1.0 | asignación dura |

Calibrar con smoke; no tratar defaults como magia no testeada.

---

## Riesgos

| Riesgo | Mitigación |
|---|---|
| Corpus chico → 1 firma/tema | Esperado; documentar en smoke |
| `ActorTranscript` sin `fecha` hoy | Extender antes de R3 real; fixtures con fecha inyectada en R1–R2 |
| Tono espera `Segmento` / `ResultadoFiltrado` | Adaptador `Argumento` → API de Tono en R3 |
| Rename `Argumento` rompe legacy | Alias; migrar tono/salida cuando consuman Argumento |
| Fragmentación de firmas | Bins de intensidad; v1.1 fusión de firmas n=1 |
| Coste: Tono × (argumentos × temas implícitos) | Un solo pass de Tono por argumento con su `tema` de cluster; compartir embedder (R6) |
| Stance con label de tópico ruidoso | `TopicoInfo.nombre` c-TF-IDF puede ser feo; v1.1 humanizar label opcional |

---

## Criterios de aceptación del umbrella

1. `procesar(transcripts_con_fecha)` → `ResultadoEnfoques` con **todos** los temas no-outlier.
2. Cada `Argumento` lleva `video_id` + `fecha` (si metadata existía).
3. `argument_shape` no cruza videos.
4. Enfoques ordenables en el tiempo por tema (`fecha_primera` ≤ `fecha_ultima` cuando hay fechas).
5. Enfoques definidos por **firmas de taxonomía Tono**, no por 2.º clustering de texto.
6. Filtrado **no** es dependencia de `discursive_approach`.
7. Cada `EnfoqueInfo` expone perfil de tono trazable a Tono.
8. Word-level **no** es obligatorio en el path actor-only.
9. HDBSCAN aparece solo en el path de **cluster** (BERTopic), no en approach.

---

## Checklist de implementación

### R0 — Diseño

- [x] Tres fronteras (shape / cluster / approach)
- [x] Filtrado fuera y **posterior** (no antes de Tono)
- [x] Decisiones 1–9 alineadas entre sí
- [x] Tono = base de approach; firmas (no HDBSCAN) como agrupación
- [x] Embedder LFM2.5-Embedding-350M congelado
- [x] HDBSCAN solo en topics_cluster
- [x] Doc revisado por coherencia lógica (2026-07-08)
- [x] Implementación R1–R4 + tests unitarios (2026-07-08)

### R1 — `argument_shape` (TDD)

- [x] DTO `Argumento` (+ `fecha`; `word_count` desde texto)
- [x] `Oracion` desde turnos `ActorTranscript` (spaCy)
- [x] Reparto temporal proporcional multi-oración
- [x] Port breakpoints + agrupación (sin depender de `words`)
- [x] `ArgumentShapeService.procesar_one` / `procesar_corpus`
- [x] Tests: no cruce de `video_id`, bounds, fechas
- [ ] Smoke ligero con JSON de `output/speech2text-smoke/` (+ fecha fixture)

### R2 — `topics_cluster` (TDD + migración)

- [x] DTOs con `Argumento` (rename desde Segmento*)
- [x] Port `descubrir_temas` → `list[Argumento]`
- [x] `TopicsClusterService` (BERTopic + HDBSCAN aquí)
- [x] Serialización versionada (`discursive_resultado_temas.v1`)
- [x] Tests (dataset pequeño + serialización; BERTopic real en smoke)

### R3 — `topics_approach` (TDD; base = Tono)

- [x] Dependencia explícita de `tono/` (sin copiar taxonomía)
- [x] Adaptador `Argumento` → input de `TonoService`
- [x] DTOs `EnfoqueInfo`, `ArgumentoConEnfoque`, `ResultadoEnfoques*`
- [x] `descubrir_enfoques`: todos los temas → Tono → **firmas** → orden temporal
- [x] Sin HDBSCAN en camino feliz
- [x] Keywords opcionales (no definen enfoque)
- [x] `TopicsApproachService` con analyzer inyectable / TonoService real
- [x] Tests: multi-fecha + perfiles de tono fake

### R4 — Orquestador

- [x] `DiscursiveApproachService.procesar`
- [x] Tests de composición con fakes (shape, cluster, approach)

### R5 — Integración

- [x] Campo `fecha` en `ActorTranscript` (+ serialización JSON opcional)
- [x] Propagar `fecha` desde `AudioVideo`/`VideoMeta` en `transcribir_turnos_actor` + `TranscribeSpeechService`
- [x] Loader `actor_transcript` tolera segmentos con `source_turn` anidado o plano
- [x] Smoke ligero `argument_shape` sobre `output/speech2text-smoke/actor_transcripts`
- [x] Cablear runner: `PipelineRunner.discover_discursive` + CLI `--discursive`
- [x] Docs `docs/componente-discursive-approach.md`
- [ ] Smoke real end-to-end con modelos pesados (Play-PoliTest)

### R6 — Posterior / limpieza

- [ ] Filtrado: selección de tema y/o enfoque **después** de approach, para alimentar salida
- [ ] Eliminar paths legacy `segmentacion`/`temas` sin callers
- [ ] Embedder compartido shape ↔ cluster ↔ Tono
- [ ] Adaptar Salida a `ResultadoEnfoques` / `Argumento` si aplica

---

## Siguiente paso

1. Aprobar este `requisitos.md` como contrato v1.  
2. **R1 TDD:** `Argumento` + `argument_shape` sobre `ActorTranscript` (fechas en fixture si hace falta).  
3. No implementar `topics_approach` hasta tener argumentos (+ fechas) y adaptador claro a Tono.
