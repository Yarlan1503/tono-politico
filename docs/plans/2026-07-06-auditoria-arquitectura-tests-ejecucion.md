# Auditoría de arquitectura, tests y ejecución — plan de remediación

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** convertir el pipeline `tono-politico` en un flujo controlado, reproducible y fluido para producción sobre playlists reales: arquitectura con contratos explícitos, tests que cubran CLI/orquestación/errores y ejecución con reanudación, cache controlado y provenance verificable.

**Architecture:** mantener el patrón actual de componentes OOP + helpers puros, pero extraer la orquestación de `main.py` a una capa `PipelineRunner` testeable con resultados estructurados (`RunResult`, `RunManifest`, `Fase1Artifacts`). El CLI debe ser una envoltura delgada que parsea argumentos, carga config validada y traduce resultados a exit codes.

**Tech Stack:** Python 3.11, uv, ruff, ty, pytest, pyannote.audio 4.x, Whisper, spaCy, BERTopic/UMAP/HDBSCAN, transformers, yt-dlp.

---

## 0. Evidencia de auditoría

### Estado de gates ejecutados

Comandos corridos desde `/home/cachorro/Documentos/Proyectos/tono-politico`:

```bash
uv run ruff check src/ tests/ main.py
uv run pytest tests/ -m "not slow" --tb=short
uv run ty check
uv run pytest tests/ -m "not slow" --cov=src/tono_politico --cov=main --cov-report=term-missing:skip-covered --quiet
```

Resultados observados:

- `ruff`: **verde**.
- `pytest -m "not slow"`: **255 passed, 5 deselected**.
- `ty check`: **falla** con 3 diagnósticos en tests:
  - `tests/test_audio.py:130`: `ruta` es `Path | None`, pero se usa `.exists()` sin narrowing.
  - `tests/test_audio.py:131`: `ruta` es `Path | None`, pero se usa `.name` sin narrowing.
  - `tests/test_salida_models.py:76`: `informe.provenance` es `Provenance | None`, pero se usa `.pipeline` sin narrowing.
- Cobertura no-slow total: **86%**.
- `main.py` **nunca se importa** en tests: coverage warning `Module main was never imported`.
- Módulos con cobertura baja o crítica:
  - `src/tono_politico/temas/descubrimiento.py`: 30%.
  - `src/tono_politico/tono/service.py`: 48%.
  - `src/tono_politico/tono/embeddings.py`: 55%.
  - `src/tono_politico/segmentacion/service.py`: 77%.
  - `src/tono_politico/diarizacion/service.py`: 80%.

### Hallazgos de arquitectura

1. **Buen patrón base:** cada componente mantiene paquete propio (`models.py`, `service.py`, helpers puros), lo cual facilita TDD y sustitución de modelos.
2. **Orquestación acoplada a CLI:** `main.py` construye servicios, ejecuta fases, hace `print`, maneja `sys.exit`, resuelve playlist y limpia cache. Eso impide testear el pipeline completo sin CLI/subprocess o mocks extensos.
3. **Configuración no tipada:** se pasa `dict` libre desde YAML. Hay defaults duplicados en `main.py`, `config/config.yaml` y docs.
4. **`data_dir` inconsistente por diseño:** `_build_ingesta()` lee `ingesta.data_dir`, mientras `_build_diarizacion()` y `_limpiar_cache()` leen `project.data_dir`. Si esos valores divergen, el pipeline puede descargar en un lugar y buscar/limpiar en otro.
5. **Fase 2 recalcula fase 1 completa:** para analizar un tópico se vuelve a descargar/transcribir/diarizar/segmentar/descubrir temas. No hay artefacto de fase 1 ni reanudación.
6. **Cleanup frágil:** `fase_1()` llama `obtener_info_playlist()` en `finally`, incluso si `_ejecutar_fase_1()` ya falló durante la obtención de metadata o si YouTube está intermitente. `fase_2()` usa `_get_playlist_name()` también en `finally`.
7. **Diarización mejoró, pero aún usa API privada:** videos usan `output.speaker_embeddings`, alineado con WhisperX; el perfil de referencia aún usa `pipeline._inferences["_embedding"]`, que es privado y puede romper con pyannote.
8. **Namespace pyannote divergente:** el código usa `pyannote-community/speaker-diarization-community-1` por smoke local; docs oficiales de pyannote y proyectos similares usan `pyannote/speaker-diarization-community-1` con token y aceptación de condiciones. Esto debe quedar explícito como `primary/fallback`, no como default silencioso.
9. **Device configurado pero no aplicado:** `EmbeddorTono` y `ClasificadorLLM` tienen `device`, pero no hacen `.to(device)` ni mueven tensores. `DiarizacionService` tampoco expone `device` ni usa `pipeline.to(...)`.
10. **Batching desaprovechado:** `EmbeddorTono.embed_batch()` itera y llama `embed()` por texto; spaCy se llama oración por oración/segmento y no usa `nlp.pipe`.
11. **Salida sin manifest runtime:** no queda una bitácora estructurada de videos procesados, omitidos, errores, thresholds/modelos usados, cache limpiado, runtime por fase ni ruta de artefactos.
12. **Docs internas desactualizadas:** `README.md`, `AGENTS.md`, `docs/configuracion.md` y `docs/componente-1-5-diarizacion.md` aún mencionan `pyannote/embedding`, `speaker_embedding_model`, `pyannote/speaker-diarization-community-1` como único pipeline, “smoke test pendiente” y “ty limpio”.

### Hallazgos de tests

1. **Base unitaria fuerte:** 260 tests recolectados, 255 no-slow verdes. Buen uso de fakes para audio/Whisper/pyannote/modelos.
2. **Hueco principal:** cero cobertura de `main.py` y del contrato CLI (`--topico`, `--tema`, `--keep-cache`, `--output`, exit codes, cleanup en errores).
3. **Tests no validan gates reales:** `pytest` y `ruff` pasan, pero `ty` falla. La suite debe incluir `uv run ty check` como gate explícito antes de cerrar.
4. **Fakes no cubren degradación end-to-end:** hay tests de `descargar_audio() -> None`, pero falta prueba de flujo completo: video fallido → `IngestaService` lo omite → manifest reporta skipped → pipeline continúa.
5. **Slow tests están separados pero sin smoke contractual de pipeline completo:** 5 tests slow existen, pero no hay un smoke controlado que verifique que fase 1/fase 2 se conectan mediante servicios fake o fixtures serializadas.
6. **Cobertura baja en módulos de mayor riesgo semántico:** descubrimiento de temas, tono/embeddings y orquestación de service.

### Hallazgos de ejecución/fluidez

1. **No hay modo resume/checkpoint:** la fase 1 real es cara; repetirla por cada tópico es fricción alta.
2. **No hay resumen final accionable:** el usuario ve logs y prints, pero no una tabla de `procesados/omitidos/fallidos`, tiempos por fase ni recomendaciones.
3. **Errores parciales no viajan por contratos:** los errores de descarga quedan en logs; no son parte de DTOs ni del resultado del run.
4. **No hay exit code taxonomy:** `sys.exit(1)` solo aparece en tópico sin segmentos; otros fallos dependen de excepciones.
5. **Cache policy binaria:** limpiar todo o `--keep-cache`; falta `run_id`, `--resume`, `--clean-cache`, `--cache-dir` y separación clara entre runtime artifacts y modelos persistentes.
6. **Progreso de modelos pesados no conectado:** pyannote soporta `ProgressHook`; Whisper/yt-dlp/BERTopic pueden reportarse por fase, pero hoy la fluidez depende de logs dispersos.

---

## 1. Recomendaciones oficiales/proyectos similares consultados

### pyannote.audio 4.x / Community-1

- README oficial de `pyannote.audio` recomienda:
  - instalar con `uv add pyannote.audio`,
  - aceptar condiciones del modelo en Hugging Face,
  - crear token HF,
  - cargar con `Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token="...")`,
  - mover a GPU con `pipeline.to(torch.device("cuda"))` si está disponible,
  - ejecutar con `ProgressHook`.
- Blog oficial de pyannote Community-1 recomienda `output.exclusive_speaker_diarization` porque en modo exclusivo solo hay un speaker activo por instante, simplificando alineación STT↔speaker.
- Implicación para este repo: mantener `exclusive_speaker_diarization`; reemplazar el acceso privado `_inferences` por salida oficial `speaker_embeddings` donde sea posible; soportar `device` y `ProgressHook`; documentar `pyannote/...` como oficial y `pyannote-community/...` como fallback validado localmente si se decide conservarlo.

### WhisperX

- `whisperx/diarize.py` usa `Pipeline.from_pretrained(model_config, token=token, cache_dir=cache_dir).to(device)`.
- Extrae `output.speaker_embeddings` y los serializa por label con `diarization.labels()`.
- Para asignación temporal usa estructura tipo interval tree para consultas de overlap eficientes en contenido largo.
- Implicación: el refactor post-smoke de usar `output.speaker_embeddings` va en la dirección correcta; falta encapsularlo en adapter y evitar API privada para perfil de referencia.

### yt-dlp

- README oficial documenta:
  - `--no-abort-on-error`: continuar con el siguiente video ante errores de descarga; es default para playlists.
  - `--ignore-errors`: ignorar errores de descarga/postprocesado, con cuidado porque puede marcar como exitoso un postproceso fallido.
  - `--retries` y `--fragment-retries`: retries por descarga/fragmento.
  - `--download-archive`: no redescargar IDs ya procesados.
  - `--continue`: reanudar fragmentos parciales.
- Implicación: el contrato `descargar_audio() -> Path | None` es correcto, pero debe emitir un resultado estructurado y, opcionalmente, usar `--download-archive` por run/cache para reanudación.

### Whisper

- Código oficial `transcribe.py` usa `word_timestamps`; en CPU, si `fp16=True`, cambia a FP32 y advierte. Pasar `fp16=False` en CPU evita warning.
- Implicación: el repo ya usa `word_timestamps=True` y `fp16=False`, bien para alineación y CPU.

### spaCy

- Documentación oficial recomienda `nlp.pipe()` para procesar lotes/streams en vez de texto por texto, deshabilitar componentes no usados y ajustar `batch_size`/`n_process` con cuidado.
- Implicación: `SegmentacionService` debe procesar segmentos/oraciones con `nlp.pipe` y cargar solo componentes necesarios para sentencización.

### BERTopic

- Best practices oficiales recomiendan:
  - precalcular embeddings,
  - fijar `random_state` en UMAP para reproducibilidad,
  - controlar número de tópicos principalmente con HDBSCAN `min_cluster_size`, no con merge posterior `nr_topics`,
  - usar `calculate_probabilities=True` solo si se necesita matriz de probabilidades, porque ralentiza.
- Implicación: `TemasService` debe exponer/reproducir seed, manejar datasets pequeños explícitamente y no depender de probabilidades costosas si solo se usa relevancia aproximada.

### Sentence Transformers / embeddings batch

- API oficial `SentenceTransformer.encode` expone `batch_size` y `normalize_embeddings`.
- Aunque el repo usa `transformers` directo para LFM2.5, la recomendación general aplica: `embed_batch()` debe tokenizar un batch real, no iterar por textos.

### Hugging Face Transformers generation

- Documentación oficial de `GenerationConfig` separa estrategia de decoding: `do_sample` habilita muestreo; `temperature` modula probabilidades cuando hay sampling.
- Implicación: para clasificación JSON determinista conviene una `GenerationConfig` explícita. Si se busca reproducibilidad, preferir `do_sample=False` o fijar seed/decoding; si se conserva sampling con `temperature=0.1`, documentarlo y testear robustez.

### uv / ty / pytest

- `uv run` ejecuta comandos dentro del entorno del proyecto y mantiene `.venv` sincronizado.
- `ty check` debe correrse desde el proyecto o con `uv run`/venv activa para descubrir dependencias.
- pytest recomienda `tmp_path` para archivos temporales por test y `monkeypatch` para cambios con teardown automático.
- Implicación: los comandos del repo deben usar consistentemente `uv run ruff`, `uv run ty`, `uv run pytest`.

---

## 2. Plan priorizado

### P0 — Gate real y documentación verdadera

#### Task 1: Corregir el gate `ty` en tests

**Objective:** que `uv run ty check` quede verde sin ignorar errores.

**Files:**
- Modify: `tests/test_audio.py:128-132`
- Modify: `tests/test_salida_models.py:73-76`

**Step 1: Write/adjust assertions with explicit narrowing**

En `tests/test_audio.py`, antes de usar `.exists()`:

```python
ruta = descargar_audio(video, playlist_mock.nombre, base_dir=tmp_path)

assert ruta is not None
assert ruta.exists()
assert ruta.name == f"{video.id}.wav"
```

En `tests/test_salida_models.py`, antes de usar `.pipeline`:

```python
assert informe.provenance is not None
assert informe.provenance.pipeline == "test"
```

**Step 2: Verify**

```bash
uv run ty check
```

Expected: `Found 0 diagnostics`.

**Step 3: Full gate**

```bash
uv run ruff check src/ tests/ main.py
uv run pytest tests/ -m "not slow" --tb=short
uv run ty check
```

Expected: ruff clean, `255 passed`, ty clean.

---

#### Task 2: Actualizar docs internas post-remediación

**Objective:** eliminar instrucciones falsas sobre `pyannote/embedding`, `speaker_embedding_model`, smoke pendiente y `ty` limpio.

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/configuracion.md`
- Modify: `docs/componente-1-5-diarizacion.md`

**Step 1: Replace stale model descriptions**

Actualizar a:

- Diarización: Community-1 con `output.speaker_embeddings`.
- Embeddings de speakers: provienen del output del pipeline, no de `pyannote/embedding` separado.
- Perfil de voz actual: `pipeline-internal` temporal; planificado migrar a perfil por `speaker_embeddings` de referencia.
- Namespace:
  - Oficial pyannote: `pyannote/speaker-diarization-community-1` con token/condiciones.
  - Fallback local validado: `pyannote-community/speaker-diarization-community-1` si el oficial falla en entorno local.
- Smoke real ya ejecutado: distancias observadas `0.075–0.131`, umbrales `0.5/0.7` con margen.
- Gate actual: `ruff` y `pytest` verdes; `ty` queda verde después de Task 1.

**Step 2: Verify stale references removed**

```bash
rg "speaker_embedding_model|pyannote/embedding|smoke test pendiente|ty check limpio" README.md AGENTS.md docs/
```

Expected: no referencias obsoletas, salvo si aparecen en una sección histórica claramente marcada como obsoleta.

---

### P1 — Orquestación testeable y contratos de ejecución ✅ COMPLETADO

#### Task 3: Introducir DTOs de ejecución del pipeline

**Objective:** representar resultados, errores parciales y manifest sin depender de logs.

**Files:**
- Create: `src/tono_politico/pipeline/models.py`
- Create: `src/tono_politico/pipeline/__init__.py`
- Test: `tests/test_pipeline_models.py`

**Models proposed:**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RunStatus = Literal["ok", "partial", "failed"]
PhaseName = Literal["ingesta", "diarizacion", "segmentacion", "temas", "filtrado", "tono", "salida"]

@dataclass(frozen=True)
class VideoRunStatus:
    video_id: str
    titulo: str = ""
    descargado: bool = False
    transcrito: bool = False
    diarizado: bool = False
    segmentos_actor: int = 0
    omitido: bool = False
    error: str | None = None

@dataclass(frozen=True)
class PhaseRunStatus:
    phase: PhaseName
    ok: bool
    elapsed_seconds: float = 0.0
    message: str = ""

@dataclass
class RunManifest:
    run_id: str
    playlist_url: str
    playlist_name: str
    status: RunStatus
    videos: list[VideoRunStatus] = field(default_factory=list)
    phases: list[PhaseRunStatus] = field(default_factory=list)
    artifacts_dir: Path | None = None
    cache_dir: Path | None = None

@dataclass
class RunResult:
    manifest: RunManifest
    exit_code: int = 0
    informe_path: Path | None = None
```

**Step 1: Write tests**

Tests should assert defaults, `status` values and serializability via `dataclasses.asdict`.

**Step 2: Run targeted tests**

```bash
uv run pytest tests/test_pipeline_models.py -v
```

---

#### Task 4: Extraer `PipelineRunner` de `main.py`

**Objective:** mover ejecución de fases a una clase testeable e inyectable.

**Files:**
- Create: `src/tono_politico/pipeline/runner.py`
- Modify: `main.py`
- Test: `tests/test_pipeline_runner.py`

**Design:**

- `main.py` solo debe:
  1. parsear args,
  2. cargar/validar config,
  3. construir `PipelineRunner`,
  4. llamar `discover()` o `analyze()`,
  5. imprimir resumen,
  6. retornar exit code.
- `PipelineRunner` recibe factories inyectables para services, útil para tests sin modelos pesados.

**Skeleton:**

```python
@dataclass
class PipelineRunner:
    cfg: Config
    factories: ServiceFactories
    keep_cache: bool = False

    def discover(self, playlist_url: str) -> RunResult:
        ...

    def analyze(self, playlist_url: str, topico_id: int, tema: str, output_path: str | None) -> RunResult:
        ...
```

**Acceptance criteria:**

- Ningún `sys.exit()` dentro de lógica de pipeline.
- `main()` puede retornar `int`; el bloque `if __name__ == "__main__"` hace `raise SystemExit(main())`.
- Fase 1 y fase 2 se prueban con fake services.

**Verification:**

```bash
uv run pytest tests/test_pipeline_runner.py -v
uv run pytest tests/ -m "not slow" --tb=short
```

---

#### Task 5: Tipar y validar config

**Objective:** reemplazar `dict` libre por config validada para eliminar defaults divergentes.

**Files:**
- Create: `src/tono_politico/config.py`
- Modify: `main.py`
- Modify: `config/config.yaml`
- Test: `tests/test_config.py`

**Config dataclasses proposed:**

```python
@dataclass(frozen=True)
class ProjectConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")

@dataclass(frozen=True)
class DiarizacionConfig:
    actor_objetivo: str = "Lilly Téllez"
    video_ref_id: str = "su9nURIj9XQ"
    pipeline: str = "pyannote/speaker-diarization-community-1"
    fallback_pipeline: str | None = "pyannote-community/speaker-diarization-community-1"
    umbral_match: float = 0.5
    umbral_ambiguo: float = 0.7
    device: str = "auto"
```

**Important migration:**

- Eliminar o deprecar `ingesta.data_dir`; usar solo `project.data_dir`.
- Si se conserva compatibilidad temporal, validar que ambos sean iguales y emitir warning.

**Tests:**

- YAML mínimo carga defaults.
- YAML con `ingesta.data_dir != project.data_dir` falla o emite warning controlado.
- Config serializable/no contiene tokens.

---

### P1 — Robustez de modelos y performance ✅ COMPLETADO

#### Task 6: Encapsular pyannote en adapter oficial/fallback

**Objective:** aislar detalles de pyannote, token, device, progress hook y fallback de namespace.

**Files:**
- Create: `src/tono_politico/diarizacion/adapter.py`
- Modify: `src/tono_politico/diarizacion/service.py`
- Test: `tests/test_diarizacion_adapter.py`

**Behavior:**

1. Intentar `primary_pipeline` oficial.
2. Si falla por acceso/gating/model-not-found y hay `fallback_pipeline`, intentar fallback.
3. Nunca loguear token.
4. Si `device="auto"`, usar CUDA cuando esté disponible.
5. Si existe `ProgressHook`, usarlo en llamadas largas.

**Test cases:**

- `from_pretrained(primary)` éxito.
- `from_pretrained(primary)` falla, fallback éxito.
- ambos fallan → error claro con instrucciones de HF token/condiciones.
- token leído desde env/cache se pasa pero no aparece en logs.

---

#### Task 7: Eliminar uso privado `pipeline._inferences` para perfil de referencia

**Objective:** construir `PerfilVozActor` desde salida pública `output.speaker_embeddings` del audio de referencia.

**Files:**
- Modify: `src/tono_politico/diarizacion/service.py`
- Test: `tests/test_diarizacion_service.py`
- Maybe Modify: `src/tono_politico/diarizacion/perfil_voz.py`

**Algorithm:**

1. Ejecutar pipeline sobre `ref_path`.
2. Leer `output.speaker_diarization.labels()` y `output.speaker_embeddings`.
3. Si hay un speaker, usar ese embedding.
4. Si hay varios speakers, elegir el label con mayor duración total en `exclusive_speaker_diarization` o `speaker_diarization`.
5. Guardar `modelo_embedding="speaker_embeddings:<pipeline_name>"`.

**Tests:**

- referencia con 1 speaker → usa ese embedding.
- referencia con 2 speakers → elige el de mayor duración.
- sin embeddings → error accionable.

---

#### Task 8: Implementar batching real en embeddings y spaCy

**Objective:** mejorar fluidez/performance sin cambiar outputs.

**Files:**
- Modify: `src/tono_politico/tono/embeddings.py`
- Modify: `src/tono_politico/segmentacion/service.py`
- Modify: `src/tono_politico/segmentacion/sentencias.py`
- Test: `tests/test_tono_embeddings.py`
- Test: `tests/test_segmentacion_service.py`

**Embedding changes:**

- `embed_batch(texts)` debe tokenizar lista completa:

```python
inputs = self._tokenizer(
    texts,
    return_tensors="pt",
    max_length=512,
    truncation=True,
    padding=True,
)
inputs = {k: v.to(self._device) for k, v in inputs.items()}
```

- Mover modelo a `self._device`.
- Mantener `embed(text)` como wrapper sobre `embed_batch([text])[0]`.

**spaCy changes:**

- Usar `nlp.pipe(texts, batch_size=...)` para segmentos.
- Deshabilitar componentes no requeridos si el modelo lo permite.

**Verification:**

- Tests unitarios con fake tokenizer/model validan que `embed_batch` llama al modelo una sola vez para N textos.
- Tests de segmentación validan mismo resultado que versión actual.

---

#### Task 9: Hacer decoding de stance reproducible

**Objective:** clasificar stance con configuración explícita y testeable.

**Files:**
- Modify: `src/tono_politico/tono/zero_shot.py`
- Modify: `config/config.yaml`
- Test: `tests/test_tono_zero_shot.py`

**Design:**

- Crear `GenerationConfig` o dataclass propia con:
  - `max_new_tokens=100`
  - `do_sample=False` por default para clasificación determinista
  - si se decide conservar sampling, exigir seed y documentar `temperature`
- Mover input tensors al device.

**Tests:**

- fake model recibe `do_sample=False` default.
- config override puede activar sampling.
- basura/JSON inválido sigue devolviendo default controlado.

---

### P2 — Flujo de ejecución, cache, resume y UX ✅ COMPLETADO

#### Task 10: Crear `RunManifest` persistente y resumen final

**Objective:** que cada corrida deje una bitácora legible y machine-readable.

**Files:**
- Create: `src/tono_politico/pipeline/manifest.py`
- Modify: `src/tono_politico/pipeline/runner.py`
- Test: `tests/test_pipeline_manifest.py`

**Artifacts:**

```text
output/runs/<run_id>/
  manifest.json
  fase1-topicos.json
  informe.json
  informe.md
```

**Manifest should include:**

- model/provider names,
- thresholds,
- config path/hash,
- playlist name/url,
- videos processed/skipped/failed,
- phase timings,
- output paths,
- cache cleanup status.

**CLI summary example:**

```text
Run: 20260706-153012-Play-PoliTest
Status: partial
Videos: 6 procesados, 1 omitido (descarga 403)
Fase 1: 12 tópicos descubiertos
Artifacts: output/runs/20260706-153012-Play-PoliTest/
Cache: limpiado (usa --keep-cache para conservar)
```

---

#### Task 11: Añadir `--run-id`, `--resume` y artefacto de fase 1

**Objective:** no repetir fase 1 para analizar múltiples tópicos.

**Files:**
- Modify: `main.py`
- Modify: `src/tono_politico/pipeline/runner.py`
- Create/Modify: serializers para `ResultadoTemas`
- Test: `tests/test_cli.py`
- Test: `tests/test_pipeline_resume.py`

**CLI proposal:**

```bash
# Descubre tópicos y guarda fase 1
uv run python main.py --playlist URL --run-id politest-001

# Reusa fase 1 para analizar tópico 3
uv run python main.py --resume output/runs/politest-001 --topico 3 --tema "seguridad"
```

**Acceptance criteria:**

- `--resume` no llama ingesta/diarización/segmentación/temas.
- Si falta artifact requerido, error claro y exit code específico.
- `--keep-cache` conserva runtime cache; sin flag, manifest registra limpieza.

---

#### Task 12: Convertir errores parciales en datos, no solo logs

**Objective:** que fallos como YouTube 403 queden en contratos y salida final.

**Files:**
- Modify: `src/tono_politico/ingesta/audio.py`
- Modify: `src/tono_politico/ingesta/service.py`
- Add: `src/tono_politico/ingesta/models.py` if needed
- Test: `tests/test_ingesta_errors.py`

**Design:**

- Introducir `DownloadResult`:

```python
@dataclass(frozen=True)
class DownloadResult:
    video_id: str
    path: Path | None
    ok: bool
    error: str | None = None
```

- `descargar_audio` puede seguir exponiendo wrapper legacy `Path | None`, pero service interno debería usar resultado estructurado.

**Tests:**

- timeout → `DownloadResult(ok=False, error="timeout...")`.
- returncode != 0 → error truncado pero informativo.
- postproceso sin archivo → error específico.
- service continúa y manifest marca video omitido.

---

#### Task 13: Usar `--download-archive` para reanudación yt-dlp

**Objective:** evitar redescargas y hacer reintentos controlados por playlist/run.

**Files:**
- Modify: `src/tono_politico/ingesta/audio.py`
- Test: `tests/test_audio.py`

**Behavior:**

- Si `run_cache_dir` existe, usar:
  - `--download-archive <run_cache_dir>/yt-dlp-archive.txt`
  - `--continue`
  - mantener `--retries 10`
- No usar `--ignore-errors` a ciegas para un solo video; para playlist batch lo maneja el service.

**Verification:**

- Test de comando generado contiene `--download-archive` cuando hay archive path.
- Fallo de un video no marca archivo como OK.

---

### P2 — Calidad semántica y escalabilidad ✅ COMPLETADO

#### Task 14: Hacer topic modeling reproducible y robusto en datasets pequeños

**Objective:** alinear `TemasService` con BERTopic best practices.

**Files:**
- Modify: `src/tono_politico/temas/descubrimiento.py`
- Modify: `src/tono_politico/temas/service.py`
- Modify: `config/config.yaml`
- Test: `tests/test_temas.py`

**Changes:**

- Exponer `random_state` para UMAP.
- Documentar/validar `min_topic_size` vs número de segmentos.
- Si `len(segmentos) < min_topic_size`, devolver tópico único controlado con warning estructurado.
- Mantener `calculate_probabilities=False` salvo que el pipeline realmente necesite matriz completa.

**Tests:**

- dataset pequeño → tópico único, no excepción.
- `random_state` se pasa al UMAP fake.
- `min_topic_size` demasiado alto produce warning/metadata.

---

#### Task 15: Indexar turnos para alineación larga

**Objective:** mantener criterio midpoint pero evitar costo lineal por segmento en videos largos.

**Files:**
- Modify: `src/tono_politico/diarizacion/alineacion.py`
- Test: `tests/test_diarizacion_alineacion.py`

**Approach:**

- Inspirado en WhisperX: construir índice ordenado por `t_start` para buscar turno por midpoint con `bisect`.
- Conservar frontera semiabierta `[inicio, fin)`.
- Si no hay match, descartar segmento.

**Tests:**

- midpoint exacto en inicio incluido.
- midpoint exacto en fin excluido.
- múltiples turnos ordenados/desordenados.
- mismo output que implementación actual en casos existentes.

---

### P3 — CLI, CI local y smoke real ✅ COMPLETADO

#### Task 16: Añadir tests de CLI sin modelos pesados

**Objective:** cubrir `main.py` y evitar regresiones de flags.

**Files:**
- Create: `tests/test_cli.py`
- Modify: `main.py`

**Tests:**

- `--topico` sin `--tema` produce error argparse.
- fase 1 llama `runner.discover`.
- fase 2 llama `runner.analyze`.
- `--keep-cache` llega al runner.
- `--config` custom se carga.
- no se llama `sys.exit` dentro de funciones testeables.

**Verification:**

```bash
uv run pytest tests/test_cli.py -v
uv run pytest tests/ -m "not slow" --cov=main --cov-report=term-missing
```

Expected: `main.py` importado y cubierto.

---

#### Task 17: Definir comandos canónicos de cierre

**Objective:** hacer que el gate local refleje el stack Astral completo del proyecto.

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Resolved: `check.sh` movido a la raíz del proyecto; `scripts/` eliminado (run_pipeline.py era obsoleto, check.sh promocionado a raíz).

**Command:**

```bash
uv run ruff check src/ tests/ main.py
uv run ruff format --check src/ tests/ main.py
uv run ty check
uv run pytest tests/ -m "not slow" --tb=short
```

**Optional slow/smoke:**

```bash
RUN_SLOW_MODELS=1 uv run pytest tests/ -m slow --tb=short
```

**Note:** `scripts/` eliminado; `check.sh` ahora vive en la raíz del proyecto.

---

#### Task 18: Smoke real controlado sobre Play-PoliTest

**Objective:** verificar control/fluidez real después de arquitectura, no solo unit tests.

**Preconditions:**

- HF token configurado localmente, sin imprimirlo.
- Modelos ya cacheados o red disponible.
- YouTube puede seguir devolviendo 403 en `71GicqtYqpQ`; debe reportarse como parcial, no bloquear.

**Commands:**

```bash
uv run python main.py --playlist "https://youtube.com/playlist?list=PLE9Zk7g9R__M" --run-id politest-smoke --keep-cache
uv run python main.py --resume output/runs/politest-smoke --topico 0 --tema "tema seleccionado" --output output/runs/politest-smoke/
```

**Acceptance criteria:**

- Run status `ok` o `partial` explícito.
- Si `71GicqtYqpQ` falla, manifest lo marca como omitido con error 403.
- Fase 2 no repite descargas/transcripción/diarización cuando usa `--resume`.
- Se generan `manifest.json`, tópicos y salida JSON/Markdown.
- Cache runtime se conserva solo si `--keep-cache`.

---

## 3. Orden recomendado de implementación

1. **Task 1** — arreglar `ty`; gate real limpio.
2. **Task 2** — docs verdaderas para no guiar mal.
3. **Task 3–4** — DTOs + runner testeable; desbloquea CLI tests.
4. **Task 5** — config tipada y `data_dir` único.
5. **Task 6–7** — pyannote adapter + eliminar `_inferences` privada.
6. **Task 10–12** — manifest, resume base y errores parciales estructurados.
7. **Task 16–17** — CLI tests + gate canónico.
8. **Task 8–9, 14–15** — performance/semántica/escalabilidad.
9. **Task 18** — smoke real de cierre.

---

## 4. Definition of Done

El plan se considera completado cuando:

- `uv run ruff check src/ tests/ main.py` pasa.
- `uv run ruff format --check src/ tests/ main.py` pasa.
- `uv run ty check` pasa.
- `uv run pytest tests/ -m "not slow" --tb=short` pasa.
- `main.py` está cubierto por tests de CLI/orquestación.
- El pipeline puede ejecutar fase 1, guardar artefactos y analizar N tópicos con `--resume` sin repetir fase 1.
- Los errores parciales de descarga/diarización/transcripción aparecen en `manifest.json` y en el resumen final.
- Diarización ya no usa `pipeline._inferences`.
- Docs internas no contienen `pyannote/embedding` ni `speaker_embedding_model` como flujo vigente.
- Smoke real sobre Play-PoliTest produce salida o `partial` explícito sin crash silencioso.

---

## 5. Riesgos y decisiones pendientes

1. **Namespace pyannote:** decidir si el default vuelve al oficial `pyannote/speaker-diarization-community-1` con fallback `pyannote-community/...`, o si se mantiene fallback como default por reproducibilidad local. Recomendación: oficial como primary + fallback explícito.
2. **Formato de artifacts fase 1:** serializar `ResultadoTemas` completo puede requerir serializers para `Segmento`, `Oracion` y scores. Implementar mínimo necesario para resume antes de embellecer.
3. **Sampling en LLM:** para clasificación determinista, recomiendo `do_sample=False`. Si el usuario prefiere variabilidad controlada, fijar seed y registrar decoding config en provenance.
4. **Cookies YouTube:** no incorporar cookies/browser auth sin autorización explícita del usuario.
5. **Scripts:** `scripts/` eliminado (run_pipeline.py obsoleto); `check.sh` promocionado a la raíz.
