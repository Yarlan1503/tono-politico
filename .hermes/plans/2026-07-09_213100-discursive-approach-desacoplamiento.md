# Hoja de Ruta: Desacoplar discursive_approach de `tono/`, `filtrado.models`, `segmentacion.models`, `temas.models`

> **Para Hermes:** Usar subagent-driven-development para implementar tarea por tarea.

**Meta:** Que `discursive_approach/` sea autocontenido: cero imports hacia `tono/`, `filtrado/`, `segmentacion/`, `temas/`.

**Arquitectura:** El acoplamiento es de 2 tipos:
1. **Motor de inferencia** — `topics_approach/service.py` usa `TonoService.procesar()`. Solución: mover `tono/` completo a `discursive_approach/tono/`.
2. **DTOs puente** — `topics_approach/adapter.py` construye `Segmento`, `ResultadoFiltrado`, `TopicoInfo` legacy solo para alimentar `TonoService`. Solución: refactorizar `TonoService` para que acepte `list[Argumento]` directamente, eliminando el adapter.

**Stack:** Python 3.11 · uv · ruff · ty · pytest

---

## Contexto

### Dependencias externas de `discursive_approach/`

| Módulo externo | Qué importa | Dónde | Líneas |
|---|---|---|---|
| `tono/` (6 archivos) | `TonoService`, `ResultadoTono`, `SegmentoConTono`, stack embeddings/zero_shot/taxonomia | `topics_approach/service.py`, `topics_approach/adapter.py` | 1063 |
| `filtrado/models.py` | `ResultadoFiltrado`, `SegmentoFiltrado`, `CriterioFiltrado` | `topics_approach/adapter.py` | 57 |
| `segmentacion/models.py` | `Segmento`, `Oracion` | `topics_approach/adapter.py`, `tono/models.py` | 48 |
| `temas/models.py` | `TopicoInfo` (legacy) | `topics_approach/adapter.py` | 56 |

### El cuello de botella: `adapter.py`

`topics_approach/adapter.py` (98 líneas) hace esto:

```
Argumento[] → argumento_a_segmento() → Segmento (legacy)
            → argumentos_a_resultado_filtrado() → ResultadoFiltrado (legacy)
            → TonoService.procesar(ResultadoFiltrado) → ResultadoTono
            → segmento_con_tono_a_perfil() → PerfilTonoArgumento
```

El adapter existe **solo** porque `TonoService.procesar()` requiere `ResultadoFiltrado` como entrada. Si `TonoService` aceptara `list[Argumento]` o `list[str]` directamente, el adapter desaparece y los DTOs puente dejan de necesitarse.

### `tono/models.py` también depende de `segmentacion/models.py`

```python
# tono/models.py L17
from ..segmentacion.models import Segmento
```

`SegmentoConTono` hereda/envuelve `Segmento`. Al mover `tono/` dentro de `discursive_approach/`, este import se actualiza a un DTO local.

### Tests afectados

| Test file | Tests | Importa de |
|---|---|---|
| `test_tono_embeddings.py` | 12 | `tono.embeddings` |
| `test_tono_models.py` | 16 | `tono.models` + `segmentacion.models` |
| `test_tono_taxonomia.py` | 12 | `tono.taxonomia` |
| `test_tono_zero_shot.py` | 19 | `tono.models` + `tono.zero_shot` |
| `test_topics_approach.py` | 6 | `topics_approach` (usa adapter indirectamente) |

Total: **65 tests** que se mueven/actualizan.

---

## Tareas

### Fase A: Mover `tono/` a `discursive_approach/tono/`

#### Task A1: `git mv tono/ → discursive_approach/tono/`

**Objetivo:** Mover el paquete completo.

**Step 1:**
```bash
git mv src/tono_politico/tono/ src/tono_politico/discursive_approach/tono/
```

**Step 2:** Actualizar imports internos de `tono/` que referencian `segmentacion.models`:
- `discursive_approach/tono/models.py` L17: `from ..segmentacion.models import Segmento` → crear DTO local o importar de `..argument_shape.models`

**Step 3:** Actualizar `discursive_approach/tono/service.py` L26: `from ..filtrado.models import ResultadoFiltrado` → ver Task B1 (refactor de signature).

**Step 4:** Crear `discursive_approach/tono/__init__.py` con exports públicos (mismo contenido que el `tono/__init__.py` actual, ajustando imports relativos).

**Verificación:** `uv run ruff check src/tono_politico/discursive_approach/tono/`

#### Task A2: Actualizar `topics_approach/service.py`

**Archivo:** `discursive_approach/topics_approach/service.py`

```python
# Antes:
from ...tono.service import TonoService
# Después:
from ..tono.service import TonoService
```

#### Task A3: Actualizar `topics_approach/adapter.py`

**Archivo:** `discursive_approach/topics_approach/adapter.py`

```python
# Antes:
from ...tono.models import ResultadoTono, SegmentoConTono
# Después:
from ..tono.models import ResultadoTono, SegmentoConTono
```

#### Task A4: Actualizar `discursive_approach/__init__.py`

Si el `__init__.py` de `discursive_approach` exponía algo de `tono/` via lazy, actualizar la ruta.

### Fase B: Eliminar el adapter (DTOs puente)

#### Task B1: Refactorizar `TonoService.procesar()` para aceptar `list[str]`

**Objetivo:** Eliminar la dependencia de `ResultadoFiltrado`/`Segmento`/`SegmentoFiltrado`.

**Archivo:** `discursive_approach/tono/service.py`

**Nuevo signature:**
```python
def procesar_textos(
    self,
    textos: list[str],
    *,
    metadatos: list[dict] | None = None,
) -> ResultadoTono:
    """Analiza tono sobre textos planos con metadatos opcionales."""
```

**Implementación:**
1. `procesar_textos` construye internamente lo que necesita (embeddings, LLM) sin requerir `Segmento` ni `ResultadoFiltrado`.
2. `SegmentoConTono` se simplifica para no envolver `Segmento` — solo guarda `texto` + `t_start` + `t_end` + `video_id` + los resultados de tono.
3. Mantener `procesar(resultado_filtrado)` como compatibilidad temporal que delega en `procesar_textos`.

**Test:**
```python
def test_tono_service_procesar_textos():
    svc = TonoService(actor="X", tema="Y", tono_analyzer=FakeAnalyzer())
    resultado = svc.procesar_textos(["texto de prueba"], metadatos=[{"video_id": "v1"}])
    assert len(resultado.segmentos) == 1
    assert resultado.segmentos[0].texto == "texto de prueba"
```

#### Task B2: Simplificar `SegmentoConTono` (eliminar herencia de `Segmento`)

**Archivo:** `discursive_approach/tono/models.py`

```python
# Antes:
@dataclass
class SegmentoConTono:
    segmento: Segmento  # dependencia de segmentacion.models
    stance: ResultadoStance
    ...

# Después:
@dataclass
class SegmentoConTono:
    texto: str
    t_start: float
    t_end: float
    video_id: str | None = None
    stance: ResultadoStance = ...
    ...
```

#### Task B3: Reescribir `topics_approach/service.py` sin adapter

**Archivo:** `discursive_approach/topics_approach/service.py`

```python
# Antes:
from .adapter import argumentos_a_resultado_filtrado, resultado_tono_a_perfiles
resultado_filtrado = argumentos_a_resultado_filtrado(argumentos, topico)
resultado_tono = self.tono_service.procesar(resultado_filtrado)
perfiles = resultado_tono_a_perfiles(resultado_tono)

# Después:
textos = [a.texto for a in argumentos]
metadatos = [{"video_id": a.video_id, "t_start": a.t_start, "t_end": a.t_end} for a in argumentos]
resultado_tono = self.tono_service.procesar_textos(textos, metadatos=metadatos)
perfiles = [segmento_con_tono_a_perfil(sct) for sct in resultado_tono.segmentos]
```

#### Task B4: Eliminar `adapter.py`

**Archivo a eliminar:**
- `discursive_approach/topics_approach/adapter.py`

Mover las 2 funciones de conversión que sí se quedan (`segmento_con_tono_a_perfil`, `resultado_tono_a_perfiles`) a `topics_approach/service.py` o `topics_approach/enfoques.py`.

#### Task B5: Eliminar imports de `filtrado.models`, `segmentacion.models`, `temas.models`

Después de Task B1-B4, ningúno debería quedar. Verificar:

```bash
grep -r 'filtrado' src/tono_politico/discursive_approach/
grep -r 'segmentacion' src/tono_politico/discursive_approach/
grep -r 'temas.models' src/tono_politico/discursive_approach/
```

Ambos deben devolver 0 resultados.

### Fase C: Mover y actualizar tests

#### Task C1: Mover tests de tono

**Archivos:**
- `git mv tests/test_tono_embeddings.py tests/test_tono_models.py tests/test_tono_taxonomia.py tests/test_tono_zero_shot.py` → mantener nombres (ya son tests del stack de tono, ahora bajo discursive_approach).

Actualizar imports:
- `from tono_politico.tono.` → `from tono_politico.discursive_approach.tono.`
- `from tono_politico.segmentacion.models import Segmento` → eliminar o usar DTO local

#### Task C2: Actualizar `test_topics_approach.py`

- Verificar que los fakes (`FakeTono`) ya no usen `ResultadoFiltrado`.
- Si el fake implementa `procesar()`, cambiarlo a `procesar_textos()`.

#### Task C3: Actualizar `test_argument_shape_smoke.py`

Si importa de `tono_politico.diarizacion.actor_transcript`, actualizar a la nueva ruta (ver hoja de ruta de speech2text).

### Fase D: Eliminar paquetes vacíos

#### Task D1: Eliminar `tono/`, `filtrado/`, `segmentacion/`, `temas/`

**Pre-verificación:**
```bash
grep -r 'tono_politico.tono' src/ tests/ main.py
grep -r 'tono_politico.filtrado' src/ tests/ main.py
grep -r 'tono_politico.segmentacion' src/ tests/ main.py
grep -r 'tono_politico.temas' src/ tests/ main.py
```

Todos deben devolver 0 resultados.

**Eliminar:**
```bash
rm -rf src/tono_politico/tono/
rm -rf src/tono_politico/filtrado/
rm -rf src/tono_politico/segmentacion/
rm -rf src/tono_politico/temas/
```

**Verificación final completa:**
```bash
uv run ruff check src/ tests/ main.py
uv run ty check
uv run pytest tests/ -q -m "not slow"
uv run python main.py --config config/config.yaml --validate-config
uv run python main.py --config config/config.yaml --dry-run
```

### Fase E: Docs

#### Task E1: Actualizar README, AGENTS, configuracion

- Estructura del código: reflejar que `tono/` ahora es `discursive_approach/tono/`.
- Eliminar referencias a `filtrado/`, `segmentacion/`, `temas/`.
- Actualizar tabla de componentes.

**Commit final:**
```bash
git add -A && git commit -m "refactor: discursive_approach autocontenido — tono/ movido, adapter eliminado, DTOs puente removidos"
```

---

## Orden de ejecución recomendado

1. **Primero la hoja de speech2text** — mueve `diarizacion/` a `speech2text/diarization/` y `PlaylistInfo` a `audio_fetcher/models.py`.
2. **Después esta hoja** — mueve `tono/` a `discursive_approach/tono/`, refactoriza el adapter, elimina DTOs puente.
3. **Al final** — verificar que `tono_politico/` solo contiene: `execution/`, `speech2text/`, `discursive_approach/`, y nada más (excepto `__init__.py`).

Resultado: un proyecto con 3 paquetes autocontenidos y cero dependencias cruzadas hacia módulos legacy.

---

## Riesgos

| Riesgo | Mitigación |
|---|---|
| `TonoService.procesar()` refactor puede romper tests de integración slow | Mantener `procesar()` como wrapper temporal que delega en `procesar_textos()` |
| `SegmentoConTono` sin `Segmento` puede romper serialización JSON | Verificar que el JSON output solo usa campos planos (texto, scores) |
| `tono/embeddings.py` importa `numpy`/`torch` pesados | El `__init__.py` de `discursive_approach` ya es lazy — mantenerlo así |
| Tests slow de tono (5 deselected) pueden necesitar actualización | Ejecutar `RUN_SLOW=1 bash check.sh` después de cada fase |
