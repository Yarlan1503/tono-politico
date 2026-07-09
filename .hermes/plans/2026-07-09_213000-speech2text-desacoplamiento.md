# Hoja de Ruta: Desacoplar speech2text de `diarizacion/` y `models.py`

> **Para Hermes:** Usar subagent-driven-development para implementar tarea por tarea.

**Meta:** Que `speech2text/` sea autocontenido: cero imports hacia `tono_politico.diarizacion` o `tono_politico.models`.

**Arquitectura:** Mover los 7 archivos reusados de `diarizacion/` a `speech2text/` como subpaquete `speech2text/diarization/`. Mover `PlaylistInfo` de `models.py` a `speech2text/audio_fetcher/models.py`. Actualizar todos los imports. Eliminar `diarizacion/` y `models.py`.

**Stack:** Python 3.11 · uv · ruff · ty · pytest

---

## Contexto

`speech2text/` importa de dos módulos externos:

1. **`diarizacion/`** — 7 archivos (976 líneas totales): `models.py`, `adapter.py`, `matching.py`, `perfil_voz.py`, `transcripcion_actor.py`, `whisper_clip.py`, `actor_transcript.py`
2. **`models.py`** — 1 DTO: `PlaylistInfo`

Después de moverlos, `diarizacion/` y `models.py` quedan vacíos y se eliminan.

**Tests afectados:** 38 tests en 8 archivos importan de `diarizacion/`:
- `test_diarizacion_adapter.py` (5)
- `test_diarizacion_matching.py` (18)
- `test_diarizacion_models.py` (8)
- `test_diarizacion_perfil_voz.py` (4)
- `test_perfil_desde_output.py` (3)
- `test_actor_transcript_serializacion.py` (4)
- `test_transcripcion_actor.py` (10)
- `test_whisper_clip_transcriber.py` (7)

**Tests que importan `PlaylistInfo` de `models`:**
- `test_audio_fetcher_models.py`
- `test_audio_fetcher_playlist.py`
- `test_speech2text_service.py`
- `test_fecha_propagacion.py`

---

## Tareas

### Task 1: Crear `speech2text/diarization/` con los 7 archivos movidos

**Objetivo:** Mover los 7 archivos de `diarizacion/` a `speech2text/diarization/` actualizando imports internos.

**Archivos:**
- Crear: `src/tono_politico/speech2text/diarization/__init__.py`
- Mover: `diarizacion/models.py` → `speech2text/diarization/models.py`
- Mover: `diarizacion/adapter.py` → `speech2text/diarization/adapter.py`
- Mover: `diarizacion/matching.py` → `speech2text/diarization/matching.py`
- Mover: `diarizacion/perfil_voz.py` → `speech2text/diarization/perfil_voz.py`
- Mover: `diarizacion/transcripcion_actor.py` → `speech2text/diarization/transcripcion_actor.py`
- Mover: `diarizacion/whisper_clip.py` → `speech2text/diarization/whisper_clip.py`
- Mover: `diarizacion/actor_transcript.py` → `speech2text/diarization/actor_transcript.py`

**Step 1:** `git mv` de cada archivo a `speech2text/diarization/`.

**Step 2:** Crear `__init__.py` exponiendo la API pública (mismos exports que el `diarizacion/__init__.py` actual, sin `DiarizacionService`).

**Step 3:** Actualizar imports internos — cada archivo movido que hacia `from .models` o `from .adapter` sigue funcionando porque están en el mismo paquete. Los imports a `tono_politico.diarizacion.X` dentro de estos archivos (si existen) se actualizan a `tono_politico.speech2text.diarization.X` o imports relativos.

**Step 4:** `uv run ruff check src/tono_politico/speech2text/diarization/`

### Task 2: Actualizar imports en `speech2text/` (consumidores)

**Objetivo:** Que los archivos de `speech2text/` que antes importaban de `diarizacion/` ahora importen del nuevo subpaquete.

**Archivos a modificar:**
- `speech2text/service.py` — `from tono_politico.diarizacion.models` → `from .diarization.models`
- `speech2text/speaker_timestamps/__init__.py` — `from tono_politico.diarizacion.models` → `from ..diarization.models`
- `speech2text/speaker_timestamps/service.py` — 4 imports `from tono_politico.diarizacion.*` → `from ..diarization.*`
- `speech2text/transcribe_speech/__init__.py` — `from tono_politico.diarizacion.models` → `from ..diarization.models`
- `speech2text/transcribe_speech/service.py` — 3 imports `from tono_politico.diarizacion.*` → `from ..diarization.*`

**Verificación:** `uv run python -c "from tono_politico.speech2text import SpeechToTextService; print('ok')"`

### Task 3: Actualizar imports en `discursive_approach/` (consumidor de DTOs)

**Objetivo:** `discursive_approach/` importa `ActorTranscript` y `ActorTranscriptSegment` de `diarizacion.models`.

**Archivos a modificar:**
- `discursive_approach/service.py` — `from ..diarizacion.models` → `from ..speech2text.diarization.models`
- `discursive_approach/argument_shape/service.py` — `from ...diarizacion.models` → `from ...speech2text.diarization.models`
- `discursive_approach/argument_shape/sentencias.py` — `from ...diarizacion.models` → `from ...speech2text.diarization.models`

**Verificación:** `uv run python -c "from tono_politico.discursive_approach import DiscursiveApproachService; print('ok')"`

### Task 4: Actualizar imports en `execution/runner.py`

**Objetivo:** `execution/runner.py` importa `guardar_actor_transcript` y `cargar_actor_transcript`.

**Archivo a modificar:**
- `execution/runner.py` — `from tono_politico.diarizacion.actor_transcript` → `from tono_politico.speech2text.diarization.actor_transcript`

**Verificación:** `uv run python -c "from tono_politico.execution.runner import ExecutionRunner; print('ok')"`

### Task 5: Mover `PlaylistInfo` de `models.py` a `speech2text/audio_fetcher/models.py`

**Objetivo:** Eliminar la última dependencia de `speech2text/` hacia `models.py`.

**Archivos:**
- Modificar: `speech2text/audio_fetcher/models.py` — añadir `PlaylistInfo` dataclass
- Modificar: `speech2text/audio_fetcher/playlist.py` — import de `..models` → `from .models import PlaylistInfo`
- Modificar: `speech2text/audio_fetcher/service.py` — import de `..models` → `from .models import PlaylistInfo`
- Modificar: `speech2text/service.py` — import de `tono_politico.models` → `from .audio_fetcher.models import PlaylistInfo`

**Step 1:** Copiar la definición de `PlaylistInfo` (con `VideoInfo` que referencia) de `models.py` a `audio_fetcher/models.py`.

**Step 2:** Actualizar los 3 imports de consumidores.

**Step 3:** `uv run ruff check src/tono_politico/speech2text/`

### Task 6: Actualizar tests que importan de `diarizacion/`

**Objetivo:** Los 8 archivos de test necesitan actualizar sus imports.

**Archivos (renombrar + actualizar imports):**
- `test_diarizacion_adapter.py` → `test_diarization_adapter.py`
- `test_diarizacion_matching.py` → `test_diarization_matching.py`
- `test_diarizacion_models.py` → `test_diarization_models.py`
- `test_diarizacion_perfil_voz.py` → `test_diarization_perfil_voz.py`
- `test_perfil_desde_output.py` — actualizar import
- `test_actor_transcript_serializacion.py` — actualizar import
- `test_transcripcion_actor.py` — actualizar import
- `test_whisper_clip_transcriber.py` — actualizar import

**Patrón:** `git mv` + `sed` de `tono_politico.diarizacion` → `tono_politico.speech2text.diarization`.

**Verificación:** `uv run pytest tests/ -q -m "not slow"` → 218 passed

### Task 7: Actualizar tests que importan `PlaylistInfo`

**Objetivo:** Tests que importan de `tono_politico.models`.

**Archivos:**
- `test_audio_fetcher_models.py` — `PlaylistInfo` ya está en `audio_fetcher/models.py`
- `test_audio_fetcher_playlist.py` — actualizar si importa de `tono_politico.models`
- `test_speech2text_service.py` — actualizar si importa de `tono_politico.models`
- `test_fecha_propagacion.py` — actualizar si importa de `tono_politico.models`

**Verificación:** `uv run pytest tests/ -q -m "not slow"` → 218 passed

### Task 8: Eliminar `diarizacion/` y `models.py`

**Objetivo:** Los paquetes ya no tienen consumidores.

**Archivos a eliminar:**
- `src/tono_politico/diarizacion/` (directorio completo — solo queda `__init__.py` vacío)
- `src/tono_politico/models.py`

**Pre-verificación:**
```bash
grep -r 'tono_politico.diarizacion' src/ tests/ main.py
grep -r 'tono_politico.models' src/ tests/ main.py
```
Ambos deben devolver 0 resultados.

**Verificación final:**
```bash
uv run ruff check src/ tests/ main.py
uv run ty check
uv run pytest tests/ -q -m "not slow"
uv run python main.py --config config/config.yaml --validate-config
uv run python main.py --config config/config.yaml --dry-run
```

### Task 9: Actualizar `execution/runner.py` imports de test

**Objetivo:** `test_execution_runner.py` tiene fakes que importan de `diarizacion`.

**Archivo:**
- `test_execution_runner.py` — actualizar imports en fakes (`GranularSpeechToText` usa `ruta_audio`, `VideoMeta`; imports de `ActorTranscript` en `_actor_transcript()`).

**Verificación:** `uv run pytest tests/test_execution_runner.py -q`

### Task 10: Actualizar docs

**Archivos:**
- `README.md` — actualizar estructura del código
- `AGENTS.md` — actualizar estructura y tabla de componentes
- `docs/componente-speech2text.md` — actualizar referencias a `diarizacion/`

**Commit final:**
```bash
git add -A && git commit -m "refactor: speech2text autocontenido — diarizacion/ y models.py eliminados"
```

---

## Riesgos

| Riesgo | Mitigación |
|---|---|
| Import circular: `discursive_approach` importa de `speech2text.diarization.models` | No hay circularidad: `speech2text` no importa de `discursive_approach` |
| `tono/models.py` importa `Segmento` de `segmentacion/models.py` (no afecta a speech2text) | Fuera del scope de esta hoja de ruta — se aborda en la hoja de discursive_approach |
| Tests con nombres de archivo renombrados pueden romper CI | Usar `git mv` para preservar historia |
