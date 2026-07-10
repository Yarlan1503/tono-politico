# Auditoría y hoja de ruta de refactorización de `speech2text`

> **Para Hermes:** ejecutar la implementación con TDD estricto y revisión por fases. No avanzar a la siguiente fase sin cerrar sus gates.

**Fecha de auditoría:** 2026-07-10  
**Módulo:** `src/tono_politico/speech2text/`  
**Control plane relacionado:** `src/tono_politico/execution/`  
**Estado del working tree:** hay una refactorización amplia sin commit; esta auditoría describe el estado actual, no pretende reconstruir el estado de `HEAD`.

**Meta:** hacer que `speech2text` sea correcto, observable, reanudable y reproducible sin alterar su frontera funcional: `playlist → audio → diarización/match → ASR actor-only → ActorTranscript`.

**Arquitectura propuesta:** conservar `SpeechToTextService` y tres submódulos autocontenidos (`audio_fetcher`, `speaker_timestamps`, `transcribe_speech`). El núcleo sólo contiene la orquestación y los DTOs de dominio. La persistencia JSON, los estados de ejecución, el manifest y la observabilidad pertenecen al control plane (`execution/`), no al núcleo de `speech2text`.

**Stack:** Python 3.11 · uv · ruff · ty · pytest · yt-dlp · pyannote Community-1 · OpenAI Whisper.

## Estructura aprobada

```text
src/tono_politico/speech2text/
├── __init__.py
├── service.py                    # orquesta los tres submódulos
├── models.py                     # DTOs del dominio de speech2text
│
├── audio_fetcher/
│   ├── __init__.py
│   ├── playlist.py               # playlist_name + metadata de vídeos
│   ├── audio.py                  # descarga y cache de audio
│   ├── service.py                # orquesta audio_fetcher
│   └── models.py                 # DTOs del dominio de audio_fetcher
│
├── speaker_timestamps/
│   ├── __init__.py
│   ├── perfil_voz.py             # construye el perfil de voz
│   ├── matching.py               # identifica turnos y valida resultados
│   ├── service.py                # carga modelo y orquesta diarización
│   └── models.py                 # DTOs del dominio de speaker_timestamps
│
└── transcribe_speech/
    ├── __init__.py
    ├── actor_clip.py             # padding y eliminación de audio no actor
    ├── transcription_clip.py     # Whisper sobre el audio editado
    ├── service.py                # orquesta transcribe_speech
    └── models.py                 # DTOs del dominio de transcribe_speech
```

No forman parte de esta estructura `errors.py`, `validation.py`, `cache.py`, `adapter.py`, `output.py`, `quality.py`, `results.py`, `requisitos.md`, `actor_transcript.py`, `whisper_clip.py` ni `transcripcion_actor.py`.

La serialización JSON de `ActorTranscript` debe vivir en el control plane o en su capa de artefactos, no como un módulo adicional del núcleo.

---

## 1. Resumen ejecutivo

`speech2text` está bien encaminado en su separación de responsabilidades y su ruta granular. La suite focalizada pasa y la cobertura de statements del paquete es alta, pero el módulo todavía no está listo para declararse robusto en producción porque los fallos relevantes se pierden como `None`, el runner puede marcar como `ok` una corrida sin perfil de voz, el resume considera suficiente un directorio vacío, y la calidad actual no representa los vídeos descubiertos que fueron omitidos.

El hallazgo de datos más importante es que el smoke real disponible contiene **7 transcripts, 195 segmentos y 1.961 palabras, pero `fecha=null` en los 7 vídeos**. Esto confirma que la propagación existe en el DTO, pero no está resuelta en la ingesta de metadata.

El gate global no está verde: `ruff` y formato pasan; `ty` falla con seis imports legacy inexistentes en `discursive_approach/topics_approach`; la colección completa de pytest falla al importar dos suites discursivas. Es una deuda del repositorio vecino, no un fallo de los 102 tests focalizados de `speech2text`, pero bloquea la definición de terminado del módulo.

---

## 2. Evidencia ejecutada

### Gates y probes

| Comando / probe | Resultado actual | Lectura |
|---|---:|---|
| `uv run ruff check src/ tests/ main.py` | pasa | No hay errores Ruff detectados |
| `uv run ruff format --check src/ tests/ main.py` | pasa; 77 archivos | Formato consistente |
| `uv run ty check` | falla con 6 diagnósticos | Imports retirados en `topics_approach` |
| Suite focalizada `speech2text` | **102 passed** | Contratos unitarios actuales pasan |
| Tests adicionales de propagación de fecha | **2 passed** | La capa de transcripción propaga fecha cuando la recibe |
| Suite focalizada con cobertura | **102 passed; 90%** | 698 statements, 71 no cubiertos |
| Tests de `execution` | **24 passed** | El runner granular está cubierto en los casos nominales |
| `bash check.sh` | falla en `ty` antes de pytest | El gate canónico no puede declararse verde |
| `uv run python main.py ... --validate-config` | pasa | El config se carga y valida |
| `uv run python main.py ... --dry-run` | pasa | Se construye el plan stage-based |
| Import probe de `main.py` | no carga `pyannote`, `whisper`, `spacy`, `torch` | Lazy loading del CLI funciona |
| `git diff --check` | pasa | No hay whitespace errors en el diff actual |

La colección global de pytest no llega a ejecutar la suite completa: falla al importar `tests/test_discursive_approach_service.py` y `tests/test_topics_approach.py` por `ModuleNotFoundError: tono_politico.tono`.

### Smoke real disponible

Artefactos inspeccionados en `output/20260709-162526/`:

- 7 archivos `ActorTranscript`.
- 195 segmentos.
- 1.961 palabras.
- 24 segmentos de una palabra.
- 41 segmentos de dos palabras o menos.
- Duración mínima de segmento: aproximadamente 0,016875 s.
- Todos los rangos inspeccionados respetan `source_turn`.
- **7/7 transcripts tienen `fecha=null`.**
- El stage `speech2text` tardó 2.280,09 s, aproximadamente 38,00 minutos.
- El `manifest.json` de esta corrida histórica no contiene `speech2text/quality.json`; la generación del quality report pertenece al cambio actual y aún necesita verificarse en un smoke nuevo.

El smoke confirma que no se debe introducir un filtro automático por número de palabras o duración: hay fórmulas cortas como `Gracias.` que deben conservarse hasta contar con una muestra etiquetada.

---

## 3. Qué está bien y debe preservarse

1. **Frontera funcional clara.** `speech2text` no intenta hacer segmentación semántica, temas ni tono.
2. **Tres responsabilidades separadas.** Descarga/cache, diarización/matching y ASR por clips están aislados.
3. **API granular.** `discover`, `ensure_perfil` y `procesar_one` permiten que el runner controle el loop y los artefactos.
4. **ASR actor-only.** Whisper se ejecuta sobre los turnos aceptados, no sobre el vídeo completo.
5. **Contrato durable explícito.** `actor_transcript.v1` conserva actor, turnos, timestamps absolutos, `source_turn` y metadata ASR.
6. **Diarización adecuada para la reconciliación.** Community-1 y `exclusive_speaker_diarization` son coherentes con el objetivo de recortar clips sin solapes.
7. **Carga perezosa.** Los imports pesados ocurren dentro de métodos de runtime y el CLI puede validar/dry-run sin cargar modelos.
8. **Frontera de ejecución clara.** La observabilidad, el manifest y los estados de corrida pertenecen a `execution/`; `speech2text` entrega dominio y DTOs.
9. **Inyección de dependencias en tests.** ffmpeg, Whisper y pyannote tienen puntos fakeables.
10. **Decisiones vigentes.** Se conservan segmentos cortos y no se recalibran thresholds sin datos etiquetados.

---

## 4. Hallazgos priorizados

### P1 — La pérdida del perfil de referencia puede terminar con estado `ok`

**Evidencia:** `SpeechToTextService.ensure_perfil()` devuelve `False` si falta o falla el audio de referencia. `ExecutionRunner._run_speech2text_granular()` simplemente no procesa vídeos y devuelve una lista vacía. Un probe con un servicio fake produjo:

```text
exit_code=0
stage_status=ok
quality=True
```

Esto confunde “playlist vacía / cero participación” con “pipeline incapaz de identificar al actor”.

**Corrección propuesta:** convertir la ausencia del perfil en un resultado estructurado de stage o en una excepción tipada (`ReferenceProfileError`) que marque `speech2text` como `failed`; conservar el detalle en manifest y quality report.

**Archivos:** `speech2text/service.py`, `speech2text/models.py`, `execution/runner.py`, tests de runner y service.

### P1 — Fallos parciales se degradan a `None` y desaparecen

**Evidencia:** `fetch_one`, `procesar_one` de diarización y `procesar_one` de ASR pueden terminar en `None` o lista vacía. El runner sólo persiste transcripts exitosos. La calidad recibe `Iterable[ActorTranscript]`, no la lista de vídeos seleccionados ni los motivos de skip.

**Impacto:** no se puede distinguir descarga fallida, audio corrupto, perfil no encontrado, actor no identificado, ASR vacío o error de modelo.

**Corrección propuesta:** si se conservan estados por vídeo, deben modelarse en `execution/` y no introducir un `results.py` en `speech2text`. `SpeechToTextService` seguirá entregando los DTOs de dominio; el runner decidirá cómo registrar éxito, skip o fallo sin contaminar `ActorTranscript`.

**Archivos:** `speech2text/service.py`, `speech2text/models.py`, subservicios y `execution/runner.py`.

### P1 — La observabilidad del runner cuenta sólo transcripts materializados

**Evidencia:** el mecanismo actual recibe transcripts, y `_run_speech2text_granular()` sólo agrega los que no son `None`. Por tanto, `total_videos` no representa los vídeos descubiertos/seleccionados y los `None` reales del runner desaparecen.

**Corrección propuesta:** mover esta responsabilidad al control plane. Si se necesita un informe, debe recibir el inventario seleccionado y los estados del runner; no debe convertirse en `quality.py` dentro de `speech2text`.

**Criterios mínimos del control plane:** vídeos descubiertos, seleccionados, exitosos, skipped, failed; motivos por vídeo; turnos diarizados; turnos aceptados; clips ASR; segmentos con texto; palabras; duraciones; modelo/thresholds/procedencia.

### P1 — `resume` considera completo un directorio de transcripts vacío

**Evidencia:** `artifact_exists(..., "actor_transcripts_dir")` sólo comprueba que el directorio exista. Un probe con un directorio vacío produjo:

```text
empty_dir_should_run=False
reason=artefacto de salida ya existe: actor_transcripts_dir
```

**Corrección propuesta:** definir completitud de `speech2text` como un artefacto verificable: manifest/quality válidos, inventario de vídeos compatible y, para cada vídeo seleccionado, estado terminal registrado. No basta `Path.exists()`.

**Archivos:** `execution/artifacts.py`, `execution/plan.py`, `execution/runner.py`, tests de resume.

### P1 — `fecha` está declarada como contrato, pero el pipeline real la pierde

**Evidencia:** el parser sólo usa `upload_date`; los 7 transcripts reales disponibles contienen `fecha=null`. La capa de transcripción sí propaga una fecha cuando `AudioVideo.fecha` la trae.

**Corrección propuesta:** normalizar explícitamente fuentes disponibles de yt-dlp (`upload_date` y fallback documentado), validar formato `YYYYMMDD`, conservar `None` sólo cuando no hay fuente confiable y registrar `metadata_date_status`/motivo. Añadir fixture realista y prueba de normalización.

**No hacer:** inventar fechas o convertir timestamps sin una decisión de zona horaria y fuente.

### P1 — El cache acepta cualquier path existente con sufijo `.wav`

**Evidencia:** `AudioFetcherService.fetch_one()` usa `destino.exists()`. Un directorio llamado `v.wav` fue aceptado como `AudioVideo` válido. `descargar_audio_result()` sólo comprueba existencia, no archivo regular ni tamaño mínimo.

**Corrección propuesta:** `is_valid_audio_cache()` debe comprobar archivo regular, tamaño > 0 y, si el coste es aceptable, duración/codec con `ffprobe`. Ante cache inválido, eliminar o apartar el path y redescargar.

**Archivos:** `audio_fetcher/audio.py`, `audio_fetcher/service.py`, tests de cache corrupto.

### P1 — El gate global está bloqueado por el refactor vecino

**Evidencia:** `ty check` reporta imports inexistentes en:

- `src/tono_politico/discursive_approach/topics_approach/adapter.py` hacia `filtrado.models`, `segmentacion.models`, `temas.models`, `tono.models`.
- `src/tono_politico/discursive_approach/topics_approach/service.py` hacia `tono.service`.

Esos paths no existen en el filesystem actual. La colección global de pytest falla por el mismo corte de `tono`.

**Corrección propuesta:** tratarlo como dependencia de release de esta hoja de ruta: desacoplar/reconstruir `topics_approach` o retirar su import del path activo antes de afirmar que `speech2text` está integrado y verificable de extremo a extremo.

### P2 — Extracción de embeddings sin validación de forma

**Evidencia:** `_extraer_embeddings()` asume que `len(speaker_embeddings)` coincide con `labels()`. Un probe con dos labels y un embedding produjo `IndexError`.

La documentación oficial de pyannote describe el orden de embeddings respecto de `speaker_diarization.labels()`, pero el código debe validar `None`, cantidad, dimensión, finitud y labels antes de confiar en la correspondencia.

**Corrección propuesta:** concentrar la consistencia de labels, embeddings, turnos y rangos en `speaker_timestamps/matching.py`. `service.py` carga el modelo y entrega la salida; `matching.py` valida y transforma esa salida en los DTOs de `speaker_timestamps`. No se crea `adapter.py` ni `output.py`.

### P2 — El parser de metadata y los subprocesses no cubren todos los fallos

**Evidencia:** una duración no numérica produce `ValueError` crudo; `obtener_info_playlist()` no traduce explícitamente `TimeoutExpired`/`FileNotFoundError` a un error de dominio. `descargar_audio_result()` sólo captura timeout.

**Corrección propuesta:** errores tipados por dependencia (`PlaylistDiscoveryError`, `AudioDownloadError`, `AudioDecodeError`), stderr truncado y sin credenciales, timeout configurable y tests para binario ausente, timeout y JSON parcialmente inválido.

### P2 — Contratos persistidos sin validación de schema/invariantes

`ActorTranscript` será un DTO de `models.py`; la carga/serialización de artefactos y la validación al entrar por `resume`/input pertenecen al control plane.

**Corrección propuesta:** validación explícita en `execution/` o en la frontera de cada submódulo; no crear `validation.py` en el núcleo. Rechazar schema desconocido, scopes no actor-only, rangos imposibles y campos obligatorios ausentes.

### P2 — Reproducibilidad y coste operativo incompletos

- El smoke real tardó aproximadamente 38 minutos para 7 vídeos.
- El ASR lanza ffmpeg y Whisper por turno; el modelo se cachea por instancia, pero no hay benchmark ni política de batching.
- La ruta de cache usa el nombre sanitizado de playlist; dos playlists distintas con el mismo nombre pueden compartir namespace.
- `manifest.json` no registra de forma suficiente configuración/modelos/thresholds por unidad.

**Corrección propuesta:** primero instrumentar y fijar correctness; después comparar por benchmark el diseño actual contra una alternativa batched (`faster-whisper`/WhisperX) sin cambiar todavía el contrato actor-only.

### P2 — Documentación y métricas cuantitativas están desactualizadas

`README.md`, `AGENTS.md` y `speech2text/requisitos.md` aún dicen 67 tests, 221 passed y que `ty check`/`bash check.sh` pasan. La suite focalizada actual reporta 102 tests y el gate global falla. También existe un comentario en `config/config.yaml` sobre un path legacy que ya no es la interfaz expuesta por `main.py`.

**Corrección propuesta:** actualizar números sólo después del gate final; mantener el smoke histórico como histórico y publicar el nuevo quality report como evidencia actual.

---

## 5. Hoja de ruta propuesta

### Fase 0 — Congelar contratos y baseline

**Objetivo:** evitar que la refactorización mezcle cambios de contenido, ejecución y modelos.

**Tareas TDD:**

1. Crear fixtures mínimos de `VideoMeta`, audio válido/inválido, resultado de perfil y `ActorTranscript`.
2. Especificar `reason_code`, estados terminales y definición de vídeo seleccionado.
3. Añadir un fixture de inventario con los 7 IDs del smoke histórico y fechas esperadas/ausentes, sin incorporar audio pesado.
4. Registrar baseline actual: 102 tests focalizados, 90% de coverage del paquete, smoke histórico 7/7 y gate global bloqueado por imports discursivos.

**Archivos:** `tests/fixtures/speech2text/`, `tests/test_speech2text_contracts.py`, `.hermes/plans/...`.

**Gate:** suite focalizada verde y decisión aprobada sobre los estados de unidad.

### Fase 1 — Robustez de audio y metadata

**Objetivo:** que `playlist.py` produzca metadata estable y que `audio.py` concentre descarga, rutas y validación de cache sin crear `cache.py`.

**Archivos:**

- `speech2text/audio_fetcher/models.py`
- `speech2text/audio_fetcher/playlist.py`
- `speech2text/audio_fetcher/audio.py`
- `speech2text/audio_fetcher/service.py`
- `tests/test_audio_fetcher_*.py`

**Secuencia RED → GREEN → REFACTOR:**

1. Test: duración inválida, fecha inválida, línea JSON inválida y metadata sin ID.
2. Test: fecha obtenida por fallback documentado y fecha ausente explícita.
3. Test: cache inexistente, directorio con sufijo `.wav`, archivo vacío y archivo regular válido.
4. Test: timeout, binario ausente, exit code no cero y archivo de salida ausente.
5. Implementar normalización y errores de dominio mínimos.
6. Ejecutar `uv run pytest tests/test_audio_fetcher_*.py -q` y luego Ruff/ty.

**Decisión pendiente:** confirmar la fuente de fecha de YouTube que se considerará canónica y si el cache se versionará por playlist ID/URL además del nombre.

### Fase 2 — Contratos de ejecución fuera de `speech2text`

**Objetivo:** que ningún skip o fallo desaparezca sin añadir `results.py` ni `quality.py` al módulo.

**Archivos:**

- `speech2text/service.py`
- `speech2text/models.py`
- `execution/runner.py`
- `execution/models.py` o `execution/manifest.py`
- `execution/artifacts.py`
- tests de `speech2text` y `execution`

**Diseño aprobado:**

- `speech2text` conserva DTOs de dominio y métodos de servicio.
- `execution` registra estados por vídeo, razones, timings y artefactos.
- La serialización de `ActorTranscript` también pertenece a la capa de artefactos.
- La observabilidad agregada, si se conserva, se implementa en `execution/`, no en `speech2text/quality.py`.

**Secuencia TDD:**

1. RED: referencia ausente → stage `failed`, exit code no cero y manifest con razón.
2. RED: descarga fallida → unidad registrada como fallida y las demás unidades continúan según `fail_fast`.
3. RED: actor no identificado → unidad registrada como skip con razón distinta a descarga.
4. RED: ASR vacío → unidad registrada como skip con métricas disponibles.
5. RED: `resume` con artefacto parcial → sólo se consideran completas las unidades realmente persistidas.
6. GREEN mínimo y migración del runner.

**Gate:** los estados de ejecución y la persistencia funcionan sin introducir nuevos archivos de control dentro de `speech2text`.

### Fase 3 — Perfil y matching con contrato validado

**Objetivo:** eliminar errores estructurales silenciosos en pyannote sin crear `adapter.py` ni `output.py`.

**Archivos:**

- `speaker_timestamps/service.py`
- `speaker_timestamps/perfil_voz.py`
- `speaker_timestamps/matching.py`
- `speaker_timestamps/models.py`
- `tests/test_speaker_timestamps_service.py`
- `tests/test_speaker_timestamps_matching.py`
- `tests/test_speaker_timestamps_models.py`
- `tests/test_perfil_voz.py`

**Secuencia:**

1. RED: `service.py` no puede cargar el pipeline o no encuentra el output requerido.
2. RED: `speaker_embeddings=None`, labels vacíos, cantidad inconsistente y dimensión inconsistente.
3. RED: embedding con NaN/Inf o vector de norma cero.
4. RED: turnos con `t_end <= t_start` o fuera de límites plausibles.
5. Implementar la carga lazy del modelo dentro de `service.py`.
6. Implementar en `matching.py` la consistencia de labels, embeddings, turnos y rangos.
7. Mantener `exclusive_speaker_diarization` y los thresholds actuales hasta disponer de muestra etiquetada.

**Gate:** errores accionables; ningún `IndexError` o `AttributeError` crudo en casos de output inválido.

### Fase 4 — Edición de clips y transcripción

**Objetivo:** separar claramente la edición del audio actor-only de la transcripción Whisper.

**Archivos:**

- `transcribe_speech/actor_clip.py`
- `transcribe_speech/transcription_clip.py`
- `transcribe_speech/service.py`
- `transcribe_speech/models.py`
- `tests/test_actor_clip.py`
- `tests/test_transcription_clip.py`
- `tests/test_transcribe_speech_service.py`

**Responsabilidades:**

- `actor_clip.py`: recibe `AudioVideo` + turnos del actor, aplica padding, elimina los intervalos donde no participa el actor y produce el audio editado.
- `transcription_clip.py`: recibe el audio editado y ejecuta Whisper con el modelo/configuración aprobados.
- `service.py`: orquesta `actor_clip → transcription_clip → ActorTranscript`.
- `models.py`: conserva el mapeo entre offsets del clip editado y timestamps absolutos del audio original.

**Contrato temporal obligatorio:** si `actor_clip.py` concatena intervalos, debe devolver junto al archivo editado una tabla de ventanas como:

```text
clip_start, clip_end  →  source_turn_start, source_turn_end
```

Sin ese mapeo, eliminar audio no actor provocaría timestamps incorrectos en `ActorTranscript`.

**Secuencia TDD:**

1. RED: padding acotado a inicio y duración del audio.
2. RED: eliminación de intervalos no actor y conservación del orden temporal.
3. RED: mapeo de timestamps del clip editado al timeline original.
4. RED: audio editado inexistente, vacío o con ffmpeg fallido.
5. RED: segmento Whisper instantáneo, fuera de rango o sin texto.
6. GREEN: implementación mínima de `actor_clip.py` y `transcription_clip.py`.
7. Benchmark sobre fixtures antes de valorar batching o un backend alternativo.

**Gate:** contrato `ActorTranscript` actor-only y turn-level preservado; segmentos cortos siguen conservándose.

### Fase 5 — Resume, manifest y procedencia

**Objetivo:** hacer reanudable una corrida incompleta a nivel de unidad, no sólo a nivel de directorio.

**Archivos:**

- `execution/artifacts.py`
- `execution/plan.py`
- `execution/runner.py`
- `execution/models.py`
- `config/config.yaml`
- tests de execution y CLI

**Cambios:**

1. `artifact_exists` debe validar contenido mínimo y compatibilidad de schema.
2. Un directorio vacío no debe satisfacer `speech2text`.
3. Guardar manifest de forma incremental o con checkpoint seguro después de cada vídeo.
4. Registrar fingerprint de configuración, modelos, versión de schema, thresholds y estado de cache.
5. Corregir coerciones peligrosas como `bool("false") == True`; validar tipos YAML en vez de convertir strings silenciosamente.
6. Decidir si `overwrite` fuerza todo el stage o sólo unidades no válidas.

**Gate:** matar/reanudar una corrida fake conserva unidades completas, no duplica transcripts y recalcula sólo lo necesario.

### Fase 6 — Cierre de integración y documentación

**Objetivo:** que el módulo pueda declararse listo con evidencia reproducible.

**Archivos:**

- `README.md`
- `AGENTS.md`
- `docs/module-speech2text.md`
- `docs/component_audio_fetcher.md`
- `docs/component_speaker_timestamps.md`
- `docs/component_transcribe_speech.md`
- `docs/configuracion.md`
- `src/tono_politico/execution/requisitos.md`

**Tareas:**

1. Actualizar número real de tests y separar suite focalizada de gate global.
2. Documentar la estructura aprobada y las fronteras de cada submódulo.
3. Documentar la política de fecha: valor, fuente y ausencia explícita.
4. Documentar la carga de pyannote dentro de `speaker_timestamps/service.py` y la consistencia de resultados dentro de `matching.py`.
5. Documentar la política yt-dlp de retries, continue y download archive sólo si queda cableada en el contrato.
6. Marcar el smoke 2026-07-09 como baseline histórico y ejecutar uno nuevo con los artefactos del control plane.
7. Corregir los imports legacy de `discursive_approach` o declarar explícitamente esa ruta fuera del paquete instalable antes del gate final.

**Gate final:**

```bash
uv run ruff check src/ tests/ main.py
uv run ruff format --check src/ tests/ main.py
uv run ty check
uv run pytest tests/ -m "not slow" --tb=short
bash check.sh
uv run python main.py --config config/config.yaml --validate-config
uv run python main.py --config config/config.yaml --dry-run
```

Después, smoke progresivo real y comparación contra el baseline: número de vídeos seleccionados, estados por unidad, fecha, turnos, segmentos, palabras, duración y tiempos por fase.

---

## 6. Definición de terminado

`speech2text` no se considerará cerrado hasta que:

- los estados de ejecución, persistencia y observabilidad estén fuera del núcleo y sean responsabilidad de `execution/`;
- `resume` no confunda directorios vacíos con artefactos completos;
- cache y output se validen como archivos/contratos, no sólo por `exists()`;
- `fecha` sea correcta cuando la fuente la ofrece y la ausencia quede explicada;
- la salida siga siendo `ActorTranscript` actor-only y turn-level;
- no se introduzcan filtros de segmentos cortos sin datos etiquetados;
- `speaker_timestamps/service.py` cargue el modelo y `matching.py` valide resultados;
- `actor_clip.py` preserve el mapeo entre audio editado y timestamps originales;
- el smoke real se ejecute con los artefactos actuales del control plane;
- `ruff`, formato, `ty`, pytest global, `check.sh`, validate-config y dry-run pasen;
- README, AGENTS, configuración y documentos de componentes coincidan con el código.

---

## 7. Decisiones abiertas que requieren aprobación antes de implementar

1. **Perfil ausente:** recomendación vigente: fallo de stage, no corrida `ok` vacía.
2. **Estados de ejecución:** pertenecen a `execution/`; no se crea `results.py` dentro de `speech2text`.
3. **Fecha:** decidir fallback exacto de yt-dlp y política de timezone si se usa timestamp.
4. **Cache:** la responsabilidad queda dentro de `audio_fetcher/audio.py`; decidir si el namespace debe incluir playlist ID/URL además del nombre sanitizado.
5. **Edición de audio:** `actor_clip.py` debe conservar el mapeo entre offsets editados y timestamps originales.
6. **Performance:** mantener Whisper como baseline y comparar alternativas sólo después de fijar correctness.
7. **Thresholds:** mantener 0.5/0.7 hasta tener muestra etiquetada de match/no-match/ambiguo.
8. **Integración global:** decidir si se corrige `discursive_approach` en paralelo o si el gate de `speech2text` tendrá un gate aislado temporal, sin declararlo como gate completo del repositorio.

---

## 8. Fuentes externas consultadas

- [pyannote-audio README — Community-1](https://github.com/pyannote/pyannote-audio/blob/main/README.md): instalación, ffmpeg, token HF, `torch.device`, `ProgressHook` y uso local.
- [Community-1 model card](https://huggingface.co/pyannote/speaker-diarization-community-1): `exclusive_speaker_diarization`, downmix/resampling y ejecución offline.
- [pyannote `DiarizeOutput`](https://github.com/pyannote/pyannote-audio/blob/6328b97b/src/pyannote/audio/pipelines/speaker_diarization.py): `speaker_embeddings` opcionales y orden respecto de `speaker_diarization.labels()`.
- [OpenAI Whisper README](https://github.com/openai/whisper/blob/main/README.md): modelo `turbo`, idioma, ffmpeg, ventanas internas de 30 s y trade-off de memoria/velocidad.
- [yt-dlp README](https://github.com/yt-dlp/yt-dlp/blob/master/README.md): retries, `--continue`, `--download-archive` y opciones de output.
- [WhisperX README](https://github.com/m-bain/whisperX/blob/main/README.md): referencia adyacente para batching, VAD, forced alignment y diarización; se usa sólo como comparación, no como cambio decidido.

---

## 9. Nota de alcance

Esta hoja de ruta no implementa todavía las correcciones. Su objetivo es cerrar el diagnóstico y ordenar la refactorización conforme a la estructura aprobada. No propone reconstruir `discursive_approach` dentro de `speech2text`, ni eliminar la decisión vigente de conservar segmentos cortos. La primera implementación recomendada es estabilizar `audio_fetcher`, después `speaker_timestamps`, y finalmente sustituir el flujo actual de ASR por `actor_clip.py → transcription_clip.py`.
