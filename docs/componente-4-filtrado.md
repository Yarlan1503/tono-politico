# Componente 4: Filtrado

> **Estado:** ✅ MVP implementado · **Tests:** 5

## Propósito

Toma el `ResultadoTemas` producido por el Componente 3 y selecciona los segmentos relevantes para un tópico elegido. Este componente no descubre tópicos ni interpreta texto libre: opera sobre el contrato ya estructurado de Temas.

Este componente responde: **“¿qué subset de segmentos debe pasar al análisis de tono para el tópico seleccionado?”**

## Arquitectura

```text
ResultadoTemas (Componente 3)
    │
    ▼ FiltradoService.procesar()
    │
    ▼ filtrar_por_topico()
        ├─ match por topico_id
        ├─ umbral min_relevancia
        └─ política explícita para outliers (-1)
            │
            ▼ ResultadoFiltrado
                ├─ criterio: CriterioFiltrado
                ├─ topico: TopicoInfo | None
                ├─ segmentos: list[SegmentoFiltrado]
                ├─ total_segmentos_entrada
                └─ total_segmentos_filtrados
```

## API

### `FiltradoService`

```python
from tono_politico.filtrado import FiltradoService

svc = FiltradoService(
    topico_id=0,
    min_relevancia=0.35,
    incluir_outliers=False,
)

resultado_filtrado = svc.procesar(resultado_temas)
```

### Función pura

```python
from tono_politico.filtrado import CriterioFiltrado, filtrar_por_topico

criterio = CriterioFiltrado(
    topico_id=0,
    min_relevancia=0.35,
    incluir_outliers=False,
)

resultado_filtrado = filtrar_por_topico(resultado_temas, criterio)
```

### Configuración

| Parámetro | Tipo | Default | Descripción |
|---|---:|---:|---|
| `topico_id` | `int` | — | Tópico elegido desde `ResultadoTemas.topicos`. No tiene default global porque depende de cada análisis. |
| `min_relevancia` | `float` | `0.35` | Probabilidad mínima de asignación del segmento al tópico. |
| `incluir_outliers` | `bool` | `False` | Permite analizar explícitamente el tópico `-1` si se solicita. |

Los defaults documentados viven en `config/config.yaml`; el YAML sigue siendo referencia canónica, no loader automático.

## Módulos internos

### `service.py` — Orquestador OOP

`FiltradoService` encapsula el tópico elegido y el umbral de relevancia. Su método `.procesar()` construye un `CriterioFiltrado` y delega en la función pura `filtrar_por_topico()`.

### `filtro.py` — Lógica pura

**`filtrar_por_topico(resultado_temas, criterio) -> ResultadoFiltrado`**

1. Busca metadata del tópico elegido en `resultado_temas.topicos`.
2. Recorre `resultado_temas.segmentos`.
3. Conserva solo segmentos cuyo `topico_id` coincide.
4. Excluye outliers `-1` por default.
5. Conserva outliers solo si `incluir_outliers=True`.
6. Aplica `probabilidad >= min_relevancia`.
7. Devuelve conteos y segmentos filtrados con provenance.

### `models.py` — DTOs locales

#### `CriterioFiltrado`

| Campo | Tipo | Descripción |
|---|---|---|
| `topico_id` | `int` | ID del tópico elegido. |
| `min_relevancia` | `float` | Umbral mínimo de probabilidad. |
| `incluir_outliers` | `bool` | Política explícita para tópico `-1`. |

#### `SegmentoFiltrado`

| Campo | Tipo | Descripción |
|---|---|---|
| `segmento` | `Segmento` | Segmento original del Componente 2. |
| `topico_id` | `int` | Tópico usado para seleccionarlo. |
| `relevancia` | `float` | Probabilidad de asignación usada como score de filtro. |

#### `ResultadoFiltrado`

| Campo | Tipo | Descripción |
|---|---|---|
| `criterio` | `CriterioFiltrado` | Criterio aplicado. |
| `topico` | `TopicoInfo | None` | Metadata del tópico si existe en `ResultadoTemas`. |
| `segmentos` | `list[SegmentoFiltrado]` | Segmentos seleccionados. |
| `total_segmentos_entrada` | `int` | Tamaño del input recibido. |
| `total_segmentos_filtrados` | `int` | Tamaño del subset seleccionado. |

## Decisiones de diseño

### Filtrado determinista por `topico_id`

Se eligió filtrar por `topico_id` + `min_relevancia` para mantener separado el contrato de Componentes:

- Temas descubre y enumera tópicos.
- Usuario/CLI elige un tópico inspeccionando `ResultadoTemas.topicos`.
- Filtrado selecciona los segmentos relevantes para ese tópico.

No usamos `BERTopic.find_topics()` ni búsqueda textual todavía porque `ResultadoTemas` no expone el modelo BERTopic entrenado ni embeddings de tópicos. Meter eso ahora acoplaría Filtrado a internals de Temas.

### Outliers excluidos por default

El tópico `-1` representa ruido/outliers en BERTopic. Por eso `FiltradoService(topico_id=-1)` no devuelve segmentos salvo que `incluir_outliers=True`. Esto evita mandar ruido al análisis de tono accidentalmente.

### Conteos y provenance

`ResultadoFiltrado` conserva el criterio aplicado y los conteos de entrada/salida. Esto será útil para el Componente 6: reportar cuántos segmentos del corpus sustentan una lectura de tono.

## Tests

`tests/test_filtrado.py` cubre:

- filtrado por `topico_id` y `min_relevancia`;
- exclusión de outliers por default;
- inclusión explícita de outliers;
- configuración de `FiltradoService`;
- cumplimiento de `ComponenteProtocol` vía `.procesar()`.

## Pendientes futuros

- Selección por keywords o query textual, si se decide ampliar el contrato de Temas para exponer embeddings/modelo entrenado.
- CLI/pipeline end-to-end para listar tópicos, elegir `topico_id` y ejecutar Filtrado.
