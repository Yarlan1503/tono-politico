# Componente 6: Salida

## Propósito

Genera el informe final del pipeline: agrega los resultados del Componente 5
en un perfil interpretable del actor político, añade metadata de provenance
y serializa todo a JSON + Markdown.

## Pipeline interno

```text
ResultadoTono → SalidaService → InformeTono
                  ├── agregacion.py    → PerfilActor
                  ├── serializacion.py → JSON + Markdown
                  └── models.py        → Provenance
```

## Salidas

### `informe.json`

```json
{
  "perfil": {
    "actor": "AMLO",
    "tema": "fracking",
    "n_segmentos": 5,
    "stance_dominante": "rechazo",
    "intensidad_promedio": 3.8,
    "logica_dominante": "populista",
    "sentimiento_dominante": "indignacion",
    "estilo_dominante": "directo",
    "funcion_dominante": "critica"
  },
  "provenance": {
    "pipeline": "tono-politico v0.1.0",
    "modelos": [
      "LiquidAI/LFM2.5-Embedding-350M",
      "LiquidAI/LFM2.5-1.2B-Instruct"
    ],
    "fecha": "2026-07-04T12:00:00+00:00",
    "advertencia_confianza": "Los scores de similitud coseno son medidas relativas..."
  },
  "segmentos": [
    {
      "texto": "No vamos a permitir el fracking...",
      "stance": {"stance": "rechazo", "confianza": 0.85},
      "intensidad_antagonica": 4,
      "logica_politica": {"populista": 0.6123, ...},
      ...
    }
  ]
}
```

### `informe.md`

```markdown
# AMLO — fracking

## Perfil agregado

| Dimensión | Resultado |
|---|---|
| Segmentos analizados | 5 |
| Stance | rechazo |
| Intensidad antagónica | 3.8 / 5 |
| Lógica política | populista |
| Sentimiento | indignacion |
| Estilo discursivo | directo |
| Función discursiva | critica |

## Provenance
- **Pipeline:** tono-politico v0.1.0
- **Modelos:** LiquidAI/LFM2.5-Embedding-350M, LiquidAI/LFM2.5-1.2B-Instruct
- **Fecha:** 2026-07-04T12:00:00+00:00

## Advertencia de confianza
> Los scores de similitud coseno son medidas relativas de proximidad
> semántica, no probabilidades calibradas.
```

## Módulos

| Archivo | Responsabilidad |
|---|---|
| `service.py` | `SalidaService` — orquestador, implementa `ComponenteProtocol` |
| `agregacion.py` | `stance_dominante`, `intensidad_promedio`, `dimension_dominante`, `generar_perfil` |
| `serializacion.py` | `perfil_a_dict`, `segmento_a_dict`, `generar_json`, `generar_markdown` |
| `models.py` | `Provenance`, `PerfilActor`, `InformeTono` |

## Diseño

- **100% puro:** sin modelos de ML, sin I/O externo (excepto escritura opcional).
- **Provenance obligatorio:** todo informe declara modelos usados y advertencia de confianza.
- **Agregación flexible:** perfil agregado + datos por segmento (no pierde granularidad).
- **JSON serializable:** toda la salida se puede convertir a JSON sin objetos custom.

## Uso

```python
from tono_politico.salida import SalidaService

# Sin output_path: solo devuelve InformeTono en memoria
svc = SalidaService()
informe = svc.procesar(resultado_tono)

# Con output_path = directorio: genera informe.json + informe.md
svc = SalidaService(output_path="output/")
informe = svc.procesar(resultado_tono)

# Con output_path = archivo específico: solo ese formato
svc = SalidaService(output_path="output/mi_informe.json")
svc.procesar(resultado_tono)
```

## Tests

35 tests, todos fast (sin modelos). Cubren:

- Agregación: stance mayoritario, intensidad promedio, dimensión dominante, perfil completo.
- Serialización: dict serializable, JSON válido, Markdown con perfil y provenance.
- Service: salida en memoria, escritura a disco (JSON, Markdown, ambos).
