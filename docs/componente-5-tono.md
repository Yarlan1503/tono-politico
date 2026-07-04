# Componente 5: Tono

## Propósito

Analiza el tono político de cada segmento filtrado a lo largo de 6 dimensiones
ortogonales. Usa una **arquitectura híbrida** que combina embeddings (para
dimensiones multi-label independientes) y LLM (para stance, que requiere
razonamiento contextual).

## Pipeline interno

```text
ResultadoFiltrado → TonoService → ResultadoTono
                      ├── EmbeddorTono (LFM2.5-Embedding-350M)
                      │   ├── Lógica política (6 labels)
                      │   ├── Sentimiento (5 emociones)
                      │   ├── Estilo discursivo (6 estilos)
                      │   ├── Función discursiva (3 labels)
                      │   └── Intensidad antagónica (5 niveles)
                      └── ClasificadorLLM (LFM2.5-1.2B-Instruct)
                          └── Stance (apoyo / rechazo)
```

## Arquitectura híbrida

### Por qué no usar un solo enfoque

**NLI zero-shot** (mDeBERTa, XLM-R): probado primero. Colapsa dimensiones
— todos los políticos salen "populista alto, estatista bajo". No distingue
dirección del stance (Trump a favor del fracking sale "antagónico").

**LLM solo** (LFM2.5-1.2B-Instruct): acierta stance cuando se le da actor +
tema explícitos, pero no puede mantener 10 dimensiones simultáneas en working
memory. Confunde "está a favor del fracking pero ataca a sus oponentes".

**Embeddings solos**: discriminan dimensiones paralelas sin colapsar, pero no
razonan sobre el tema específico (Milei sale "en contra" del fracking por
keyword matching con "pobreza" y "opresión").

### Solución híbrida

Cada enfoque hace lo que hace mejor:

- **Embeddings**: cada label se evalúa independientemente contra el texto del
  segmento mediante similitud coseno. No hay competencia entre labels, no hay
  working memory limitada. Un segmento puede ser simultáneamente populista Y
  corporativista sin que uno excluya al otro.

- **LLM**: razona stance con contexto completo del actor y tema. El prompt
  incluye actor, tema a evaluar y dos ejemplos few-shot balanceados (uno pro,
  uno anti) para evitar sesgo direccional.

## Dimensiones

### Stance (vía LLM)

| Label | Descripción |
|---|---|
| `apoyo` | El discurso promueve, defiende o justifica el tema |
| `rechazo` | El discurso critica, condena o se opone al tema |

El LLM recibe actor + tema + texto + few-shot. Si falla (JSON garbage, timeout,
OOM), registra warning y devuelve `apoyo` con `confianza=0.0`. Ningún segmento
mata el batch.

### Intensidad antagónica (vía embeddings, 5 niveles)

| Nivel | Descripción |
|---|---|
| 1 | Conciliador, colaborativo, sin confrontación |
| 2 | Firme pero respetuoso |
| 3 | Combativo, señala responsables |
| 4 | Confrontacional, ataca directamente a adversarios |
| 5 | Beligerante, enemistad existencial, divide en buenos y malos |

### Lógica política (vía embeddings, 6 labels multi-label)

| Label | Descripción |
|---|---|
| `nacionalista` | Defiende la soberanía frente a intereses extranjeros |
| `globalista` | Favorece apertura, libre comercio, integración global |
| `populista` | Alinea con valores del pueblo sobre expertos |
| `tecnocrata` | Confía en expertise técnica sobre voluntad popular |
| `corporativista` | Apoya empresas privadas, mercado libre |
| `estatista` | Defiende al Estado como rector de la economía |

### Sentimiento (vía embeddings, 5 emociones políticas)

| Label | Descripción |
|---|---|
| `esperanza` | Optimismo, promesa de un mañana mejor |
| `angustia` | Preocupación, temor, descripción de peligro |
| `indignacion` | Rabia moral ante una injusticia |
| `orgullo` | Exaltación de la grandeza nacional o de identidad |
| `empatia` | Compasión hacia quienes sufren |

### Estilo discursivo (vía embeddings, 6 estilos)

| Label | Descripción |
|---|---|
| `directo` | Lenguaje cotidiano, frases cortas, sin adornos |
| `academico` | Datos, citas, estructura argumentativa formal |
| `confrontativo` | Provocador, busca el choque frontal |
| `conciliador` | Diplomático, inclusivo, busca consensos |
| `catastrofista` | Alarmista, todo es crisis inminente |
| `testimonial` | Anécdotas, historias de gente común |

### Función discursiva (vía embeddings, 3 labels)

| Label | Descripción |
|---|---|
| `critica` | Ataca, denuncia o señala responsables |
| `propuesta` | Ofrece soluciones, plantea alternativas |
| `narrativa_personal` | Construye la imagen del político |

## Módulos

| Archivo | Responsabilidad |
|---|---|
| `service.py` | `TonoService` — orquestador híbrido (lazy-load ambos modelos) |
| `embeddings.py` | `EmbeddorTono`, `mean_pooling`, `cosine_similarity`, `cosine_similarity_batch` |
| `zero_shot.py` | `ClasificadorLLM`, `construir_prompt_stance`, `parsear_stance` |
| `taxonomia.py` | 25 prototipos textuales en 5 dimensiones |
| `models.py` | DTOs: `EtiquetaScore`, `Resultado*`, `SegmentoConTono`, `ResultadoTono` |

## Hallazgo crítico: sentence-transformers vs mean pooling manual

El wrapper `SentenceTransformer` produce **embeddings degenerados** con
`LFM2.5-Embedding-350M`: todas las similitudes coseno entre textos diferentes
dan exactamente 1.0. Esto no afecta a los Componentes 2 y 3 (que operan con
distancias relativas: percentil 95 en breakpoints, UMAP en BERTopic), pero
sí afecta al Componente 5 donde las similitudes absolutas importan.

La solución: cargar el modelo con `AutoModel` directo y aplicar mean pooling
manual sobre `last_hidden_state` con `attention_mask`.

## Calibración de prototipos

Los 25 prototipos de `taxonomia.py` son descripciones ricas (3-4 oraciones,
~200-300 chars) calibradas empíricamente con citas reales de 5 políticos:

| Actor | Stance esperado | Lógica esperada | Sentimiento esperado |
|---|---|---|---|
| AMLO | rechazo | populista + nacionalista | angustia / indignación |
| Sheinbaum | evaluando | tecnócrata + populista | orgullo |
| Calderón | apoyo | globalista + corporativista | esperanza |
| Trump | apoyo | populista + nacionalista | esperanza / indignación |
| Milei | apoyo | populista + corporativista | indignación |

## Tests

- **56 tests fast** (sin modelos): mean pooling, cosine similarity, prompt
  construction, JSON parsing, manejo de errores, mapeo de scores.
- **5 tests slow** (modelos reales): validan que los embeddings no son
  degenerados, que el LLM acierta AMLO→rechazo y Trump→apoyo, y que el
  pipeline completo produce un `ResultadoTono` válido.
