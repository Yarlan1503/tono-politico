# Tono Político

Herramienta de análisis NLP para determinar el tono de un actor político configurable respecto a un tema, usando transcripciones de videos de YouTube como fuente.

## Tres lecturas de tono

1. **Sentimiento** — positivo / negativo / neutral
2. **Stance** — favor / contra / neutro respecto al tema
3. **Tono retórico** — populista / técnico / institucional (eje primario) + confrontativo / conciliador / emocional / nacionalista / victimizante (eje secundario multi-label)

## Arquitectura

```
1. Ingesta        →  YouTube playlist → Whisper → transcripciones con timestamps + pausas
2. Segmentación   →  Segmentos crudos → segmentos semánticos coherentes (pausas + embeddings)
3. Temas          →  BERTopic → descubrimiento automático de temas predominantes
4. Filtrado       →  Tema seleccionado → subset de segmentos relevantes
5. Tono           →  3 modelos zero-shot sobre los segmentos filtrados
6. Salida         →  Agregación → JSON
```

## Configuración

```bash
# Crear entorno virtual con uv
uv venv
source .venv/bin/activate  # o usar automáticamente con uv run

# Instalar dependencias
uv pip install -e .
```

## Uso (próximamente)

```bash
# Descubrir temas predominantes
python -m tono_politico --playlist "URL" --actor "Sheinbaum" --descubrir-temas

# Analizar tono sobre un tema específico
python -m tono_politico --playlist "URL" --actor "Sheinbaum" --tema "seguridad pública"
```
