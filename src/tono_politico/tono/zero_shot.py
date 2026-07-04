"""Clasificación de stance (apoyo/rechazo) vía LFM2.5-1.2B-Instruct.

El LLM solo clasifica stance — la intensidad antagónica (1-5) se evalúa
con embeddings en clasificacion.py, que demostró mejor gradación.

El prompt incluye actor + tema explícitos para que el modelo sepa qué
evaluar, y un ejemplo few-shot de un caso "a favor pero confrontacional"
para evitar la confusión entre posición sobre el tema y posición sobre
los adversarios.

Funciones puras:
- construir_prompt_stance(actor, tema, texto) → messages
- parsear_stance(raw_json) → ResultadoStance

Clase:
- ClasificadorLLM: lazy-load del modelo y expone clasificar_stance()
"""

from __future__ import annotations

import json
import logging

from .models import ResultadoStance

logger = logging.getLogger(__name__)

LLM_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"

_SYSTEM_PROMPT = """Eres un analista experto en discurso político. Recibes un segmento de discurso, el actor que lo pronuncia y el tema a evaluar.
Tu tarea es determinar si el discurso expresa APOYO o RECHAZO hacia el tema indicado.

IMPORTANTE: Evalúa la posición sobre el TEMA ESPECÍFICO, no sobre los adversarios mencionados. Un político puede estar a favor del fracking y atacar a sus oponentes al mismo tiempo. Si el discurso promueve, defiende o justifica el tema, es APOYO. Si el discurso critica, condena o se opone al tema, es RECHAZO.
EJEMPLO A (a favor del fracking, tono confrontativo):
Actor: Milei, Tema: fracking
Texto: "Vamos a explotar Vaca Muerta y nadie nos lo va a impedir. Los políticos corruptos que se oponen hundieron al país en la miseria."
Respuesta: {"stance": "apoyo", "confianza": 0.9}

EJEMPLO B (en contra del fracking, tono de denuncia):
Actor: AMLO, Tema: fracking
Texto: "No vamos a permitir el fracking porque contamina los mantos acuíferos y destruye la tierra de nuestros campesinos."
Respuesta: {"stance": "rechazo", "confianza": 0.9}

Devuelves EXCLUSIVAMENTE un JSON: {"stance": "apoyo" | "rechazo", "confianza": <float 0-1>}"""


def construir_prompt_stance(
    actor: str,
    tema: str,
    texto: str,
) -> list[dict[str, str]]:
    """Construye los mensajes para el LLM en formato chat.

    Args:
        actor: Nombre del actor político (ej. "AMLO").
        tema: Tema a evaluar (ej. "fracking").
        texto: Texto del segmento a clasificar.

    Returns:
        Lista de mensajes [{"role": "system", ...}, {"role": "user", ...}].
    """
    user_content = (
        f"ACTOR: {actor}\n"
        f"TEMA A EVALUAR: {tema}\n\n"
        f'"{texto}"'
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def parsear_stance(raw: str) -> ResultadoStance:
    """Parsea la respuesta JSON del LLM a un ResultadoStance.

    Args:
        raw: Texto crudo devuelto por el modelo.

    Returns:
        ResultadoStance con stance y confianza.

    Raises:
        ValueError: Si no encuentra JSON válido o el stance no es
            "apoyo" / "rechazo".
    """
    # Buscar el JSON dentro del texto
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError(
            f"No se pudo parsear JSON de la respuesta: {raw[:200]}"
        )

    json_str = raw[start:end]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"No se pudo parsear JSON de la respuesta: {e}"
        ) from e

    stance = data.get("stance", "")
    if stance not in ("apoyo", "rechazo"):
        raise ValueError(
            f"stance inválido: '{stance}'. Debe ser 'apoyo' o 'rechazo'."
        )

    confianza = float(data.get("confianza", 0.5))
    confianza = max(0.0, min(1.0, confianza))

    return ResultadoStance(stance=stance, confianza=confianza)


class ClasificadorLLM:
    """Wrapper de LFM2.5-1.2B-Instruct para clasificación de stance.

    Carga perezosa: el modelo se carga en el primer clasificar_stance().

    Attributes:
        model_name: Nombre del modelo en HuggingFace.
        device: "cpu" o "cuda".
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Carga perezosa del modelo y tokenizer."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Cargando modelo: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
        )
        self._model.eval()

    def clasificar_stance(
        self,
        texto: str,
        actor: str,
        tema: str,
    ) -> ResultadoStance:
        """Clasifica el stance del texto sobre el tema indicado.

        Si el modelo falla (JSON garbage, timeout, OOM, etc.), registra
        un warning y devuelve un ResultadoStance con confianza 0.0
        para no interrumpir el procesamiento del batch.

        Args:
            texto: Texto del segmento.
            actor: Nombre del actor político.
            tema: Tema a evaluar.

        Returns:
            ResultadoStance con stance (apoyo/rechazo) y confianza.
            En caso de error: apoyo con confianza 0.0.
        """
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        messages = construir_prompt_stance(actor, tema, texto)
        chat = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        input_ids = self._tokenizer(chat, return_tensors="pt").input_ids

        try:
            with torch.no_grad():
                output = self._model.generate(  # ty: ignore[invalid-argument-type]
                    input_ids,
                    max_new_tokens=100,
                    temperature=0.1,
                    top_k=50,
                    repetition_penalty=1.05,
                    do_sample=True,
                )

            decoded = self._tokenizer.decode(
                output[0][input_ids.shape[-1]:],
                skip_special_tokens=True,
            )
            if isinstance(decoded, list):
                decoded = decoded[0]
            response = decoded.strip()

            logger.debug(f"LLM response: {response[:200]}")
            return parsear_stance(response)

        except (ValueError, RuntimeError, OSError, IndexError, AttributeError) as e:
            logger.warning(
                f"Error clasificando stance (actor={actor}, tema={tema}): "
                f"{type(e).__name__}: {e}. Devolviendo default."
            )
            return ResultadoStance(stance="apoyo", confianza=0.0)
