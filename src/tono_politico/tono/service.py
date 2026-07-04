"""Componente 5: Tono — service orquestador.

Combina dos enfoques complementarios:

1. Embeddings (LFM2.5-Embedding-350M con mean pooling) para las dimensiones
   multi-label: lógica política, sentimiento, estilo discursivo, función
   discursiva e intensidad antagónica. Cada label es independiente — no
   compite con otras en working memory del LLM.

2. LLM (LFM2.5-1.2B-Instruct) para stance (apoyo/rechazo), que requiere
   razonar sobre el tema específico con contexto del actor.

Función pública:
- mapear_scores(scores_dict) → (logica, sentimiento, estilo, funcion, intensidad)

Clase:
- TonoService: orquestador OOP que implementa ComponenteProtocol
"""

from __future__ import annotations

import logging

import numpy as np

from ..filtrado.models import ResultadoFiltrado
from .embeddings import EmbeddorTono, cosine_similarity_batch
from .models import (
    ResultadoEstiloDiscursivo,
    ResultadoFuncionDiscursiva,
    ResultadoLogicaPolitica,
    ResultadoSentimiento,
    ResultadoTono,
    SegmentoConTono,
)
from .taxonomia import prototipos_de
from .zero_shot import ClasificadorLLM

logger = logging.getLogger(__name__)


def mapear_scores(
    scores: dict[str, dict[str, float]],
) -> tuple[
    ResultadoLogicaPolitica,
    ResultadoSentimiento,
    ResultadoEstiloDiscursivo,
    ResultadoFuncionDiscursiva,
    int,
]:
    """Mapea un dict de scores crudos a los DTOs correspondientes.

    Args:
        scores: Dict con 5 keys (dimensiones), cada una con un dict
            de label → score.

    Returns:
        Tupla (logica, sentimiento, estilo, funcion, intensidad).
    """
    logica = ResultadoLogicaPolitica(**scores["logica_politica"])
    sentimiento = ResultadoSentimiento(**scores["sentimiento"])
    estilo = ResultadoEstiloDiscursivo(**scores["estilo_discursivo"])
    funcion = ResultadoFuncionDiscursiva(**scores["funcion_discursiva"])

    # Intensidad: el nivel con mayor score (más conservador en empates)
    intensidad_scores = scores["intensidad"]
    intensidad = max(
        intensidad_scores,
        key=lambda k: intensidad_scores[k],
    )
    intensidad = int(intensidad)

    return logica, sentimiento, estilo, funcion, intensidad


class TonoService:
    """Service del Componente 5: análisis de tono político.

    Recibe ResultadoFiltrado del Componente 4 y produce ResultadoTono con
    6 dimensiones por segmento: stance, intensidad, lógica política,
    sentimiento, estilo discursivo y función discursiva.

    Attributes:
        actor: Nombre del actor político analizado.
        tema: Tema/objetivo a evaluar (ej. "fracking").
    """

    def __init__(
        self,
        actor: str,
        tema: str,
    ) -> None:
        self.actor = actor
        self.tema = tema

        # Modelos lazy-load
        self._embeddor: EmbeddorTono | None = None
        self._clasificador: ClasificadorLLM | None = None
        # Cache de prototipos embebidos (se llena en el primer procesar())
        self._proto_embs: dict[str, dict[str, np.ndarray]] = {}
        self._proto_dims: list[str] = []

    def procesar(self, input_data: ResultadoFiltrado) -> ResultadoTono:
        """Analiza el tono político de cada segmento filtrado.

        Args:
            input_data: ResultadoFiltrado del Componente 4.

        Returns:
            ResultadoTono con todos los segmentos analizados.
        """
        if not input_data.segmentos:
            logger.info("Sin segmentos para analizar tono")
            return ResultadoTono(tema=self.tema, actor=self.actor)

        # Cargar modelos y prototipos
        embeddor = self._get_embeddor()
        clasificador = self._get_clasificador()
        proto_embs = self._get_proto_embs(embeddor)

        segmentos_con_tono: list[SegmentoConTono] = []

        for seg_filtrado in input_data.segmentos:
            texto = seg_filtrado.segmento.texto

            # 1. Embeddings para 5 dimensiones
            text_emb = embeddor.embed(texto)
            scores: dict[str, dict[str, float]] = {}

            for dim in self._proto_dims:
                proto_labels = list(proto_embs[dim].keys())
                proto_matrix = np.array([
                    proto_embs[dim][label] for label in proto_labels
                ])
                sims = cosine_similarity_batch(text_emb, proto_matrix)

                scores[dim] = {
                    label: float(sim)
                    for label, sim in zip(proto_labels, sims, strict=True)
                }

            logica, sent, estilo, func, intensidad = mapear_scores(scores)

            # 2. LLM para stance
            stance = clasificador.clasificar_stance(
                texto=texto,
                actor=self.actor,
                tema=self.tema,
            )

            # 3. Construir SegmentoConTono
            segmentos_con_tono.append(
                SegmentoConTono(
                    segmento=seg_filtrado.segmento,
                    stance=stance,
                    intensidad_antagonica=intensidad,
                    logica_politica=logica,
                    sentimiento=sent,
                    estilo_discursivo=estilo,
                    funcion_discursiva=func,
                )
            )

        logger.info(
            f"Análisis de tono completo: {len(segmentos_con_tono)} segmentos "
            f"analizados para actor={self.actor}, tema={self.tema}"
        )

        return ResultadoTono(
            tema=self.tema,
            actor=self.actor,
            segmentos=segmentos_con_tono,
        )

    def _get_embeddor(self) -> EmbeddorTono:
        if self._embeddor is None:
            self._embeddor = EmbeddorTono()
        return self._embeddor

    def _get_clasificador(self) -> ClasificadorLLM:
        if self._clasificador is None:
            self._clasificador = ClasificadorLLM()
        return self._clasificador

    def _get_proto_embs(
        self, embeddor: EmbeddorTono
    ) -> dict[str, dict[str, np.ndarray]]:
        """Embebe todos los prototipos una sola vez (cache)."""
        if self._proto_embs:
            return self._proto_embs

        from .taxonomia import todas_las_dimensiones

        self._proto_dims = todas_las_dimensiones()

        for dim in self._proto_dims:
            protos = prototipos_de(dim)
            proto_embs: dict[str, np.ndarray] = {}
            for label, texto_proto in protos.items():
                proto_embs[label] = embeddor.embed(texto_proto)
            self._proto_embs[dim] = proto_embs

        logger.info(
            f"Prototipos embebidos: {sum(len(v) for v in self._proto_embs.values())} "
            f"prototipos en {len(self._proto_embs)} dimensiones"
        )

        return self._proto_embs
