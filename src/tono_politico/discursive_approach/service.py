"""Orquestador: shape → cluster → approaches."""

from __future__ import annotations

import logging
from typing import Any

from ..speech2text.diarization.models import ActorTranscript
from .argument_shape.models import Argumento
from .argument_shape.service import ArgumentShapeService
from .topics_approach.models import ResultadoEnfoques
from .topics_approach.service import TopicsApproachService
from .topics_cluster.models import ResultadoTemas
from .topics_cluster.service import TopicsClusterService

logger = logging.getLogger(__name__)


class DiscursiveApproachService:
    """Umbrella: ActorTranscript[] → ResultadoEnfoques."""

    def __init__(
        self,
        actor: str,
        shape_service: Any | None = None,
        cluster_service: Any | None = None,
        approach_service: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.actor = actor
        self.shape_service = shape_service or ArgumentShapeService()
        self.cluster_service = cluster_service or TopicsClusterService()
        self.approach_service = approach_service or TopicsApproachService(actor=actor)

    def shape_one(self, transcript: ActorTranscript) -> list[Argumento]:
        return self.shape_service.procesar_one(transcript)

    def shape_corpus(self, transcripts: list[ActorTranscript]) -> list[Argumento]:
        return self.shape_service.procesar_corpus(transcripts)

    def cluster(self, argumentos: list[Argumento]) -> ResultadoTemas:
        return self.cluster_service.procesar(argumentos)

    def approaches(self, resultado: ResultadoTemas) -> ResultadoEnfoques:
        return self.approach_service.procesar(resultado)

    def procesar(self, transcripts: list[ActorTranscript]) -> ResultadoEnfoques:
        """shape_corpus → cluster → approaches."""
        argumentos = self.shape_corpus(transcripts)
        temas = self.cluster(argumentos)
        enfoques = self.approaches(temas)
        logger.info(
            "discursive_approach completo: %s args → %s temas → %s enfoques",
            len(argumentos),
            temas.num_topicos,
            enfoques.num_enfoques_total,
        )
        return enfoques
