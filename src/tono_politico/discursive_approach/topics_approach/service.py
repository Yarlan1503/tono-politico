"""TopicsApproachService: ResultadoTemas → ResultadoEnfoques (base = Tono)."""

from __future__ import annotations

import logging
from typing import Protocol

from ...tono.service import TonoService
from ..topics_cluster.models import ResultadoTemas, TopicoInfo
from .adapter import (
    argumentos_a_resultado_filtrado,
    resultado_tono_a_perfiles,
)
from .enfoques import descubrir_enfoques_tema
from .models import PerfilTonoArgumento, ResultadoEnfoques, ResultadoEnfoquesTema

logger = logging.getLogger(__name__)


class TonoAnalyzer(Protocol):
    """Analiza textos de argumentos respecto a un tema → perfiles de tono."""

    def analizar(
        self,
        textos: list[str],
        actor: str,
        tema: str,
    ) -> list[PerfilTonoArgumento]: ...


class TonoServiceAnalyzer:
    """Adapter que reutiliza TonoService (un tema por llamada)."""

    def __init__(self, argumentos_por_texto: dict[str, object] | None = None) -> None:
        # reserved for cache; real call uses full Argumento list via service
        pass

    def analizar_argumentos(
        self,
        argumentos: list,
        actor: str,
        topico: TopicoInfo,
    ) -> list[PerfilTonoArgumento]:
        tono = TonoService(actor=actor, tema=topico.nombre)
        filtrado = argumentos_a_resultado_filtrado(argumentos, topico)
        # filtrado.topico must be temas.TopicoInfo-like; adapter proxy works for fields
        resultado = tono.procesar(filtrado)  # type: ignore[arg-type]
        return resultado_tono_a_perfiles(resultado)


class TopicsApproachService:
    """Orquesta Tono + firmas de tono por cada tema no-outlier."""

    def __init__(
        self,
        actor: str,
        tono_analyzer: TonoAnalyzer | None = None,
        tono_service_factory: object | None = None,
    ) -> None:
        """
        Args:
            actor: Actor del pipeline (stance LLM).
            tono_analyzer: Fake o analyzer custom (tests). Si None, usa TonoService real.
            tono_service_factory: reservado.
        """
        self.actor = actor
        self._analyzer = tono_analyzer
        self._real_adapter = TonoServiceAnalyzer()

    def procesar(self, resultado: ResultadoTemas) -> ResultadoEnfoques:
        """Todos los temas no-outlier → enfoques por firmas de tono."""
        topicos_validos = [t for t in resultado.topicos if t.id != -1]
        por_tema: list[ResultadoEnfoquesTema] = []
        total_enfoques = 0

        for topico in topicos_validos:
            at_list = [a for a in resultado.argumentos if a.topico_id == topico.id]
            if not at_list:
                por_tema.append(ResultadoEnfoquesTema(topico=topico))
                continue

            perfiles = self._analizar(at_list, topico)
            if len(perfiles) != len(at_list):
                raise RuntimeError(
                    f"Tono devolvió {len(perfiles)} perfiles para {len(at_list)} argumentos"
                )
            items = list(zip(at_list, perfiles, strict=True))
            tema_res = descubrir_enfoques_tema(topico, items)
            total_enfoques += len(tema_res.enfoques)
            por_tema.append(tema_res)

        logger.info(
            "topics_approach: %s temas, %s enfoques totales",
            len(por_tema),
            total_enfoques,
        )
        return ResultadoEnfoques(
            por_tema=por_tema,
            num_temas=len(por_tema),
            num_enfoques_total=total_enfoques,
        )

    def _analizar(self, at_list: list, topico: TopicoInfo) -> list[PerfilTonoArgumento]:
        argumentos = [a.argumento for a in at_list]
        if self._analyzer is not None:
            textos = [a.texto for a in argumentos]
            return self._analyzer.analizar(textos, self.actor, topico.nombre)
        return self._real_adapter.analizar_argumentos(argumentos, self.actor, topico)
