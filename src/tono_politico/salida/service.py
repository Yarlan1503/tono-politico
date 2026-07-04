"""Componente 6: Salida — service orquestador.

Recibe ResultadoTono del Componente 5, genera el perfil agregado,
construye el provenance, serializa a JSON + Markdown y opcionalmente
escribe a disco.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from ..tono.embeddings import EMBEDDING_MODEL
from ..tono.models import ResultadoTono
from ..tono.zero_shot import LLM_MODEL
from .agregacion import generar_perfil
from .models import InformeTono, Provenance
from .serializacion import generar_json, generar_markdown

logger = logging.getLogger(__name__)


class SalidaService:
    """Service del Componente 6: salida del pipeline.

    Genera el informe final con perfil agregado, segmentos detallados
    y metadata de provenance. Opcionalmente escribe JSON + Markdown a disco.

    Attributes:
        output_path: Ruta donde escribir los archivos. Si es un archivo
            (.json o .md), escribe solo ese formato. Si es un directorio,
            escribe informe.json + informe.md. Si es None, no escribe.
    """

    def __init__(
        self,
        output_path: str | Path | None = None,
    ) -> None:
        self.output_path = Path(output_path) if output_path else None

    def procesar(self, input_data: ResultadoTono) -> InformeTono:
        """Genera el informe final del pipeline.

        Args:
            input_data: ResultadoTono del Componente 5.

        Returns:
            InformeTono con perfil, segmentos y provenance.
        """
        resultado = input_data

        # 1. Generar perfil agregado
        perfil = generar_perfil(resultado)

        # 2. Construir provenance
        provenance = self._build_provenance()

        # 3. Construir informe
        informe = InformeTono(
            perfil=perfil,
            segmentos=resultado.segmentos,
            provenance=provenance,
        )

        # 4. Escriir a disco si hay output_path
        if self.output_path is not None:
            self._escribir(resultado, perfil, provenance)

        logger.info(
            f"Informe generado: {perfil.actor} / {perfil.tema} "
            f"({perfil.n_segmentos} segmentos)"
        )

        return informe

    def _build_provenance(self) -> Provenance:
        """Construye el Provenance con metadata del pipeline."""
        return Provenance(
            pipeline="tono-politico v0.1.0",
            modelos=[EMBEDDING_MODEL, LLM_MODEL],
            fecha=datetime.now(UTC).isoformat(),
        )

    def _escribir(
        self,
        resultado: ResultadoTono,
        perfil,
        provenance: Provenance,
    ) -> None:
        """Escribe JSON y/o Markdown a disco según output_path."""
        assert self.output_path is not None

        if self.output_path.suffix == ".json":
            json_str = generar_json(resultado, perfil, provenance)
            self.output_path.write_text(json_str, encoding="utf-8")
            logger.info(f"JSON escrito: {self.output_path}")

        elif self.output_path.suffix == ".md":
            md_str = generar_markdown(resultado, perfil, provenance)
            self.output_path.write_text(md_str, encoding="utf-8")
            logger.info(f"Markdown escrito: {self.output_path}")

        else:
            # Es un directorio: escribir ambos
            self.output_path.mkdir(parents=True, exist_ok=True)
            json_path = self.output_path / "informe.json"
            md_path = self.output_path / "informe.md"

            json_path.write_text(
                generar_json(resultado, perfil, provenance), encoding="utf-8"
            )
            md_path.write_text(
                generar_markdown(resultado, perfil, provenance), encoding="utf-8"
            )
            logger.info(f"Archivos escritos: {json_path}, {md_path}")
