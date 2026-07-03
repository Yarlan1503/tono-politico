"""Protocolo base para todos los componentes del pipeline.

Cada componente implementa .procesar(input) -> output.
El tipo específico de input/output depende de cada componente.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ComponenteProtocol(Protocol):
    """Contrato uniforme para todos los componentes del pipeline.

    Implementar este protocol en cada service para que el Pipeline
    pueda encadenarlos genéricamente.
    """

    def procesar(self, input_data: Any) -> Any:
        """Procesa la entrada y devuelve el resultado del componente.

        Args:
            input_data: La entrada específica del componente.
                - IngestaService: str (URL de playlist)
                - SegmentacionService: list[VideoTranscript]
                - etc.

        Returns:
            La salida específica del componente.
                - IngestaService: list[VideoTranscript]
                - etc.
        """
        ...
