"""Tests para agrupacion.py: agrupación de oraciones en segmentos."""

from __future__ import annotations

from tono_politico.models import WordTimestamp
from tono_politico.segmentacion.agrupacion import agrupar_segmentos
from tono_politico.segmentacion.breakpoints import Breakpoint
from tono_politico.segmentacion.models import Oracion

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def w(word: str, start: float, end: float) -> WordTimestamp:
    """Crea una WordTimestamp mínima."""
    return WordTimestamp(word=word, start=start, end=end, probability=0.9)


def oracion(texto: str, t_start: float, t_end: float, n_words: int = 3) -> Oracion:
    """Crea una Oracion con words simuladas."""
    words = [
        w(f"palabra{i}", t_start + i, t_start + i + 0.5)
        for i in range(n_words)
    ]
    return Oracion(texto=texto, t_start=t_start, t_end=t_end, words=words)


# ──────────────────────────────────────────────────────────
# Tests: casos edge
# ──────────────────────────────────────────────────────────

class TestCasosEdge:
    def test_input_vacio_devuelve_vacio(self):
        """Sin oraciones, no hay segmentos."""
        assert agrupar_segmentos([], []) == []

    def test_una_oracion_sin_breakpoints_un_segmento(self):
        """Una sola oración produce un segmento."""
        oraciones = [oracion("Hola mundo.", 0.0, 2.0)]
        segmentos = agrupar_segmentos(oraciones, [])
        assert len(segmentos) == 1
        assert segmentos[0].texto == "Hola mundo."

    def test_sin_breakpoints_todo_junto(self):
        """Sin breakpoints, todas las oraciones van en un segmento."""
        oraciones = [
            oracion("Uno.", 0.0, 1.0),
            oracion("Dos.", 1.0, 2.0),
            oracion("Tres.", 2.0, 3.0),
        ]
        segmentos = agrupar_segmentos(oraciones, [])
        assert len(segmentos) == 1
        assert len(segmentos[0].oraciones) == 3


# ──────────────────────────────────────────────────────────
# Tests: agrupación por breakpoints
# ──────────────────────────────────────────────────────────

class TestAgrupacionPorBreakpoints:
    def test_un_breakpoint_divide_en_dos(self):
        """Un breakpoint en índice 2 divide en 2 segmentos."""
        oraciones = [
            oracion("A1.", 0.0, 1.0),
            oracion("A2.", 1.0, 2.0),
            oracion("B1.", 2.0, 3.0),
            oracion("B2.", 3.0, 4.0),
        ]
        bps = [Breakpoint(indice=2, intensidad=0.9)]
        segmentos = agrupar_segmentos(oraciones, bps)

        assert len(segmentos) == 2
        assert len(segmentos[0].oraciones) == 2
        assert len(segmentos[1].oraciones) == 2

    def test_dos_breakpoints_dividen_en_tres(self):
        """Dos breakpoints dividen en 3 segmentos."""
        oraciones = [
            oracion("A1.", 0.0, 1.0),
            oracion("B1.", 1.0, 2.0),
            oracion("C1.", 2.0, 3.0),
        ]
        bps = [
            Breakpoint(indice=1, intensidad=0.9),
            Breakpoint(indice=2, intensidad=0.8),
        ]
        segmentos = agrupar_segmentos(oraciones, bps, min_oraciones=1)

        assert len(segmentos) == 3
        assert segmentos[0].oraciones[0].texto == "A1."
        assert segmentos[1].oraciones[0].texto == "B1."
        assert segmentos[2].oraciones[0].texto == "C1."


# ──────────────────────────────────────────────────────────
# Tests: timestamps y texto
# ──────────────────────────────────────────────────────────

class TestTimestampsYTexto:
    def test_t_start_y_t_end_correctos(self):
        """t_start = primer oración, t_end = última del segmento."""
        oraciones = [
            oracion("Primera.", 10.0, 15.0),
            oracion("Segunda.", 16.0, 20.0),
        ]
        segmentos = agrupar_segmentos(oraciones, [])

        assert segmentos[0].t_start == 10.0
        assert segmentos[0].t_end == 20.0

    def test_texto_concatena_oraciones(self):
        """El texto concatena las oraciones con espacio."""
        oraciones = [
            oracion("Hola.", 0.0, 1.0),
            oracion("Mundo.", 1.0, 2.0),
        ]
        segmentos = agrupar_segmentos(oraciones, [])

        assert segmentos[0].texto == "Hola. Mundo."

    def test_word_count_suma_palabras(self):
        """word_count es la suma de words de todas las oraciones."""
        oraciones = [
            oracion("Uno.", 0.0, 1.0, n_words=2),
            oracion("Dos.", 1.0, 2.0, n_words=4),
        ]
        segmentos = agrupar_segmentos(oraciones, [])

        assert segmentos[0].word_count == 6


# ──────────────────────────────────────────────────────────
# Tests: guardrails
# ──────────────────────────────────────────────────────────

class TestGuardrails:
    def test_max_oraciones_subdivide(self):
        """Si un bloque excede max_oraciones, se subdivide."""
        oraciones = [
            oracion(f"S{i}.", float(i), float(i + 1)) for i in range(6)
        ]
        segmentos = agrupar_segmentos(oraciones, [], max_oraciones=3)

        for seg in segmentos:
            assert len(seg.oraciones) <= 3

    def test_max_palabras_subdivide(self):
        """Si un bloque excede max_palabras, se subdivide."""
        oraciones = [
            oracion("Larga.", float(i), float(i + 1), n_words=5)
            for i in range(4)
        ]
        segmentos = agrupar_segmentos(oraciones, [], max_palabras=10)

        for seg in segmentos:
            assert seg.word_count <= 10

    def test_min_oraciones_fusiona_con_anterior(self):
        """Si un bloque tiene menos de min_oraciones, se fusiona."""
        oraciones = [
            oracion("A1.", 0.0, 1.0),
            oracion("A2.", 1.0, 2.0),
            oracion("B1.", 2.0, 3.0),
            oracion("C1.", 3.0, 4.0),
            oracion("C2.", 4.0, 5.0),
        ]
        bps = [
            Breakpoint(indice=2, intensidad=0.8),
            Breakpoint(indice=3, intensidad=0.8),
        ]
        segmentos = agrupar_segmentos(oraciones, bps, min_oraciones=2)

        for seg in segmentos:
            assert len(seg.oraciones) >= 2


# ──────────────────────────────────────────────────────────
# Tests: video_id
# ──────────────────────────────────────────────────────────

class TestVideoId:
    def test_video_id_se_propaga(self):
        """video_id se asigna a cada segmento."""
        oraciones = [oracion("Test.", 0.0, 1.0)]
        segmentos = agrupar_segmentos(oraciones, [], video_id="vid123")
        assert segmentos[0].video_id == "vid123"
