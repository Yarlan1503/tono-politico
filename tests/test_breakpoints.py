"""Tests para breakpoints.py: detección de breakpoints semánticos.

Estándar LangChain: distancia coseno + percentil 95.
Sin señal acústica — Whisper ya segmenta acústicamente.
"""

from __future__ import annotations

from tono_politico.segmentacion.breakpoints import detectar_breakpoints
from tono_politico.segmentacion.models import Oracion

# ──────────────────────────────────────────────────────────
# Fake embedding model
# ──────────────────────────────────────────────────────────

class FakeEmbeddingModel:
    """Modelo que devuelve vectores controlables por test."""

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.vectors = vectors or {}
        self.encode_calls: list[str] = []

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.encode_calls.extend(texts)
        return [self.vectors.get(t, [1.0]) for t in texts]


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def oracion(texto: str, t_start: float, t_end: float) -> Oracion:
    """Crea una Oracion mínima sin words para tests de breakpoints."""
    return Oracion(texto=texto, t_start=t_start, t_end=t_end, words=[])


# ──────────────────────────────────────────────────────────
# Tests: casos edge
# ──────────────────────────────────────────────────────────

class TestCasosEdge:
    def test_input_vacio_sin_breakpoints(self):
        """Lista vacía no produce breakpoints."""
        assert detectar_breakpoints([], FakeEmbeddingModel()) == []

    def test_una_sola_oracion_sin_breakpoints(self):
        """Una sola oración no produce breakpoints."""
        oraciones = [oracion("Hola mundo.", 0.0, 2.0)]
        assert detectar_breakpoints(oraciones, FakeEmbeddingModel()) == []

    def test_dos_oraciones_sin_breakpoints(self):
        """Con 2 oraciones no hay percentil útil."""
        oraciones = [
            oracion("Idea A.", 0.0, 2.0),
            oracion("Idea B.", 2.5, 4.0),
        ]
        model = FakeEmbeddingModel({
            "Idea A.": [1.0, 0.0],
            "Idea B.": [0.0, 1.0],
        })
        assert detectar_breakpoints(oraciones, model) == []


# ──────────────────────────────────────────────────────────
# Tests: semántico (distancia coseno + percentil)
# ──────────────────────────────────────────────────────────

class TestBreakpointSemantico:
    def test_oraciones_similares_no_generan_bp(self):
        """3 oraciones similares → distancias bajas → no hay breakpoint."""
        oraciones = [
            oracion("La economía crece.", 0.0, 2.0),
            oracion("El PIB aumenta.", 2.0, 4.0),
            oracion("Las exportaciones suben.", 4.0, 6.0),
        ]
        model = FakeEmbeddingModel({
            "La economía crece.": [1.0, 0.0],
            "El PIB aumenta.": [0.99, 0.01],
            "Las exportaciones suben.": [0.98, 0.02],
        })
        bps = detectar_breakpoints(
            oraciones, model, breakpoint_percentile=95,
        )
        assert bps == []

    def test_cambio_topico_genera_bp_semantico(self):
        """Un cambio drástico de tópico genera breakpoint."""
        oraciones = [
            oracion("La economía crece.", 0.0, 2.0),
            oracion("El PIB aumenta.", 2.0, 4.0),
            oracion("El fútbol fue intenso.", 4.0, 6.0),
            oracion("El gol fue espectacular.", 6.0, 8.0),
        ]
        model = FakeEmbeddingModel({
            "La economía crece.": [1.0, 0.0],
            "El PIB aumenta.": [0.99, 0.01],
            "El fútbol fue intenso.": [0.0, 1.0],
            "El gol fue espectacular.": [0.01, 0.99],
        })
        bps = detectar_breakpoints(
            oraciones, model, breakpoint_percentile=95,
        )
        assert len(bps) == 1
        assert bps[0].indice == 2  # cortar antes de "El fútbol..."

    def test_intensidad_es_distancia_coseno(self):
        """La intensidad de un bp es la distancia coseno."""
        oraciones = [
            oracion("Tema A1.", 0.0, 1.0),
            oracion("Tema A2.", 1.0, 2.0),
            oracion("Tema B1.", 2.0, 3.0),
        ]
        model = FakeEmbeddingModel({
            "Tema A1.": [1.0, 0.0],
            "Tema A2.": [1.0, 0.0],
            "Tema B1.": [0.0, 1.0],
        })
        bps = detectar_breakpoints(
            oraciones, model, breakpoint_percentile=95,
        )
        assert len(bps) == 1
        assert abs(bps[0].intensidad - 1.0) < 0.01

    def test_varios_cambios_generan_varios_bps(self):
        """Múltiples cambios de tópico generan múltiples breakpoints."""
        oraciones = [
            oracion("Economía 1.", 0.0, 1.0),
            oracion("Economía 2.", 1.0, 2.0),
            oracion("Deporte 1.", 2.0, 3.0),
            oracion("Deporte 2.", 3.0, 4.0),
            oracion("Salud 1.", 4.0, 5.0),
            oracion("Salud 2.", 5.0, 6.0),
        ]
        model = FakeEmbeddingModel({
            "Economía 1.": [1.0, 0.0, 0.0],
            "Economía 2.": [0.99, 0.01, 0.0],
            "Deporte 1.": [0.0, 1.0, 0.0],
            "Deporte 2.": [0.0, 0.99, 0.01],
            "Salud 1.": [0.0, 0.0, 1.0],
            "Salud 2.": [0.01, 0.0, 0.99],
        })
        bps = detectar_breakpoints(
            oraciones, model, breakpoint_percentile=95,
        )
        indices = [bp.indice for bp in bps]
        assert 2 in indices  # Economía → Deporte
        assert 4 in indices  # Deporte → Salud


# ──────────────────────────────────────────────────────────
# Tests: configurabilidad
# ──────────────────────────────────────────────────────────

class TestConfigurabilidad:
    def test_percentil_mas_bajo_genera_mas_breakpoints(self):
        """Un percentil más bajo baja el umbral → más breakpoints."""
        oraciones = [
            oracion(f"Frase {i}.", float(i), float(i + 1))
            for i in range(6)
        ]
        model = FakeEmbeddingModel({
            "Frase 0.": [1.0, 0.0],
            "Frase 1.": [0.9, 0.1],
            "Frase 2.": [0.7, 0.3],
            "Frase 3.": [0.5, 0.5],
            "Frase 4.": [0.3, 0.7],
            "Frase 5.": [0.1, 0.9],
        })
        bps_alto = detectar_breakpoints(
            oraciones, model, breakpoint_percentile=95,
        )
        bps_bajo = detectar_breakpoints(
            oraciones, model, breakpoint_percentile=50,
        )
        assert len(bps_bajo) >= len(bps_alto)
