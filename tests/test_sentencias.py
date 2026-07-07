"""Tests para sentencias.py: extracción de oraciones con spaCy."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tono_politico.models import SegmentoRaw, WordTimestamp
from tono_politico.segmentacion.sentencias import extraer_oraciones

# ──────────────────────────────────────────────────────────
# Fake spaCy (simula doc.sents con char offsets)
# ──────────────────────────────────────────────────────────


@dataclass
class FakeSent:
    text: str
    start_char: int
    end_char: int


@dataclass
class FakeDoc:
    sents: list[FakeSent]


class FakeSpacyNlp:
    """NLP que divide texto por punto seguido de espacio (suficiente para tests)."""

    def _process(self, text: str) -> FakeDoc:
        sents: list[FakeSent] = []
        pos = 0
        parts = text.split(". ")
        for i, part in enumerate(parts):
            sent_text = part + ("." if not part.endswith(".") and i < len(parts) - 1 else "")
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            start = text.find(sent_text, pos)
            end = start + len(sent_text)
            sents.append(FakeSent(text=sent_text, start_char=start, end_char=end))
            pos = end
        return FakeDoc(sents=sents)

    def __call__(self, text: str) -> FakeDoc:
        return self._process(text)

    def pipe(self, texts, batch_size=50):
        return [self._process(t) for t in texts]


@pytest.fixture
def fake_nlp() -> FakeSpacyNlp:
    return FakeSpacyNlp()


# ──────────────────────────────────────────────────────────
# Fixtures: segmentos crudos
# ──────────────────────────────────────────────────────────


@pytest.fixture
def segmentos_crudos() -> list[SegmentoRaw]:
    """Un SegmentoRaw con 2 oraciones y 7 words."""
    return [
        SegmentoRaw(
            texto="La economía va bien. Tenemos crecimiento.",
            t_start=10.0,
            t_end=18.0,
            pausa_antes=0.0,
            words=[
                WordTimestamp(word="La", start=10.0, end=10.2, probability=0.95),
                WordTimestamp(word="economía", start=10.2, end=10.8, probability=0.92),
                WordTimestamp(word="va", start=10.8, end=11.0, probability=0.88),
                WordTimestamp(word="bien.", start=11.0, end=11.5, probability=0.91),
                WordTimestamp(word="Tenemos", start=12.0, end=12.4, probability=0.94),
                WordTimestamp(word="crecimiento", start=12.4, end=13.1, probability=0.89),
                WordTimestamp(word=".", start=13.1, end=13.2, probability=0.87),
            ],
        )
    ]


@pytest.fixture
def dos_segmentos_crudos() -> list[SegmentoRaw]:
    """Dos SegmentoRaw para verificar que se procesan ambos."""
    return [
        SegmentoRaw(
            texto="Hola mundo.",
            t_start=0.0,
            t_end=2.0,
            pausa_antes=0.0,
            words=[
                WordTimestamp(word="Hola", start=0.0, end=1.0, probability=0.9),
                WordTimestamp(word="mundo.", start=1.0, end=2.0, probability=0.9),
            ],
        ),
        SegmentoRaw(
            texto="Adiós. Buenas noches.",
            t_start=5.0,
            t_end=9.0,
            pausa_antes=3.0,
            words=[
                WordTimestamp(word="Adiós.", start=5.0, end=6.0, probability=0.9),
                WordTimestamp(word="Buenas", start=6.5, end=7.0, probability=0.9),
                WordTimestamp(word="noches.", start=7.0, end=8.0, probability=0.9),
            ],
        ),
    ]


# ──────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────


class TestExtraerOraciones:
    def test_input_vacio_devuelve_vacio(self, fake_nlp):
        """Input vacío devuelve lista vacía."""
        assert extraer_oraciones([], fake_nlp) == []

    def test_segmento_sin_words_se_omite(self, fake_nlp):
        """Un SegmentoRaw sin words se omite."""
        seg = SegmentoRaw(texto="Hola.", t_start=0.0, t_end=1.0, pausa_antes=0.0)
        assert extraer_oraciones([seg], fake_nlp) == []

    def test_divide_segmento_en_dos_oraciones(self, segmentos_crudos, fake_nlp):
        """Un segmento con 2 oraciones debe producir 2 Oracion."""
        oraciones = extraer_oraciones(segmentos_crudos, fake_nlp)

        assert len(oraciones) == 2
        assert oraciones[0].texto == "La economía va bien."
        assert oraciones[1].texto == "Tenemos crecimiento."

    def test_cada_oracion_tiene_timestamps_correctos(self, segmentos_crudos, fake_nlp):
        """t_start/t_end de cada oración vienen de sus words."""
        oraciones = extraer_oraciones(segmentos_crudos, fake_nlp)

        # Oración 1: "La economía va bien." → words[0..3]
        assert oraciones[0].t_start == 10.0
        assert oraciones[0].t_end == 11.5

        # Oración 2: "Tenemos crecimiento." → words[4..6]
        assert oraciones[1].t_start == 12.0
        assert oraciones[1].t_end == 13.2

    def test_cada_oracion_tiene_sus_words(self, segmentos_crudos, fake_nlp):
        """Las words se asignan a la oración correcta."""
        oraciones = extraer_oraciones(segmentos_crudos, fake_nlp)

        assert len(oraciones[0].words) == 4  # La, economía, va, bien.
        assert oraciones[0].words[0].word == "La"

        assert len(oraciones[1].words) == 3  # Tenemos, crecimiento, .
        assert oraciones[1].words[0].word == "Tenemos"

    def test_procesa_multiples_segmentos(self, dos_segmentos_crudos, fake_nlp):
        """Múltiples SegmentoRaw se procesan secuencialmente."""
        oraciones = extraer_oraciones(dos_segmentos_crudos, fake_nlp)

        # Segmento 1: "Hola mundo." → 1 oración
        # Segmento 2: "Adiós. Buenas noches." → 2 oraciones
        assert len(oraciones) == 3
        assert oraciones[0].texto == "Hola mundo."
        assert oraciones[1].texto == "Adiós."
        assert oraciones[2].texto == "Buenas noches."

    def test_oraciones_en_orden_cronologico(self, dos_segmentos_crudos, fake_nlp):
        """Las oraciones deben estar en orden temporal."""
        oraciones = extraer_oraciones(dos_segmentos_crudos, fake_nlp)

        for i in range(len(oraciones) - 1):
            assert oraciones[i].t_start <= oraciones[i + 1].t_start
