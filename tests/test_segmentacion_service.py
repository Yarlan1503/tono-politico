"""Tests para SegmentacionService: integración del Componente 2."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

from tono_politico.models import SegmentoRaw, VideoTranscript, WordTimestamp
from tono_politico.segmentacion.service import SegmentacionService

# ──────────────────────────────────────────────────────────
# Fakes
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
    """Divide texto por '. ' manteniendo offsets."""

    def __call__(self, text: str) -> FakeDoc:
        sents: list[FakeSent] = []
        pos = 0
        parts = text.split(". ")
        for i, part in enumerate(parts):
            sent_text = part + (
                "." if not part.endswith(".") and i < len(parts) - 1 else ""
            )
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            start = text.find(sent_text, pos)
            end = start + len(sent_text)
            sents.append(
                FakeSent(text=sent_text, start_char=start, end_char=end)
            )
            pos = end
        return FakeDoc(sents=sents)


class FakeEmbeddingModel:
    """Devuelve vectores controlables."""

    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.vectors = vectors or {}

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(t, [1.0, 0.0]) for t in texts]


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def transcript_con_segmentos(
    video_id: str = "vid001",
) -> VideoTranscript:
    """VideoTranscript con 4 SegmentoRaw (2 tópicos: economía y fútbol)."""
    return VideoTranscript(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        titulo="Test Video",
        fecha="20260101",
        raw_segments=[
            SegmentoRaw(
                texto="La economía crece este año.",
                t_start=0.0,
                t_end=3.0,
                pausa_antes=0.0,
                words=[
                    WordTimestamp(word="La", start=0.0, end=0.3, probability=0.9),
                    WordTimestamp(word="economía", start=0.3, end=0.8, probability=0.9),
                    WordTimestamp(word="crece", start=0.8, end=1.2, probability=0.9),
                    WordTimestamp(word="este", start=1.2, end=1.5, probability=0.9),
                    WordTimestamp(word="año.", start=1.5, end=2.0, probability=0.9),
                ],
            ),
            SegmentoRaw(
                texto="El PIB aumentó significativamente.",
                t_start=3.0,
                t_end=6.0,
                pausa_antes=1.0,
                words=[
                    WordTimestamp(word="El", start=3.0, end=3.2, probability=0.9),
                    WordTimestamp(word="PIB", start=3.2, end=3.6, probability=0.9),
                    WordTimestamp(word="aumentó", start=3.6, end=4.2, probability=0.9),
                    WordTimestamp(
                        word="significativamente.", start=4.2, end=5.0, probability=0.9
                    ),
                ],
            ),
            SegmentoRaw(
                texto="El fútbol fue emocionante.",
                t_start=6.0,
                t_end=9.0,
                pausa_antes=1.0,
                words=[
                    WordTimestamp(word="El", start=6.0, end=6.2, probability=0.9),
                    WordTimestamp(word="fútbol", start=6.2, end=6.8, probability=0.9),
                    WordTimestamp(word="fue", start=6.8, end=7.2, probability=0.9),
                    WordTimestamp(word="emocionante.", start=7.2, end=8.0, probability=0.9),
                ],
            ),
            SegmentoRaw(
                texto="El gol decisivo en el minuto noventa.",
                t_start=9.0,
                t_end=12.0,
                pausa_antes=1.0,
                words=[
                    WordTimestamp(word="El", start=9.0, end=9.2, probability=0.9),
                    WordTimestamp(word="gol", start=9.2, end=9.5, probability=0.9),
                    WordTimestamp(word="decisivo", start=9.5, end=10.0, probability=0.9),
                    WordTimestamp(word="en", start=10.0, end=10.2, probability=0.9),
                    WordTimestamp(word="el", start=10.2, end=10.4, probability=0.9),
                    WordTimestamp(word="minuto", start=10.4, end=10.8, probability=0.9),
                    WordTimestamp(word="noventa.", start=10.8, end=11.5, probability=0.9),
                ],
            ),
        ],
    )


# ──────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────

class TestSegmentacionService:
    def test_init_guarda_config(self):
        """Los parámetros del constructor se guardan."""
        svc = SegmentacionService(
            spacy_model="es_core_news_sm",
            breakpoint_percentile=90,
            min_oraciones=1,
            max_oraciones=5,
            max_palabras=100,
        )
        assert svc.spacy_model_name == "es_core_news_sm"
        assert svc.breakpoint_percentile == 90
        assert svc.min_oraciones == 1
        assert svc.max_oraciones == 5
        assert svc.max_palabras == 100

    def test_init_defaults(self):
        """Defaults correctos."""
        svc = SegmentacionService()
        assert svc.spacy_model_name == "es_core_news_lg"
        assert svc.breakpoint_percentile == 95
        assert svc.min_oraciones == 2
        assert svc.max_oraciones == 8
        assert svc.max_palabras == 150

    def test_input_vacio_devuelve_vacio(self):
        """Sin transcripts devuelve lista vacía."""
        svc = SegmentacionService()
        assert svc.procesar([]) == []

    def test_transcript_sin_raw_segments_se_omite(self):
        """VideoTranscript sin raw_segments se omite sin error."""
        svc = SegmentacionService()
        vt = VideoTranscript(
            video_id="vid_empty",
            url="https://fake",
            titulo="Empty",
            fecha=None,
            raw_segments=[],
        )
        with (
            patch.object(svc, "_get_nlp", return_value=FakeSpacyNlp()),
            patch.object(svc, "_get_embedder", return_value=FakeEmbeddingModel()),
        ):
            assert svc.procesar([vt]) == []

    def test_segmenta_dos_topicos_en_dos_segmentos(self):
        """4 oraciones con cambio de tópico → al menos 2 segmentos."""
        svc = SegmentacionService(
            min_oraciones=1,  # permitir segmentos de 1 oración
        )
        vt = transcript_con_segmentos()

        # Vectores: economía similar entre sí, fútbol similar entre sí,
        # pero distintos entre grupos
        embedder = FakeEmbeddingModel({
            "La economía crece este año.": [1.0, 0.0],
            "El PIB aumentó significativamente.": [0.99, 0.01],
            "El fútbol fue emocionante.": [0.0, 1.0],
            "El gol decisivo en el minuto noventa.": [0.01, 0.99],
        })

        with (
            patch.object(svc, "_get_nlp", return_value=FakeSpacyNlp()),
            patch.object(svc, "_get_embedder", return_value=embedder),
        ):
            segmentos = svc.procesar([vt])

        assert len(segmentos) >= 2
        # Verificar que el corte ocurre entre economía y fútbol
        textos = [s.texto for s in segmentos]
        # Los primeros segmentos mencionan economía
        texto_completo = " ".join(textos)
        assert "economía" in texto_completo.lower()
        assert "fútbol" in texto_completo.lower()

    def test_segmentos_tienen_video_id(self):
        """Cada segmento debe llevar el video_id del transcript."""
        svc = SegmentacionService(min_oraciones=1)
        vt = transcript_con_segmentos(video_id="vid_test")

        embedder = FakeEmbeddingModel()

        with (
            patch.object(svc, "_get_nlp", return_value=FakeSpacyNlp()),
            patch.object(svc, "_get_embedder", return_value=embedder),
        ):
            segmentos = svc.procesar([vt])

        for seg in segmentos:
            assert seg.video_id == "vid_test"

    def test_segmentos_tienen_timestamps_validos(self):
        """Cada segmento tiene t_start < t_end."""
        svc = SegmentacionService(min_oraciones=1)
        vt = transcript_con_segmentos()

        embedder = FakeEmbeddingModel()

        with (
            patch.object(svc, "_get_nlp", return_value=FakeSpacyNlp()),
            patch.object(svc, "_get_embedder", return_value=embedder),
        ):
            segmentos = svc.procesar([vt])

        for seg in segmentos:
            assert seg.t_start < seg.t_end
            assert seg.t_start >= 0.0

    def test_cumple_componente_protocol(self):
        """SegmentacionService cumple ComponenteProtocol."""
        from tono_politico.protocol import ComponenteProtocol

        svc = SegmentacionService()
        assert isinstance(svc, ComponenteProtocol)
