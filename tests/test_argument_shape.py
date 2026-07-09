"""Tests TDD para discursive_approach.argument_shape."""

from __future__ import annotations

from tono_politico.diarizacion.models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
)
from tono_politico.discursive_approach.argument_shape.agrupacion import agrupar_argumentos
from tono_politico.discursive_approach.argument_shape.breakpoints import (
    Breakpoint,
    detectar_breakpoints,
)
from tono_politico.discursive_approach.argument_shape.models import Argumento, Oracion
from tono_politico.discursive_approach.argument_shape.sentencias import (
    extraer_oraciones_de_transcript,
)
from tono_politico.discursive_approach.argument_shape.service import ArgumentShapeService


class FakeNlp:
    """spaCy mínimo: parte por '. ' si hay más de una oración."""

    def pipe(self, texts, batch_size: int = 50):
        return [self(t) for t in texts]

    def __call__(self, text: str):
        parts = [
            p.strip() for p in text.replace("!", ".").replace("?", ".").split(".") if p.strip()
        ]
        if not parts:
            parts = [text]

        # rebuild with periods for char offsets
        class _Span:
            def __init__(self, start: int, end: int, t: str):
                self.start_char = start
                self.end_char = end
                self.text = t

        class _Doc:
            def __init__(self, sents):
                self.sents = sents

        sents = []
        cursor = 0
        full = text
        for i, p in enumerate(parts):
            # find p in full from cursor
            idx = full.find(p, cursor)
            if idx < 0:
                idx = cursor
            start = idx
            end = idx + len(p)
            # include trailing period if present
            if end < len(full) and full[end] == ".":
                end += 1
            sents.append(_Span(start, end, full[start:end].strip() or p))
            cursor = end
        return _Doc(sents)


class FakeEmbedder:
    def __init__(self, vectors: dict[str, list[float]] | None = None):
        self.vectors = vectors or {}

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(t, [1.0, 0.0]) for t in texts]


def _transcript(
    video_id: str = "vid1",
    fecha: str | None = "20240115",
    segments: list[ActorTranscriptSegment] | None = None,
) -> ActorTranscript:
    if segments is None:
        segments = [
            ActorTranscriptSegment(
                text="La economía crece. El empleo mejora.",
                t_start=0.0,
                t_end=10.0,
                speaker="ACTOR",
                source_turn_start=0.0,
                source_turn_end=10.0,
                word_count=6,
            )
        ]
    return ActorTranscript(
        schema_version="actor_transcript.v1",
        video_id=video_id,
        actor="AMLO",
        scope="actor_only",
        asr=AsrMetadata(provider="whisper", model="large-v3-turbo", language="es"),
        segments=segments,
        fecha=fecha,
    )


class TestModels:
    def test_argumento_lleva_fecha_y_video_id(self):
        arg = Argumento(
            texto="hola",
            t_start=0.0,
            t_end=1.0,
            oraciones=[],
            word_count=1,
            video_id="v",
            fecha="20240101",
        )
        assert arg.fecha == "20240101"
        assert arg.video_id == "v"


class TestSentencias:
    def test_extrae_oraciones_sin_word_level(self):
        tr = _transcript(
            segments=[
                ActorTranscriptSegment(
                    text="Primera idea. Segunda idea.",
                    t_start=0.0,
                    t_end=10.0,
                    speaker="A",
                    source_turn_start=0.0,
                    source_turn_end=10.0,
                    word_count=4,
                )
            ]
        )
        oraciones = extraer_oraciones_de_transcript(tr, FakeNlp())
        assert len(oraciones) == 2
        assert oraciones[0].t_start == 0.0
        assert oraciones[-1].t_end == 10.0
        # reparto proporcional: primera mitad ~0-5
        assert oraciones[0].t_end <= oraciones[1].t_start + 1e-6 or oraciones[0].t_end <= 5.5

    def test_turnos_vacios(self):
        tr = _transcript(segments=[])
        assert extraer_oraciones_de_transcript(tr, FakeNlp()) == []


class TestBreakpoints:
    def test_menos_de_tres_oraciones_sin_bp(self):
        oras = [
            Oracion("A.", 0.0, 1.0),
            Oracion("B.", 1.0, 2.0),
        ]
        assert detectar_breakpoints(oras, FakeEmbedder()) == []

    def test_detecta_corte_semantico(self):
        oras = [
            Oracion("economia uno", 0.0, 1.0),
            Oracion("economia dos", 1.0, 2.0),
            Oracion("seguridad tema", 2.0, 3.0),
            Oracion("seguridad dos", 3.0, 4.0),
        ]
        model = FakeEmbedder(
            {
                "economia uno": [1.0, 0.0],
                "economia dos": [0.99, 0.01],
                "seguridad tema": [0.0, 1.0],
                "seguridad dos": [0.01, 0.99],
            }
        )
        bps = detectar_breakpoints(oras, model, breakpoint_percentile=75)
        assert any(b.indice == 2 for b in bps)


class TestAgrupacion:
    def test_word_count_desde_texto_sin_words(self):
        oras = [
            Oracion("uno dos tres", 0.0, 1.0),
            Oracion("cuatro cinco", 1.0, 2.0),
        ]
        args = agrupar_argumentos(oras, [], min_oraciones=1, video_id="v", fecha="20240101")
        assert len(args) == 1
        assert args[0].word_count == 5
        assert args[0].fecha == "20240101"

    def test_breakpoint_divide_bloques(self):
        oras = [
            Oracion("a b", 0.0, 1.0),
            Oracion("c d", 1.0, 2.0),
            Oracion("e f", 2.0, 3.0),
            Oracion("g h", 3.0, 4.0),
        ]
        bps = [Breakpoint(indice=2, intensidad=0.9)]
        args = agrupar_argumentos(oras, bps, min_oraciones=1, video_id="v", fecha=None)
        assert len(args) == 2
        assert args[0].oraciones[0].texto == "a b"

    def test_no_mezcla_videos_es_por_video_id_param(self):
        oras = [Oracion("solo una", 0.0, 1.0), Oracion("dos", 1.0, 2.0)]
        args = agrupar_argumentos(oras, [], min_oraciones=1, video_id="vidX", fecha="20240202")
        assert all(a.video_id == "vidX" for a in args)


class TestArgumentShapeService:
    def test_procesar_one_propaga_fecha_y_video(self):
        svc = ArgumentShapeService()
        svc._nlp = FakeNlp()  # type: ignore[attr-defined]
        svc._embedder = FakeEmbedder(  # type: ignore[attr-defined]
            {
                "La economía crece.": [1.0, 0.0],
                "El empleo mejora.": [0.95, 0.05],
            }
        )
        # lower min so short text still groups
        svc.min_oraciones = 1
        tr = _transcript()
        args = svc.procesar_one(tr)
        assert len(args) >= 1
        assert all(a.video_id == "vid1" for a in args)
        assert all(a.fecha == "20240115" for a in args)

    def test_procesar_corpus_no_cruza_videos(self):
        svc = ArgumentShapeService()
        svc._nlp = FakeNlp()  # type: ignore[attr-defined]
        svc._embedder = FakeEmbedder()  # type: ignore[attr-defined]
        svc.min_oraciones = 1
        tr1 = _transcript(video_id="A", fecha="20240101")
        tr2 = _transcript(video_id="B", fecha="20240102")
        args = svc.procesar_corpus([tr1, tr2])
        ids = {a.video_id for a in args}
        assert ids == {"A", "B"}
        assert all(a.fecha is not None for a in args)

    def test_procesar_one_vacio(self):
        svc = ArgumentShapeService()
        svc._nlp = FakeNlp()  # type: ignore[attr-defined]
        svc._embedder = FakeEmbedder()  # type: ignore[attr-defined]
        tr = _transcript(segments=[])
        assert svc.procesar_one(tr) == []
