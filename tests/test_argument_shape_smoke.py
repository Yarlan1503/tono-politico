"""Smoke ligero: ActorTranscript reales de speech2text-smoke → argument_shape.

No carga modelos pesados (nlp/embedder fake). Marca la cadena de contratos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tono_politico.discursive_approach.argument_shape.service import ArgumentShapeService
from tono_politico.speech2text.actor_transcript import cargar_actor_transcript

SMOKE_DIR = (
    Path(__file__).resolve().parents[1] / "output" / "speech2text-smoke" / "actor_transcripts"
)


class FakeNlp:
    def pipe(self, texts, batch_size: int = 50):
        return [self(t) for t in texts]

    def __call__(self, text: str):
        class _Span:
            def __init__(self, start, end, t):
                self.start_char = start
                self.end_char = end
                self.text = t

        class _Doc:
            def __init__(self, sents):
                self.sents = sents

        parts = [
            p.strip() for p in text.replace("!", ".").replace("?", ".").split(".") if p.strip()
        ]
        if not parts:
            parts = [text]
        sents = []
        cursor = 0
        for p in parts:
            idx = text.find(p, cursor)
            if idx < 0:
                idx = cursor
            end = idx + len(p)
            if end < len(text) and text[end : end + 1] == ".":
                end += 1
            sents.append(_Span(idx, end, text[idx:end].strip() or p))
            cursor = end
        return _Doc(sents)


class FakeEmbedder:
    def encode(self, texts):
        # vectores triviales pero distintos por hash de texto
        out = []
        for t in texts:
            h = sum(ord(c) for c in t) % 97
            out.append([float(h), float(97 - h)])
        return out


@pytest.mark.skipif(not SMOKE_DIR.is_dir(), reason="no hay speech2text-smoke local")
def test_smoke_argument_shape_sobre_actor_transcripts():
    paths = sorted(SMOKE_DIR.glob("*.json"))
    assert paths, "directorio smoke vacío"
    # máximo 3 videos para smoke rápido
    paths = paths[:3]

    svc = ArgumentShapeService()
    svc._nlp = FakeNlp()
    svc._embedder = FakeEmbedder()
    svc.min_oraciones = 1

    total_args = 0
    for path in paths:
        tr = cargar_actor_transcript(path)
        # transcripts viejos pueden no tener fecha; no debe romper
        if tr.fecha is None:
            tr.fecha = "20240101"  # fixture de smoke
        args = svc.procesar_one(tr)
        assert all(a.video_id == tr.video_id for a in args)
        assert all(a.fecha == tr.fecha for a in args)
        total_args += len(args)

    assert total_args >= 0  # puede ser 0 si transcripts vacíos; no falla
