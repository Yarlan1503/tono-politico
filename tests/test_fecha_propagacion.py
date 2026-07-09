"""Propagación VideoMeta/AudioVideo.fecha → ActorTranscript → Argumento."""

from __future__ import annotations

from pathlib import Path

from tono_politico.discursive_approach.argument_shape.service import ArgumentShapeService
from tono_politico.speech2text.audio_fetcher.models import AudioVideo
from tono_politico.speech2text.models import TurnoOrador
from tono_politico.speech2text.transcribe_speech.service import TranscribeSpeechService
from tono_politico.speech2text.transcribe_speech.transcripcion_actor import (
    ClipTranscriptSegment,
    transcribir_turnos_actor,
)


class FakeTranscriber:
    def transcribir_clip(self, audio_path, *, t_start, t_end, modelo, idioma):
        return [ClipTranscriptSegment(text="Un argumento corto. Otro más.", t_start=0.0, t_end=1.0)]


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

        # una sola oración por simplicidad
        return _Doc([_Span(0, len(text), text)])


class FakeEmbedder:
    def encode(self, texts):
        return [[1.0, 0.0] for _ in texts]


def test_transcribir_turnos_actor_acepta_fecha(tmp_path: Path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    tx = transcribir_turnos_actor(
        wav,
        [TurnoOrador("vid", "SPEAKER_00", 0.0, 2.0)],
        video_id="vid",
        actor="Actor",
        transcriptor=FakeTranscriber(),
        fecha="20240315",
    )
    assert tx.fecha == "20240315"


def test_transcribe_speech_propaga_fecha_de_audio(tmp_path: Path):
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"x")
    audio = AudioVideo(
        video_id="v1",
        url="https://example.com/v1",
        titulo="T",
        fecha="20240520",
        audio_path=wav,
        duracion=10.0,
    )
    svc = TranscribeSpeechService(actor="A", transcriptor=FakeTranscriber())
    tx = svc.procesar_one(
        audio,
        [TurnoOrador("v1", "SPEAKER_00", 0.0, 2.0)],
    )
    assert tx is not None
    assert tx.fecha == "20240520"


def test_argument_shape_copia_fecha_a_argumentos(tmp_path: Path):
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"x")
    audio = AudioVideo(
        video_id="v1",
        url="u",
        titulo="T",
        fecha="20240601",
        audio_path=wav,
        duracion=5.0,
    )
    tx = TranscribeSpeechService(actor="A", transcriptor=FakeTranscriber()).procesar_one(
        audio,
        [TurnoOrador("v1", "SPEAKER_00", 0.0, 2.0)],
    )
    assert tx is not None
    shape = ArgumentShapeService()
    shape._nlp = FakeNlp()
    shape._embedder = FakeEmbedder()
    shape.min_oraciones = 1
    args = shape.procesar_one(tx)
    assert args
    assert all(a.fecha == "20240601" for a in args)
    assert all(a.video_id == "v1" for a in args)
