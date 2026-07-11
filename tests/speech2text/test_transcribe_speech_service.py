"""Tests de TranscribeSpeechService."""

from __future__ import annotations

from pathlib import Path

from tono_politico.speech2text.audio_fetcher.models import AudioVideo, PlaylistInfo
from tono_politico.speech2text.models import (
    ActorTranscript,
    TurnoOrador,
)
from tono_politico.speech2text.transcribe_speech.models import ClipTranscriptSegment
from tono_politico.speech2text.transcribe_speech.service import TranscribeSpeechService


class FakeTranscriber:
    def __init__(self, text: str = "hola mundo") -> None:
        self.text = text
        self.calls: list[tuple[float, float]] = []

    def transcribir_clip(
        self,
        audio_path: Path,
        *,
        t_start: float,
        t_end: float,
        modelo: str,
        idioma: str,
    ) -> list[ClipTranscriptSegment]:
        self.calls.append((t_start, t_end))
        if not self.text:
            return []
        return [ClipTranscriptSegment(text=self.text, t_start=0.0, t_end=t_end - t_start)]


def _audio(tmp_path: Path) -> AudioVideo:
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"x")
    return AudioVideo(
        video_id="v1",
        url="https://example.com/v1",
        titulo="T",
        fecha="20260101",
        audio_path=wav,
        duracion=30.0,
        fecha_fuente="upload_date",
        playlist=PlaylistInfo(
            nombre="Play-PoliTest",
            nombre_cache="Play-PoliTest",
            playlist_id="PLE9Zk7g9R__M",
            url="https://www.youtube.com/playlist?list=PLE9Zk7g9R__M",
        ),
    )


def _turno(video_id: str = "v1") -> TurnoOrador:
    return TurnoOrador(
        video_id=video_id,
        speaker_id="SPEAKER_00",
        t_start=1.0,
        t_end=3.0,
    )


class TestTranscribeSpeechService:
    def test_sin_turnos_none(self, tmp_path: Path) -> None:
        svc = TranscribeSpeechService(transcriptor=FakeTranscriber())
        assert svc.procesar_one(_audio(tmp_path), []) is None

    def test_con_texto(self, tmp_path: Path) -> None:
        fake = FakeTranscriber("discurso político")
        svc = TranscribeSpeechService(actor="Actor", transcriptor=fake)
        tx = svc.procesar_one(_audio(tmp_path), [_turno()])
        assert isinstance(tx, ActorTranscript)
        assert tx.video_id == "v1"
        assert tx.actor == "Actor"
        assert len(tx.segments) == 1
        assert "discurso" in tx.segments[0].text
        assert fake.calls == [(1.0, 3.0)]
        assert tx.fecha == "20260101"

    def test_con_texto_conserva_source_metadata(self, tmp_path: Path) -> None:
        svc = TranscribeSpeechService(actor="Actor", transcriptor=FakeTranscriber())

        tx = svc.procesar_one(_audio(tmp_path), [_turno()])

        assert tx is not None
        assert tx.source is not None
        assert tx.source.playlist_name == "Play-PoliTest"
        assert tx.source.video_title == "T"
        assert tx.source.upload_date == "20260101"
        assert tx.source.date_source == "upload_date"

    def test_sin_texto_none(self, tmp_path: Path) -> None:
        svc = TranscribeSpeechService(transcriptor=FakeTranscriber(""))
        assert svc.procesar_one(_audio(tmp_path), [_turno()]) is None
