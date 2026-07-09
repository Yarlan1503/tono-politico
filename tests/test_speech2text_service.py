"""Tests de SpeechToTextService (orquestador)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from tono_politico.diarizacion.models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
    TurnoOrador,
)
from tono_politico.models import PlaylistInfo
from tono_politico.speech2text.audio_fetcher.models import AudioVideo, VideoMeta
from tono_politico.speech2text.service import SpeechToTextService


def _meta(video_id: str) -> VideoMeta:
    return VideoMeta(
        video_id=video_id,
        url=f"https://example.com/{video_id}",
        titulo=f"T {video_id}",
        fecha=None,
        duracion=10.0,
    )


def _audio(tmp_path: Path, video_id: str) -> AudioVideo:
    wav = tmp_path / f"{video_id}.wav"
    wav.write_bytes(b"x")
    return AudioVideo(
        video_id=video_id,
        url=f"https://example.com/{video_id}",
        titulo="T",
        fecha=None,
        audio_path=wav,
        duracion=10.0,
    )


def _tx(video_id: str = "v1") -> ActorTranscript:
    return ActorTranscript(
        schema_version="actor_transcript.v1",
        video_id=video_id,
        actor="Actor",
        scope="actor_only",
        asr=AsrMetadata(provider="whisper", model="large-v3-turbo", language="es"),
        segments=[
            ActorTranscriptSegment(
                text="hola",
                t_start=0.0,
                t_end=1.0,
                speaker="SPEAKER_00",
                source_turn_start=0.0,
                source_turn_end=1.0,
                word_count=1,
            )
        ],
    )


class TestSpeechToTextService:
    def test_discover_delega(self, tmp_path: Path) -> None:
        fetcher = MagicMock()
        playlist = PlaylistInfo(nombre="P", url="u", videos=[])
        metas = [_meta("a")]
        fetcher.discover.return_value = (playlist, metas)
        svc = SpeechToTextService(
            data_dir=tmp_path,
            audio_fetcher=fetcher,
            speaker_timestamps=MagicMock(),
            transcribe_speech=MagicMock(),
        )
        p, m = svc.discover("url")
        assert p.nombre == "P"
        assert m[0].video_id == "a"

    def test_procesar_one_pipeline(self, tmp_path: Path) -> None:
        fetcher = MagicMock()
        speakers = MagicMock()
        asr = MagicMock()

        audio = _audio(tmp_path, "v1")
        fetcher.fetch_one.return_value = audio
        speakers.procesar_one.return_value = [
            TurnoOrador(
                video_id="v1",
                speaker_id="SPEAKER_00",
                t_start=0.0,
                t_end=1.0,
            )
        ]
        asr.procesar_one.return_value = _tx("v1")

        svc = SpeechToTextService(
            data_dir=tmp_path,
            audio_fetcher=fetcher,
            speaker_timestamps=speakers,
            transcribe_speech=asr,
        )
        out = svc.procesar_one(_meta("v1"), "P")
        assert out is not None
        assert out.video_id == "v1"
        fetcher.fetch_one.assert_called_once()
        speakers.procesar_one.assert_called_once_with(audio)
        asr.procesar_one.assert_called_once()

    def test_procesar_one_skip_sin_audio(self, tmp_path: Path) -> None:
        fetcher = MagicMock()
        fetcher.fetch_one.return_value = None
        svc = SpeechToTextService(
            data_dir=tmp_path,
            audio_fetcher=fetcher,
            speaker_timestamps=MagicMock(),
            transcribe_speech=MagicMock(),
        )
        assert svc.procesar_one(_meta("x"), "P") is None

    def test_ensure_perfil(self, tmp_path: Path) -> None:
        fetcher = MagicMock()
        speakers = MagicMock()
        ref = _audio(tmp_path, "ref")
        fetcher.fetch_one.return_value = ref
        svc = SpeechToTextService(
            data_dir=tmp_path,
            video_ref_id="ref",
            audio_fetcher=fetcher,
            speaker_timestamps=speakers,
            transcribe_speech=MagicMock(),
        )
        ok = svc.ensure_perfil("P", [_meta("ref"), _meta("other")])
        assert ok is True
        speakers.build_perfil.assert_called_once_with(ref)

    def test_ensure_perfil_ref_ausente(self, tmp_path: Path) -> None:
        svc = SpeechToTextService(
            data_dir=tmp_path,
            video_ref_id="missing",
            audio_fetcher=MagicMock(),
            speaker_timestamps=MagicMock(),
            transcribe_speech=MagicMock(),
        )
        assert svc.ensure_perfil("P", [_meta("a")]) is False
