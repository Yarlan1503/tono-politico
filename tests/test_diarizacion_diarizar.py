"""Tests para diarizar(): pure function que ejecuta pyannote y extrae turnos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tono_politico.diarizacion.diarizacion import diarizar
from tono_politico.diarizacion.models import TurnoOrador

# ──────────────────────────────────────────────────────────
# Fakes que simulan la API de pyannote
# ──────────────────────────────────────────────────────────


@dataclass
class FakeSegment:
    """Simula pyannote.core.Segment con start/end."""

    start: float
    end: float


@dataclass
class FakeExclusiveDiarization:
    """Simula output.exclusive_speaker_diarization con itertracks."""

    tracks: list[tuple[FakeSegment, str, str]]  # (segment, track, label)

    def itertracks(self, yield_label=True):
        for seg, track, label in self.tracks:
            if yield_label:
                yield seg, track, label
            else:
                yield seg, track


@dataclass
class FakeDiarizationOutput:
    """Simula el objeto de salida de Pipeline.__call__."""

    _exclusive: FakeExclusiveDiarization

    @property
    def exclusive_speaker_diarization(self) -> FakeExclusiveDiarization:
        return self._exclusive


class FakePipeline:
    """Simula pyannote.audio.Pipeline — callable sobre un path de audio."""

    def __init__(self, tracks_by_audio: dict[str, list[tuple[float, float, str]]]):
        """Args:
        tracks_by_audio: {audio_path_str: [(start, end, speaker_label), ...]}
        """
        self._tracks = tracks_by_audio

    def __call__(self, audio_path, **kwargs) -> FakeDiarizationOutput:
        key = str(audio_path)
        raw_tracks = self._tracks.get(key, [])
        tracks = [
            (FakeSegment(s, e), f"track_{i}", label) for i, (s, e, label) in enumerate(raw_tracks)
        ]
        return FakeDiarizationOutput(FakeExclusiveDiarization(tracks))


# ──────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────


class TestDiarizar:
    """diarizar(): audio WAV + pipeline → list[TurnoOrador]."""

    def test_un_speaker_varios_turnos(self):
        """Un audio con un solo orador produce turnos con el mismo speaker_id."""
        audio = Path("/fake/audio.wav")
        pipeline = FakePipeline(
            {
                str(audio): [
                    (0.0, 3.5, "SPEAKER_00"),
                    (4.0, 8.2, "SPEAKER_00"),
                    (9.0, 12.0, "SPEAKER_00"),
                ],
            }
        )

        turnos = diarizar(audio, pipeline, video_id="abc123")

        assert len(turnos) == 3
        assert all(isinstance(t, TurnoOrador) for t in turnos)
        assert all(t.video_id == "abc123" for t in turnos)
        assert all(t.speaker_id == "SPEAKER_00" for t in turnos)
        assert turnos[0].t_start == 0.0
        assert turnos[0].t_end == 3.5
        assert turnos[2].t_start == 9.0

    def test_dos_speakers_intercalados(self):
        """Dos oradores intercalados producen turnos etiquetados correctamente."""
        audio = Path("/fake/audio.wav")
        pipeline = FakePipeline(
            {
                str(audio): [
                    (0.0, 5.0, "SPEAKER_00"),
                    (5.5, 10.0, "SPEAKER_01"),
                    (10.5, 15.0, "SPEAKER_00"),
                ],
            }
        )

        turnos = diarizar(audio, pipeline, video_id="vid1")

        assert len(turnos) == 3
        assert turnos[0].speaker_id == "SPEAKER_00"
        assert turnos[1].speaker_id == "SPEAKER_01"
        assert turnos[2].speaker_id == "SPEAKER_00"

    def test_sin_turnos(self):
        """Un audio sin actividad de habla devuelve lista vacía."""
        audio = Path("/fake/silence.wav")
        pipeline = FakePipeline({str(audio): []})

        turnos = diarizar(audio, pipeline, video_id="empty")

        assert turnos == []

    def test_video_id_se_propaga(self):
        """Cada turno lleva el video_id correcto."""
        audio = Path("/fake/v.wav")
        pipeline = FakePipeline(
            {
                str(audio): [(0.0, 2.0, "SPEAKER_00")],
            }
        )

        turnos = diarizar(audio, pipeline, video_id="xyz789")

        assert turnos[0].video_id == "xyz789"

    def test_tres_speakers(self):
        """Tres oradores distintos en el mismo audio."""
        audio = Path("/fake/debate.wav")
        pipeline = FakePipeline(
            {
                str(audio): [
                    (0.0, 3.0, "SPEAKER_00"),
                    (3.0, 6.0, "SPEAKER_01"),
                    (6.0, 9.0, "SPEAKER_02"),
                ],
            }
        )

        turnos = diarizar(audio, pipeline, video_id="debate")

        speaker_ids = {t.speaker_id for t in turnos}
        assert speaker_ids == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}

    def test_audio_path_como_string(self):
        """diarizar acepta Path o str para audio_path."""
        pipeline = FakePipeline(
            {
                "/fake/str.wav": [(0.0, 1.0, "SPEAKER_00")],
            }
        )

        turnos = diarizar("/fake/str.wav", pipeline, video_id="s1")

        assert len(turnos) == 1
