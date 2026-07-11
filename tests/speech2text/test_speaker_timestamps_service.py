"""Tests de SpeakerTimestampsService (con mocks de pyannote)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tono_politico.speech2text.audio_fetcher.models import AudioVideo
from tono_politico.speech2text.models import PerfilVozActor, SpeakerMatch
from tono_politico.speech2text.speaker_timestamps.service import (
    PyannotePipelineLoadError,
    SpeakerTimestampsService,
    _extraer_turnos,
    load_pyannote_pipeline,
    run_pyannote_pipeline,
)


def _audio(tmp_path: Path, video_id: str = "vid1") -> AudioVideo:
    wav = tmp_path / f"{video_id}.wav"
    wav.write_bytes(b"RIFF")
    return AudioVideo(
        video_id=video_id,
        url=f"https://example.com/{video_id}",
        titulo="T",
        fecha=None,
        audio_path=wav,
        duracion=10.0,
    )


def _perfil() -> PerfilVozActor:
    return PerfilVozActor(
        actor="Actor",
        video_id_referencia="ref",
        embedding=[1.0, 0.0],
        modelo_embedding="test",
        duracion_segundos=10.0,
    )


class TestExtraerTurnos:
    def test_from_exclusive(self) -> None:
        seg1 = SimpleNamespace(start=0.0, end=1.5)
        seg2 = SimpleNamespace(start=2.0, end=3.0)
        exclusive = MagicMock()
        exclusive.itertracks.return_value = [
            (seg1, None, "SPEAKER_00"),
            (seg2, None, "SPEAKER_01"),
        ]
        output = SimpleNamespace(exclusive_speaker_diarization=exclusive)
        turnos = _extraer_turnos(output, "v1")
        assert len(turnos) == 2
        assert turnos[0].speaker_id == "SPEAKER_00"
        assert turnos[0].t_end == 1.5


class TestSpeakerTimestampsService:
    def test_procesar_one_sin_perfil_raises(self, tmp_path: Path) -> None:
        svc = SpeakerTimestampsService()
        with pytest.raises(RuntimeError, match="Perfil"):
            svc.procesar_one(_audio(tmp_path))

    def test_procesar_one_filtra_actor(self, tmp_path: Path) -> None:
        svc = SpeakerTimestampsService(actor="Actor")
        svc.set_perfil(_perfil())

        seg_a = SimpleNamespace(start=0.0, end=1.0)
        seg_b = SimpleNamespace(start=1.0, end=2.0)
        exclusive = MagicMock()
        exclusive.itertracks.return_value = [
            (seg_a, None, "SPEAKER_00"),
            (seg_b, None, "SPEAKER_01"),
        ]
        diar = MagicMock()
        diar.labels.return_value = ["SPEAKER_00", "SPEAKER_01"]
        output = SimpleNamespace(
            exclusive_speaker_diarization=exclusive,
            speaker_diarization=diar,
            speaker_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        )

        matches = [
            SpeakerMatch(
                speaker_id="SPEAKER_00",
                distancia=0.1,
                aceptado=True,
                es_ambiguo=False,
            ),
            SpeakerMatch(
                speaker_id="SPEAKER_01",
                distancia=0.9,
                aceptado=False,
                es_ambiguo=False,
            ),
        ]

        with (
            patch.object(svc, "_get_pipeline", return_value="pipe"),
            patch(
                "tono_politico.speech2text.speaker_timestamps.service.run_pyannote_pipeline",
                return_value=output,
            ),
            patch(
                "tono_politico.speech2text.speaker_timestamps.service.identificar_actor",
                return_value=matches,
            ),
        ):
            turnos = svc.procesar_one(_audio(tmp_path))

        assert len(turnos) == 1
        assert turnos[0].speaker_id == "SPEAKER_00"

    def test_procesar_one_rechaza_embeddings_inconsistentes(self, tmp_path: Path) -> None:
        svc = SpeakerTimestampsService()
        svc.set_perfil(_perfil())
        exclusive = MagicMock()
        exclusive.itertracks.return_value = [
            (SimpleNamespace(start=0.0, end=1.0), None, "SPEAKER_00"),
        ]
        diar = MagicMock()
        diar.labels.return_value = ["SPEAKER_00", "SPEAKER_01"]
        output = SimpleNamespace(
            exclusive_speaker_diarization=exclusive,
            speaker_diarization=diar,
            speaker_embeddings=[[1.0, 0.0]],
        )
        with (
            patch.object(svc, "_get_pipeline", return_value="pipe"),
            patch(
                "tono_politico.speech2text.speaker_timestamps.service.run_pyannote_pipeline",
                return_value=output,
            ),
            pytest.raises(ValueError, match="embeddings"),
        ):
            svc.procesar_one(_audio(tmp_path))

    def test_procesar_one_rechaza_turno_invalido(self, tmp_path: Path) -> None:
        svc = SpeakerTimestampsService()
        svc.set_perfil(_perfil())
        exclusive = MagicMock()
        exclusive.itertracks.return_value = [
            (SimpleNamespace(start=2.0, end=1.0), None, "SPEAKER_00"),
        ]
        output = SimpleNamespace(exclusive_speaker_diarization=exclusive)

        with pytest.raises(ValueError, match="t_end"):
            _extraer_turnos(output, "vid1")


FAKE_CREDENTIAL = "HF_TEST_VALUE"


@dataclass
class FakeLoadedPipeline:
    name: str
    to_calls: list[Any]
    call_kwargs: list[dict[str, Any]]

    def to(self, device: Any) -> None:
        self.to_calls.append(device)

    def __call__(self, audio_path: str, **kwargs: Any) -> str:
        self.call_kwargs.append(kwargs)
        return f"output:{audio_path}"


class FakePipelineClass:
    calls: list[tuple[str, str | None]] = []
    failures: dict[str, Exception] = {}

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.failures = {}

    @classmethod
    def from_pretrained(cls, name: str, token: str | None = None) -> FakeLoadedPipeline:
        cls.calls.append((name, token))
        if name in cls.failures:
            raise cls.failures[name]
        return FakeLoadedPipeline(name=name, to_calls=[], call_kwargs=[])


class FakeCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeDevice:
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"device({self.name})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeDevice) and self.name == other.name


class FakeTorch:
    def __init__(self, cuda_available: bool):
        self.cuda = FakeCuda(cuda_available)

    def device(self, target: str) -> FakeDevice:
        return FakeDevice(target)


class FakeProgressHook:
    entered = False
    exited = False

    def __enter__(self):
        type(self).entered = True
        return "HOOK"

    def __exit__(self, exc_type, exc, tb) -> None:
        type(self).exited = True


def test_load_primary_pipeline_pasa_token_y_device_auto_cuda():
    FakePipelineClass.reset()

    loaded = load_pyannote_pipeline(
        primary_pipeline="pyannote/speaker-diarization-community-1",
        fallback_pipeline="pyannote-community/speaker-diarization-community-1",
        token=FAKE_CREDENTIAL,
        device="auto",
        pipeline_cls=FakePipelineClass,
        torch_module=FakeTorch(cuda_available=True),
    )

    assert loaded.pipeline_name == "pyannote/speaker-diarization-community-1"
    assert loaded.used_fallback is False
    assert FakePipelineClass.calls == [
        ("pyannote/speaker-diarization-community-1", FAKE_CREDENTIAL)
    ]
    assert loaded.pipeline.to_calls == [FakeDevice("cuda")]


def test_load_fallback_si_primary_falla(caplog):
    FakePipelineClass.reset()
    FakePipelineClass.failures = {
        "pyannote/speaker-diarization-community-1": RuntimeError("gated primary"),
    }

    with caplog.at_level(logging.WARNING):
        loaded = load_pyannote_pipeline(
            primary_pipeline="pyannote/speaker-diarization-community-1",
            fallback_pipeline="pyannote-community/speaker-diarization-community-1",
            token=FAKE_CREDENTIAL,
            device="cpu",
            pipeline_cls=FakePipelineClass,
            torch_module=FakeTorch(cuda_available=True),
        )

    assert loaded.pipeline_name == "pyannote-community/speaker-diarization-community-1"
    assert loaded.used_fallback is True
    assert FakePipelineClass.calls == [
        ("pyannote/speaker-diarization-community-1", FAKE_CREDENTIAL),
        ("pyannote-community/speaker-diarization-community-1", FAKE_CREDENTIAL),
    ]
    assert loaded.pipeline.to_calls == [FakeDevice("cpu")]
    assert "gated primary" in caplog.text
    assert FAKE_CREDENTIAL not in caplog.text


def test_load_ambos_fallan_error_accionable_sin_filtrar_token(caplog):
    FakePipelineClass.reset()
    FakePipelineClass.failures = {
        "pyannote/speaker-diarization-community-1": RuntimeError("gated primary"),
        "pyannote-community/speaker-diarization-community-1": RuntimeError("fallback missing"),
    }

    with caplog.at_level(logging.WARNING):
        with pytest.raises(PyannotePipelineLoadError) as excinfo:
            load_pyannote_pipeline(
                primary_pipeline="pyannote/speaker-diarization-community-1",
                fallback_pipeline="pyannote-community/speaker-diarization-community-1",
                token=FAKE_CREDENTIAL,
                device="cpu",
                pipeline_cls=FakePipelineClass,
                torch_module=FakeTorch(cuda_available=False),
            )

    message = str(excinfo.value)
    assert "pyannote/speaker-diarization-community-1" in message
    assert "pyannote-community/speaker-diarization-community-1" in message
    assert "Hugging Face" in message
    assert FAKE_CREDENTIAL not in message
    assert FAKE_CREDENTIAL not in caplog.text


def test_run_pipeline_usa_progress_hook_si_existe():
    pipeline = FakeLoadedPipeline(name="model", to_calls=[], call_kwargs=[])
    FakeProgressHook.entered = False
    FakeProgressHook.exited = False

    output = run_pyannote_pipeline(
        pipeline,
        "audio.wav",
        progress_hook_cls=FakeProgressHook,
    )

    assert output == "output:audio.wav"
    assert FakeProgressHook.entered is True
    assert FakeProgressHook.exited is True
    assert pipeline.call_kwargs == [{"hook": "HOOK"}]


def test_run_pipeline_sin_progress_hook_llama_directo():
    pipeline = FakeLoadedPipeline(name="model", to_calls=[], call_kwargs=[])

    output = run_pyannote_pipeline(pipeline, "audio.wav", progress_hook_cls=None)

    assert output == "output:audio.wav"
    assert pipeline.call_kwargs == [{}]
