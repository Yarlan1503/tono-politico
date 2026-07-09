"""Tests RED para adaptador real Whisper + ffmpeg sobre clips temporales."""

from __future__ import annotations

import subprocess
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest


def _sut():
    return import_module("tono_politico.speech2text.diarization.whisper_clip")


def _audio(tmp_path: Path) -> Path:
    path = tmp_path / "video.wav"
    path.write_bytes(b"fake-audio")
    return path


class FakeRunner:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []
        self.output_paths: list[Path] = []

    def __call__(
        self,
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is True
        assert capture_output is True
        assert text is True
        self.calls.append(cmd)
        output_path = Path(cmd[-1])
        self.output_paths.append(output_path)
        if self.fail:
            raise subprocess.CalledProcessError(1, cmd, stderr="ffmpeg boom")
        output_path.write_bytes(b"clip-wav")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


class RefusesExistingOutputRunner(FakeRunner):
    def __call__(
        self,
        cmd: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
    ) -> subprocess.CompletedProcess[str]:
        output_path = Path(cmd[-1])
        if output_path.exists():
            raise subprocess.CalledProcessError(1, cmd, stderr="File exists")
        return super().__call__(cmd, check=check, capture_output=capture_output, text=text)


class FakeWhisperModel:
    def __init__(self, *, fail: bool = False, segments: list[dict[str, Any]] | None = None) -> None:
        self.fail = fail
        self.segments = segments or []
        self.calls: list[dict[str, Any]] = []
        self.saw_existing_temp = False

    def transcribe(self, audio_path: str, **kwargs: Any) -> dict[str, Any]:
        path = Path(audio_path)
        self.saw_existing_temp = path.exists()
        self.calls.append({"audio_path": path, **kwargs})
        if self.fail:
            raise RuntimeError("whisper boom")
        return {"segments": self.segments}


class FakeModelLoader:
    def __init__(self, model: FakeWhisperModel) -> None:
        self.model = model
        self.calls: list[str] = []

    def __call__(self, modelo: str) -> FakeWhisperModel:
        self.calls.append(modelo)
        return self.model


def test_construye_comando_ffmpeg_para_wav_temporal_normalizado(tmp_path: Path):
    audio_path = _audio(tmp_path)
    runner = FakeRunner()
    model = FakeWhisperModel(segments=[{"text": "Hola", "start": 0.0, "end": 1.0}])
    loader = FakeModelLoader(model)

    transcriber = _sut().WhisperFfmpegClipTranscriber(
        temp_dir=tmp_path,
        runner=runner,
        model_loader=loader,
    )

    transcriber.transcribir_clip(
        audio_path,
        t_start=12.5,
        t_end=16.5,
        modelo="large-v3-turbo",
        idioma="es",
    )

    cmd = runner.calls[0]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-ss") + 1] == "12.5"
    assert cmd[cmd.index("-i") + 1] == str(audio_path)
    assert cmd[cmd.index("-t") + 1] == "4.0"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-c:a") + 1] == "pcm_s16le"
    assert "-vn" in cmd
    assert Path(cmd[-1]).suffix == ".wav"


def test_llama_whisper_con_el_temporal_y_no_con_el_audio_original(tmp_path: Path):
    audio_path = _audio(tmp_path)
    runner = FakeRunner()
    model = FakeWhisperModel(segments=[{"text": " Texto normalizado ", "start": 0.2, "end": 1.4}])
    loader = FakeModelLoader(model)
    transcriber = _sut().WhisperFfmpegClipTranscriber(
        temp_dir=tmp_path,
        runner=runner,
        model_loader=loader,
    )

    segments = transcriber.transcribir_clip(
        audio_path,
        t_start=1.0,
        t_end=3.0,
        modelo="large-v3-turbo",
        idioma="es",
    )

    assert loader.calls == ["large-v3-turbo"]
    assert model.saw_existing_temp is True
    assert model.calls[0]["audio_path"] != audio_path
    assert model.calls[0]["language"] == "es"
    assert model.calls[0]["word_timestamps"] is False
    assert model.calls[0]["fp16"] is False
    assert model.calls[0]["verbose"] is False
    assert segments == [
        _sut().ClipTranscriptSegment(text="Texto normalizado", t_start=0.2, t_end=1.4)
    ]


def test_normaliza_y_omite_segmentos_vacios_sin_exponer_words_ni_probability(tmp_path: Path):
    audio_path = _audio(tmp_path)
    runner = FakeRunner()
    model = FakeWhisperModel(
        segments=[
            {"text": "   ", "start": 0.0, "end": 0.5},
            {
                "text": " Segmento útil ",
                "start": 0.5,
                "end": 2.0,
                "words": [{"word": "Segmento", "probability": 0.99}],
                "probability": 0.99,
            },
        ]
    )
    transcriber = _sut().WhisperFfmpegClipTranscriber(
        temp_dir=tmp_path,
        runner=runner,
        model_loader=FakeModelLoader(model),
    )

    segments = transcriber.transcribir_clip(
        audio_path,
        t_start=0.0,
        t_end=3.0,
        modelo="large-v3-turbo",
        idioma="es",
    )

    assert segments == [_sut().ClipTranscriptSegment(text="Segmento útil", t_start=0.5, t_end=2.0)]
    assert not hasattr(segments[0], "words")
    assert not hasattr(segments[0], "probability")


def test_limpia_temporal_aunque_whisper_falle(tmp_path: Path):
    audio_path = _audio(tmp_path)
    runner = FakeRunner()
    model = FakeWhisperModel(fail=True)
    transcriber = _sut().WhisperFfmpegClipTranscriber(
        temp_dir=tmp_path,
        runner=runner,
        model_loader=FakeModelLoader(model),
    )

    with pytest.raises(RuntimeError, match="whisper boom"):
        transcriber.transcribir_clip(
            audio_path,
            t_start=0.0,
            t_end=1.0,
            modelo="large-v3-turbo",
            idioma="es",
        )

    assert runner.output_paths
    assert not runner.output_paths[0].exists()


def test_ffmpeg_falla_con_mensaje_claro_y_limpia_temporal(tmp_path: Path):
    audio_path = _audio(tmp_path)
    runner = FakeRunner(fail=True)
    model = FakeWhisperModel()
    transcriber = _sut().WhisperFfmpegClipTranscriber(
        temp_dir=tmp_path,
        runner=runner,
        model_loader=FakeModelLoader(model),
    )

    with pytest.raises(RuntimeError, match="ffmpeg boom"):
        transcriber.transcribir_clip(
            audio_path,
            t_start=0.0,
            t_end=1.0,
            modelo="large-v3-turbo",
            idioma="es",
        )

    assert runner.output_paths
    assert not runner.output_paths[0].exists()
    assert model.calls == []


def test_no_entrega_a_ffmpeg_un_output_temporal_preexistente(tmp_path: Path):
    audio_path = _audio(tmp_path)
    runner = RefusesExistingOutputRunner()
    model = FakeWhisperModel(segments=[{"text": "Hola", "start": 0.0, "end": 1.0}])
    transcriber = _sut().WhisperFfmpegClipTranscriber(
        temp_dir=tmp_path,
        runner=runner,
        model_loader=FakeModelLoader(model),
    )

    segments = transcriber.transcribir_clip(
        audio_path,
        t_start=0.0,
        t_end=1.0,
        modelo="large-v3-turbo",
        idioma="es",
    )

    assert segments == [_sut().ClipTranscriptSegment(text="Hola", t_start=0.0, t_end=1.0)]


def test_rechaza_audio_inexistente_y_rango_temporal_invalido(tmp_path: Path):
    transcriber = _sut().WhisperFfmpegClipTranscriber(
        temp_dir=tmp_path,
        runner=FakeRunner(),
        model_loader=FakeModelLoader(FakeWhisperModel()),
    )

    with pytest.raises(FileNotFoundError, match="Audio"):
        transcriber.transcribir_clip(
            tmp_path / "missing.wav",
            t_start=0.0,
            t_end=1.0,
            modelo="large-v3-turbo",
            idioma="es",
        )

    audio_path = _audio(tmp_path)
    with pytest.raises(ValueError, match="t_end"):
        transcriber.transcribir_clip(
            audio_path,
            t_start=2.0,
            t_end=1.0,
            modelo="large-v3-turbo",
            idioma="es",
        )
