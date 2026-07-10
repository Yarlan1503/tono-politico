"""Edición temporal con ffmpeg y transcripción Whisper de un clip."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from .models import ClipTranscriptSegment

ModelLoader = Callable[[str], Any]
Runner = Callable[..., subprocess.CompletedProcess[str]]


class WhisperFfmpegClipTranscriber:
    """Normaliza un rango a WAV mono 16 kHz y lo transcribe con Whisper."""

    def __init__(
        self,
        *,
        temp_dir: Path | str | None = None,
        ffmpeg_bin: str = "ffmpeg",
        runner: Runner = subprocess.run,
        model_loader: ModelLoader | None = None,
    ) -> None:
        self.temp_dir = Path(temp_dir) if temp_dir is not None else None
        self.ffmpeg_bin = ffmpeg_bin
        self._runner = runner
        self._model_loader = model_loader or _load_whisper_model
        self._models: dict[str, Any] = {}

    def transcribir_clip(
        self,
        audio_path: Path,
        *,
        t_start: float,
        t_end: float,
        modelo: str,
        idioma: str,
    ) -> list[ClipTranscriptSegment]:
        _validar_clip(Path(audio_path), t_start, t_end)
        temp_path = self._crear_temp_wav()
        try:
            self._recortar_con_ffmpeg(Path(audio_path), temp_path, t_start, t_end)
            resultado = self._modelo(modelo).transcribe(
                str(temp_path),
                language=idioma,
                word_timestamps=False,
                fp16=False,
                verbose=False,
            )
            return _normalizar_segmentos(resultado)
        finally:
            temp_path.unlink(missing_ok=True)

    def _modelo(self, modelo: str) -> Any:
        if modelo not in self._models:
            self._models[modelo] = self._model_loader(modelo)
        return self._models[modelo]

    def _crear_temp_wav(self) -> Path:
        if self.temp_dir is not None:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(suffix=".wav", dir=self.temp_dir)
        os.close(fd)
        temp_path = Path(path)
        temp_path.unlink(missing_ok=True)
        return temp_path

    def _recortar_con_ffmpeg(
        self,
        audio_path: Path,
        temp_path: Path,
        t_start: float,
        t_end: float,
    ) -> None:
        cmd = [
            self.ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(t_start),
            "-i",
            str(audio_path),
            "-t",
            str(t_end - t_start),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(temp_path),
        ]
        try:
            self._runner(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg no está instalado o no está en PATH") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or str(exc)
            raise RuntimeError(f"ffmpeg falló al recortar clip: {stderr}") from exc


def _load_whisper_model(modelo: str) -> Any:
    import whisper  # type: ignore[import-not-found]

    return whisper.load_model(modelo)


def _validar_clip(audio_path: Path, t_start: float, t_end: float) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio no encontrado: {audio_path}")
    if t_start < 0 or t_end <= t_start:
        raise ValueError("t_end debe ser mayor que t_start y t_start no puede ser negativo")


def _normalizar_segmentos(resultado: Any) -> list[ClipTranscriptSegment]:
    if not isinstance(resultado, dict):
        return []
    data = cast(dict[str, Any], resultado)
    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list):
        return []

    segments: list[ClipTranscriptSegment] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        segment = cast(dict[str, Any], raw_segment)
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        t_start = float(segment.get("start", 0.0))
        t_end = float(segment.get("end", 0.0))
        if t_end <= t_start:
            continue
        segments.append(ClipTranscriptSegment(text=text, t_start=t_start, t_end=t_end))
    return segments
