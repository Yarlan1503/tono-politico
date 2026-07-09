#!/usr/bin/env python3
"""Smoke end-to-end de SpeechToTextService sobre Play-PoliTest.

Uso:
    uv run python scripts_smoke_speech2text.py

Escribe:
    output/speech2text-smoke/
      summary.json
      actor_transcripts/<video_id>.json
      run.log  (vía stdout del proceso)
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tono_politico.speech2text import SpeechToTextService

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLE9Zk7g9R__M"
DATA_DIR = Path("data/speech2text-smoke")
OUT_DIR = Path("output/speech2text-smoke")
ACTOR = "Lilly Téllez"
VIDEO_REF_ID = "su9nURIj9XQ"


def _to_jsonable(obj: object) -> object:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("smoke_speech2text")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tx_dir = OUT_DIR / "actor_transcripts"
    tx_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    started_iso = datetime.now(UTC).isoformat()
    log.info("=== speech2text smoke START ===")
    log.info("playlist=%s", PLAYLIST_URL)
    log.info("data_dir=%s out_dir=%s", DATA_DIR, OUT_DIR)

    svc = SpeechToTextService(
        data_dir=DATA_DIR,
        actor=ACTOR,
        video_ref_id=VIDEO_REF_ID,
        whisper_model="large-v3-turbo",
        idioma="es",
        umbral_match=0.5,
        umbral_ambiguo=0.7,
        device="auto",
    )

    per_video: list[dict[str, Any]] = []
    transcripts_ok = 0
    errors: list[str] = []

    try:
        playlist, metas = svc.discover(PLAYLIST_URL)
        log.info(
            "discover: playlist=%r videos=%s",
            playlist.nombre,
            len(metas),
        )
        for m in metas:
            log.info(
                "  meta %s | %s | fecha=%s | dur=%.1fs",
                m.video_id,
                m.titulo[:60],
                m.fecha,
                m.duracion,
            )

        if not metas:
            raise RuntimeError("Playlist vacía o inaccesible")

        if not svc.ensure_perfil(playlist.nombre, metas):
            raise RuntimeError("No se pudo construir el perfil de voz del actor")

        for i, meta in enumerate(metas, start=1):
            log.info(
                "--- video %s/%s: %s ---",
                i,
                len(metas),
                meta.video_id,
            )
            t0 = time.perf_counter()
            status = "ok"
            err: str | None = None
            n_segments = 0
            n_words = 0
            try:
                tx = svc.procesar_one(meta, playlist.nombre)
                if tx is None:
                    status = "skip"
                    log.warning("skip: sin ActorTranscript para %s", meta.video_id)
                else:
                    transcripts_ok += 1
                    n_segments = len(tx.segments)
                    n_words = sum(s.word_count or 0 for s in tx.segments)
                    out_path = tx_dir / f"{meta.video_id}.json"
                    out_path.write_text(
                        json.dumps(_to_jsonable(tx), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    log.info(
                        "ok: %s segmentos=%s words≈%s → %s",
                        meta.video_id,
                        n_segments,
                        n_words,
                        out_path,
                    )
            except Exception as exc:  # noqa: BLE001 — smoke: capturar y seguir
                status = "error"
                err = f"{type(exc).__name__}: {exc}"
                errors.append(f"{meta.video_id}: {err}")
                log.error("error en %s: %s", meta.video_id, err)
                log.error(traceback.format_exc())

            per_video.append(
                {
                    "video_id": meta.video_id,
                    "titulo": meta.titulo,
                    "fecha": meta.fecha,
                    "duracion": meta.duracion,
                    "status": status,
                    "n_segments": n_segments,
                    "n_words": n_words,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                    "error": err,
                }
            )

        elapsed = round(time.perf_counter() - started, 2)
        summary = {
            "started_at": started_iso,
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_s": elapsed,
            "playlist_url": PLAYLIST_URL,
            "playlist_name": playlist.nombre,
            "actor": ACTOR,
            "video_ref_id": VIDEO_REF_ID,
            "n_videos": len(metas),
            "n_ok": transcripts_ok,
            "n_skip": sum(1 for v in per_video if v["status"] == "skip"),
            "n_error": sum(1 for v in per_video if v["status"] == "error"),
            "total_segments": sum(int(v["n_segments"]) for v in per_video),
            "total_words": sum(int(v["n_words"]) for v in per_video),
            "videos": per_video,
            "errors": errors,
            "data_dir": str(DATA_DIR),
            "out_dir": str(OUT_DIR),
        }
        summary_path = OUT_DIR / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("=== speech2text smoke DONE ===")
        log.info(
            "ok=%s skip=%s error=%s segments=%s elapsed=%.1fs",
            summary["n_ok"],
            summary["n_skip"],
            summary["n_error"],
            summary["total_segments"],
            elapsed,
        )
        log.info("summary → %s", summary_path)
        return 0 if summary["n_error"] == 0 and summary["n_ok"] > 0 else 1

    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.perf_counter() - started, 2)
        fatal = {
            "started_at": started_iso,
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_s": elapsed,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "videos": per_video,
            "errors": errors,
        }
        (OUT_DIR / "summary.json").write_text(
            json.dumps(fatal, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.error("FATAL: %s", exc)
        log.error(traceback.format_exc())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
