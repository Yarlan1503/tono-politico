"""Serialización y persistencia de RunManifest en disco.

Cada corrida del pipeline deja una bitácora legible y machine-readable
en ``output/runs/<run_id>/manifest.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import RunManifest, RunResult

logger = logging.getLogger(__name__)


def manifest_to_dict(manifest: RunManifest) -> dict:
    """Convierte un RunManifest a diccionario serializable a JSON.

    Los ``Path`` se convierten a ``str``; los ``None`` se mantienen o se omiten.
    """
    data: dict = {
        "run_id": manifest.run_id,
        "playlist_url": manifest.playlist_url,
        "playlist_name": manifest.playlist_name,
        "status": manifest.status,
    }
    if manifest.artifacts_dir is not None:
        data["artifacts_dir"] = str(manifest.artifacts_dir)
    if manifest.cache_dir is not None:
        data["cache_dir"] = str(manifest.cache_dir)

    data["videos"] = [
        {
            "video_id": v.video_id,
            "titulo": v.titulo,
            "descargado": v.descargado,
            "transcrito": v.transcrito,
            "diarizado": v.diarizado,
            "segmentos_actor": v.segmentos_actor,
            "omitido": v.omitido,
            "error": v.error,
        }
        for v in manifest.videos
    ]
    data["phases"] = [
        {
            "phase": p.phase,
            "ok": p.ok,
            "elapsed_seconds": round(p.elapsed_seconds, 3),
            "message": p.message,
        }
        for p in manifest.phases
    ]
    return data


def manifest_to_json(manifest: RunManifest) -> str:
    """Serializa un RunManifest a JSON pretty-printed."""
    return json.dumps(manifest_to_dict(manifest), indent=2, ensure_ascii=False)


def guardar_manifest(
    manifest: RunManifest,
    output_base: Path | str,
) -> Path:
    """Persiste ``manifest.json`` en ``<output_base>/<run_id>/manifest.json``.

    Crea el directorio si no existe. Devuelve la ruta absoluta del archivo.
    """
    output_base = Path(output_base)
    run_dir = output_base / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(manifest_to_json(manifest), encoding="utf-8")
    manifest.artifacts_dir = run_dir

    logger.info("Manifest guardado: %s", manifest_path)
    return manifest_path


def resumen_final(result: RunResult) -> str:
    """Genera un resumen legible para el CLI.

    Ejemplo::

        Run: 20260706-153012-Play-PoliTest
        Status: partial
        Videos: 6 procesados, 1 omitido (descarga 403)
        Fase 1: 12 tópicos descubiertos
        Artifacts: output/runs/20260706-153012-Play-PoliTest/
        Cache: limpiado (usa --keep-cache para conservar)
    """
    m = result.manifest
    procesados = sum(1 for v in m.videos if not v.omitido)
    omitidos = sum(1 for v in m.videos if v.omitido)

    lineas = [
        f"Run: {m.run_id}",
        f"Status: {m.status}",
        f"Videos: {procesados} procesado{'s' if procesados != 1 else ''}, "
        f"{omitidos} omitido{'s' if omitidos != 1 else ''}",
    ]

    # Fases ejecutadas
    for p in m.phases:
        status_icon = "✅" if p.ok else "❌"
        lineas.append(
            f"  {status_icon} {p.phase} ({p.elapsed_seconds:.1f}s)"
            + (f" — {p.message}" if p.message else "")
        )

    if m.artifacts_dir is not None:
        lineas.append(f"Artifacts: {m.artifacts_dir}")

    if result.informe_path is not None:
        lineas.append(f"Informe: {result.informe_path}")

    lineas.append("Cache: limpiado (usa --keep-cache para conservar)")

    return "\n".join(lineas)
