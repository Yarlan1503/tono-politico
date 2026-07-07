"""Serialización de ResultadoTemas para persistencia y resume.

Permite guardar/cargar el resultado de Fase 1 para que ``--resume``
no tenga que repetir ingesta/diarización/segmentación/temas.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..models import WordTimestamp
from ..segmentacion.models import Oracion, Segmento
from .models import ResultadoTemas, SegmentoTematizado, TopicoInfo

logger = logging.getLogger(__name__)


def _word_to_dict(w: WordTimestamp) -> dict:
    return {
        "word": w.word,
        "start": w.start,
        "end": w.end,
        "probability": w.probability,
    }


def _word_from_dict(d: dict) -> WordTimestamp:
    return WordTimestamp(
        word=d["word"],
        start=d["start"],
        end=d["end"],
        probability=d.get("probability"),
    )


def _oracion_to_dict(o: Oracion) -> dict:
    return {
        "texto": o.texto,
        "t_start": o.t_start,
        "t_end": o.t_end,
        "words": [_word_to_dict(w) for w in o.words],
    }


def _oracion_from_dict(d: dict) -> Oracion:
    return Oracion(
        texto=d["texto"],
        t_start=d["t_start"],
        t_end=d["t_end"],
        words=[_word_from_dict(w) for w in d.get("words", [])],
    )


def _segmento_to_dict(s: Segmento) -> dict:
    return {
        "texto": s.texto,
        "t_start": s.t_start,
        "t_end": s.t_end,
        "oraciones": [_oracion_to_dict(o) for o in s.oraciones],
        "word_count": s.word_count,
        "video_id": s.video_id,
    }


def _segmento_from_dict(d: dict) -> Segmento:
    return Segmento(
        texto=d["texto"],
        t_start=d["t_start"],
        t_end=d["t_end"],
        oraciones=[_oracion_from_dict(o) for o in d.get("oraciones", [])],
        word_count=d.get("word_count", 0),
        video_id=d.get("video_id", ""),
    )


def _segmento_tematizado_to_dict(st: SegmentoTematizado) -> dict:
    return {
        "segmento": _segmento_to_dict(st.segmento),
        "topico_id": st.topico_id,
        "probabilidad": st.probabilidad,
    }


def _segmento_tematizado_from_dict(d: dict) -> SegmentoTematizado:
    return SegmentoTematizado(
        segmento=_segmento_from_dict(d["segmento"]),
        topico_id=d["topico_id"],
        probabilidad=d["probabilidad"],
    )


def _topico_to_dict(t: TopicoInfo) -> dict:
    return {
        "id": t.id,
        "nombre": t.nombre,
        "palabras_clave": t.palabras_clave,
        "num_segmentos": t.num_segmentos,
        "representatividad": t.representatividad,
    }


def _topico_from_dict(d: dict) -> TopicoInfo:
    return TopicoInfo(
        id=d["id"],
        nombre=d["nombre"],
        palabras_clave=d.get("palabras_clave", []),
        num_segmentos=d.get("num_segmentos", 0),
        representatividad=d.get("representatividad", 0.0),
    )


# ──────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────


def resultado_temas_to_dict(resultado: ResultadoTemas) -> dict:
    """Serializa un ResultadoTemas a diccionario."""
    return {
        "segmentos": [_segmento_tematizado_to_dict(s) for s in resultado.segmentos],
        "topicos": [_topico_to_dict(t) for t in resultado.topicos],
        "num_topicos": resultado.num_topicos,
    }


def resultado_temas_to_json(resultado: ResultadoTemas) -> str:
    """Serializa un ResultadoTemas a JSON pretty-printed."""
    return json.dumps(resultado_temas_to_dict(resultado), indent=2, ensure_ascii=False)


def resultado_temas_from_json(json_str: str) -> ResultadoTemas:
    """Deserializa un ResultadoTemas desde JSON."""
    data = json.loads(json_str)
    return ResultadoTemas(
        segmentos=[_segmento_tematizado_from_dict(s) for s in data.get("segmentos", [])],
        topicos=[_topico_from_dict(t) for t in data.get("topicos", [])],
        num_topicos=data.get("num_topicos", 0),
    )


def guardar_fase1(resultado: ResultadoTemas, run_dir: Path | str) -> Path:
    """Persiste ``fase1-topicos.json`` en ``<run_dir>/fase1-topicos.json``.

    Crea el directorio si no existe. Devuelve la ruta del archivo.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "fase1-topicos.json"
    path.write_text(resultado_temas_to_json(resultado), encoding="utf-8")
    logger.info("Fase 1 guardada: %s", path)
    return path


def cargar_fase1(run_dir: Path | str) -> ResultadoTemas:
    """Carga ``fase1-topicos.json`` desde un directorio de run.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    path = Path(run_dir) / "fase1-topicos.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró fase1-topicos.json en {run_dir}. "
            "Ejecuta discover primero (sin --resume)."
        )
    return resultado_temas_from_json(path.read_text(encoding="utf-8"))
