"""Serialización de ResultadoTemas (discursive_approach)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..argument_shape.models import Argumento, Oracion
from .models import ArgumentoTematizado, ResultadoTemas, TopicoInfo

logger = logging.getLogger(__name__)


def _oracion_to_dict(o: Oracion) -> dict:
    return {"texto": o.texto, "t_start": o.t_start, "t_end": o.t_end}


def _oracion_from_dict(d: dict) -> Oracion:
    return Oracion(texto=d["texto"], t_start=d["t_start"], t_end=d["t_end"])


def _argumento_to_dict(a: Argumento) -> dict:
    return {
        "texto": a.texto,
        "t_start": a.t_start,
        "t_end": a.t_end,
        "oraciones": [_oracion_to_dict(o) for o in a.oraciones],
        "word_count": a.word_count,
        "video_id": a.video_id,
        "fecha": a.fecha,
    }


def _argumento_from_dict(d: dict) -> Argumento:
    return Argumento(
        texto=d["texto"],
        t_start=d["t_start"],
        t_end=d["t_end"],
        oraciones=[_oracion_from_dict(o) for o in d.get("oraciones", [])],
        word_count=d.get("word_count", 0),
        video_id=d.get("video_id", ""),
        fecha=d.get("fecha"),
    )


def resultado_temas_to_dict(resultado: ResultadoTemas) -> dict:
    return {
        "schema_version": "discursive_resultado_temas.v1",
        "argumentos": [
            {
                "argumento": _argumento_to_dict(a.argumento),
                "topico_id": a.topico_id,
                "probabilidad": a.probabilidad,
            }
            for a in resultado.argumentos
        ],
        "topicos": [
            {
                "id": t.id,
                "nombre": t.nombre,
                "palabras_clave": t.palabras_clave,
                "num_argumentos": t.num_argumentos,
                "representatividad": t.representatividad,
            }
            for t in resultado.topicos
        ],
        "num_topicos": resultado.num_topicos,
    }


def resultado_temas_to_json(resultado: ResultadoTemas) -> str:
    return json.dumps(resultado_temas_to_dict(resultado), indent=2, ensure_ascii=False)


def resultado_temas_from_json(json_str: str) -> ResultadoTemas:
    data = json.loads(json_str)
    return ResultadoTemas(
        argumentos=[
            ArgumentoTematizado(
                argumento=_argumento_from_dict(item["argumento"]),
                topico_id=item["topico_id"],
                probabilidad=item["probabilidad"],
            )
            for item in data.get("argumentos", [])
        ],
        topicos=[
            TopicoInfo(
                id=t["id"],
                nombre=t["nombre"],
                palabras_clave=t.get("palabras_clave", []),
                num_argumentos=t.get("num_argumentos", 0),
                representatividad=t.get("representatividad", 0.0),
            )
            for t in data.get("topicos", [])
        ],
        num_topicos=data.get("num_topicos", 0),
    )


def guardar_resultado_temas(resultado: ResultadoTemas, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resultado_temas_to_json(resultado), encoding="utf-8")
    logger.info("ResultadoTemas guardado: %s", path)
    return path


def cargar_resultado_temas(path: Path | str) -> ResultadoTemas:
    path = Path(path)
    return resultado_temas_from_json(path.read_text(encoding="utf-8"))
