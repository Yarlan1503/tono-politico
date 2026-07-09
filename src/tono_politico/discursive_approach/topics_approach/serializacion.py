"""Serialización de ResultadoEnfoques."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..argument_shape.models import Argumento, Oracion
from .models import (
    EnfoqueInfo,
    PerfilTonoArgumento,
    ResultadoEnfoques,
)


def _oracion_to_dict(o: Oracion) -> dict[str, Any]:
    return {"texto": o.texto, "t_start": o.t_start, "t_end": o.t_end}


def _argumento_to_dict(a: Argumento) -> dict[str, Any]:
    return {
        "texto": a.texto,
        "t_start": a.t_start,
        "t_end": a.t_end,
        "oraciones": [_oracion_to_dict(o) for o in a.oraciones],
        "word_count": a.word_count,
        "video_id": a.video_id,
        "fecha": a.fecha,
    }


def _perfil_to_dict(p: PerfilTonoArgumento | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return {
        "stance": p.stance,
        "intensidad": p.intensidad,
        "logica_dominante": p.logica_dominante,
        "sentimiento_dominante": p.sentimiento_dominante,
        "estilo_dominante": p.estilo_dominante,
        "funcion_dominante": p.funcion_dominante,
    }


def _enfoque_to_dict(e: EnfoqueInfo) -> dict[str, Any]:
    return {
        "id": e.id,
        "topico_id": e.topico_id,
        "nombre": e.nombre,
        "palabras_clave": e.palabras_clave,
        "num_argumentos": e.num_argumentos,
        "fecha_primera": e.fecha_primera,
        "fecha_ultima": e.fecha_ultima,
        "stance_dominante": e.stance_dominante,
        "intensidad_media": e.intensidad_media,
        "logica_dominante": e.logica_dominante,
        "sentimiento_dominante": e.sentimiento_dominante,
        "estilo_dominante": e.estilo_dominante,
        "funcion_dominante": e.funcion_dominante,
    }


def resultado_enfoques_to_dict(resultado: ResultadoEnfoques) -> dict[str, Any]:
    return {
        "schema_version": "discursive_resultado_enfoques.v1",
        "num_temas": resultado.num_temas,
        "num_enfoques_total": resultado.num_enfoques_total,
        "por_tema": [
            {
                "topico": {
                    "id": tema.topico.id,
                    "nombre": tema.topico.nombre,
                    "palabras_clave": tema.topico.palabras_clave,
                    "num_argumentos": tema.topico.num_argumentos,
                    "representatividad": tema.topico.representatividad,
                },
                "enfoques": [_enfoque_to_dict(e) for e in tema.enfoques],
                "argumentos": [
                    {
                        "argumento": _argumento_to_dict(a.argumento),
                        "topico_id": a.topico_id,
                        "enfoque_id": a.enfoque_id,
                        "probabilidad_topico": a.probabilidad_topico,
                        "probabilidad_enfoque": a.probabilidad_enfoque,
                        "tono": _perfil_to_dict(a.tono),
                    }
                    for a in tema.argumentos
                ],
            }
            for tema in resultado.por_tema
        ],
    }


def resultado_enfoques_to_json(resultado: ResultadoEnfoques) -> str:
    return json.dumps(resultado_enfoques_to_dict(resultado), indent=2, ensure_ascii=False)


def guardar_resultado_enfoques(resultado: ResultadoEnfoques, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(resultado_enfoques_to_json(resultado), encoding="utf-8")
    return path
