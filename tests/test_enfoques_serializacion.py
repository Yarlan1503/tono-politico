"""Serialización de ResultadoEnfoques."""

from __future__ import annotations

from pathlib import Path

from tono_politico.discursive_approach.argument_shape.models import Argumento
from tono_politico.discursive_approach.topics_approach.models import (
    ArgumentoConEnfoque,
    EnfoqueInfo,
    PerfilTonoArgumento,
    ResultadoEnfoques,
    ResultadoEnfoquesTema,
)
from tono_politico.discursive_approach.topics_approach.serializacion import (
    guardar_resultado_enfoques,
    resultado_enfoques_to_dict,
)
from tono_politico.discursive_approach.topics_cluster.models import TopicoInfo


def test_resultado_enfoques_to_dict_y_guardar(tmp_path: Path) -> None:
    topico = TopicoInfo(id=0, nombre="seguridad", num_argumentos=1)
    arg = Argumento(
        texto="texto",
        t_start=0.0,
        t_end=1.0,
        word_count=1,
        video_id="v1",
        fecha="20240101",
    )
    perfil = PerfilTonoArgumento(
        stance="rechazo",
        intensidad=4,
        logica_dominante="populista",
        sentimiento_dominante="indignacion",
        estilo_dominante="confrontativo",
        funcion_dominante="critica",
    )
    resultado = ResultadoEnfoques(
        por_tema=[
            ResultadoEnfoquesTema(
                topico=topico,
                enfoques=[
                    EnfoqueInfo(
                        id=0,
                        topico_id=0,
                        nombre="rechazo|populista|indignacion|confrontativo|critica|alta",
                        num_argumentos=1,
                        fecha_primera="20240101",
                        fecha_ultima="20240101",
                        stance_dominante="rechazo",
                        intensidad_media=4.0,
                        logica_dominante="populista",
                        sentimiento_dominante="indignacion",
                        estilo_dominante="confrontativo",
                        funcion_dominante="critica",
                    )
                ],
                argumentos=[
                    ArgumentoConEnfoque(
                        argumento=arg,
                        topico_id=0,
                        enfoque_id=0,
                        probabilidad_topico=1.0,
                        probabilidad_enfoque=1.0,
                        tono=perfil,
                    )
                ],
            )
        ],
        num_temas=1,
        num_enfoques_total=1,
    )

    data = resultado_enfoques_to_dict(resultado)
    assert data["schema_version"] == "discursive_resultado_enfoques.v1"
    assert data["num_temas"] == 1
    assert data["por_tema"][0]["enfoques"][0]["stance_dominante"] == "rechazo"

    path = guardar_resultado_enfoques(resultado, tmp_path / "enfoques.json")
    assert path.exists()
    assert "discursive_resultado_enfoques" in path.read_text(encoding="utf-8")
