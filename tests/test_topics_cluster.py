"""Tests topics_cluster (discursive_approach)."""

from __future__ import annotations

from tono_politico.discursive_approach.argument_shape.models import Argumento
from tono_politico.discursive_approach.topics_cluster.descubrimiento import (
    _resultado_topico_unico,
    descubrir_temas,
)
from tono_politico.discursive_approach.topics_cluster.serializacion import (
    resultado_temas_from_json,
    resultado_temas_to_json,
)
from tono_politico.discursive_approach.topics_cluster.service import TopicsClusterService


def _arg(texto: str, video_id: str = "v", fecha: str = "20240101") -> Argumento:
    return Argumento(
        texto=texto,
        t_start=0.0,
        t_end=1.0,
        oraciones=[],
        word_count=len(texto.split()),
        video_id=video_id,
        fecha=fecha,
    )


class TestDescubrimientoSinBertopic:
    def test_dataset_pequeno_topico_unico(self):
        args = [_arg("uno"), _arg("dos")]
        res = _resultado_topico_unico(args)
        assert res.num_topicos == 1
        assert all(a.topico_id == 0 for a in res.argumentos)
        assert res.topicos[0].num_argumentos == 2

    def test_descubrir_temas_dataset_pequeno_sin_cargar_bertopic(self):
        class Boom:
            def encode(self, texts):
                raise AssertionError("no debería embedir")

        args = [_arg("a"), _arg("b")]
        res = descubrir_temas(args, Boom(), min_topic_size=3)
        assert res.num_topicos == 1


class TestSerializacion:
    def test_roundtrip(self):
        args = [_arg("texto de prueba", fecha="20240501")]
        res = _resultado_topico_unico(args)
        loaded = resultado_temas_from_json(resultado_temas_to_json(res))
        assert loaded.num_topicos == 1
        assert loaded.argumentos[0].argumento.fecha == "20240501"
        assert loaded.argumentos[0].argumento.texto == "texto de prueba"


class TestTopicsClusterService:
    def test_vacio(self):
        svc = TopicsClusterService()
        res = svc.procesar([])
        assert res.num_topicos == 0
        assert res.argumentos == []

    def test_pequeno_sin_modelo(self):
        svc = TopicsClusterService(min_topic_size=5)
        # force empty embedder path via small data — no model load
        res = svc.procesar([_arg("x"), _arg("y")])
        assert res.num_topicos == 1
