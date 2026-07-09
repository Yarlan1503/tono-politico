"""Tests topics_approach: firmas de tono (sin HDBSCAN)."""

from __future__ import annotations

from tono_politico.discursive_approach.argument_shape.models import Argumento
from tono_politico.discursive_approach.topics_approach.enfoques import (
    descubrir_enfoques_tema,
    firma_de_perfil,
    intensidad_bin,
)
from tono_politico.discursive_approach.topics_approach.models import PerfilTonoArgumento
from tono_politico.discursive_approach.topics_approach.service import TopicsApproachService
from tono_politico.discursive_approach.topics_cluster.models import (
    ArgumentoTematizado,
    ResultadoTemas,
    TopicoInfo,
)


def _perfil(
    stance: str = "rechazo",
    intensidad: int = 4,
    logica: str = "populista",
    sent: str = "indignacion",
    estilo: str = "confrontativo",
    func: str = "critica",
) -> PerfilTonoArgumento:
    return PerfilTonoArgumento(
        stance=stance,
        intensidad=intensidad,
        logica_dominante=logica,
        sentimiento_dominante=sent,
        estilo_dominante=estilo,
        funcion_dominante=func,
    )


def _arg(texto: str, fecha: str, t: float = 0.0) -> Argumento:
    return Argumento(
        texto=texto,
        t_start=t,
        t_end=t + 1.0,
        word_count=3,
        video_id="v",
        fecha=fecha,
    )


class FakeTono:
    def __init__(self, perfiles: list[PerfilTonoArgumento]):
        self.perfiles = perfiles
        self.calls: list[tuple[str, str, int]] = []

    def analizar(self, textos, actor, tema):
        self.calls.append((actor, tema, len(textos)))
        assert len(textos) == len(self.perfiles)
        return list(self.perfiles)


class TestFirmas:
    def test_intensidad_bin(self):
        assert intensidad_bin(1) == "1-2"
        assert intensidad_bin(2) == "1-2"
        assert intensidad_bin(3) == "3"
        assert intensidad_bin(5) == "4-5"

    def test_misma_firma_mismo_enfoque(self):
        topico = TopicoInfo(id=0, nombre="corrupcion", num_argumentos=2)
        p = _perfil()
        items = [
            (ArgumentoTematizado(_arg("a", "20240101", 0.0), 0, 1.0), p),
            (ArgumentoTematizado(_arg("b", "20240201", 0.0), 0, 1.0), p),
        ]
        res = descubrir_enfoques_tema(topico, items)
        assert len(res.enfoques) == 1
        assert res.enfoques[0].num_argumentos == 2
        assert res.enfoques[0].fecha_primera == "20240101"
        assert res.enfoques[0].fecha_ultima == "20240201"
        assert all(a.probabilidad_enfoque == 1.0 for a in res.argumentos)

    def test_firmas_distintas_dos_enfoques(self):
        topico = TopicoInfo(id=1, nombre="economia", num_argumentos=2)
        items = [
            (
                ArgumentoTematizado(_arg("critica", "20240101"), 1, 1.0),
                _perfil(stance="rechazo", func="critica"),
            ),
            (
                ArgumentoTematizado(_arg("propuesta", "20240301"), 1, 1.0),
                _perfil(stance="apoyo", func="propuesta", estilo="directo", intensidad=2),
            ),
        ]
        res = descubrir_enfoques_tema(topico, items)
        assert len(res.enfoques) == 2
        # orden temporal de argumentos
        fecha_0 = res.argumentos[0].argumento.fecha
        fecha_1 = res.argumentos[1].argumento.fecha
        assert fecha_0 is not None
        assert fecha_1 is not None
        assert fecha_0 <= fecha_1

    def test_firma_incluye_bins(self):
        f1 = firma_de_perfil(_perfil(intensidad=4))
        f2 = firma_de_perfil(_perfil(intensidad=5))
        assert f1 == f2  # mismo bin 4-5


class TestTopicsApproachService:
    def test_omite_outliers(self):
        topico = TopicoInfo(id=0, nombre="tema-a", num_argumentos=1)
        outlier = TopicoInfo(id=-1, nombre="Outlier", num_argumentos=1)
        args = [
            ArgumentoTematizado(_arg("ok", "20240101"), 0, 1.0),
            ArgumentoTematizado(_arg("noise", "20240101"), -1, 0.0),
        ]
        resultado = ResultadoTemas(
            argumentos=args,
            topicos=[topico, outlier],
            num_topicos=1,
        )
        fake = FakeTono([_perfil()])
        svc = TopicsApproachService(actor="AMLO", tono_analyzer=fake)
        out = svc.procesar(resultado)
        assert out.num_temas == 1
        assert out.por_tema[0].topico.id == 0
        assert fake.calls[0][1] == "tema-a"
        assert out.num_enfoques_total == 1

    def test_todos_los_temas(self):
        t0 = TopicoInfo(id=0, nombre="A", num_argumentos=1)
        t1 = TopicoInfo(id=1, nombre="B", num_argumentos=1)
        args = [
            ArgumentoTematizado(_arg("a", "20240101"), 0, 1.0),
            ArgumentoTematizado(_arg("b", "20240201"), 1, 1.0),
        ]
        resultado = ResultadoTemas(argumentos=args, topicos=[t0, t1], num_topicos=2)

        class MultiFake:
            def analizar(self, textos, actor, tema):
                return [_perfil(logica="populista" if tema == "A" else "estatista")]

        svc = TopicsApproachService(actor="X", tono_analyzer=MultiFake())
        out = svc.procesar(resultado)
        assert out.num_temas == 2
        assert out.num_enfoques_total == 2
