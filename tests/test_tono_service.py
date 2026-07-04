"""Tests para tono/service.py — TonoService orquestador.

Los tests de lógica pura (mapear_scores, intensidad) no requieren modelos.
Los tests de TonoService son de integración (marca slow).
"""

from __future__ import annotations

import pytest

from tono_politico.filtrado.models import CriterioFiltrado, ResultadoFiltrado, SegmentoFiltrado
from tono_politico.protocol import ComponenteProtocol
from tono_politico.segmentacion.models import Oracion, Segmento
from tono_politico.tono.service import TonoService, mapear_scores


# ============================================================
# HELPERS
# ============================================================
def segmento(texto: str, video_id: str = "vid001") -> Segmento:
    return Segmento(
        texto=texto,
        t_start=0.0,
        t_end=5.0,
        oraciones=[Oracion(texto=texto, t_start=0.0, t_end=5.0, words=[])],
        word_count=len(texto.split()),
        video_id=video_id,
    )


def resultado_filtrado(textos: list[str]) -> ResultadoFiltrado:
    segs = [
        SegmentoFiltrado(segmento=segmento(t), topico_id=0, relevancia=0.8)
        for t in textos
    ]
    return ResultadoFiltrado(
        criterio=CriterioFiltrado(topico_id=0),
        topico=None,
        segmentos=segs,
        total_segmentos_entrada=len(segs),
        total_segmentos_filtrados=len(segs),
    )


# ============================================================
# MAPEAR SCORES — función pura
# ============================================================
class TestMapearScores:
    def test_mapea_logica_politica(self):
        scores = {
            "logica_politica": {
                "nacionalista": 0.55,
                "globalista": 0.40,
                "populista": 0.61,
                "tecnocrata": 0.35,
                "corporativista": 0.48,
                "estatista": 0.50,
            },
            "sentimiento": {
                "esperanza": 0.45,
                "angustia": 0.60,
                "indignacion": 0.52,
                "orgullo": 0.38,
                "empatia": 0.41,
            },
            "estilo_discursivo": {
                "directo": 0.67,
                "academico": 0.45,
                "confrontativo": 0.58,
                "conciliador": 0.52,
                "catastrofista": 0.50,
                "testimonial": 0.43,
            },
            "funcion_discursiva": {
                "critica": 0.55,
                "propuesta": 0.38,
                "narrativa_personal": 0.47,
            },
            "intensidad": {
                "1": 0.40,
                "2": 0.45,
                "3": 0.55,
                "4": 0.50,
                "5": 0.60,
            },
        }
        logica, sent, estilo, func, intensidad = mapear_scores(scores)

        assert logica.populista == 0.61
        assert logica.nacionalista == 0.55
        assert sent.angustia == 0.60
        assert estilo.directo == 0.67
        assert func.critica == 0.55

    def test_intensidad_devuelve_nivel_maximo(self):
        scores = {
            "logica_politica": {k: 0.1 for k in [
                "nacionalista", "globalista", "populista",
                "tecnocrata", "corporativista", "estatista",
            ]},
            "sentimiento": {k: 0.1 for k in [
                "esperanza", "angustia", "indignacion", "orgullo", "empatia",
            ]},
            "estilo_discursivo": {k: 0.1 for k in [
                "directo", "academico", "confrontativo",
                "conciliador", "catastrofista", "testimonial",
            ]},
            "funcion_discursiva": {k: 0.1 for k in [
                "critica", "propuesta", "narrativa_personal",
            ]},
            "intensidad": {"1": 0.3, "2": 0.4, "3": 0.5, "4": 0.8, "5": 0.6},
        }
        _, _, _, _, intensidad = mapear_scores(scores)
        assert intensidad == 4

    def test_intensidad_empate_devuelve_menor(self):
        """En caso de empate, devuelve el nivel menor (más conservador)."""
        scores = {
            "logica_politica": {k: 0.1 for k in [
                "nacionalista", "globalista", "populista",
                "tecnocrata", "corporativista", "estatista",
            ]},
            "sentimiento": {k: 0.1 for k in [
                "esperanza", "angustia", "indignacion", "orgullo", "empatia",
            ]},
            "estilo_discursivo": {k: 0.1 for k in [
                "directo", "academico", "confrontativo",
                "conciliador", "catastrofista", "testimonial",
            ]},
            "funcion_discursiva": {k: 0.1 for k in [
                "critica", "propuesta", "narrativa_personal",
            ]},
            "intensidad": {"1": 0.5, "2": 0.5, "3": 0.5, "4": 0.5, "5": 0.5},
        }
        _, _, _, _, intensidad = mapear_scores(scores)
        assert intensidad == 1


# ============================================================
# TONOSERVICE — constructor y protocol
# ============================================================
class TestTonoServiceInit:
    def test_guarda_config_del_constructor(self):
        svc = TonoService(actor="AMLO", tema="fracking")
        assert svc.actor == "AMLO"
        assert svc.tema == "fracking"

    def test_implementa_componente_protocol(self):
        svc = TonoService(actor="X", tema="Y")
        assert isinstance(svc, ComponenteProtocol)

    def test_procesar_vacio_devuelve_vacio(self):
        svc = TonoService(actor="AMLO", tema="fracking")
        rf = ResultadoFiltrado(
            criterio=CriterioFiltrado(topico_id=0),
            topico=None,
        )
        resultado = svc.procesar(rf)
        assert resultado.tema == "fracking"
        assert resultado.actor == "AMLO"
        assert resultado.segmentos == []


# ============================================================
# TONOSERVICE — integración (requiere modelos)
# ============================================================
@pytest.mark.slow
class TestTonoServiceIntegracion:
    """Tests de integración que cargan modelos reales."""

    def test_procesa_un_segmento_completo(self):
        svc = TonoService(actor="AMLO", tema="fracking")
        rf = resultado_filtrado([
            "No vamos a permitir el fracking porque daña la naturaleza "
            "y contamina los mantos acuíferos."
        ])
        resultado = svc.procesar(rf)

        assert resultado.tema == "fracking"
        assert resultado.actor == "AMLO"
        assert len(resultado.segmentos) == 1

        seg = resultado.segmentos[0]
        assert seg.stance.stance in ("apoyo", "rechazo")
        assert seg.intensidad_antagonica in (1, 2, 3, 4, 5)
        # Lógica política: todos los scores deben ser floats
        assert isinstance(seg.logica_politica.populista, float)
        assert isinstance(seg.sentimiento.angustia, float)
        assert isinstance(seg.estilo_discursivo.directo, float)
        assert isinstance(seg.funcion_discursiva.critica, float)
