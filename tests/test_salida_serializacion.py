"""Tests para salida/serializacion.py — JSON y Markdown."""

from __future__ import annotations

import json

from tono_politico.salida.models import PerfilActor, Provenance
from tono_politico.salida.serializacion import (
    generar_json,
    generar_markdown,
    perfil_a_dict,
    segmento_a_dict,
)
from tono_politico.segmentacion.models import Oracion, Segmento
from tono_politico.tono.models import (
    ResultadoEstiloDiscursivo,
    ResultadoFuncionDiscursiva,
    ResultadoLogicaPolitica,
    ResultadoSentimiento,
    ResultadoStance,
    ResultadoTono,
    SegmentoConTono,
)


def _seg_tono(texto: str = "El pueblo exige justicia.") -> SegmentoConTono:
    seg = Segmento(
        texto=texto, t_start=0.0, t_end=5.0,
        oraciones=[Oracion(texto=texto, t_start=0.0, t_end=5.0, words=[])],
        word_count=4, video_id="v1",
    )
    return SegmentoConTono(
        segmento=seg,
        stance=ResultadoStance(stance="rechazo", confianza=0.85),
        intensidad_antagonica=4,
        logica_politica=ResultadoLogicaPolitica(
            nacionalista=0.55, globalista=0.30, populista=0.68,
            tecnocrata=0.25, corporativista=0.40, estatista=0.50,
        ),
        sentimiento=ResultadoSentimiento(
            esperanza=0.30, angustia=0.65, indignacion=0.55,
            orgullo=0.35, empatia=0.40,
        ),
        estilo_discursivo=ResultadoEstiloDiscursivo(
            directo=0.67, academico=0.35, confrontativo=0.58,
            conciliador=0.40, catastrofista=0.45, testimonial=0.30,
        ),
        funcion_discursiva=ResultadoFuncionDiscursiva(
            critica=0.60, propuesta=0.35, narrativa_personal=0.30,
        ),
    )


def _perfil() -> PerfilActor:
    return PerfilActor(
        actor="AMLO", tema="fracking", n_segmentos=1,
        stance_dominante="rechazo", intensidad_promedio=4.0,
        logica_dominante="populista", sentimiento_dominante="angustia",
        estilo_dominante="directo", funcion_dominante="critica",
    )


def _provenance() -> Provenance:
    return Provenance(
        pipeline="tono-politico v0.1.0",
        modelos=["LFM2.5-Embedding-350M", "LFM2.5-1.2B-Instruct"],
        fecha="2026-07-04T12:00:00",
    )


# ============================================================
# PERFIL A DICT
# ============================================================
class TestPerfilADict:
    def test_devuelve_dict_serializable(self):
        d = perfil_a_dict(_perfil())
        assert isinstance(d, dict)
        assert d["actor"] == "AMLO"
        assert d["tema"] == "fracking"
        assert d["stance_dominante"] == "rechazo"
        assert d["n_segmentos"] == 1

    def test_es_json_serializable(self):
        d = perfil_a_dict(_perfil())
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ============================================================
# SEGMENTO A DICT
# ============================================================
class TestSegmentoADict:
    def test_devuelve_dict_con_texto_y_scores(self):
        d = segmento_a_dict(_seg_tono())
        assert isinstance(d, dict)
        assert "texto" in d
        assert "stance" in d
        assert "logica_politica" in d
        assert "sentimiento" in d
        assert d["stance"]["stance"] == "rechazo"

    def test_es_json_serializable(self):
        d = segmento_a_dict(_seg_tono())
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ============================================================
# GENERAR JSON
# ============================================================
class TestGenerarJSON:
    def test_devuelve_json_valido(self):
        seg = _seg_tono()
        resultado = ResultadoTono(
            tema="fracking", actor="AMLO", segmentos=[seg]
        )
        perfil = _perfil()
        prov = _provenance()

        json_str = generar_json(resultado, perfil, prov)
        assert isinstance(json_str, str)

        data = json.loads(json_str)
        assert data["perfil"]["actor"] == "AMLO"
        assert data["provenance"]["pipeline"] == "tono-politico v0.1.0"
        assert len(data["segmentos"]) == 1
        assert data["segmentos"][0]["stance"]["stance"] == "rechazo"

    def test_json_vacio_si_no_hay_segmentos(self):
        resultado = ResultadoTono(tema="X", actor="Y")
        perfil = PerfilActor(
            actor="Y", tema="X", n_segmentos=0,
            stance_dominante="apoyo", intensidad_promedio=0.0,
            logica_dominante="populista", sentimiento_dominante="esperanza",
            estilo_dominante="directo", funcion_dominante="propuesta",
        )
        json_str = generar_json(resultado, perfil, _provenance())
        data = json.loads(json_str)
        assert data["perfil"]["n_segmentos"] == 0
        assert data["segmentos"] == []


# ============================================================
# GENERAR MARKDOWN
# ============================================================
class TestGenerarMarkdown:
    def test_contiene_titulo_con_actor_y_tema(self):
        resultado = ResultadoTono(tema="fracking", actor="AMLO")
        md = generar_markdown(resultado, _perfil(), _provenance())
        assert "# AMLO — fracking" in md

    def test_contiene_tabla_de_perfil(self):
        resultado = ResultadoTono(tema="fracking", actor="AMLO")
        md = generar_markdown(resultado, _perfil(), _provenance())
        assert "Stance" in md
        assert "rechazo" in md
        assert "Intensidad" in md
        assert "populista" in md

    def test_contiene_provenance(self):
        resultado = ResultadoTono(tema="fracking", actor="AMLO")
        md = generar_markdown(resultado, _perfil(), _provenance())
        assert "Provenance" in md or "provenance" in md.lower()
        assert "LFM2.5" in md

    def test_contiene_advertencia(self):
        resultado = ResultadoTono(tema="fracking", actor="AMLO")
        md = generar_markdown(resultado, _perfil(), _provenance())
        assert "relativ" in md.lower() or "confianza" in md.lower()
