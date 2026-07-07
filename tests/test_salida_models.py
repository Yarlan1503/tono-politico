"""Tests para salida/models.py — DTOs del Componente 6."""

from __future__ import annotations

from tono_politico.salida.models import InformeTono, PerfilActor, Provenance


class TestProvenance:
    def test_crear_provenance_completo(self):
        prov = Provenance(
            pipeline="tono-politico v0.1.0",
            modelos=["LFM2.5-Embedding-350M", "LFM2.5-1.2B-Instruct"],
            fecha="2026-07-04T12:00:00",
            marco_teorico="docs/marco-teorico.md",
            advertencia_confianza="Scores de similitud coseno son relativos, no probabilidades.",
        )
        assert prov.pipeline == "tono-politico v0.1.0"
        assert len(prov.modelos) == 2
        assert "LFM2.5-Embedding-350M" in prov.modelos

    def test_provenance_tiene_defaults_razonables(self):
        prov = Provenance(
            pipeline="tono-politico",
            modelos=["m1"],
            fecha="2026-01-01",
        )
        assert prov.marco_teorico != ""
        assert prov.advertencia_confianza != ""


class TestPerfilActor:
    def test_crear_perfil_completo(self):
        perfil = PerfilActor(
            actor="AMLO",
            tema="fracking",
            n_segmentos=5,
            stance_dominante="rechazo",
            intensidad_promedio=3.8,
            logica_dominante="populista",
            sentimiento_dominante="indignacion",
            estilo_dominante="directo",
            funcion_dominante="critica",
        )
        assert perfil.actor == "AMLO"
        assert perfil.tema == "fracking"
        assert perfil.n_segmentos == 5
        assert perfil.stance_dominante == "rechazo"
        assert perfil.intensidad_promedio == 3.8
        assert perfil.logica_dominante == "populista"
        assert perfil.sentimiento_dominante == "indignacion"
        assert perfil.estilo_dominante == "directo"
        assert perfil.funcion_dominante == "critica"


class TestInformeTono:
    def test_crear_informe_minimo(self):
        perfil = PerfilActor(
            actor="X",
            tema="Y",
            n_segmentos=0,
            stance_dominante="apoyo",
            intensidad_promedio=1.0,
            logica_dominante="populista",
            sentimiento_dominante="esperanza",
            estilo_dominante="directo",
            funcion_dominante="propuesta",
        )
        prov = Provenance(
            pipeline="test",
            modelos=[],
            fecha="2026-01-01",
        )
        informe = InformeTono(perfil=perfil, segmentos=[], provenance=prov)
        assert informe.perfil.actor == "X"
        assert informe.segmentos == []
        assert informe.provenance is not None
        assert informe.provenance.pipeline == "test"
