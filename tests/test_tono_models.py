"""Tests para el Componente 5: Tono — DTOs (models.py)."""

from __future__ import annotations

from tono_politico.tono.models import (
    EtiquetaScore,
    ResultadoEstiloDiscursivo,
    ResultadoFuncionDiscursiva,
    ResultadoLogicaPolitica,
    ResultadoSentimiento,
    ResultadoStance,
    ResultadoTono,
    SegmentoConTono,
)


class TestEtiquetaScore:
    def test_crear_etiqueta_con_score(self):
        et = EtiquetaScore(etiqueta="populista", score=0.72)
        assert et.etiqueta == "populista"
        assert et.score == 0.72

    def test_score_acepta_cero(self):
        et = EtiquetaScore(etiqueta="tecnocrata", score=0.0)
        assert et.score == 0.0


class TestResultadoLogicaPolitica:
    def test_contiene_seis_etiquetas(self):
        resultado = ResultadoLogicaPolitica(
            nacionalista=0.55,
            globalista=0.40,
            populista=0.61,
            tecnocrata=0.35,
            corporativista=0.48,
            estatista=0.50,
        )
        assert resultado.nacionalista == 0.55
        assert resultado.globalista == 0.40
        assert resultado.populista == 0.61
        assert resultado.tecnocrata == 0.35
        assert resultado.corporativista == 0.48
        assert resultado.estatista == 0.50

    def test_to_scores_devuelve_lista_ordenada(self):
        resultado = ResultadoLogicaPolitica(
            nacionalista=0.55,
            globalista=0.40,
            populista=0.61,
            tecnocrata=0.35,
            corporativista=0.48,
            estatista=0.50,
        )
        scores = resultado.to_scores()
        assert len(scores) == 6
        assert all(isinstance(s, EtiquetaScore) for s in scores)
        etiquetas = [s.etiqueta for s in scores]
        assert "nacionalista" in etiquetas
        assert "populista" in etiquetas

    def test_dominante_devuelve_etiqueta_mayor(self):
        resultado = ResultadoLogicaPolitica(
            nacionalista=0.40,
            globalista=0.70,
            populista=0.30,
            tecnocrata=0.20,
            corporativista=0.50,
            estatista=0.35,
        )
        dom = resultado.dominante()
        assert dom.etiqueta == "globalista"
        assert dom.score == 0.70


class TestResultadoSentimiento:
    def test_contiene_cinco_emociones(self):
        resultado = ResultadoSentimiento(
            esperanza=0.45,
            angustia=0.60,
            indignacion=0.52,
            orgullo=0.38,
            empatia=0.41,
        )
        assert resultado.esperanza == 0.45
        assert resultado.angustia == 0.60
        assert resultado.indignacion == 0.52
        assert resultado.orgullo == 0.38
        assert resultado.empatia == 0.41

    def test_dominante_devuelve_emocion_mayor(self):
        resultado = ResultadoSentimiento(
            esperanza=0.30,
            angustia=0.65,
            indignacion=0.50,
            orgullo=0.40,
            empatia=0.35,
        )
        assert resultado.dominante().etiqueta == "angustia"


class TestResultadoEstiloDiscursivo:
    def test_contiene_seis_estilos(self):
        resultado = ResultadoEstiloDiscursivo(
            directo=0.67,
            academico=0.45,
            confrontativo=0.58,
            conciliador=0.52,
            catastrofista=0.50,
            testimonial=0.43,
        )
        assert resultado.directo == 0.67
        assert resultado.academico == 0.45
        assert resultado.confrontativo == 0.58
        assert resultado.conciliador == 0.52
        assert resultado.catastrofista == 0.50
        assert resultado.testimonial == 0.43

    def test_dominante_devuelve_estilo_mayor(self):
        resultado = ResultadoEstiloDiscursivo(
            directo=0.45,
            academico=0.51,
            confrontativo=0.58,
            conciliador=0.40,
            catastrofista=0.35,
            testimonial=0.30,
        )
        assert resultado.dominante().etiqueta == "confrontativo"


class TestResultadoFuncionDiscursiva:
    def test_contiene_tres_funciones(self):
        resultado = ResultadoFuncionDiscursiva(
            critica=0.55,
            propuesta=0.38,
            narrativa_personal=0.47,
        )
        assert resultado.critica == 0.55
        assert resultado.propuesta == 0.38
        assert resultado.narrativa_personal == 0.47

    def test_dominante_devuelve_funcion_mayor(self):
        resultado = ResultadoFuncionDiscursiva(
            critica=0.30,
            propuesta=0.65,
            narrativa_personal=0.40,
        )
        assert resultado.dominante().etiqueta == "propuesta"


class TestResultadoStance:
    def test_crea_stance_con_valores(self):
        resultado = ResultadoStance(stance="apoyo", confianza=0.85)
        assert resultado.stance == "apoyo"
        assert resultado.confianza == 0.85

    def test_stance_valido_solo_apoyo_o_rechazo(self):
        ResultadoStance(stance="apoyo", confianza=0.7)
        ResultadoStance(stance="rechazo", confianza=0.7)


class TestSegmentoConTono:
    def test_crear_segmento_con_tono_completo(self):
        from tono_politico.segmentacion.models import Segmento

        seg = Segmento(texto="El pueblo exige justicia.", t_start=0.0, t_end=5.0)
        stance = ResultadoStance(stance="rechazo", confianza=0.82)
        intensidad = 4
        logica = ResultadoLogicaPolitica(
            nacionalista=0.55,
            globalista=0.40,
            populista=0.61,
            tecnocrata=0.35,
            corporativista=0.48,
            estatista=0.50,
        )
        sentimiento = ResultadoSentimiento(
            esperanza=0.30,
            angustia=0.65,
            indignacion=0.50,
            orgullo=0.40,
            empatia=0.35,
        )
        estilo = ResultadoEstiloDiscursivo(
            directo=0.67,
            academico=0.45,
            confrontativo=0.58,
            conciliador=0.52,
            catastrofista=0.50,
            testimonial=0.43,
        )
        funcion = ResultadoFuncionDiscursiva(
            critica=0.55,
            propuesta=0.38,
            narrativa_personal=0.47,
        )

        sct = SegmentoConTono(
            segmento=seg,
            stance=stance,
            intensidad_antagonica=intensidad,
            logica_politica=logica,
            sentimiento=sentimiento,
            estilo_discursivo=estilo,
            funcion_discursiva=funcion,
        )

        assert sct.segmento == seg
        assert sct.stance.stance == "rechazo"
        assert sct.intensidad_antagonica == 4
        assert sct.logica_politica.populista == 0.61
        assert sct.sentimiento.angustia == 0.65
        assert sct.estilo_discursivo.directo == 0.67
        assert sct.funcion_discursiva.critica == 0.55


class TestResultadoTono:
    def test_crear_resultado_vacio(self):
        resultado = ResultadoTono(
            tema="fracking",
            actor="AMLO",
            segmentos=[],
        )
        assert resultado.tema == "fracking"
        assert resultado.actor == "AMLO"
        assert resultado.segmentos == []

    def test_crear_resultado_con_segmentos(self):
        from tono_politico.segmentacion.models import Segmento

        seg = Segmento(texto="El pueblo exige justicia.", t_start=0.0, t_end=5.0)
        stance = ResultadoStance(stance="apoyo", confianza=0.90)
        logica = ResultadoLogicaPolitica(
            nacionalista=0.50,
            globalista=0.30,
            populista=0.60,
            tecnocrata=0.20,
            corporativista=0.35,
            estatista=0.45,
        )
        sentimiento = ResultadoSentimiento(
            esperanza=0.55,
            angustia=0.30,
            indignacion=0.40,
            orgullo=0.50,
            empatia=0.35,
        )
        estilo = ResultadoEstiloDiscursivo(
            directo=0.60,
            academico=0.30,
            confrontativo=0.50,
            conciliador=0.45,
            catastrofista=0.35,
            testimonial=0.40,
        )
        funcion = ResultadoFuncionDiscursiva(
            critica=0.40,
            propuesta=0.60,
            narrativa_personal=0.30,
        )

        sct = SegmentoConTono(
            segmento=seg,
            stance=stance,
            intensidad_antagonica=3,
            logica_politica=logica,
            sentimiento=sentimiento,
            estilo_discursivo=estilo,
            funcion_discursiva=funcion,
        )

        resultado = ResultadoTono(
            tema="fracking",
            actor="AMLO",
            segmentos=[sct],
        )

        assert len(resultado.segmentos) == 1
        assert resultado.segmentos[0].stance.stance == "apoyo"
        assert resultado.segmentos[0].intensidad_antagonica == 3
