"""Tests para salida/agregacion.py — funciones puras de agregación."""

from __future__ import annotations

import pytest

from tono_politico.salida.agregacion import (
    dimension_dominante,
    generar_perfil,
    intensidad_promedio,
    stance_dominante,
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


def seg_tono(
    texto: str = "texto",
    stance: str = "apoyo",
    confianza: float = 0.8,
    intensidad: int = 3,
    nacionalista: float = 0.5,
    globalista: float = 0.3,
    populista: float = 0.6,
    tecnocrata: float = 0.2,
    corporativista: float = 0.4,
    estatista: float = 0.45,
    esperanza: float = 0.4,
    angustia: float = 0.3,
    indignacion: float = 0.5,
    orgullo: float = 0.35,
    empatia: float = 0.3,
    directo: float = 0.5,
    academico: float = 0.3,
    confrontativo: float = 0.4,
    conciliador: float = 0.35,
    catastrofista: float = 0.3,
    testimonial: float = 0.25,
    critica: float = 0.5,
    propuesta: float = 0.4,
    narrativa_personal: float = 0.3,
) -> SegmentoConTono:
    """Factory de SegmentoConTono con defaults personalizables."""
    seg = Segmento(
        texto=texto,
        t_start=0.0,
        t_end=5.0,
        oraciones=[Oracion(texto=texto, t_start=0.0, t_end=5.0, words=[])],
        word_count=len(texto.split()),
    )
    return SegmentoConTono(
        segmento=seg,
        stance=ResultadoStance(stance=stance, confianza=confianza),
        intensidad_antagonica=intensidad,
        logica_politica=ResultadoLogicaPolitica(
            nacionalista=nacionalista,
            globalista=globalista,
            populista=populista,
            tecnocrata=tecnocrata,
            corporativista=corporativista,
            estatista=estatista,
        ),
        sentimiento=ResultadoSentimiento(
            esperanza=esperanza,
            angustia=angustia,
            indignacion=indignacion,
            orgullo=orgullo,
            empatia=empatia,
        ),
        estilo_discursivo=ResultadoEstiloDiscursivo(
            directo=directo,
            academico=academico,
            confrontativo=confrontativo,
            conciliador=conciliador,
            catastrofista=catastrofista,
            testimonial=testimonial,
        ),
        funcion_discursiva=ResultadoFuncionDiscursiva(
            critica=critica,
            propuesta=propuesta,
            narrativa_personal=narrativa_personal,
        ),
    )


# ============================================================
# STANCE DOMINANTE
# ============================================================
class TestStanceDominante:
    def test_mayoria_rechazo(self):
        segmentos = [
            seg_tono(stance="rechazo"),
            seg_tono(stance="rechazo"),
            seg_tono(stance="apoyo"),
        ]
        assert stance_dominante(segmentos) == "rechazo"

    def test_mayoria_apoyo(self):
        segmentos = [
            seg_tono(stance="apoyo"),
            seg_tono(stance="apoyo"),
            seg_tono(stance="rechazo"),
        ]
        assert stance_dominante(segmentos) == "apoyo"

    def test_empate_devuelve_apoyo(self):
        """En caso de empate, devuelve 'apoyo' (default conservador)."""
        segmentos = [seg_tono(stance="apoyo"), seg_tono(stance="rechazo")]
        assert stance_dominante(segmentos) == "apoyo"

    def test_lista_vacia_devuelve_apoyo(self):
        assert stance_dominante([]) == "apoyo"


# ============================================================
# INTENSIDAD PROMEDIO
# ============================================================
class TestIntensidadPromedio:
    def test_promedio_simple(self):
        segmentos = [seg_tono(intensidad=2), seg_tono(intensidad=4)]
        assert intensidad_promedio(segmentos) == 3.0

    def test_un_solo_segmento(self):
        segmentos = [seg_tono(intensidad=5)]
        assert intensidad_promedio(segmentos) == 5.0

    def test_lista_vacia_devuelve_cero(self):
        assert intensidad_promedio([]) == 0.0


# ============================================================
# DIMENSION DOMINANTE
# ============================================================
class TestDimensionDominante:
    def test_logica_politica_dominante(self):
        segmentos = [
            seg_tono(populista=0.7, nacionalista=0.4),
            seg_tono(populista=0.6, nacionalista=0.5),
        ]
        result = dimension_dominante(segmentos, "logica_politica")
        assert result == "populista"

    def test_sentimiento_dominante(self):
        segmentos = [
            seg_tono(indignacion=0.8, esperanza=0.3),
            seg_tono(indignacion=0.7, esperanza=0.4),
        ]
        result = dimension_dominante(segmentos, "sentimiento")
        assert result == "indignacion"

    def test_estilo_dominante(self):
        segmentos = [
            seg_tono(directo=0.8, academico=0.3),
            seg_tono(directo=0.7, academico=0.4),
        ]
        result = dimension_dominante(segmentos, "estilo_discursivo")
        assert result == "directo"

    def test_funcion_dominante(self):
        segmentos = [
            seg_tono(critica=0.7, propuesta=0.3),
            seg_tono(critica=0.6, propuesta=0.4),
        ]
        result = dimension_dominante(segmentos, "funcion_discursiva")
        assert result == "critica"

    def test_lista_varia_lanza_value_error(self):
        with pytest.raises(ValueError):
            dimension_dominante([], "logica_politica")


# ============================================================
# GENERAR PERFIL
# ============================================================
class TestGenerarPerfil:
    def test_genera_perfil_completo(self):
        segmentos = [
            seg_tono(stance="rechazo", intensidad=4, populista=0.7),
            seg_tono(stance="rechazo", intensidad=5, populista=0.6),
        ]
        resultado_tono = ResultadoTono(
            tema="fracking", actor="AMLO", segmentos=segmentos
        )
        perfil = generar_perfil(resultado_tono)

        assert perfil.actor == "AMLO"
        assert perfil.tema == "fracking"
        assert perfil.n_segmentos == 2
        assert perfil.stance_dominante == "rechazo"
        assert perfil.intensidad_promedio == 4.5
        assert perfil.logica_dominante == "populista"

    def test_genera_perfil_sin_segmentos(self):
        resultado_tono = ResultadoTono(tema="X", actor="Y")
        perfil = generar_perfil(resultado_tono)
        assert perfil.n_segmentos == 0
        assert perfil.intensidad_promedio == 0.0
