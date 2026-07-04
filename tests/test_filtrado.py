"""Tests para el Componente 4: Filtrado."""

from __future__ import annotations

from tono_politico.filtrado import CriterioFiltrado, FiltradoService, filtrar_por_topico
from tono_politico.protocol import ComponenteProtocol
from tono_politico.segmentacion.models import Oracion, Segmento
from tono_politico.temas.models import ResultadoTemas, SegmentoTematizado, TopicoInfo


def segmento(texto: str, video_id: str = "vid001") -> Segmento:
    """Crea un Segmento mínimo para pruebas."""
    return Segmento(
        texto=texto,
        t_start=0.0,
        t_end=5.0,
        oraciones=[Oracion(texto=texto, t_start=0.0, t_end=5.0, words=[])],
        word_count=len(texto.split()),
        video_id=video_id,
    )


def resultado_temas_ejemplo() -> ResultadoTemas:
    """ResultadoTemas con dos tópicos y distintas relevancias."""
    seg_economia_alta = segmento("La economía crece por inversión pública.")
    seg_economia_baja = segmento("El presupuesto tendrá ajustes menores.")
    seg_seguridad = segmento("La seguridad pública exige coordinación federal.")

    return ResultadoTemas(
        segmentos=[
            SegmentoTematizado(
                segmento=seg_economia_alta,
                topico_id=0,
                probabilidad=0.82,
            ),
            SegmentoTematizado(
                segmento=seg_economia_baja,
                topico_id=0,
                probabilidad=0.30,
            ),
            SegmentoTematizado(
                segmento=seg_seguridad,
                topico_id=1,
                probabilidad=0.91,
            ),
        ],
        topicos=[
            TopicoInfo(
                id=0,
                nombre="economía inversión presupuesto",
                palabras_clave=["economía", "inversión", "presupuesto"],
                num_segmentos=2,
                representatividad=2 / 3,
            ),
            TopicoInfo(
                id=1,
                nombre="seguridad coordinación federal",
                palabras_clave=["seguridad", "coordinación", "federal"],
                num_segmentos=1,
                representatividad=1 / 3,
            ),
        ],
        num_topicos=2,
    )


def resultado_temas_con_outlier() -> ResultadoTemas:
    """ResultadoTemas con un segmento outlier (-1)."""
    base = resultado_temas_ejemplo()
    seg_outlier = segmento("Comentario lateral sin tema claro.")

    return ResultadoTemas(
        segmentos=[
            *base.segmentos,
            SegmentoTematizado(
                segmento=seg_outlier,
                topico_id=-1,
                probabilidad=0.65,
            ),
        ],
        topicos=[
            *base.topicos,
            TopicoInfo(
                id=-1,
                nombre="Outlier",
                palabras_clave=[],
                num_segmentos=1,
                representatividad=0.25,
            ),
        ],
        num_topicos=base.num_topicos,
    )


class TestFiltrarPorTopico:
    def test_filtra_segmentos_por_topico_y_relevancia_minima(self):
        resultado_temas = resultado_temas_ejemplo()
        criterio = CriterioFiltrado(topico_id=0, min_relevancia=0.35)

        resultado = filtrar_por_topico(resultado_temas, criterio)

        assert resultado.criterio == criterio
        assert resultado.topico == resultado_temas.topicos[0]
        assert resultado.total_segmentos_entrada == 3
        assert resultado.total_segmentos_filtrados == 1
        assert [s.segmento.texto for s in resultado.segmentos] == [
            "La economía crece por inversión pública."
        ]
        assert resultado.segmentos[0].topico_id == 0
        assert resultado.segmentos[0].relevancia == 0.82

    def test_excluye_outliers_por_default(self):
        resultado_temas = resultado_temas_con_outlier()
        criterio = CriterioFiltrado(topico_id=-1, min_relevancia=0.0)

        resultado = filtrar_por_topico(resultado_temas, criterio)

        assert resultado.topico is not None
        assert resultado.topico.id == -1
        assert resultado.total_segmentos_entrada == 4
        assert resultado.total_segmentos_filtrados == 0
        assert resultado.segmentos == []

    def test_permite_outliers_cuando_se_solicitan_explicitamente(self):
        resultado_temas = resultado_temas_con_outlier()
        criterio = CriterioFiltrado(
            topico_id=-1,
            min_relevancia=0.0,
            incluir_outliers=True,
        )

        resultado = filtrar_por_topico(resultado_temas, criterio)

        assert resultado.total_segmentos_entrada == 4
        assert resultado.total_segmentos_filtrados == 1
        assert [s.segmento.texto for s in resultado.segmentos] == [
            "Comentario lateral sin tema claro."
        ]
        assert resultado.segmentos[0].topico_id == -1
        assert resultado.segmentos[0].relevancia == 0.65


class TestFiltradoService:
    def test_init_guarda_config(self):
        svc = FiltradoService(
            topico_id=1,
            min_relevancia=0.5,
            incluir_outliers=True,
        )

        assert svc.topico_id == 1
        assert svc.min_relevancia == 0.5
        assert svc.incluir_outliers is True

    def test_procesar_filtra_con_config_del_constructor(self):
        svc = FiltradoService(topico_id=1, min_relevancia=0.9)

        assert isinstance(svc, ComponenteProtocol)

        resultado = svc.procesar(resultado_temas_ejemplo())

        assert resultado.criterio == CriterioFiltrado(
            topico_id=1,
            min_relevancia=0.9,
            incluir_outliers=False,
        )
        assert resultado.topico is not None
        assert resultado.topico.id == 1
        assert resultado.total_segmentos_entrada == 3
        assert resultado.total_segmentos_filtrados == 1
        assert [s.segmento.texto for s in resultado.segmentos] == [
            "La seguridad pública exige coordinación federal."
        ]
