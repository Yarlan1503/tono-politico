"""Tests para TemasService y descubrimiento de temas."""

from __future__ import annotations

from unittest.mock import patch

from tono_politico.segmentacion.models import Oracion, Segmento
from tono_politico.temas.models import ResultadoTemas, SegmentoTematizado, TopicoInfo
from tono_politico.temas.service import TemasService

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def segmento(texto: str, video_id: str = "vid001") -> Segmento:
    """Crea un Segmento mínimo."""
    return Segmento(
        texto=texto,
        t_start=0.0,
        t_end=5.0,
        oraciones=[
            Oracion(texto=texto, t_start=0.0, t_end=5.0, words=[]),
        ],
        word_count=3,
        video_id=video_id,
    )


def segmentos_ejemplo() -> list[Segmento]:
    """6 segmentos: 3 sobre economía, 3 sobre deporte."""
    return [
        segmento("La economía crece este año significativamente."),
        segmento("El PIB aumentó y las exportaciones subieron."),
        segmento("Los indicadores económicos son positivos ahora."),
        segmento("El fútbol fue emocionante en el estadio."),
        segmento("El gol decisivo cambió el partido completo."),
        segmento("Los jugadores celebraron la victoria agónicamente."),
    ]


# ──────────────────────────────────────────────────────────
# Tests: DTOs
# ──────────────────────────────────────────────────────────

class TestDTOs:
    def test_segmento_tematizado(self):
        st = SegmentoTematizado(
            segmento=segmento("test"),
            topico_id=0,
            probabilidad=0.85,
        )
        assert st.topico_id == 0
        assert st.probabilidad == 0.85

    def test_topico_info_defaults(self):
        ti = TopicoInfo(id=0, nombre="economía")
        assert ti.palabras_clave == []
        assert ti.num_segmentos == 0
        assert ti.representatividad == 0.0

    def test_resultado_temas_defaults(self):
        rt = ResultadoTemas()
        assert rt.segmentos == []
        assert rt.topicos == []
        assert rt.num_topicos == 0


# ──────────────────────────────────────────────────────────
# Tests: TemasService init
# ──────────────────────────────────────────────────────────

class TestTemasServiceInit:
    def test_init_guarda_config(self):
        svc = TemasService(
            min_topic_size=2,
            n_neighbors=5,
            n_components=3,
        )
        assert svc.min_topic_size == 2
        assert svc.n_neighbors == 5
        assert svc.n_components == 3

    def test_init_defaults(self):
        svc = TemasService()
        assert svc.min_topic_size == 3
        assert svc.n_neighbors == 10
        assert svc.n_components == 5
        assert svc.embedding_model_name == "LiquidAI/LFM2.5-Embedding-350M"

    def test_cumple_componente_protocol(self):
        from tono_politico.protocol import ComponenteProtocol

        svc = TemasService()
        assert isinstance(svc, ComponenteProtocol)


# ──────────────────────────────────────────────────────────
# Tests: procesar (con BERTopic mockeado)
# ──────────────────────────────────────────────────────────

class TestProcesar:
    def test_input_vacio_devuelve_vacio(self):
        """Sin segmentos devuelve ResultadoTemas vacío."""
        svc = TemasService()
        resultado = svc.procesar([])
        assert resultado.num_topicos == 0
        assert resultado.segmentos == []

    def test_pocos_segmentos_todo_outlier(self):
        """Menos segmentos que min_topic_size → todos outlier (-1)."""
        svc = TemasService(min_topic_size=10)
        segs = [segmento("Uno."), segmento("Dos.")]

        with patch.object(svc, "_get_embedder"):
            resultado = svc.procesar(segs)

        assert resultado.num_topicos == 0
        assert len(resultado.segmentos) == 2
        assert all(s.topico_id == -1 for s in resultado.segmentos)

    def test_descubre_dos_topicos(self):
        """6 segmentos (2 grupos) → BERTopic encuentra 2 tópicos."""
        segs = segmentos_ejemplo()
        svc = TemasService(min_topic_size=2)

        # Mock del resultado de descubrir_temas
        resultado_mock = ResultadoTemas(
            segmentos=[
                SegmentoTematizado(segmento=s, topico_id=0, probabilidad=0.9)
                for s in segs[:3]
            ] + [
                SegmentoTematizado(segmento=s, topico_id=1, probabilidad=0.85)
                for s in segs[3:]
            ],
            topicos=[
                TopicoInfo(
                    id=0,
                    nombre="economía crecimiento pib",
                    palabras_clave=["economía", "crecimiento", "pib"],
                    num_segmentos=3,
                    representatividad=0.5,
                ),
                TopicoInfo(
                    id=1,
                    nombre="fútbol gol partido",
                    palabras_clave=["fútbol", "gol", "partido"],
                    num_segmentos=3,
                    representatividad=0.5,
                ),
            ],
            num_topicos=2,
        )

        with patch.object(svc, "_get_embedder"), \
             patch(
                 "tono_politico.temas.service.descubrir_temas",
                 return_value=resultado_mock,
             ):
            resultado = svc.procesar(segs)

        assert resultado.num_topicos == 2
        assert len(resultado.segmentos) == 6
        assert len(resultado.topicos) == 2

        # Verificar asignación de tópicos
        topicos_seg_0_2 = {s.topico_id for s in resultado.segmentos[:3]}
        topicos_seg_3_5 = {s.topico_id for s in resultado.segmentos[3:]}
        assert topicos_seg_0_2 == {0}
        assert topicos_seg_3_5 == {1}

    def test_propaga_video_id(self):
        """Los segmentos tematizados conservan su video_id."""
        segs = [
            segmento("Texto A.", video_id="vid_a"),
            segmento("Texto B.", video_id="vid_b"),
        ]
        svc = TemasService(min_topic_size=10)

        with patch.object(svc, "_get_embedder"):
            resultado = svc.procesar(segs)

        ids = {st.segmento.video_id for st in resultado.segmentos}
        assert ids == {"vid_a", "vid_b"}

    def test_representatividad_suma_uno(self):
        """La suma de representatividad de todos los tópicos = 1.0."""
        segs = segmentos_ejemplo()
        svc = TemasService(min_topic_size=2)

        resultado_mock = ResultadoTemas(
            segmentos=[
                SegmentoTematizado(segmento=s, topico_id=0, probabilidad=0.9)
                for s in segs[:3]
            ] + [
                SegmentoTematizado(segmento=s, topico_id=1, probabilidad=0.85)
                for s in segs[3:]
            ],
            topicos=[
                TopicoInfo(id=0, nombre="A", num_segmentos=3, representatividad=0.5),
                TopicoInfo(id=1, nombre="B", num_segmentos=3, representatividad=0.5),
            ],
            num_topicos=2,
        )

        with patch.object(svc, "_get_embedder"), \
             patch(
                 "tono_politico.temas.service.descubrir_temas",
                 return_value=resultado_mock,
             ):
            resultado = svc.procesar(segs)

        total = sum(t.representatividad for t in resultado.topicos)
        assert abs(total - 1.0) < 0.01
