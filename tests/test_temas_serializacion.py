"""Tests para serialización de ResultadoTemas (persistencia y resume)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tono_politico.models import WordTimestamp
from tono_politico.segmentacion.models import Oracion, Segmento
from tono_politico.temas.models import ResultadoTemas, SegmentoTematizado, TopicoInfo
from tono_politico.temas.serializacion import (
    cargar_fase1,
    guardar_fase1,
    resultado_temas_from_json,
    resultado_temas_to_dict,
    resultado_temas_to_json,
)

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def _segmento(
    texto: str = "Hola mundo.",
    video_id: str = "v1",
    t_start: float = 0.0,
    t_end: float = 5.0,
) -> Segmento:
    return Segmento(
        texto=texto,
        t_start=t_start,
        t_end=t_end,
        oraciones=[
            Oracion(
                texto=texto,
                t_start=t_start,
                t_end=t_end,
                words=[WordTimestamp(word="Hola", start=t_start, end=t_start + 0.5)],
            ),
        ],
        word_count=2,
        video_id=video_id,
    )


def _resultado() -> ResultadoTemas:
    return ResultadoTemas(
        segmentos=[
            SegmentoTematizado(
                segmento=_segmento("Economía crece."), topico_id=0, probabilidad=0.9
            ),
            SegmentoTematizado(
                segmento=_segmento("Fútbol emocionante.", video_id="v2"),
                topico_id=1,
                probabilidad=0.8,
            ),
        ],
        topicos=[
            TopicoInfo(
                id=0,
                nombre="economía",
                palabras_clave=["pib", "crecimiento"],
                num_segmentos=1,
                representatividad=0.5,
            ),
            TopicoInfo(
                id=1,
                nombre="fútbol",
                palabras_clave=["gol", "partido"],
                num_segmentos=1,
                representatividad=0.5,
            ),
        ],
        num_topicos=2,
    )


# ──────────────────────────────────────────────────────────
# resultado_temas_to_dict
# ──────────────────────────────────────────────────────────


class TestResultadoTemasToDict:
    def test_serializa_topicos(self):
        d = resultado_temas_to_dict(_resultado())
        assert len(d["topicos"]) == 2
        assert d["topicos"][0]["id"] == 0
        assert d["topicos"][0]["palabras_clave"] == ["pib", "crecimiento"]

    def test_serializa_segmentos(self):
        d = resultado_temas_to_dict(_resultado())
        assert len(d["segmentos"]) == 2
        assert d["segmentos"][0]["topico_id"] == 0
        assert d["segmentos"][0]["segmento"]["video_id"] == "v1"

    def test_serializa_num_topicos(self):
        d = resultado_temas_to_dict(_resultado())
        assert d["num_topicos"] == 2


# ──────────────────────────────────────────────────────────
# Round-trip: to_json → from_json
# ──────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_round_trip_preserva_topicos(self):
        original = _resultado()
        j = resultado_temas_to_json(original)
        recovered = resultado_temas_from_json(j)

        assert recovered.num_topicos == original.num_topicos
        assert len(recovered.topicos) == len(original.topicos)
        assert recovered.topicos[0].id == original.topicos[0].id
        assert recovered.topicos[0].palabras_clave == original.topicos[0].palabras_clave

    def test_round_trip_preserva_segmentos(self):
        original = _resultado()
        j = resultado_temas_to_json(original)
        recovered = resultado_temas_from_json(j)

        assert len(recovered.segmentos) == len(original.segmentos)
        assert recovered.segmentos[0].topico_id == original.segmentos[0].topico_id
        assert recovered.segmentos[0].segmento.video_id == original.segmentos[0].segmento.video_id
        assert recovered.segmentos[0].segmento.texto == original.segmentos[0].segmento.texto

    def test_round_trip_preserva_oraciones_y_words(self):
        original = _resultado()
        j = resultado_temas_to_json(original)
        recovered = resultado_temas_from_json(j)

        seg = recovered.segmentos[0].segmento
        assert len(seg.oraciones) == 1
        assert seg.oraciones[0].words[0].word == "Hola"

    def test_round_trip_preserva_probabilidad(self):
        original = _resultado()
        j = resultado_temas_to_json(original)
        recovered = resultado_temas_from_json(j)
        assert recovered.segmentos[0].probabilidad == pytest.approx(0.9)


# ──────────────────────────────────────────────────────────
# guardar_fase1 / cargar_fase1
# ──────────────────────────────────────────────────────────


class TestGuardarCargarFase1:
    def test_guardar_crea_archivo(self, tmp_path: Path):
        path = guardar_fase1(_resultado(), tmp_path / "runs" / "run-001")
        assert path.exists()
        assert path.name == "fase1-topicos.json"

    def test_cargar_recupera_estructura(self, tmp_path: Path):
        run_dir = tmp_path / "runs" / "run-001"
        guardar_fase1(_resultado(), run_dir)

        recovered = cargar_fase1(run_dir)
        assert recovered.num_topicos == 2
        assert len(recovered.segmentos) == 2

    def test_cargar_falla_claramente_si_no_existe(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="fase1-topicos"):
            cargar_fase1(tmp_path / "inexistente")
