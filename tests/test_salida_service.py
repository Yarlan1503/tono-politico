"""Tests para salida/service.py — SalidaService orquestador."""

from __future__ import annotations

import json

from tono_politico.protocol import ComponenteProtocol
from tono_politico.salida.models import InformeTono
from tono_politico.salida.service import SalidaService
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


def _resultado_tono(n: int = 2) -> ResultadoTono:
    """Crea un ResultadoTono con n segmentos sintéticos."""
    segmentos = []
    for i in range(n):
        seg = Segmento(
            texto=f"Segmento {i} del discurso.",
            t_start=float(i * 5),
            t_end=float((i + 1) * 5),
            oraciones=[Oracion(texto=f"Segmento {i}.", t_start=0, t_end=5, words=[])],
            word_count=3,
            video_id="v1",
        )
        segmentos.append(SegmentoConTono(
            segmento=seg,
            stance=ResultadoStance(stance="rechazo" if i % 2 == 0 else "apoyo", confianza=0.8),
            intensidad_antagonica=3 + (i % 2),
            logica_politica=ResultadoLogicaPolitica(
                nacionalista=0.55, globalista=0.30, populista=0.65,
                tecnocrata=0.25, corporativista=0.40, estatista=0.50,
            ),
            sentimiento=ResultadoSentimiento(
                esperanza=0.35, angustia=0.60, indignacion=0.55,
                orgullo=0.30, empatia=0.35,
            ),
            estilo_discursivo=ResultadoEstiloDiscursivo(
                directo=0.65, academico=0.35, confrontativo=0.55,
                conciliador=0.40, catastrofista=0.45, testimonial=0.30,
            ),
            funcion_discursiva=ResultadoFuncionDiscursiva(
                critica=0.60, propuesta=0.35, narrativa_personal=0.30,
            ),
        ))
    return ResultadoTono(tema="fracking", actor="AMLO", segmentos=segmentos)


class TestSalidaServiceInit:
    def test_implementa_componente_protocol(self):
        svc = SalidaService()
        assert isinstance(svc, ComponenteProtocol)

    def test_output_path_none_por_default(self):
        svc = SalidaService()
        assert svc.output_path is None


class TestSalidaServiceProcesar:
    def test_procesar_devuelve_informe(self):
        svc = SalidaService()
        resultado = _resultado_tono(2)
        informe = svc.procesar(resultado)

        assert isinstance(informe, InformeTono)
        assert informe.perfil.actor == "AMLO"
        assert informe.perfil.n_segmentos == 2
        assert len(informe.segmentos) == 2
        assert informe.provenance is not None

    def test_procesar_resultado_vacio(self):
        svc = SalidaService()
        resultado = ResultadoTono(tema="X", actor="Y")
        informe = svc.procesar(resultado)

        assert informe.perfil.n_segmentos == 0
        assert len(informe.segmentos) == 0

    def test_procesar_genera_json_en_disco(self, tmp_path):
        output = tmp_path / "informe.json"
        svc = SalidaService(output_path=output)
        svc.procesar(_resultado_tono(1))

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert data["perfil"]["actor"] == "AMLO"

    def test_procesar_genera_markdown_en_disco(self, tmp_path):
        output_md = tmp_path / "informe.md"
        svc = SalidaService(output_path=output_md)
        svc.procesar(_resultado_tono(1))

        assert output_md.exists()
        content = output_md.read_text(encoding="utf-8")
        assert "# AMLO — fracking" in content

    def test_procesar_genera_ambos_archivos(self, tmp_path):
        """Si output_path es un directorio, genera ambos archivos."""
        svc = SalidaService(output_path=tmp_path)
        svc.procesar(_resultado_tono(1))

        assert (tmp_path / "informe.json").exists()
        assert (tmp_path / "informe.md").exists()
