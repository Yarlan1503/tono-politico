"""Tests para la orquestación testeable del pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tono_politico.config import Config, ProjectConfig
from tono_politico.filtrado.models import CriterioFiltrado, ResultadoFiltrado
from tono_politico.models import PlaylistInfo, SegmentoRaw, VideoTranscript
from tono_politico.pipeline.runner import PipelineRunner, ServiceFactories
from tono_politico.salida.models import InformeTono, PerfilActor, Provenance
from tono_politico.temas.models import ResultadoTemas, TopicoInfo
from tono_politico.tono.models import ResultadoTono


class FakeService:
    def __init__(self, output: Any):
        self.output = output
        self.calls: list[tuple[Any, ...]] = []

    def procesar(self, *args: Any) -> Any:
        self.calls.append(args)
        return self.output


class FailingService:
    def __init__(self, message: str):
        self.message = message
        self.calls: list[tuple[Any, ...]] = []

    def procesar(self, *args: Any) -> Any:
        self.calls.append(args)
        raise RuntimeError(self.message)


class FakeSalidaService(FakeService):
    def __init__(self, output: Any, output_path: Path | None = Path("output")):
        super().__init__(output)
        self.output_path = output_path


def _transcript(video_id: str = "vid-1", titulo: str = "Video 1") -> VideoTranscript:
    return VideoTranscript(
        video_id=video_id,
        url=f"https://youtu.be/{video_id}",
        titulo=titulo,
        fecha=None,
        raw_segments=[SegmentoRaw("texto", 0.0, 2.0, 0.0)],
    )


def _resultado_temas() -> ResultadoTemas:
    return ResultadoTemas(
        topicos=[
            TopicoInfo(
                id=0,
                nombre="seguridad",
                palabras_clave=["seguridad"],
                num_segmentos=1,
            )
        ],
        num_topicos=1,
    )


def _resultado_filtrado(total: int = 1) -> ResultadoFiltrado:
    return ResultadoFiltrado(
        criterio=CriterioFiltrado(topico_id=0),
        topico=TopicoInfo(id=0, nombre="seguridad"),
        total_segmentos_entrada=1,
        total_segmentos_filtrados=total,
    )


def _informe() -> InformeTono:
    return InformeTono(
        perfil=PerfilActor(
            actor="Lilly Téllez",
            tema="seguridad",
            n_segmentos=0,
            stance_dominante="rechazo",
            intensidad_promedio=0.0,
            logica_dominante="populista",
            sentimiento_dominante="indignacion",
            estilo_dominante="directo",
            funcion_dominante="critica",
        ),
        provenance=Provenance(
            pipeline="test",
            modelos=[],
            fecha="2026-07-06T00:00:00",
        ),
    )


def _factories(
    *,
    playlist_name: str = "Play-PoliTest",
    ingesta: Any | None = None,
    diarizacion: Any | None = None,
    segmentacion: Any | None = None,
    temas: Any | None = None,
    filtrado: Any | None = None,
    tono: Any | None = None,
    salida: FakeSalidaService | None = None,
) -> ServiceFactories:
    transcript = _transcript()
    ingesta = ingesta or FakeService([transcript])
    diarizacion = diarizacion or FakeService([transcript])
    segmentacion = segmentacion or FakeService(["segmento-1"])
    temas = temas or FakeService(_resultado_temas())
    filtrado = filtrado or FakeService(_resultado_filtrado())
    tono = tono or FakeService(ResultadoTono(tema="seguridad", actor="Lilly Téllez"))
    salida = salida or FakeSalidaService(_informe())

    return ServiceFactories(
        build_ingesta=lambda _cfg: ingesta,
        build_diarizacion=lambda _cfg: diarizacion,
        build_segmentacion=lambda _cfg: segmentacion,
        build_temas=lambda _cfg: temas,
        build_filtrado=lambda _cfg, _topico_id: filtrado,
        build_tono=lambda _cfg, _actor, _tema: tono,
        build_salida=lambda _cfg, _output_path: salida,
        get_playlist_info=lambda url: PlaylistInfo(nombre=playlist_name, url=url, videos=[]),
    )


def _cfg(tmp_path: Path) -> Config:
    return Config(project=ProjectConfig(data_dir=tmp_path))


class TestPipelineRunnerDiscover:
    def test_discover_ejecuta_fase_1_con_services_inyectados(self, tmp_path: Path):
        transcript = _transcript("vid-1")
        actor_transcript = _transcript("vid-1")
        ingesta = FakeService([transcript])
        diarizacion = FakeService([actor_transcript])
        segmentacion = FakeService(["segmento-1"])
        temas = FakeService(_resultado_temas())
        factories = _factories(
            ingesta=ingesta,
            diarizacion=diarizacion,
            segmentacion=segmentacion,
            temas=temas,
        )
        runner = PipelineRunner(cfg=_cfg(tmp_path), factories=factories, keep_cache=True)

        result = runner.discover("playlist-url")

        assert result.exit_code == 0
        assert result.manifest.status == "ok"
        assert result.manifest.playlist_url == "playlist-url"
        assert result.manifest.playlist_name == "Play-PoliTest"
        assert [phase.phase for phase in result.manifest.phases] == [
            "ingesta",
            "diarizacion",
            "segmentacion",
            "temas",
        ]
        assert ingesta.calls == [("playlist-url",)]
        assert diarizacion.calls == [([transcript], "Play-PoliTest")]
        assert segmentacion.calls == [([actor_transcript],)]
        assert temas.calls == [(["segmento-1"],)]
        assert result.manifest.videos[0].video_id == "vid-1"
        assert result.manifest.videos[0].transcrito is True
        assert result.manifest.videos[0].diarizado is True
        assert result.manifest.videos[0].segmentos_actor == 1

    def test_discover_limpia_cache_runtime_si_keep_cache_es_false(self, tmp_path: Path):
        cache_dir = tmp_path / "Play-PoliTest"
        cache_dir.mkdir()
        (cache_dir / "transcripcion.json").write_text("{}", encoding="utf-8")
        runner = PipelineRunner(cfg=_cfg(tmp_path), factories=_factories(), keep_cache=False)

        result = runner.discover("playlist-url")

        assert result.exit_code == 0
        assert result.manifest.cache_dir == cache_dir
        assert not cache_dir.exists()

    def test_discover_conserva_cache_runtime_si_keep_cache_es_true(self, tmp_path: Path):
        cache_dir = tmp_path / "Play-PoliTest"
        cache_dir.mkdir()
        runner = PipelineRunner(cfg=_cfg(tmp_path), factories=_factories(), keep_cache=True)

        result = runner.discover("playlist-url")

        assert result.exit_code == 0
        assert result.manifest.cache_dir == cache_dir
        assert cache_dir.exists()

    def test_discover_si_una_fase_falla_registra_manifest_y_limpia_cache(
        self,
        tmp_path: Path,
    ):
        cache_dir = tmp_path / "Play-PoliTest"
        cache_dir.mkdir()
        segmentacion = FailingService("segmentación rota")
        runner = PipelineRunner(
            cfg=_cfg(tmp_path),
            factories=_factories(segmentacion=segmentacion),
            keep_cache=False,
        )

        result = runner.discover("playlist-url")

        assert result.exit_code == 1
        assert result.manifest.status == "failed"
        assert result.manifest.phases[-1].phase == "segmentacion"
        assert result.manifest.phases[-1].ok is False
        assert "segmentación rota" in result.manifest.phases[-1].message
        assert not cache_dir.exists()

    def test_discover_si_get_playlist_info_falla_no_borra_data_dir(self, tmp_path: Path):
        data_dir = tmp_path
        sibling = tmp_path / "otra_cosa"
        sibling.mkdir()
        factories = _factories()
        factories = ServiceFactories(
            build_ingesta=factories.build_ingesta,
            build_diarizacion=factories.build_diarizacion,
            build_segmentacion=factories.build_segmentacion,
            build_temas=factories.build_temas,
            build_filtrado=factories.build_filtrado,
            build_tono=factories.build_tono,
            build_salida=factories.build_salida,
            get_playlist_info=lambda url: (_ for _ in ()).throw(
                RuntimeError("playlist info rota")
            ),
        )
        runner = PipelineRunner(cfg=_cfg(data_dir), factories=factories, keep_cache=False)

        result = runner.discover("playlist-url")

        assert result.exit_code == 1
        assert result.manifest.status == "failed"
        assert data_dir.exists()
        assert sibling.exists()


class TestPipelineRunnerAnalyze:
    def test_analyze_ejecuta_fase_2_y_devuelve_informe_path(self, tmp_path: Path):
        filtrado = FakeService(_resultado_filtrado(total=1))
        tono = FakeService(ResultadoTono(tema="seguridad", actor="Lilly Téllez"))
        salida = FakeSalidaService(_informe(), output_path=Path("output"))
        runner = PipelineRunner(
            cfg=_cfg(tmp_path),
            factories=_factories(filtrado=filtrado, tono=tono, salida=salida),
            keep_cache=True,
        )

        result = runner.analyze(
            playlist_url="playlist-url",
            topico_id=0,
            tema="seguridad",
            output_path="output",
        )

        assert result.exit_code == 0
        assert result.manifest.status == "ok"
        assert [phase.phase for phase in result.manifest.phases] == [
            "ingesta",
            "diarizacion",
            "segmentacion",
            "temas",
            "filtrado",
            "tono",
            "salida",
        ]
        assert filtrado.calls == [(_resultado_temas(),)]
        resultado_filtrado = filtrado.output
        assert tono.calls == [(resultado_filtrado,)]
        assert salida.calls == [(tono.output,)]
        assert result.informe_path == Path("output")

    def test_analyze_sin_segmentos_filtrados_falla_sin_sys_exit(self, tmp_path: Path):
        filtrado = FakeService(_resultado_filtrado(total=0))
        tono = FakeService(ResultadoTono(tema="seguridad", actor="Lilly Téllez"))
        salida = FakeSalidaService(_informe())
        runner = PipelineRunner(
            cfg=_cfg(tmp_path),
            factories=_factories(filtrado=filtrado, tono=tono, salida=salida),
            keep_cache=True,
        )

        result = runner.analyze("playlist-url", topico_id=0, tema="seguridad", output_path=None)

        assert result.exit_code == 1
        assert result.manifest.status == "failed"
        assert result.manifest.phases[-1].phase == "filtrado"
        assert result.manifest.phases[-1].ok is False
        assert "No hay segmentos" in result.manifest.phases[-1].message
        assert tono.calls == []
        assert salida.calls == []

    def test_analyze_si_una_fase_falla_registra_manifest_y_limpia_cache(
        self,
        tmp_path: Path,
    ):
        cache_dir = tmp_path / "Play-PoliTest"
        cache_dir.mkdir()
        tono = FailingService("tono roto")
        salida = FakeSalidaService(_informe())
        runner = PipelineRunner(
            cfg=_cfg(tmp_path),
            factories=_factories(tono=tono, salida=salida),
            keep_cache=False,
        )

        result = runner.analyze("playlist-url", topico_id=0, tema="seguridad", output_path=None)

        assert result.exit_code == 1
        assert result.manifest.status == "failed"
        assert result.manifest.phases[-1].phase == "tono"
        assert result.manifest.phases[-1].ok is False
        assert "tono roto" in result.manifest.phases[-1].message
        assert salida.calls == []
        assert not cache_dir.exists()

    def test_analyze_si_get_playlist_info_falla_no_borra_data_dir(self, tmp_path: Path):
        data_dir = tmp_path
        sibling = tmp_path / "otra_cosa"
        sibling.mkdir()
        factories = _factories()
        factories = ServiceFactories(
            build_ingesta=factories.build_ingesta,
            build_diarizacion=factories.build_diarizacion,
            build_segmentacion=factories.build_segmentacion,
            build_temas=factories.build_temas,
            build_filtrado=factories.build_filtrado,
            build_tono=factories.build_tono,
            build_salida=factories.build_salida,
            get_playlist_info=lambda url: (_ for _ in ()).throw(
                RuntimeError("playlist info rota")
            ),
        )
        runner = PipelineRunner(cfg=_cfg(data_dir), factories=factories, keep_cache=False)

        result = runner.analyze("playlist-url", topico_id=0, tema="seguridad", output_path=None)

        assert result.exit_code == 1
        assert result.manifest.status == "failed"
        assert data_dir.exists()
        assert sibling.exists()
