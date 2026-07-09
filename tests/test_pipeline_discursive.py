"""Tests: PipelineRunner.discover_discursive (speech2text → discursive_approach)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tono_politico.config import Config, ProjectConfig
from tono_politico.diarizacion.models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
)
from tono_politico.discursive_approach.argument_shape.models import Argumento
from tono_politico.discursive_approach.topics_approach.models import (
    ResultadoEnfoques,
    ResultadoEnfoquesTema,
)
from tono_politico.discursive_approach.topics_cluster.models import (
    ArgumentoTematizado,
    ResultadoTemas,
    TopicoInfo,
)
from tono_politico.models import PlaylistInfo
from tono_politico.pipeline.runner import PipelineRunner, ServiceFactories


class FakeSpeech2Text:
    def __init__(self, transcripts: list[ActorTranscript]):
        self.transcripts = transcripts
        self.calls: list[str] = []

    def procesar(self, url: str) -> list[ActorTranscript]:
        self.calls.append(url)
        return self.transcripts


class FakeDiscursive:
    def __init__(self) -> None:
        self.shape_calls: list[Any] = []
        self.cluster_calls: list[Any] = []
        self.approach_calls: list[Any] = []

    def shape_corpus(self, transcripts: list[ActorTranscript]) -> list[Argumento]:
        self.shape_calls.append(transcripts)
        return [
            Argumento(
                texto=seg.text,
                t_start=seg.t_start,
                t_end=seg.t_end,
                word_count=seg.word_count,
                video_id=tr.video_id,
                fecha=tr.fecha,
            )
            for tr in transcripts
            for seg in tr.segments
        ]

    def cluster(self, argumentos: list[Argumento]) -> ResultadoTemas:
        self.cluster_calls.append(argumentos)
        return ResultadoTemas(
            argumentos=[
                ArgumentoTematizado(argumento=a, topico_id=0, probabilidad=1.0) for a in argumentos
            ],
            topicos=[
                TopicoInfo(
                    id=0,
                    nombre="tema-unico",
                    num_argumentos=len(argumentos),
                    representatividad=1.0,
                )
            ],
            num_topicos=1 if argumentos else 0,
        )

    def approaches(self, resultado: ResultadoTemas) -> ResultadoEnfoques:
        self.approach_calls.append(resultado)
        topico = resultado.topicos[0] if resultado.topicos else TopicoInfo(id=0, nombre="x")
        return ResultadoEnfoques(
            por_tema=[ResultadoEnfoquesTema(topico=topico, enfoques=[], argumentos=[])],
            num_temas=1 if resultado.topicos else 0,
            num_enfoques_total=0,
        )


def _tx(video_id: str = "v1", fecha: str | None = "20240101") -> ActorTranscript:
    return ActorTranscript(
        schema_version="actor_transcript.v1",
        video_id=video_id,
        actor="Actor",
        scope="actor_only",
        asr=AsrMetadata(provider="whisper", model="m", language="es"),
        segments=[
            ActorTranscriptSegment(
                text="hola mundo político",
                t_start=0.0,
                t_end=1.0,
                speaker="SPEAKER_00",
                source_turn_start=0.0,
                source_turn_end=1.0,
                word_count=3,
            )
        ],
        fecha=fecha,
    )


def _legacy_factories(
    speech: FakeSpeech2Text,
    discursive: FakeDiscursive,
    playlist_name: str = "Play-Disc",
) -> ServiceFactories:
    # factories legacy mínimas (no se usan en discover_discursive)
    dummy = type("D", (), {"procesar": staticmethod(lambda *a, **k: None)})()
    return ServiceFactories(
        build_ingesta=lambda _c: dummy,
        build_diarizacion=lambda _c: dummy,
        build_segmentacion=lambda _c: dummy,
        build_temas=lambda _c: dummy,
        build_filtrado=lambda _c, _t: dummy,
        build_tono=lambda _c, _a, _t: dummy,
        build_salida=lambda _c, _o: dummy,
        get_playlist_info=lambda url: PlaylistInfo(nombre=playlist_name, url=url, videos=[]),
        build_speech2text=lambda _c: speech,
        build_discursive=lambda _c: discursive,
    )


def test_discover_discursive_ejecuta_fases_y_persiste(tmp_path: Path):
    speech = FakeSpeech2Text([_tx("v1"), _tx("v2", fecha="20240202")])
    discursive = FakeDiscursive()
    cfg = Config(project=ProjectConfig(data_dir=tmp_path, output_dir=tmp_path / "out"))
    runner = PipelineRunner(
        cfg=cfg,
        factories=_legacy_factories(speech, discursive),
        keep_cache=True,
        run_id="run-disc-1",
    )

    result = runner.discover_discursive("https://playlist.example")

    assert result.exit_code == 0
    assert speech.calls == ["https://playlist.example"]
    assert len(discursive.shape_calls) == 1
    assert len(discursive.cluster_calls) == 1
    assert len(discursive.approach_calls) == 1

    phases = [p.phase for p in result.manifest.phases]
    assert phases == [
        "speech2text",
        "argument_shape",
        "topics_cluster",
        "topics_approach",
    ]
    assert all(p.ok for p in result.manifest.phases)

    assert result.manifest.playlist_name == "Play-Disc"
    assert len(result.manifest.videos) == 2
    assert result.manifest.videos[0].segmentos_actor == 1

    assert runner.last_resultado_enfoques is not None
    assert runner.last_resultado_enfoques.num_temas == 1

    # artefactos (guardar_manifest: output_dir/<run_id>/)
    run_dir = tmp_path / "out" / "run-disc-1"
    assert (run_dir / "discursive-enfoques.json").exists()
    assert (run_dir / "discursive-temas.json").exists()


def test_discover_discursive_falla_sin_factories(tmp_path: Path):
    dummy = type("D", (), {"procesar": staticmethod(lambda *a, **k: None)})()
    factories = ServiceFactories(
        build_ingesta=lambda _c: dummy,
        build_diarizacion=lambda _c: dummy,
        build_segmentacion=lambda _c: dummy,
        build_temas=lambda _c: dummy,
        build_filtrado=lambda _c, _t: dummy,
        build_tono=lambda _c, _a, _t: dummy,
        build_salida=lambda _c, _o: dummy,
        get_playlist_info=lambda url: PlaylistInfo(nombre="P", url=url, videos=[]),
    )
    runner = PipelineRunner(
        cfg=Config(project=ProjectConfig(data_dir=tmp_path, output_dir=tmp_path / "out")),
        factories=factories,
        keep_cache=True,
    )
    result = runner.discover_discursive("url")
    assert result.exit_code == 1
    assert result.manifest.status == "failed"
