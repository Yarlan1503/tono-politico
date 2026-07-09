"""Tests del orquestador DiscursiveApproachService."""

from __future__ import annotations

from tono_politico.diarizacion.models import (
    ActorTranscript,
    ActorTranscriptSegment,
    AsrMetadata,
)
from tono_politico.discursive_approach.argument_shape.models import Argumento
from tono_politico.discursive_approach.service import DiscursiveApproachService
from tono_politico.discursive_approach.topics_approach.models import ResultadoEnfoques
from tono_politico.discursive_approach.topics_cluster.models import (
    ArgumentoTematizado,
    ResultadoTemas,
    TopicoInfo,
)


class FakeShape:
    def procesar_one(self, tr):
        return [
            Argumento(
                texto=s.text,
                t_start=s.t_start,
                t_end=s.t_end,
                word_count=s.word_count,
                video_id=tr.video_id,
                fecha=tr.fecha,
            )
            for s in tr.segments
        ]

    def procesar_corpus(self, trs):
        out = []
        for tr in trs:
            out.extend(self.procesar_one(tr))
        return out


class FakeCluster:
    def procesar(self, argumentos):
        return ResultadoTemas(
            argumentos=[
                ArgumentoTematizado(argumento=a, topico_id=0, probabilidad=1.0) for a in argumentos
            ],
            topicos=[
                TopicoInfo(
                    id=0,
                    nombre="unico",
                    num_argumentos=len(argumentos),
                    representatividad=1.0,
                )
            ],
            num_topicos=1 if argumentos else 0,
        )


class FakeApproach:
    def procesar(self, resultado: ResultadoTemas) -> ResultadoEnfoques:
        return ResultadoEnfoques(por_tema=[], num_temas=resultado.num_topicos, num_enfoques_total=0)


def test_procesar_encadena_shape_cluster_approach():
    tr = ActorTranscript(
        schema_version="actor_transcript.v1",
        video_id="v1",
        actor="AMLO",
        scope="actor_only",
        asr=AsrMetadata("w", "m", "es"),
        segments=[
            ActorTranscriptSegment("hola mundo", 0.0, 1.0, "A", 0.0, 1.0, 2),
        ],
        fecha="20240101",
    )
    shape = FakeShape()
    cluster = FakeCluster()
    approach = FakeApproach()
    svc = DiscursiveApproachService(
        actor="AMLO",
        shape_service=shape,  # type: ignore[arg-type]
        cluster_service=cluster,  # type: ignore[arg-type]
        approach_service=approach,  # type: ignore[arg-type]
    )
    out = svc.procesar([tr])
    assert isinstance(out, ResultadoEnfoques)
    assert out.num_temas == 1


def test_shape_one_delega():
    svc = DiscursiveApproachService(
        actor="X",
        shape_service=FakeShape(),  # type: ignore[arg-type]
        cluster_service=FakeCluster(),  # type: ignore[arg-type]
        approach_service=FakeApproach(),  # type: ignore[arg-type]
    )
    tr = ActorTranscript(
        schema_version="v",
        video_id="v",
        actor="X",
        scope="actor_only",
        asr=AsrMetadata("w", "m", "es"),
        segments=[ActorTranscriptSegment("texto", 0.0, 2.0, "A", 0.0, 2.0, 1)],
        fecha="20240202",
    )
    args = svc.shape_one(tr)
    assert len(args) == 1
    assert args[0].fecha == "20240202"
