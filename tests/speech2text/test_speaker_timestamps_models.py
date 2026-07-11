"""DTOs canónicos y compatibilidad de speaker_timestamps."""

from __future__ import annotations

from tono_politico.speech2text.models import (
    PerfilVozActor as CompatibilityPerfilVozActor,
)
from tono_politico.speech2text.models import (
    SpeakerMatch as CompatibilitySpeakerMatch,
)
from tono_politico.speech2text.models import (
    TurnoOrador as CompatibilityTurnoOrador,
)
from tono_politico.speech2text.speaker_timestamps.models import (
    PerfilVozActor,
    SpeakerMatch,
    TurnoOrador,
)


def test_diarization_dtos_have_one_canonical_module():
    assert TurnoOrador.__module__ == "tono_politico.speech2text.speaker_timestamps.models"
    assert PerfilVozActor.__module__ == "tono_politico.speech2text.speaker_timestamps.models"
    assert SpeakerMatch.__module__ == "tono_politico.speech2text.speaker_timestamps.models"
    assert CompatibilityTurnoOrador is TurnoOrador
    assert CompatibilityPerfilVozActor is PerfilVozActor
    assert CompatibilitySpeakerMatch is SpeakerMatch


class TestTurnoOrador:
    """Contrato y estructura de TurnoOrador."""

    def test_crear_turno_basico(self):
        t = TurnoOrador(
            video_id="abc123",
            speaker_id="SPEAKER_00",
            t_start=1.5,
            t_end=5.3,
        )
        assert t.video_id == "abc123"
        assert t.speaker_id == "SPEAKER_00"
        assert t.t_start == 1.5
        assert t.t_end == 5.3

    def test_turno_duracion(self):
        t = TurnoOrador("v1", "SPEAKER_01", 0.0, 3.0)
        assert t.t_end - t.t_start == 3.0

    def test_dos_turnos_mismo_speaker_distinto_video(self):
        t1 = TurnoOrador("video_a", "SPEAKER_00", 0.0, 2.0)
        t2 = TurnoOrador("video_b", "SPEAKER_00", 1.0, 4.0)
        assert t1.speaker_id == t2.speaker_id
        assert t1.video_id != t2.video_id


class TestPerfilVozActor:
    """Contrato y estructura de PerfilVozActor."""

    def test_crear_perfil_basico(self):
        perfil = PerfilVozActor(
            actor="Lilly Téllez",
            video_id_referencia="su9nURIj9XQ",
            embedding=[0.1, 0.2, 0.3],
            modelo_embedding="pipeline-internal",
            duracion_segundos=30.0,
        )
        assert perfil.actor == "Lilly Téllez"
        assert perfil.video_id_referencia == "su9nURIj9XQ"
        assert len(perfil.embedding) == 3
        assert perfil.modelo_embedding == "pipeline-internal"
        assert perfil.duracion_segundos == 30.0

    def test_embedding_es_lista_de_floats(self):
        perfil = PerfilVozActor(
            actor="X",
            video_id_referencia="v",
            embedding=[1.0, -0.5, 0.0],
            modelo_embedding="m",
            duracion_segundos=10.0,
        )
        assert all(isinstance(x, float) for x in perfil.embedding)


class TestSpeakerMatch:
    """Estados semánticos del matching del speaker."""

    def test_match_aceptado(self):
        match = SpeakerMatch(
            speaker_id="SPEAKER_00",
            distancia=0.35,
            aceptado=True,
            es_ambiguo=False,
        )
        assert match.speaker_id == "SPEAKER_00"
        assert match.distancia == 0.35
        assert match.aceptado is True
        assert match.es_ambiguo is False

    def test_match_ambiguo_descartado(self):
        match = SpeakerMatch(
            speaker_id="SPEAKER_01",
            distancia=0.6,
            aceptado=False,
            es_ambiguo=True,
        )
        assert match.aceptado is False
        assert match.es_ambiguo is True

    def test_match_rechazado_directo(self):
        match = SpeakerMatch(
            speaker_id="SPEAKER_02",
            distancia=0.9,
            aceptado=False,
            es_ambiguo=False,
        )
        assert match.aceptado is False
        assert match.es_ambiguo is False
