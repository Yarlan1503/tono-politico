"""Tests para los DTOs del Componente 1.5: Diarización."""

from __future__ import annotations

from tono_politico.diarizacion import PerfilVozActor, SpeakerMatch, TurnoOrador

# ──────────────────────────────────────────────────────────
# TurnoOrador
# ──────────────────────────────────────────────────────────


class TestTurnoOrador:
    """Contrato y estructura de TurnoOrador."""

    def test_crear_turno_basico(self):
        """Un turno se crea con los 4 campos obligatorios."""
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
        """La diferencia t_end - t_start es coherente."""
        t = TurnoOrador("v1", "SPEAKER_01", 0.0, 3.0)
        assert t.t_end - t.t_start == 3.0

    def test_dos_turnos_mismo_speaker_distinto_video(self):
        """El mismo speaker_id puede aparecer en videos distintos."""
        t1 = TurnoOrador("video_a", "SPEAKER_00", 0.0, 2.0)
        t2 = TurnoOrador("video_b", "SPEAKER_00", 1.0, 4.0)
        assert t1.speaker_id == t2.speaker_id
        assert t1.video_id != t2.video_id


# ──────────────────────────────────────────────────────────
# PerfilVozActor
# ──────────────────────────────────────────────────────────


class TestPerfilVozActor:
    """Contrato y estructura de PerfilVozActor."""

    def test_crear_perfil_basico(self):
        """Un perfil se crea con todos los campos."""
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
        """El embedding siempre es una lista de floats."""
        perfil = PerfilVozActor(
            actor="X",
            video_id_referencia="v",
            embedding=[1.0, -0.5, 0.0],
            modelo_embedding="m",
            duracion_segundos=10.0,
        )
        assert all(isinstance(x, float) for x in perfil.embedding)


# ──────────────────────────────────────────────────────────
# SpeakerMatch
# ──────────────────────────────────────────────────────────


class TestSpeakerMatch:
    """Contrato y estructura de SpeakerMatch."""

    def test_match_aceptado(self):
        """Un speaker aceptado como el actor."""
        m = SpeakerMatch(
            speaker_id="SPEAKER_00",
            distancia=0.35,
            aceptado=True,
            es_ambiguo=False,
        )
        assert m.speaker_id == "SPEAKER_00"
        assert m.distancia == 0.35
        assert m.aceptado is True
        assert m.es_ambiguo is False

    def test_match_ambiguo_descartado(self):
        """Un speaker ambiguo no se acepta."""
        m = SpeakerMatch(
            speaker_id="SPEAKER_01",
            distancia=0.6,
            aceptado=False,
            es_ambiguo=True,
        )
        assert m.aceptado is False
        assert m.es_ambiguo is True

    def test_match_rechazado_directo(self):
        """Un speaker claramente distinto al actor."""
        m = SpeakerMatch(
            speaker_id="SPEAKER_02",
            distancia=0.9,
            aceptado=False,
            es_ambiguo=False,
        )
        assert m.aceptado is False
        assert m.es_ambiguo is False
