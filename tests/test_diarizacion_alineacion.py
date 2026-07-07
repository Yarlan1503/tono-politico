"""Tests para filtrar_por_actor(): midpoint de segmento ↔ turnos del actor."""

from __future__ import annotations

from tono_politico.diarizacion.alineacion import filtrar_por_actor
from tono_politico.diarizacion.models import TurnoOrador
from tono_politico.models import SegmentoRaw, VideoTranscript, WordTimestamp

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────


def _seg(texto: str, t_start: float, t_end: float) -> SegmentoRaw:
    """SegmentoRaw mínimo con una palabra para preservar estructura."""
    return SegmentoRaw(
        texto=texto,
        t_start=t_start,
        t_end=t_end,
        pausa_antes=0.0,
        words=[WordTimestamp(word=texto.split()[0] if texto else "x", start=t_start, end=t_end)],
    )


def _transcript(segments: list[SegmentoRaw], video_id: str = "v1") -> VideoTranscript:
    return VideoTranscript(
        video_id=video_id,
        url=f"https://youtube.com/watch?v={video_id}",
        titulo="Test video",
        fecha="20260101",
        raw_segments=segments,
    )


def _turno(video_id: str, speaker: str, t_start: float, t_end: float) -> TurnoOrador:
    return TurnoOrador(video_id=video_id, speaker_id=speaker, t_start=t_start, t_end=t_end)


# ──────────────────────────────────────────────────────────
# Tests: midpoint criterion
# ──────────────────────────────────────────────────────────


class TestMidpointCriterio:
    """filtrar_por_actor(): conserva segmentos cuyo midpoint cae en un turno del actor."""

    def test_segmento_dentro_del_turno(self):
        """Midpoint cae dentro del turno del actor → se conserva."""
        transcript = _transcript([_seg("Hola mundo", 2.0, 5.0)])
        turnos = [_turno("v1", "SPEAKER_00", 0.0, 10.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 1
        assert resultado.raw_segments[0].texto == "Hola mundo"

    def test_segmento_fuera_del_turno(self):
        """Midpoint cae fuera de cualquier turno → se descarta."""
        transcript = _transcript([_seg("Otra persona", 15.0, 18.0)])
        turnos = [_turno("v1", "SPEAKER_00", 0.0, 10.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 0

    def test_midpoint_exacto_en_inicio(self):
        """Midpoint == t_start del turno (frontera inclusiva izquierda)."""
        # midpoint = (2.0 + 4.0) / 2 = 3.0, turno empieza en 3.0
        transcript = _transcript([_seg("En la frontera", 2.0, 4.0)])
        turnos = [_turno("v1", "SPEAKER_00", 3.0, 10.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 1

    def test_midpoint_justo_antes_del_inicio(self):
        """Midpoint < t_start del turno → descartado."""
        # midpoint = (1.0 + 3.0) / 2 = 2.0, turno empieza en 2.01
        transcript = _transcript([_seg("Casi", 1.0, 3.0)])
        turnos = [_turno("v1", "SPEAKER_00", 2.01, 10.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 0

    def test_midpoint_exacto_en_fin(self):
        """Midpoint == t_end del turno → fuera (frontera exclusiva derecha)."""
        # midpoint = (10.0 + 12.0) / 2 = 11.0, turno termina en 11.0
        transcript = _transcript([_seg("En el fin", 10.0, 12.0)])
        turnos = [_turno("v1", "SPEAKER_00", 0.0, 11.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 0

    def test_segmento_cruza_cambio_de_speaker(self):
        """Segmento que cruza de actor a otro: midpoint decide."""
        # Segmento de 0.0 a 10.0, midpoint = 5.0
        # Turno del actor: 0.0-4.0 (antes del midpoint)
        # Turno de otro: 4.0-10.0 (después del midpoint)
        transcript = _transcript([_seg("Cruce de speakers", 0.0, 10.0)])
        turnos = [_turno("v1", "SPEAKER_00", 0.0, 4.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        # midpoint = 5.0 cae fuera del turno del actor (0.0-4.0) → descartado
        assert len(resultado.raw_segments) == 0


# ──────────────────────────────────────────────────────────
# Tests: múltiples turnos y segmentos
# ──────────────────────────────────────────────────────────


class TestMultiplesTurnos:
    """filtrar_por_actor() con turnos intercalados del actor."""

    def test_actor_habla_dos_veces(self):
        """Actor habla en dos rangos; segmentos en ambos se conservan."""
        transcript = _transcript(
            [
                _seg("Primera parte", 1.0, 3.0),  # midpoint 2.0 → en turno 0-4
                _seg("Entrevistador", 5.0, 7.0),  # midpoint 6.0 → fuera
                _seg("Segunda parte", 9.0, 11.0),  # midpoint 10.0 → en turno 8-12
            ]
        )
        turnos = [
            _turno("v1", "SPEAKER_00", 0.0, 4.0),
            _turno("v1", "SPEAKER_00", 8.0, 12.0),
        ]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 2
        assert resultado.raw_segments[0].texto == "Primera parte"
        assert resultado.raw_segments[1].texto == "Segunda parte"

    def test_tres_segmentos_uno_por_rango(self):
        """Tres segmentos, cada uno cae en un turno distinto del actor."""
        transcript = _transcript(
            [
                _seg("Uno", 0.5, 1.5),  # midpoint 1.0 → turno A
                _seg("Dos", 5.0, 6.0),  # midpoint 5.5 → turno B
                _seg("Tres", 9.0, 10.0),  # midpoint 9.5 → turno C
            ]
        )
        turnos = [
            _turno("v1", "SPEAKER_00", 0.0, 2.0),
            _turno("v1", "SPEAKER_00", 4.0, 7.0),
            _turno("v1", "SPEAKER_00", 8.0, 11.0),
        ]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 3


# ──────────────────────────────────────────────────────────
# Tests: preservación de metadata y words
# ──────────────────────────────────────────────────────────


class TestPreservacionEstructura:
    """filtrar_por_actor() preserva metadata del transcript y words de segmentos."""

    def test_metadata_se_preserva(self):
        """El VideoTranscript filtrado conserva video_id, url, titulo, fecha."""
        transcript = _transcript([_seg("Hola", 0.0, 1.0)])
        transcript.titulo = "Mi intervención"
        turnos = [_turno("v1", "SPEAKER_00", 0.0, 2.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert resultado.video_id == "v1"
        assert resultado.url == "https://youtube.com/watch?v=v1"
        assert resultado.titulo == "Mi intervención"
        assert resultado.fecha == "20260101"

    def test_words_se_preservan(self):
        """Los WordTimestamp de los segmentos conservados se mantienen intactos."""
        seg = SegmentoRaw(
            texto="Hola mundo cruel",
            t_start=0.0,
            t_end=3.0,
            pausa_antes=0.0,
            words=[
                WordTimestamp(word="Hola", start=0.0, end=1.0),
                WordTimestamp(word="mundo", start=1.0, end=2.0),
                WordTimestamp(word="cruel", start=2.0, end=3.0),
            ],
        )
        transcript = _transcript([seg])
        turnos = [_turno("v1", "SPEAKER_00", 0.0, 5.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 1
        assert len(resultado.raw_segments[0].words) == 3
        assert resultado.raw_segments[0].words[1].word == "mundo"

    def test_pausa_antes_se_preserva(self):
        """El campo pausa_antes del segmento se mantiene."""
        seg = SegmentoRaw(texto="Hola", t_start=2.0, t_end=4.0, pausa_antes=1.5)
        transcript = _transcript([seg])
        turnos = [_turno("v1", "SPEAKER_00", 0.0, 10.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert resultado.raw_segments[0].pausa_antes == 1.5

    def test_orden_se_preserva(self):
        """Los segmentos conservados mantienen el orden cronológico original."""
        transcript = _transcript(
            [
                _seg("A", 1.0, 2.0),
                _seg("B", 3.0, 4.0),
                _seg("C", 5.0, 6.0),
            ]
        )
        turnos = [
            _turno("v1", "SPEAKER_00", 0.0, 2.5),  # A sí, B no, C sí
            _turno("v1", "SPEAKER_00", 4.5, 7.0),  # C sí
        ]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 2
        assert resultado.raw_segments[0].texto == "A"
        assert resultado.raw_segments[1].texto == "C"


# ──────────────────────────────────────────────────────────
# Tests: edge cases
# ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """filtrar_por_actor() con casos límite."""

    def test_sin_turnos_de_actor(self):
        """Lista vacía de turnos → transcript sin segmentos."""
        transcript = _transcript([_seg("Hola", 0.0, 1.0)])

        resultado = filtrar_por_actor(transcript, [])

        assert len(resultado.raw_segments) == 0

    def test_sin_segmentos(self):
        """Transcript sin segmentos → transcript vacío."""
        transcript = _transcript([])
        turnos = [_turno("v1", "SPEAKER_00", 0.0, 10.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 0

    def test_no_hay_match_ninguno(self):
        """Ningún segmento coincide con el actor → todos descartados."""
        transcript = _transcript(
            [
                _seg("Uno", 0.0, 2.0),
                _seg("Dos", 4.0, 6.0),
            ]
        )
        turnos = [_turno("v1", "SPEAKER_00", 10.0, 20.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 0

    def test_todos_coinciden(self):
        """Todos los segmentos coinciden con el actor → todos conservados."""
        transcript = _transcript(
            [
                _seg("Uno", 1.0, 2.0),
                _seg("Dos", 3.0, 4.0),
                _seg("Tres", 5.0, 6.0),
            ]
        )
        turnos = [_turno("v1", "SPEAKER_00", 0.0, 10.0)]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 3

    def test_turnos_de_otro_video_se_ignoran(self):
        """Solo se usan turnos del mismo video_id que el transcript."""
        transcript = _transcript([_seg("Hola", 2.0, 3.0)], video_id="v1")
        turnos = [
            _turno("v2", "SPEAKER_00", 0.0, 10.0),  # distinto video → ignorado
        ]

        resultado = filtrar_por_actor(transcript, turnos)

        assert len(resultado.raw_segments) == 0
