"""Tests para tono/zero_shot.py — clasificación de stance vía LLM.

Funciones puras (prompt construction, JSON parsing) se testean sin modelo.
ClasificadorLLM es de integración (marca slow).
"""

from __future__ import annotations

import pytest

from tono_politico.tono.models import ResultadoStance
from tono_politico.tono.zero_shot import (
    ClasificadorLLM,
    construir_prompt_stance,
    parsear_stance,
)


# ============================================================
# CONSTRUIR PROMPT STANCE
# ============================================================
class TestConstruirPromptStance:
    def test_devuelve_lista_de_messages(self):
        messages = construir_prompt_stance(
            actor="AMLO",
            tema="fracking",
            texto="No al fracking.",
        )
        assert isinstance(messages, list)
        assert len(messages) == 2  # system + user

    def test_system_message_rol_correcto(self):
        messages = construir_prompt_stance("AMLO", "fracking", "texto")
        assert messages[0]["role"] == "system"

    def test_user_message_contiene_actor(self):
        messages = construir_prompt_stance("AMLO", "fracking", "texto")
        user_content = messages[1]["content"]
        assert "AMLO" in user_content

    def test_user_message_contiene_tema(self):
        messages = construir_prompt_stance("AMLO", "fracking", "texto")
        user_content = messages[1]["content"]
        assert "fracking" in user_content

    def test_user_message_contiene_texto(self):
        texto = "El pueblo exige justicia social ya."
        messages = construir_prompt_stance("AMLO", "fracking", texto)
        user_content = messages[1]["content"]
        assert texto in user_content

    def test_system_prompt_contiene_definicion_apoyo(self):
        messages = construir_prompt_stance("X", "Y", "Z")
        system_content = messages[0]["content"]
        assert "apoyo" in system_content.lower()
        assert "rechazo" in system_content.lower()

    def test_system_prompt_contiene_ejemplo_pro_confrontativo(self):
        """El system prompt debe incluir un ejemplo a favor pero confrontacional."""
        messages = construir_prompt_stance("X", "Y", "Z")
        system_content = messages[0]["content"]
        # Debe mencionar que un discurso puede estar a favor del tema
        # pero atacar a adversarios al mismo tiempo
        assert "a favor" in system_content.lower() or "apoyo" in system_content.lower()
        assert "adversario" in system_content.lower() or "atac" in system_content.lower()


# ============================================================
# PARSEAR STANCE
# ============================================================
class TestParsearStance:
    def test_parsea_json_apoyo(self):
        raw = '{"stance": "apoyo", "confianza": 0.85}'
        result = parsear_stance(raw)
        assert isinstance(result, ResultadoStance)
        assert result.stance == "apoyo"
        assert result.confianza == 0.85

    def test_parsea_json_rechazo(self):
        raw = '{"stance": "rechazo", "confianza": 0.72}'
        result = parsear_stance(raw)
        assert result.stance == "rechazo"
        assert result.confianza == 0.72

    def test_parsea_json_con_texto_alrededor(self):
        raw = 'Aquí está el análisis:\n{"stance": "apoyo", "confianza": 0.9}\nListo.'
        result = parsear_stance(raw)
        assert result.stance == "apoyo"
        assert result.confianza == 0.9

    def test_valor_default_confianza_si_falta(self):
        raw = '{"stance": "rechazo"}'
        result = parsear_stance(raw)
        assert result.stance == "rechazo"
        assert result.confianza == 0.5

    def test_value_error_si_no_encuentra_json(self):
        with pytest.raises(ValueError, match="No se pudo parsear"):
            parsear_stance("Esto no es JSON")

    def test_value_error_si_stance_invalido(self):
        with pytest.raises(ValueError, match="stance inválido"):
            parsear_stance('{"stance": "neutral", "confianza": 0.5}')

    def test_clamp_confianza_entre_0_y_1(self):
        raw = '{"stance": "apoyo", "confianza": 1.5}'
        result = parsear_stance(raw)
        assert result.confianza == 1.0

        raw2 = '{"stance": "apoyo", "confianza": -0.3}'
        result2 = parsear_stance(raw2)
        assert result2.confianza == 0.0


# ============================================================
# CLASIFICADOR LLM (integración — requiere modelo)
# ============================================================
@pytest.mark.slow
class TestClasificadorLLM:
    """Tests de integración que cargan el modelo real.

    Se ejecutan solo con: pytest tests/test_tono_zero_shot.py -m slow
    """

    def test_clasificar_stance_amlo_rechazo(self):
        clf = ClasificadorLLM()
        resultado = clf.clasificar_stance(
            texto=(
                "No vamos a permitir el fracking porque daña la naturaleza "
                "y contamina los mantos acuíferos."
            ),
            actor="AMLO",
            tema="fracking",
        )
        assert isinstance(resultado, ResultadoStance)
        assert resultado.stance == "rechazo"

    def test_clasificar_stance_trump_apoyo(self):
        """Caso crítico: texto a favor del fracking pero con ataques."""
        clf = ClasificadorLLM()
        resultado = clf.clasificar_stance(
            texto=(
                "We're gonna drill, baby, drill! We're gonna frack like "
                "never before. Sleepy Joe wants to end fracking."
            ),
            actor="Trump",
            tema="fracking",
        )
        assert resultado.stance == "apoyo"


# ============================================================
# MANEJO DE ERRORES — clasificar_stance no debe explotar
# ============================================================
class TestManejoErroresClasificarStance:
    """Verifica que clasificar_stance captura errores y devuelve default."""

    def test_respuesta_vacia_devuelve_default(self, monkeypatch):
        """Si el modelo devuelve string vacío, no debe explotar."""
        clf = ClasificadorLLM()
        monkeypatch.setattr(clf, "_load", lambda: None)

        # Simular modelo que devuelve vacío
        class _FakeModel:
            def generate(self, *a, **kw):
                return [""]

        class _FakeTok:
            def apply_chat_template(self, *a, **kw):
                return "chat"

            def __call__(self, *a, **kw):
                class R:
                    input_ids = [0]
                return R()

            def decode(self, *a, **kw):
                return ""

            def strip(self):
                return ""

        clf._model = _FakeModel()
        clf._tokenizer = _FakeTok()

        resultado = clf.clasificar_stance("texto", "Actor", "tema")
        assert resultado.stance == "apoyo"
        assert resultado.confianza == 0.0

    def test_respuesta_garbage_devuelve_default(self, monkeypatch):
        """Si el modelo devuelve texto sin JSON, no debe explotar."""
        clf = ClasificadorLLM()
        monkeypatch.setattr(clf, "_load", lambda: None)

        class _FakeModel:
            def generate(self, *a, **kw):
                return [""]

        class _FakeTok:
            def apply_chat_template(self, *a, **kw):
                return "chat"

            def __call__(self, *a, **kw):
                class R:
                    input_ids = [0]
                return R()

            def decode(self, *a, **kw):
                return "no hay json aqui, solo palabras"

        clf._model = _FakeModel()
        clf._tokenizer = _FakeTok()

        resultado = clf.clasificar_stance("texto", "Actor", "tema")
        assert resultado.stance == "apoyo"
        assert resultado.confianza == 0.0

    def test_excepcion_runtime_devuelve_default(self, monkeypatch):
        """Si el modelo lanza RuntimeError, no debe explotar."""
        clf = ClasificadorLLM()
        monkeypatch.setattr(clf, "_load", lambda: None)

        class _FakeModel:
            def generate(self, *a, **kw):
                raise RuntimeError("CUDA out of memory")

        class _FakeTok:
            def apply_chat_template(self, *a, **kw):
                return "chat"

            def __call__(self, *a, **kw):
                class R:
                    input_ids = [0]
                return R()

        clf._model = _FakeModel()
        clf._tokenizer = _FakeTok()

        resultado = clf.clasificar_stance("texto", "Actor", "tema")
        assert resultado.stance == "apoyo"
        assert resultado.confianza == 0.0
