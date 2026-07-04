"""Tests para tono/taxonomia.py — integridad de los datos de prototipos."""

from __future__ import annotations

from tono_politico.tono.taxonomia import (
    STANCE_LABELS,
    prototipos_de,
    todas_las_dimensiones,
)


class TestEstructuraDimensiones:
    def test_hay_exactamente_cinco_dimensiones(self):
        dims = todas_las_dimensiones()
        assert len(dims) == 5
        assert set(dims) == {
            "logica_politica",
            "sentimiento",
            "estilo_discursivo",
            "funcion_discursiva",
            "intensidad",
        }

    def test_logica_politica_tiene_seis_labels(self):
        labels = prototipos_de("logica_politica")
        assert len(labels) == 6
        assert set(labels.keys()) == {
            "nacionalista",
            "globalista",
            "populista",
            "tecnocrata",
            "corporativista",
            "estatista",
        }

    def test_sentimiento_tiene_cinco_labels(self):
        labels = prototipos_de("sentimiento")
        assert len(labels) == 5
        assert set(labels.keys()) == {
            "esperanza",
            "angustia",
            "indignacion",
            "orgullo",
            "empatia",
        }

    def test_estilo_discursivo_tiene_seis_labels(self):
        labels = prototipos_de("estilo_discursivo")
        assert len(labels) == 6
        assert set(labels.keys()) == {
            "directo",
            "academico",
            "confrontativo",
            "conciliador",
            "catastrofista",
            "testimonial",
        }

    def test_funcion_discursiva_tiene_tres_labels(self):
        labels = prototipos_de("funcion_discursiva")
        assert len(labels) == 3
        assert set(labels.keys()) == {
            "critica",
            "propuesta",
            "narrativa_personal",
        }

    def test_intensidad_tiene_cinco_niveles(self):
        labels = prototipos_de("intensidad")
        assert len(labels) == 5
        assert set(labels.keys()) == {"1", "2", "3", "4", "5"}


class TestPrototipos:
    def test_todos_los_prototipos_son_no_vacios(self):
        for dim in todas_las_dimensiones():
            for label, prototipo in prototipos_de(dim).items():
                assert isinstance(prototipo, str), f"{dim}.{label} no es str"
                assert len(prototipo) > 30, (
                    f"{dim}.{label} prototipo muy corto ({len(prototipo)} chars)"
                )

    def test_prototipos_no_son_duplicados(self):
        """Ningún prototipo debe ser idéntico a otro."""
        todos = []
        for dim in todas_las_dimensiones():
            for prototipo in prototipos_de(dim).values():
                todos.append(prototipo)
        assert len(todos) == len(set(todos)), "Hay prototipos duplicados"

    def test_prototipos_intensidad_escalan(self):
        """Los prototipos de intensidad deben ir de conciliador a beligerante."""
        protos = prototipos_de("intensidad")
        assert "conciliador" in protos["1"].lower()
        assert "beligerant" in protos["5"].lower() or "existencial" in protos["5"].lower()


class TestStance:
    def test_stance_labels_son_apoyo_y_rechazo(self):
        assert set(STANCE_LABELS) == {"apoyo", "rechazo"}
        assert len(STANCE_LABELS) == 2


class TestPrototiposDe:
    def test_devuelve_dict_str_str_para_dimension_valida(self):
        result = prototipos_de("sentimiento")
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, str)
            assert isinstance(v, str)

    def test_keyerror_para_dimension_inexistente(self):
        try:
            prototipos_de("no_existe")
            assert False, "Debería haber lanzado KeyError"
        except KeyError:
            pass
