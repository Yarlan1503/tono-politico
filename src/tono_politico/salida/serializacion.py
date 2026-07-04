"""Serialización de DTOs a JSON y Markdown.

Funciones puras que convierten los DTOs del pipeline en formatos
consumibles: JSON para dashboards/programas, Markdown para humanos.
"""

from __future__ import annotations

import json

from ..tono.models import ResultadoTono, SegmentoConTono
from .models import PerfilActor, Provenance


# ============================================================
# DICT (intermedio para JSON)
# ============================================================
def perfil_a_dict(perfil: PerfilActor) -> dict:
    """Convierte un PerfilActor a dict serializable."""
    return {
        "actor": perfil.actor,
        "tema": perfil.tema,
        "n_segmentos": perfil.n_segmentos,
        "stance_dominante": perfil.stance_dominante,
        "intensidad_promedio": round(perfil.intensidad_promedio, 2),
        "logica_dominante": perfil.logica_dominante,
        "sentimiento_dominante": perfil.sentimiento_dominante,
        "estilo_dominante": perfil.estilo_dominante,
        "funcion_dominante": perfil.funcion_dominante,
    }


def segmento_a_dict(seg: SegmentoConTono) -> dict:
    """Convierte un SegmentoConTono a dict serializable."""
    return {
        "texto": seg.segmento.texto,
        "video_id": seg.segmento.video_id,
        "t_start": seg.segmento.t_start,
        "t_end": seg.segmento.t_end,
        "stance": {
            "stance": seg.stance.stance,
            "confianza": seg.stance.confianza,
        },
        "intensidad_antagonica": seg.intensidad_antagonica,
        "logica_politica": {
            e.etiqueta: round(e.score, 4)
            for e in seg.logica_politica.to_scores()
        },
        "sentimiento": {
            e.etiqueta: round(e.score, 4)
            for e in seg.sentimiento.to_scores()
        },
        "estilo_discursivo": {
            e.etiqueta: round(e.score, 4)
            for e in seg.estilo_discursivo.to_scores()
        },
        "funcion_discursiva": {
            e.etiqueta: round(e.score, 4)
            for e in seg.funcion_discursiva.to_scores()
        },
    }


def provenance_a_dict(prov: Provenance) -> dict:
    """Convierte un Provenance a dict serializable."""
    return {
        "pipeline": prov.pipeline,
        "modelos": prov.modelos,
        "fecha": prov.fecha,
        "marco_teorico": prov.marco_teorico,
        "advertencia_confianza": prov.advertencia_confianza,
    }


# ============================================================
# JSON
# ============================================================
def generar_json(
    resultado: ResultadoTono,
    perfil: PerfilActor,
    provenance: Provenance,
) -> str:
    """Genera el JSON completo del informe.

    Args:
        resultado: ResultadoTono del Componente 5.
        perfil: PerfilActor agregado.
        provenance: Metadata de generación.

    Returns:
        String JSON con indent=2.
    """
    data = {
        "perfil": perfil_a_dict(perfil),
        "provenance": provenance_a_dict(provenance),
        "segmentos": [
            segmento_a_dict(seg) for seg in resultado.segmentos
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


# ============================================================
# MARKDOWN
# ============================================================
def generar_markdown(
    resultado: ResultadoTono,
    perfil: PerfilActor,
    provenance: Provenance,
) -> str:
    """Genera un reporte Markdown legible para humanos.

    Args:
        resultado: ResultadoTono del Componente 5.
        perfil: PerfilActor agregado.
        provenance: Metadata de generación.

    Returns:
        String con el reporte en Markdown.
    """
    lines: list[str] = []

    # Título
    lines.append(f"# {perfil.actor} — {perfil.tema}")
    lines.append("")

    # Tabla de perfil
    lines.append("## Perfil agregado")
    lines.append("")
    lines.append("| Dimensión | Resultado |")
    lines.append("|---|---|")
    lines.append(f"| Segmentos analizados | {perfil.n_segmentos} |")
    lines.append(f"| Stance | {perfil.stance_dominante} |")
    lines.append(f"| Intensidad antagónica | {perfil.intensidad_promedio:.1f} / 5 |")
    lines.append(f"| Lógica política | {perfil.logica_dominante} |")
    lines.append(f"| Sentimiento | {perfil.sentimiento_dominante} |")
    lines.append(f"| Estilo discursivo | {perfil.estilo_dominante} |")
    lines.append(f"| Función discursiva | {perfil.funcion_dominante} |")
    lines.append("")

    # Provenance
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- **Pipeline:** {provenance.pipeline}")
    lines.append(f"- **Modelos:** {', '.join(provenance.modelos)}")
    lines.append(f"- **Fecha:** {provenance.fecha}")
    lines.append("")

    # Advertencia
    lines.append("## Advertencia de confianza")
    lines.append("")
    lines.append(f"> {provenance.advertencia_confianza}")
    lines.append("")

    return "\n".join(lines)
