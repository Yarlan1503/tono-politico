"""Firmas de tono + orden temporal (sin HDBSCAN)."""

from __future__ import annotations

from collections import Counter, defaultdict

from ..topics_cluster.models import ArgumentoTematizado, TopicoInfo
from .models import (
    ArgumentoConEnfoque,
    EnfoqueInfo,
    PerfilTonoArgumento,
    ResultadoEnfoquesTema,
)


def intensidad_bin(intensidad: int) -> str:
    """Bin de intensidad: 1–2 / 3 / 4–5."""
    if intensidad <= 2:
        return "1-2"
    if intensidad == 3:
        return "3"
    return "4-5"


def firma_de_perfil(perfil: PerfilTonoArgumento) -> tuple[str, ...]:
    """Firma determinista del perfil de tono."""
    return (
        perfil.stance,
        perfil.logica_dominante,
        perfil.sentimiento_dominante,
        perfil.estilo_dominante,
        perfil.funcion_dominante,
        intensidad_bin(perfil.intensidad),
    )


def nombre_desde_firma(firma: tuple[str, ...]) -> str:
    """Etiqueta legible corta para un enfoque."""
    return " · ".join(firma)


def descubrir_enfoques_tema(
    topico: TopicoInfo,
    items: list[tuple[ArgumentoTematizado, PerfilTonoArgumento]],
) -> ResultadoEnfoquesTema:
    """Agrupa por firma de tono y ordena por (fecha, t_start)."""
    if not items:
        return ResultadoEnfoquesTema(topico=topico)

    # firma → list of (ArgumentoTematizado, Perfil)
    buckets: dict[tuple[str, ...], list[tuple[ArgumentoTematizado, PerfilTonoArgumento]]] = (
        defaultdict(list)
    )
    for at, perfil in items:
        buckets[firma_de_perfil(perfil)].append((at, perfil))

    # IDs estables por orden de primera aparición temporal
    def sort_key(pair: tuple[ArgumentoTematizado, PerfilTonoArgumento]) -> tuple:
        a = pair[0].argumento
        return (a.fecha or "", a.t_start)

    firmas_orden = sorted(
        buckets.keys(),
        key=lambda f: min(sort_key(p) for p in buckets[f]),
    )

    firma_a_id = {f: i for i, f in enumerate(firmas_orden)}
    enfoques: list[EnfoqueInfo] = []
    argumentos_out: list[ArgumentoConEnfoque] = []

    for firma in firmas_orden:
        miembros = sorted(buckets[firma], key=sort_key)
        eid = firma_a_id[firma]
        perfiles = [p for _, p in miembros]
        fechas = [m[0].argumento.fecha for m in miembros if m[0].argumento.fecha]
        enfoques.append(
            EnfoqueInfo(
                id=eid,
                topico_id=topico.id,
                nombre=nombre_desde_firma(firma),
                palabras_clave=[],
                num_argumentos=len(miembros),
                fecha_primera=min(fechas) if fechas else None,
                fecha_ultima=max(fechas) if fechas else None,
                stance_dominante=_mode([p.stance for p in perfiles]),
                intensidad_media=sum(p.intensidad for p in perfiles) / len(perfiles),
                logica_dominante=firma[1],
                sentimiento_dominante=firma[2],
                estilo_dominante=firma[3],
                funcion_dominante=firma[4],
            )
        )
        for at, perfil in miembros:
            argumentos_out.append(
                ArgumentoConEnfoque(
                    argumento=at.argumento,
                    topico_id=topico.id,
                    enfoque_id=eid,
                    probabilidad_topico=at.probabilidad,
                    probabilidad_enfoque=1.0,
                    tono=perfil,
                )
            )

    argumentos_out.sort(
        key=lambda x: (x.argumento.fecha or "", x.argumento.t_start),
    )
    return ResultadoEnfoquesTema(
        topico=topico,
        enfoques=enfoques,
        argumentos=argumentos_out,
    )


def _mode(values: list[str]) -> str:
    if not values:
        return ""
    return Counter(values).most_common(1)[0][0]
