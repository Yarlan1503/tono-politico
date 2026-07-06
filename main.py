#!/usr/bin/env python3
"""Punto de entrada del pipeline tono-politico.

Ejecuta el pipeline completo de 7 componentes leyendo la configuración
desde config/config.yaml.

Flujo de dos fases:

FASE 1 (siempre): Ingesta → Diarización → Segmentación → Temas
    Descarga, transcribe, identifica al actor, segmenta y descubre tópicos.
    Imprime los tópicos encontrados en consola.

FASE 2 (con --topico N): Filtrado → Tono → Salida
    Filtra por el tópico seleccionado, analiza tono y genera informe.

Uso:
    # Fase 1: descubrir tópicos
    uv run python main.py --playlist URL

    # Fase 2: analizar un tópico específico
    uv run python main.py --playlist URL --topico 0 --tema "fracking"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ──────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────

CONFIG_PATH = Path("config/config.yaml")


def cargar_config(path: Path = CONFIG_PATH) -> dict:
    """Carga config/config.yaml y devuelve el diccionario."""
    if not path.exists():
        raise FileNotFoundError(f"Config no encontrado: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────
# Factory de services desde config
# ──────────────────────────────────────────────────────────


def _build_ingesta(cfg: dict):
    from tono_politico.ingesta import IngestaService

    c = cfg.get("ingesta", {})
    return IngestaService(
        data_dir=Path(c.get("data_dir", "data")),
        whisper_model=c.get("whisper_model", "large-v3-turbo"),
        idioma=c.get("idioma", "es"),
    )


def _build_diarizacion(cfg: dict):
    from tono_politico.diarizacion import DiarizacionService

    c = cfg.get("diarizacion", {})
    ref = c.get("referencia_voz", {})
    return DiarizacionService(
        actor=c.get("actor_objetivo", "Lilly Téllez"),
        video_ref_id=ref.get("video_id", "su9nURIj9XQ"),
        data_dir=Path(cfg.get("project", {}).get("data_dir", "data")),
        pipeline_name=c.get("pipeline", "pyannote/speaker-diarization-community-1"),
        embedding_model=c.get("speaker_embedding_model", "pyannote/embedding"),
        umbral_match=c.get("umbral_match", 0.5),
        umbral_ambiguo=c.get("umbral_ambiguo", 0.7),
    )


def _build_segmentacion(cfg: dict):
    from tono_politico.segmentacion import SegmentacionService

    c = cfg.get("segmentacion", {})
    return SegmentacionService(
        spacy_model=c.get("spacy_model", "es_core_news_lg"),
        breakpoint_percentile=c.get("breakpoint_percentile", 95),
        min_oraciones=c.get("min_oraciones", 2),
        max_oraciones=c.get("max_oraciones", 8),
        max_palabras=c.get("max_palabras", 150),
    )


def _build_temas(cfg: dict):
    from tono_politico.temas import TemasService

    c = cfg.get("temas", {})
    return TemasService(
        min_topic_size=c.get("min_topic_size", 3),
        n_neighbors=c.get("n_neighbors", 10),
        n_components=c.get("n_components", 5),
        embedding_model_name=c.get("embedding_model", "LiquidAI/LFM2.5-Embedding-350M"),
    )


def _build_filtrado(cfg: dict, topico_id: int):
    from tono_politico.filtrado import FiltradoService

    c = cfg.get("filtrado", {})
    return FiltradoService(
        topico_id=topico_id,
        min_relevancia=c.get("min_relevancia", 0.35),
        incluir_outliers=c.get("incluir_outliers", False),
    )


def _build_tono(cfg: dict, actor: str, tema: str):
    from tono_politico.tono import TonoService

    return TonoService(actor=actor, tema=tema)


def _build_salida(cfg: dict, output_path: str | None):
    from tono_politico.salida import SalidaService

    c = cfg.get("salida", {})
    fmt = c.get("formatos", ["json", "markdown"])

    if output_path:
        return SalidaService(output_path=output_path)
    elif "json" in fmt and "markdown" in fmt:
        return SalidaService(output_path=Path("output"))
    else:
        return SalidaService(output_path=None)


# ──────────────────────────────────────────────────────────
# Fases del pipeline
# ──────────────────────────────────────────────────────────


def fase_1(cfg: dict, playlist_url: str) -> None:
    """Ingesta → Diarización → Segmentación → Temas.

    Imprime los tópicos descubiertos para que el usuario elija con --topico.
    """
    logging.info("═══ FASE 1: Ingesta → Diarización → Segmentación → Temas ═══")

    # 1. Ingesta
    logging.info("─ Componente 1: Ingesta ─")
    svc_ingesta = _build_ingesta(cfg)
    transcripts = svc_ingesta.procesar(playlist_url)
    logging.info(f"  {len(transcripts)} transcripciones")

    # Necesitamos el nombre de la playlist para diarización
    from tono_politico.ingesta.playlist import obtener_info_playlist

    info = obtener_info_playlist(playlist_url)
    nombre_playlist = info.nombre

    # 1.5 Diarización
    logging.info("─ Componente 1.5: Diarización ─")
    svc_diarizacion = _build_diarizacion(cfg)
    transcripts_actor = svc_diarizacion.procesar(transcripts, nombre_playlist)
    total_segs_actor = sum(len(t.raw_segments) for t in transcripts_actor)
    logging.info(f"  {total_segs_actor} segmentos del actor")

    # 2. Segmentación
    logging.info("─ Componente 2: Segmentación ─")
    svc_segmentacion = _build_segmentacion(cfg)
    segmentos = svc_segmentacion.procesar(transcripts_actor)
    logging.info(f"  {len(segmentos)} segmentos semánticos")

    # 3. Temas
    logging.info("─ Componente 3: Temas ─")
    svc_temas = _build_temas(cfg)
    resultado_temas = svc_temas.procesar(segmentos)

    # Imprimir tópicos
    print("\n" + "=" * 60)
    print(f"Tópicos descubiertos: {resultado_temas.num_topicos}")
    print("=" * 60)
    for topico in resultado_temas.topicos:
        palabras = ", ".join(topico.palabras_clave[:8])
        print(
            f"  [{topico.id}] {topico.nombre} "
            f"({topico.num_segmentos} segmentos, "
            f"{topico.representatividad:.1%})\n"
            f"      Palabras: {palabras}"
        )
    print("=" * 60)
    print(
        "\nPara analizar un tópico, ejecuta:\n"
        "  uv run python main.py --playlist URL --topico N "
        "--tema \"descripcion del tema\"\n"
    )


def fase_2(
    cfg: dict,
    playlist_url: str,
    topico_id: int,
    tema: str,
    output_path: str | None,
) -> None:
    """Ingesta → Diarización → Segmentación → Temas → Filtrado → Tono → Salida."""
    logging.info("═══ PIPELINE COMPLETO (Fases 1 + 2) ═══")

    # Repetir fase 1
    logging.info("─ Componente 1: Ingesta ─")
    svc_ingesta = _build_ingesta(cfg)
    transcripts = svc_ingesta.procesar(playlist_url)

    from tono_politico.ingesta.playlist import obtener_info_playlist

    info = obtener_info_playlist(playlist_url)
    nombre_playlist = info.nombre

    logging.info("─ Componente 1.5: Diarización ─")
    svc_diarizacion = _build_diarizacion(cfg)
    transcripts_actor = svc_diarizacion.procesar(transcripts, nombre_playlist)

    logging.info("─ Componente 2: Segmentación ─")
    svc_segmentacion = _build_segmentacion(cfg)
    segmentos = svc_segmentacion.procesar(transcripts_actor)

    logging.info("─ Componente 3: Temas ─")
    svc_temas = _build_temas(cfg)
    resultado_temas = svc_temas.procesar(segmentos)

    # 4. Filtrado
    logging.info(f"─ Componente 4: Filtrado (tópico {topico_id}) ─")
    svc_filtrado = _build_filtrado(cfg, topico_id)
    resultado_filtrado = svc_filtrado.procesar(resultado_temas)
    logging.info(
        f"  {resultado_filtrado.total_segmentos_filtrados}/"
        f"{resultado_filtrado.total_segmentos_entrada} segmentos filtrados"
    )

    if resultado_filtrado.total_segmentos_filtrados == 0:
        logging.error(
            f"No hay segmentos para el tópico {topico_id}. "
            f"Verifica el ID con la fase 1."
        )
        sys.exit(1)

    # 5. Tono
    actor = cfg.get("diarizacion", {}).get("actor_objetivo", "Lilly Téllez")
    logging.info(f"─ Componente 5: Tono (actor='{actor}', tema='{tema}') ─")
    svc_tono = _build_tono(cfg, actor, tema)
    resultado_tono = svc_tono.procesar(resultado_filtrado)
    logging.info(f"  {len(resultado_tono.segmentos)} segmentos analizados")

    # 6. Salida
    logging.info("─ Componente 6: Salida ─")
    svc_salida = _build_salida(cfg, output_path)
    informe = svc_salida.procesar(resultado_tono)
    logging.info(f"  Informe generado: {informe.perfil.actor} / {informe.perfil.tema}")

    print(f"\n✅ Pipeline completo. Informe: {svc_salida.output_path or '(sin disco)'}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="tono-politico: análisis del tono político en YouTube",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--playlist",
        required=True,
        help="URL de la playlist de YouTube",
    )
    parser.add_argument(
        "--topico",
        type=int,
        default=None,
        help="ID del tópico a analizar (Fase 2). Sin esto, solo descubre tópicos.",
    )
    parser.add_argument(
        "--tema",
        type=str,
        default=None,
        help="Descripción del tema a evaluar (ej. 'fracking'). Requerido con --topico.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta de salida (archivo .json/.md o directorio). Default: output/",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help=f"Ruta del config YAML (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Logging DEBUG en vez de INFO",
    )

    args = parser.parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Cargar config
    cfg = cargar_config(Path(args.config))

    # Validar argumentos
    if args.topico is not None and not args.tema:
        parser.error("--tema es obligatorio cuando se usa --topico")

    # Ejecutar
    if args.topico is not None:
        fase_2(cfg, args.playlist, args.topico, args.tema, args.output)
    else:
        fase_1(cfg, args.playlist)


if __name__ == "__main__":
    main()
