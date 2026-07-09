"""Runner testeable del pipeline completo.

La clase de este módulo mueve la orquestación fuera del CLI para que las fases
puedan probarse con services inyectables y sin cargar modelos pesados.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tono_politico.config import Config
from tono_politico.models import PlaylistInfo, VideoTranscript
from tono_politico.temas.models import ResultadoTemas
from tono_politico.temas.serializacion import cargar_fase1, guardar_fase1

from .manifest import guardar_manifest
from .models import PhaseName, PhaseRunStatus, RunManifest, RunResult, VideoRunStatus

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceFactories:
    """Factories inyectables para construir los services del pipeline."""

    build_ingesta: Callable[[Config], Any]
    build_diarizacion: Callable[[Config], Any]
    build_segmentacion: Callable[[Config], Any]
    build_temas: Callable[[Config], Any]
    build_filtrado: Callable[[Config, int], Any]
    build_tono: Callable[[Config, str, str], Any]
    build_salida: Callable[[Config, str | None], Any]
    get_playlist_info: Callable[[str], PlaylistInfo]
    # Path discursive_approach (opcional; requerido por discover_discursive)
    build_speech2text: Callable[[Config], Any] | None = None
    build_discursive: Callable[[Config], Any] | None = None


@dataclass
class Fase1Resultado:
    """Resultado interno de Fase 1 usado por discover/analyze."""

    resultado_temas: ResultadoTemas
    manifest: RunManifest


@dataclass
class PipelineRunner:
    """Orquesta el pipeline sin depender de argparse ni sys.exit."""

    cfg: Config
    factories: ServiceFactories
    keep_cache: bool = False
    run_id: str | None = None
    last_resultado_temas: ResultadoTemas | None = field(default=None, init=False)
    last_resultado_enfoques: Any | None = field(default=None, init=False)
    last_resultado_temas_discursive: Any | None = field(default=None, init=False)
    _active_manifest: RunManifest | None = field(default=None, init=False, repr=False)

    def discover(self, playlist_url: str) -> RunResult:
        """Ejecuta Fase 1: Ingesta → Diarización → Segmentación → Temas.

        Persiste ``fase1-topicos.json`` junto al manifest para que
        ``analyze_resume`` pueda reutilizarlo sin re-ejecutar Fase 1.
        """
        self._active_manifest = None
        manifest: RunManifest | None = None
        try:
            fase_1 = self._ejecutar_fase_1(playlist_url)
            manifest = fase_1.manifest
            manifest_path = self._persistir_manifest(manifest)
            self._persistir_fase1(fase_1.resultado_temas, manifest)
            return RunResult(manifest=manifest, exit_code=0, manifest_path=manifest_path)
        except Exception:
            manifest = self._active_manifest or _failure_manifest(playlist_url)
            manifest.status = "failed"
            manifest_path = self._persistir_manifest(manifest)
            return RunResult(
                manifest=manifest,
                exit_code=1,
                manifest_path=manifest_path,
            )
        finally:
            if manifest is not None:
                self._limpiar_cache(manifest.playlist_name)

    def discover_discursive(self, playlist_url: str) -> RunResult:
        """Fase 1 path nuevo: speech2text → argument_shape → topics_cluster → topics_approach.

        Persiste ``discursive-temas.json`` y ``discursive-enfoques.json`` en el run dir.
        No ejecuta filtrado ni salida (etapas posteriores al umbrella).
        """
        self._active_manifest = None
        manifest: RunManifest | None = None
        try:
            if self.factories.build_speech2text is None or self.factories.build_discursive is None:
                raise RuntimeError(
                    "discover_discursive requiere factories.build_speech2text y build_discursive"
                )

            info = self.factories.get_playlist_info(playlist_url)
            manifest = self._crear_manifest(playlist_url, info)

            svc_s2t = self.factories.build_speech2text(self.cfg)
            transcripts = self._run_phase(
                manifest,
                "speech2text",
                lambda: svc_s2t.procesar(playlist_url),
            )
            manifest.videos = _video_statuses_actor(transcripts)

            svc_disc = self.factories.build_discursive(self.cfg)
            argumentos = self._run_phase(
                manifest,
                "argument_shape",
                lambda: svc_disc.shape_corpus(transcripts),
            )
            resultado_temas = self._run_phase(
                manifest,
                "topics_cluster",
                lambda: svc_disc.cluster(argumentos),
            )
            resultado_enfoques = self._run_phase(
                manifest,
                "topics_approach",
                lambda: svc_disc.approaches(resultado_temas),
            )

            self.last_resultado_temas_discursive = resultado_temas
            self.last_resultado_enfoques = resultado_enfoques

            manifest_path = self._persistir_manifest(manifest)
            self._persistir_discursive(resultado_temas, resultado_enfoques, manifest)
            return RunResult(manifest=manifest, exit_code=0, manifest_path=manifest_path)
        except Exception:
            manifest = self._active_manifest or _failure_manifest(playlist_url)
            manifest.status = "failed"
            manifest_path = self._persistir_manifest(manifest)
            return RunResult(
                manifest=manifest,
                exit_code=1,
                manifest_path=manifest_path,
            )
        finally:
            if manifest is not None:
                self._limpiar_cache(manifest.playlist_name)

    def analyze(
        self,
        playlist_url: str,
        topico_id: int,
        tema: str,
        output_path: str | None,
    ) -> RunResult:
        """Ejecuta Fase 1 + Filtrado → Tono → Salida."""
        self._active_manifest = None
        manifest: RunManifest | None = None
        try:
            fase_1 = self._ejecutar_fase_1(playlist_url)
            manifest = fase_1.manifest

            svc_filtrado = self.factories.build_filtrado(self.cfg, topico_id)
            resultado_filtrado = self._run_phase(
                manifest,
                "filtrado",
                lambda: svc_filtrado.procesar(fase_1.resultado_temas),
            )

            if resultado_filtrado.total_segmentos_filtrados == 0:
                manifest.status = "failed"
                manifest.phases[-1] = PhaseRunStatus(
                    phase="filtrado",
                    ok=False,
                    elapsed_seconds=manifest.phases[-1].elapsed_seconds,
                    message=f"No hay segmentos para el tópico {topico_id}",
                )
                return RunResult(manifest=manifest, exit_code=1)

            actor = self.cfg.diarizacion.actor_objetivo
            svc_tono = self.factories.build_tono(self.cfg, actor, tema)
            resultado_tono = self._run_phase(
                manifest,
                "tono",
                lambda: svc_tono.procesar(resultado_filtrado),
            )

            svc_salida = self.factories.build_salida(self.cfg, output_path)
            self._run_phase(manifest, "salida", lambda: svc_salida.procesar(resultado_tono))

            informe_path = getattr(svc_salida, "output_path", None)
            manifest_path = self._persistir_manifest(manifest)
            return RunResult(
                manifest=manifest,
                exit_code=0,
                informe_path=Path(informe_path) if informe_path is not None else None,
                manifest_path=manifest_path,
            )
        except Exception:
            manifest = self._active_manifest or _failure_manifest(playlist_url)
            manifest.status = "failed"
            manifest_path = self._persistir_manifest(manifest)
            return RunResult(
                manifest=manifest,
                exit_code=1,
                manifest_path=manifest_path,
            )
        finally:
            if manifest is not None:
                self._limpiar_cache(manifest.playlist_name)

    def _ejecutar_fase_1(self, playlist_url: str) -> Fase1Resultado:
        info = self.factories.get_playlist_info(playlist_url)
        manifest = self._crear_manifest(playlist_url, info)

        svc_ingesta = self.factories.build_ingesta(self.cfg)
        transcripts = self._run_phase(
            manifest,
            "ingesta",
            lambda: svc_ingesta.procesar(playlist_url),
        )

        svc_diarizacion = self.factories.build_diarizacion(self.cfg)
        transcripts_actor = self._run_phase(
            manifest,
            "diarizacion",
            lambda: svc_diarizacion.procesar(transcripts, info.nombre),
        )

        manifest.videos = _video_statuses(transcripts, transcripts_actor)

        svc_segmentacion = self.factories.build_segmentacion(self.cfg)
        segmentos = self._run_phase(
            manifest,
            "segmentacion",
            lambda: svc_segmentacion.procesar(transcripts_actor),
        )

        svc_temas = self.factories.build_temas(self.cfg)
        resultado_temas = self._run_phase(
            manifest,
            "temas",
            lambda: svc_temas.procesar(segmentos),
        )
        self.last_resultado_temas = resultado_temas

        return Fase1Resultado(resultado_temas=resultado_temas, manifest=manifest)

    def _crear_manifest(self, playlist_url: str, info: PlaylistInfo) -> RunManifest:
        manifest = RunManifest(
            run_id=self.run_id if self.run_id else _run_id(),
            playlist_url=playlist_url,
            playlist_name=info.nombre,
            status="ok",
            cache_dir=self._cache_dir(info.nombre),
        )
        self._active_manifest = manifest
        return manifest

    def _run_phase(
        self,
        manifest: RunManifest,
        phase: PhaseName,
        fn: Callable[[], Any],
    ) -> Any:
        started = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:
            manifest.status = "failed"
            manifest.phases.append(
                PhaseRunStatus(
                    phase=phase,
                    ok=False,
                    elapsed_seconds=time.perf_counter() - started,
                    message=f"{type(exc).__name__}: {exc}",
                )
            )
            raise
        manifest.phases.append(
            PhaseRunStatus(
                phase=phase,
                ok=True,
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        return result

    def _cache_dir(self, playlist_name: str) -> Path:
        return self.cfg.project.data_dir / playlist_name

    def _persistir_manifest(self, manifest: RunManifest) -> Path | None:
        """Persiste el manifest en ``output/runs/<run_id>/manifest.json``.

        Si la escritura falla, registra warning y devuelve None
        (no debe interrumpir el flujo del pipeline).
        """
        output_base = self.cfg.project.output_dir
        try:
            return guardar_manifest(manifest, output_base)
        except Exception:
            logger.warning("No se pudo persistir el manifest", exc_info=True)
            return None

    def _persistir_fase1(self, resultado: ResultadoTemas, manifest: RunManifest) -> None:
        """Persiste ``fase1-topicos.json`` junto al manifest."""
        if manifest.artifacts_dir is None:
            return
        try:
            guardar_fase1(resultado, manifest.artifacts_dir)
        except Exception:
            logger.warning("No se pudo persistir fase1-topicos.json", exc_info=True)

    def _persistir_discursive(
        self,
        resultado_temas: Any,
        resultado_enfoques: Any,
        manifest: RunManifest,
    ) -> None:
        """Persiste artefactos del path discursive_approach."""
        if manifest.artifacts_dir is None:
            return
        try:
            from tono_politico.discursive_approach.topics_approach.serializacion import (
                guardar_resultado_enfoques,
            )
            from tono_politico.discursive_approach.topics_cluster.serializacion import (
                guardar_resultado_temas,
            )

            guardar_resultado_temas(
                resultado_temas,
                manifest.artifacts_dir / "discursive-temas.json",
            )
            guardar_resultado_enfoques(
                resultado_enfoques,
                manifest.artifacts_dir / "discursive-enfoques.json",
            )
        except Exception:
            logger.warning("No se pudo persistir artefactos discursive", exc_info=True)

    def analyze_resume(
        self,
        run_dir: str | Path,
        topico_id: int,
        tema: str,
        output_path: str | None,
    ) -> RunResult:
        """Ejecuta solo Fase 2 cargando Fase 1 desde disco.

        No llama ingesta/diarización/segmentación/temas.
        """
        run_dir = Path(run_dir)
        manifest: RunManifest | None = None
        try:
            resultado_temas = cargar_fase1(run_dir)

            manifest = RunManifest(
                run_id=run_dir.name,
                playlist_url="(resume)",
                playlist_name="(resume)",
                status="ok",
                artifacts_dir=run_dir,
            )
            self._active_manifest = manifest

            svc_filtrado = self.factories.build_filtrado(self.cfg, topico_id)
            resultado_filtrado = self._run_phase(
                manifest,
                "filtrado",
                lambda: svc_filtrado.procesar(resultado_temas),
            )

            if resultado_filtrado.total_segmentos_filtrados == 0:
                manifest.status = "failed"
                manifest.phases[-1] = PhaseRunStatus(
                    phase="filtrado",
                    ok=False,
                    elapsed_seconds=manifest.phases[-1].elapsed_seconds,
                    message=f"No hay segmentos para el tópico {topico_id}",
                )
                manifest_path = self._persistir_manifest(manifest)
                return RunResult(
                    manifest=manifest,
                    exit_code=1,
                    manifest_path=manifest_path,
                )

            actor = self.cfg.diarizacion.actor_objetivo
            svc_tono = self.factories.build_tono(self.cfg, actor, tema)
            resultado_tono = self._run_phase(
                manifest,
                "tono",
                lambda: svc_tono.procesar(resultado_filtrado),
            )

            svc_salida = self.factories.build_salida(self.cfg, output_path)
            self._run_phase(manifest, "salida", lambda: svc_salida.procesar(resultado_tono))

            informe_path = getattr(svc_salida, "output_path", None)
            manifest_path = self._persistir_manifest(manifest)
            return RunResult(
                manifest=manifest,
                exit_code=0,
                informe_path=Path(informe_path) if informe_path is not None else None,
                manifest_path=manifest_path,
            )
        except Exception:
            manifest = manifest or RunManifest(
                run_id=run_dir.name,
                playlist_url="(resume)",
                playlist_name="(resume)",
                status="failed",
                artifacts_dir=run_dir,
            )
            manifest.status = "failed"
            manifest_path = self._persistir_manifest(manifest)
            return RunResult(
                manifest=manifest,
                exit_code=1,
                manifest_path=manifest_path,
            )

    def _limpiar_cache(self, playlist_name: str) -> None:
        if not playlist_name:
            logger.warning(
                "Se omitió limpieza de cache: playlist_name vacío "
                "(posible fallo antes de obtener metadata)."
            )
            return
        playlist_dir = self._cache_dir(playlist_name)
        if not playlist_dir.exists():
            logger.debug("Cache no encontrado: %s", playlist_dir)
            return
        if self.keep_cache:
            logger.info("Cache conservado (--keep-cache): %s", playlist_dir)
            return
        logger.info("Limpiando cache runtime: %s", playlist_dir)
        shutil.rmtree(playlist_dir)


def _run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _failure_manifest(playlist_url: str) -> RunManifest:
    return RunManifest(
        run_id=_run_id(),
        playlist_url=playlist_url,
        playlist_name="",
        status="failed",
    )


def _video_statuses(
    transcripts: list[VideoTranscript],
    transcripts_actor: list[VideoTranscript],
) -> list[VideoRunStatus]:
    actor_by_id = {transcript.video_id: transcript for transcript in transcripts_actor}
    statuses: list[VideoRunStatus] = []
    for transcript in transcripts:
        actor_transcript = actor_by_id.get(transcript.video_id)
        segmentos_actor = len(actor_transcript.raw_segments) if actor_transcript else 0
        statuses.append(
            VideoRunStatus(
                video_id=transcript.video_id,
                titulo=transcript.titulo,
                descargado=True,
                transcrito=True,
                diarizado=actor_transcript is not None,
                segmentos_actor=segmentos_actor,
                omitido=segmentos_actor == 0,
            )
        )
    return statuses


def _video_statuses_actor(transcripts: list[Any]) -> list[VideoRunStatus]:
    """Estados de video a partir de ActorTranscript (path speech2text)."""
    statuses: list[VideoRunStatus] = []
    for transcript in transcripts:
        n_seg = len(getattr(transcript, "segments", []) or [])
        statuses.append(
            VideoRunStatus(
                video_id=getattr(transcript, "video_id", ""),
                titulo="",
                descargado=True,
                transcrito=n_seg > 0,
                diarizado=True,
                segmentos_actor=n_seg,
                omitido=n_seg == 0,
            )
        )
    return statuses
