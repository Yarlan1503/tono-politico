"""DTOs del control de ejecución stage-based."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

SCHEMA_VERSION = "tono-politico.run.v1"

StageName: TypeAlias = Literal[
    "speech2text",
    "argument_shape",
    "topics_cluster",
    "topics_approach",
]
ArtifactKey: TypeAlias = Literal[
    "playlist_url",
    "actor_transcripts_dir",
    "argumentos_path",
    "temas_path",
    "enfoques_path",
]
StageStatus: TypeAlias = Literal["ok", "failed", "skipped"]

STAGE_NAMES: tuple[StageName, ...] = (
    "speech2text",
    "argument_shape",
    "topics_cluster",
    "topics_approach",
)


@dataclass(frozen=True)
class RunSettings:
    id: str | None = None
    stages: list[StageName] = field(default_factory=lambda: list(STAGE_NAMES))
    resume: bool = True
    overwrite: bool = False
    keep_cache: bool = False
    fail_fast: bool = False
    max_videos: int | None = None
    only_video_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InputConfig:
    playlist_url: str | None = None
    actor_transcripts_dir: Path | None = None
    argumentos_path: Path | None = None
    temas_path: Path | None = None
    enfoques_path: Path | None = None


@dataclass(frozen=True)
class OutputConfig:
    base_dir: Path = Path("output")
    run_dir: Path | None = None
    persist_resolved_config: bool = True
    persist_manifest: bool = True


@dataclass(frozen=True)
class ProjectExecutionConfig:
    data_dir: Path = Path("data")
    idioma: str = "es"
    random_state: int = 42


@dataclass(frozen=True)
class AudioFetcherExecutionConfig:
    enabled: bool = True
    force_download: bool = False
    playlist_dir_template: str = "{playlist}"
    audio_dir_template: str = "videos-{playlist}"


@dataclass(frozen=True)
class ReferenceVoiceConfig:
    origen: str = "misma_playlist"
    max_audios: int = 1
    video_id: str = "su9nURIj9XQ"
    url: str | None = "https://www.youtube.com/watch?v=su9nURIj9XQ&list=PLE9Zk7g9R__M&index=8"
    cache: str = "solo_ejecucion"


@dataclass(frozen=True)
class SpeakerTimestampsExecutionConfig:
    enabled: bool = True
    actor_objetivo: str = "Lilly Téllez"
    pipeline: str = "pyannote/speaker-diarization-community-1"
    fallback_pipeline: str | None = "pyannote-community/speaker-diarization-community-1"
    device: str = "auto"
    umbral_match: float = 0.5
    umbral_ambiguo: float = 0.7
    match_ambiguo: str = "descartar_como_otro_speaker"
    referencia_voz: ReferenceVoiceConfig = field(default_factory=ReferenceVoiceConfig)


@dataclass(frozen=True)
class TranscribeSpeechExecutionConfig:
    enabled: bool = True
    whisper_model: str = "large-v3-turbo"
    idioma: str = "es"
    word_timestamps: bool = False
    force_retranscribe: bool = False
    skip_existing_transcripts: bool = True


@dataclass(frozen=True)
class SpeechToTextExecutionConfig:
    enabled: bool = True
    audio_fetcher: AudioFetcherExecutionConfig = field(default_factory=AudioFetcherExecutionConfig)
    speaker_timestamps: SpeakerTimestampsExecutionConfig = field(
        default_factory=SpeakerTimestampsExecutionConfig
    )
    transcribe_speech: TranscribeSpeechExecutionConfig = field(
        default_factory=TranscribeSpeechExecutionConfig
    )


@dataclass(frozen=True)
class DiscursiveInputConfig:
    source: str = "previous_stage"
    actor_transcripts_dir: Path | None = None


@dataclass(frozen=True)
class ArgumentShapeExecutionConfig:
    enabled: bool = True
    force: bool = False
    spacy_model: str = "es_core_news_lg"
    embedding_model: str = "LiquidAI/LFM2.5-Embedding-350M"
    breakpoint_percentile: int = 95
    min_oraciones: int = 2
    max_oraciones: int = 8
    max_palabras: int = 150


@dataclass(frozen=True)
class UmapExecutionConfig:
    metric: str = "cosine"
    random_state: int = 42


@dataclass(frozen=True)
class HdbscanExecutionConfig:
    min_samples: int = 1
    metric: str = "euclidean"
    cluster_selection_method: str = "eom"


@dataclass(frozen=True)
class BertopicExecutionConfig:
    language: str = "spanish"
    calculate_probabilities: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class TopicsClusterExecutionConfig:
    enabled: bool = True
    force: bool = False
    embedding_model: str = "LiquidAI/LFM2.5-Embedding-350M"
    min_topic_size: int = 3
    n_neighbors: int = 10
    n_components: int = 5
    umap: UmapExecutionConfig = field(default_factory=UmapExecutionConfig)
    hdbscan: HdbscanExecutionConfig = field(default_factory=HdbscanExecutionConfig)
    bertopic: BertopicExecutionConfig = field(default_factory=BertopicExecutionConfig)


@dataclass(frozen=True)
class TopicsApproachExecutionConfig:
    enabled: bool = True
    force: bool = False


@dataclass(frozen=True)
class DiscursiveApproachExecutionConfig:
    enabled: bool = True
    input: DiscursiveInputConfig = field(default_factory=DiscursiveInputConfig)
    argument_shape: ArgumentShapeExecutionConfig = field(
        default_factory=ArgumentShapeExecutionConfig
    )
    topics_cluster: TopicsClusterExecutionConfig = field(
        default_factory=TopicsClusterExecutionConfig
    )
    topics_approach: TopicsApproachExecutionConfig = field(
        default_factory=TopicsApproachExecutionConfig
    )


@dataclass(frozen=True)
class RunConfig:
    schema_version: str = SCHEMA_VERSION
    run: RunSettings = field(default_factory=RunSettings)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    project: ProjectExecutionConfig = field(default_factory=ProjectExecutionConfig)
    speech2text: SpeechToTextExecutionConfig = field(default_factory=SpeechToTextExecutionConfig)
    discursive_approach: DiscursiveApproachExecutionConfig = field(
        default_factory=DiscursiveApproachExecutionConfig
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> RunConfig:
        schema_version = str(data.get("schema_version", ""))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version debe ser {SCHEMA_VERSION!r}: {schema_version!r}")

        run_data = _section(data, "run")
        input_data = _section(data, "input")
        output_data = _section(data, "output")
        project_data = _section(data, "project")
        speech_data = _section(data, "speech2text")
        discursive_data = _section(data, "discursive_approach")

        stages_raw = run_data.get("stages", list(STAGE_NAMES))
        stages = _stages(stages_raw)

        speaker_data = _section(speech_data, "speaker_timestamps")
        ref_data = _section(speaker_data, "referencia_voz")
        discursive_input_data = _section(discursive_data, "input")
        argument_shape_data = _section(discursive_data, "argument_shape")
        topics_cluster_data = _section(discursive_data, "topics_cluster")

        return cls(
            schema_version=schema_version,
            run=RunSettings(
                id=_optional_str(run_data.get("id")),
                stages=stages,
                resume=bool(run_data.get("resume", True)),
                overwrite=bool(run_data.get("overwrite", False)),
                keep_cache=bool(run_data.get("keep_cache", False)),
                fail_fast=bool(run_data.get("fail_fast", False)),
                max_videos=_optional_int(run_data.get("max_videos")),
                only_video_ids=_string_list(
                    run_data.get("only_video_ids", []),
                    "run.only_video_ids",
                ),
            ),
            input=InputConfig(
                playlist_url=_optional_str(input_data.get("playlist_url")),
                actor_transcripts_dir=_optional_path(input_data.get("actor_transcripts_dir")),
                argumentos_path=_optional_path(input_data.get("argumentos_path")),
                temas_path=_optional_path(input_data.get("temas_path")),
                enfoques_path=_optional_path(input_data.get("enfoques_path")),
            ),
            output=OutputConfig(
                base_dir=_path(output_data.get("base_dir", "output")),
                run_dir=_optional_path(output_data.get("run_dir")),
                persist_resolved_config=bool(output_data.get("persist_resolved_config", True)),
                persist_manifest=bool(output_data.get("persist_manifest", True)),
            ),
            project=ProjectExecutionConfig(
                data_dir=_path(project_data.get("data_dir", "data")),
                idioma=str(project_data.get("idioma", "es")),
                random_state=int(project_data.get("random_state", 42)),
            ),
            speech2text=_speech2text_from_mapping(speech_data, speaker_data, ref_data),
            discursive_approach=_discursive_from_mapping(
                discursive_data,
                discursive_input_data,
                argument_shape_data,
                topics_cluster_data,
            ),
        )


@dataclass(frozen=True)
class ArtifactPaths:
    run_dir: Path
    manifest_path: Path
    resolved_config_path: Path
    actor_transcripts_dir: Path
    argumentos_path: Path
    temas_path: Path
    enfoques_path: Path


@dataclass(frozen=True)
class StageSpec:
    name: StageName
    enabled: bool
    should_run: bool
    requires: list[ArtifactKey]
    produces: list[ArtifactKey]
    skip_reason: str | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    run_id: str
    config: RunConfig
    artifacts: ArtifactPaths
    stages: list[StageSpec]


@dataclass(frozen=True)
class StageResult:
    stage: StageName
    status: StageStatus
    message: str = ""
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int
    plan: ExecutionPlan
    stage_results: list[StageResult] = field(default_factory=list)
    manifest_path: Path | None = None


def _speech2text_from_mapping(
    speech_data: Mapping[str, Any],
    speaker_data: Mapping[str, Any],
    ref_data: Mapping[str, Any],
) -> SpeechToTextExecutionConfig:
    audio_data = _section(speech_data, "audio_fetcher")
    transcribe_data = _section(speech_data, "transcribe_speech")
    return SpeechToTextExecutionConfig(
        enabled=bool(speech_data.get("enabled", True)),
        audio_fetcher=AudioFetcherExecutionConfig(
            enabled=bool(audio_data.get("enabled", True)),
            force_download=bool(audio_data.get("force_download", False)),
            playlist_dir_template=str(audio_data.get("playlist_dir_template", "{playlist}")),
            audio_dir_template=str(audio_data.get("audio_dir_template", "videos-{playlist}")),
        ),
        speaker_timestamps=SpeakerTimestampsExecutionConfig(
            enabled=bool(speaker_data.get("enabled", True)),
            actor_objetivo=str(speaker_data.get("actor_objetivo", "Lilly Téllez")),
            pipeline=str(speaker_data.get("pipeline", "pyannote/speaker-diarization-community-1")),
            fallback_pipeline=_optional_str(
                speaker_data.get(
                    "fallback_pipeline",
                    "pyannote-community/speaker-diarization-community-1",
                )
            ),
            device=str(speaker_data.get("device", "auto")),
            umbral_match=float(speaker_data.get("umbral_match", 0.5)),
            umbral_ambiguo=float(speaker_data.get("umbral_ambiguo", 0.7)),
            match_ambiguo=str(speaker_data.get("match_ambiguo", "descartar_como_otro_speaker")),
            referencia_voz=ReferenceVoiceConfig(
                origen=str(ref_data.get("origen", "misma_playlist")),
                max_audios=int(ref_data.get("max_audios", 1)),
                video_id=str(ref_data.get("video_id", "su9nURIj9XQ")),
                url=_optional_str(ref_data.get("url", ReferenceVoiceConfig().url)),
                cache=str(ref_data.get("cache", "solo_ejecucion")),
            ),
        ),
        transcribe_speech=TranscribeSpeechExecutionConfig(
            enabled=bool(transcribe_data.get("enabled", True)),
            whisper_model=str(transcribe_data.get("whisper_model", "large-v3-turbo")),
            idioma=str(transcribe_data.get("idioma", "es")),
            word_timestamps=bool(transcribe_data.get("word_timestamps", False)),
            force_retranscribe=bool(transcribe_data.get("force_retranscribe", False)),
            skip_existing_transcripts=bool(transcribe_data.get("skip_existing_transcripts", True)),
        ),
    )


def _discursive_from_mapping(
    discursive_data: Mapping[str, Any],
    input_data: Mapping[str, Any],
    argument_shape_data: Mapping[str, Any],
    topics_cluster_data: Mapping[str, Any],
) -> DiscursiveApproachExecutionConfig:
    topics_approach_data = _section(discursive_data, "topics_approach")
    return DiscursiveApproachExecutionConfig(
        enabled=bool(discursive_data.get("enabled", True)),
        input=DiscursiveInputConfig(
            source=str(input_data.get("source", "previous_stage")),
            actor_transcripts_dir=_optional_path(input_data.get("actor_transcripts_dir")),
        ),
        argument_shape=ArgumentShapeExecutionConfig(
            enabled=bool(argument_shape_data.get("enabled", True)),
            force=bool(argument_shape_data.get("force", False)),
            spacy_model=str(argument_shape_data.get("spacy_model", "es_core_news_lg")),
            embedding_model=str(
                argument_shape_data.get("embedding_model", "LiquidAI/LFM2.5-Embedding-350M")
            ),
            breakpoint_percentile=int(argument_shape_data.get("breakpoint_percentile", 95)),
            min_oraciones=int(argument_shape_data.get("min_oraciones", 2)),
            max_oraciones=int(argument_shape_data.get("max_oraciones", 8)),
            max_palabras=int(argument_shape_data.get("max_palabras", 150)),
        ),
        topics_cluster=_topics_cluster_from_mapping(topics_cluster_data),
        topics_approach=TopicsApproachExecutionConfig(
            enabled=bool(topics_approach_data.get("enabled", True)),
            force=bool(topics_approach_data.get("force", False)),
        ),
    )


def _topics_cluster_from_mapping(data: Mapping[str, Any]) -> TopicsClusterExecutionConfig:
    umap_data = _section(data, "umap")
    hdbscan_data = _section(data, "hdbscan")
    bertopic_data = _section(data, "bertopic")
    return TopicsClusterExecutionConfig(
        enabled=bool(data.get("enabled", True)),
        force=bool(data.get("force", False)),
        embedding_model=str(data.get("embedding_model", "LiquidAI/LFM2.5-Embedding-350M")),
        min_topic_size=int(data.get("min_topic_size", 3)),
        n_neighbors=int(data.get("n_neighbors", 10)),
        n_components=int(data.get("n_components", 5)),
        umap=UmapExecutionConfig(
            metric=str(umap_data.get("metric", "cosine")),
            random_state=int(umap_data.get("random_state", 42)),
        ),
        hdbscan=HdbscanExecutionConfig(
            min_samples=int(hdbscan_data.get("min_samples", 1)),
            metric=str(hdbscan_data.get("metric", "euclidean")),
            cluster_selection_method=str(hdbscan_data.get("cluster_selection_method", "eom")),
        ),
        bertopic=BertopicExecutionConfig(
            language=str(bertopic_data.get("language", "spanish")),
            calculate_probabilities=bool(bertopic_data.get("calculate_probabilities", False)),
            verbose=bool(bertopic_data.get("verbose", False)),
        ),
    )


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Sección '{key}' debe ser un mapping")
    return value


def _path(value: Any) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return _path(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{path} debe ser una lista")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{path} debe contener solo strings")
        items.append(item)
    return items


def _stages(value: Any) -> list[StageName]:
    if not isinstance(value, list) or not value:
        raise ValueError("run.stages debe ser una lista no vacía")
    stages: list[StageName] = []
    valid = set(STAGE_NAMES)
    for item in value:
        if item not in valid:
            raise ValueError(f"run.stages contiene stage inválido: {item!r}")
        stages.append(item)
    return stages
