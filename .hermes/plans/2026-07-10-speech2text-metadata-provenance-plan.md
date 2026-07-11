# Speech2Text Metadata and Provenance Implementation Plan

> **For Hermes:** Use the `subagent-driven-development` skill to implement this plan task-by-task, with TDD and a spec-compliance review after each task.

**Goal:** Make `speech2text` structurally self-contained and preserve playlist identity, video title, upload date, date provenance, and reference-video provenance from yt-dlp discovery through durable artifacts.

**Architecture:** Keep domain metadata in `speech2text` and execution state in `execution`. Introduce `speaker_timestamps/models.py` as the canonical home for diarization DTOs. Thread a typed `PlaylistInfo` through the granular service API instead of reducing it to a cache-directory string. Preserve existing `actor_transcript.v1` readers by making new transcript source metadata optional, while versioning the manifest/checkpoint contracts when their shape changes.

**Tech Stack:** Python 3.11, uv, dataclasses, yt-dlp JSON output, pytest, ruff, ty, existing pyannote/Whisper services.

---

## Evidence and current findings

Collected on 2026-07-10:

- Last committed baseline: `524743b refactor(speech2text): auditoría completa Fases 0-6 — estructura target, observabilidad en execution, checkpoint incremental`.
- Focused gate after that commit: `182 passed, 1 skipped` with the two known `discursive_approach` import suites excluded.
- The real quick test `output/20260710-205855/` completed with 3/3 units successful, but all three persisted transcripts had `fecha=null`.
- The quick-test manifest contained `units`, `artifacts`, and `config_fingerprint`, but no playlist identity or per-video title/date metadata.
- A fresh metadata-only probe against the test playlist returned 7 records with usable values:
  - playlist: `Play-PoliTest`
  - titles: present
  - `upload_date`: `20260511` / `20260627`
  - `playlist_title`: present
- Therefore the first implementation task must trace the loss between yt-dlp JSON, `VideoMeta`, `AudioVideo`, `ActorTranscript`, and artifact serialization. Do not assume the parser alone is the defect.
- `codegraph_explore` currently reports removed legacy paths (`diarizacion/`, `ingesta/`, etc.); that index is stale relative to the filesystem. Use current files and imports as the source of truth until the index is refreshed.

## Decisions for this plan

1. **Canonical playlist identity:** `PlaylistInfo` will preserve the original display name and a separate sanitized cache name. The original name must never be overwritten by the filesystem-safe value.
2. **Canonical date:** prefer yt-dlp `upload_date`; fall back to valid `release_date`; do not convert `timestamp` until a timezone policy is explicitly approved. Record `fecha_fuente` and a missing/invalid status.
3. **Video title:** preserve the existing `VideoMeta.titulo`, then propagate it into `AudioVideo`, transcript source metadata, unit results, quality metrics, manifest, and checkpoint.
4. **Transcript compatibility:** keep `actor_transcript.v1` readable. Add an optional source/provenance block; old transcripts without it remain valid.
5. **Operational schemas:** write `execution_manifest.v2` and `speech2text_checkpoint.v2` because the new top-level provenance is part of the durable execution contract. Accept old manifests/checkpoints where a loader exists.
6. **Reference video:** record the reference video separately in the stage provenance. It is not counted as a selected output unit unless it is also selected by `run.only_video_ids`/`max_videos`.

---

## Phase 0 — Freeze contracts and reproduce the loss

### Task 0.1: Add failing end-to-end metadata propagation tests

**Files:**
- Test: `tests/test_speech2text_metadata_propagation.py` (create)
- Modify: `tests/test_execution_runner.py`

Write fake-service tests that discover:

```python
PlaylistInfo(
    nombre="Play PoliTest",
    nombre_cache="Play_PoliTest",
    playlist_id="PLE9Zk7g9R__M",
    url="https://www.youtube.com/playlist?list=PLE9Zk7g9R__M",
)
```

and videos with title/date/date-source. Assert the current implementation fails to expose all of the following in durable output:

- original playlist name;
- sanitized cache name;
- video title;
- upload date and date source;
- reference-video metadata;
- per-unit metadata for failed/skipped units as well as successful units.

Run:

```bash
uv run pytest tests/test_speech2text_metadata_propagation.py -q
```

Expected: FAIL because current `PlaylistInfo` has only `nombre`, `ActorTranscript` has no title/source block, and manifest/checkpoint unit records contain only execution fields.

### Task 0.2: Establish the focused baseline

Run and record:

```bash
uv run ruff check src/ tests/ main.py
uv run ruff format --check src/ tests/ main.py
uv run pytest tests/ -m "not slow" --tb=short \
  --ignore=tests/test_discursive_approach_service.py \
  --ignore=tests/test_topics_approach.py -q
```

Expected baseline: Ruff and format pass; pytest reports the current focused suite as green with the two known downstream import failures excluded.

---

## Phase 1 — Move diarization DTOs into `speaker_timestamps/models.py`

### Task 1.1: Create the canonical diarization DTO module

**Files:**
- Create: `src/tono_politico/speech2text/speaker_timestamps/models.py`
- Modify: `src/tono_politico/speech2text/models.py`

Move exactly these definitions without changing fields or validation semantics:

- `TurnoOrador`
- `PerfilVozActor`
- `SpeakerMatch`

Keep these in the umbrella module:

- `ActorTranscript`
- `ActorTranscriptSegment`
- `AsrMetadata`

Initially preserve compatibility re-exports from `speech2text.models` so existing consumers do not break while canonical imports migrate.

### Task 1.2: Migrate imports and package exports

**Files:**
- Modify: `src/tono_politico/speech2text/speaker_timestamps/__init__.py`
- Modify: `src/tono_politico/speech2text/speaker_timestamps/service.py`
- Modify: `src/tono_politico/speech2text/speaker_timestamps/matching.py`
- Modify: `src/tono_politico/speech2text/speaker_timestamps/perfil_voz.py`
- Modify: `src/tono_politico/speech2text/transcribe_speech/service.py`
- Modify: `src/tono_politico/speech2text/__init__.py`

Canonical imports must come from `speaker_timestamps.models`. Avoid a circular import between `speech2text.models`, `speaker_timestamps.__init__`, and `speaker_timestamps.service`.

### Task 1.3: Add model-location tests

**Files:**
- Create: `tests/test_speaker_timestamps_models.py`
- Modify: `tests/test_speech2text_contracts.py`
- Modify: `tests/test_diarizacion_models.py` or its current replacement tests

Assert:

- DTOs are defined in `speaker_timestamps.models`;
- `speaker_timestamps` exports them;
- old umbrella imports remain valid if compatibility re-exports are retained;
- there is only one class definition for each DTO.

Run:

```bash
uv run pytest tests/test_speaker_timestamps_models.py \
  tests/test_speaker_timestamps_service.py \
  tests/test_diarizacion_matching.py \
  tests/test_perfil_desde_output.py -q
```

Expected: PASS with the new module and no import cycles.

---

## Phase 2 — Make playlist identity and date provenance explicit

### Task 2.1: Extend `PlaylistInfo` without losing backward compatibility

**Files:**
- Modify: `src/tono_politico/speech2text/audio_fetcher/models.py`
- Test: `tests/test_audio_fetcher_models.py`

Define:

```python
@dataclass(frozen=True)
class PlaylistInfo:
    nombre: str                         # original/display name
    nombre_cache: str | None = None     # sanitized filesystem name
    playlist_id: str | None = None
    url: str | None = None

    @property
    def cache_name(self) -> str:
        return self.nombre_cache or sanitizar_nombre_directorio(self.nombre)
```

If importing the sanitizer into the DTO creates an undesirable cycle, implement the fallback at the service boundary and document it. Existing fake fixtures that construct `PlaylistInfo(nombre="P")` must continue to work.

### Task 2.2: Add date-source metadata to `VideoMeta`

**Files:**
- Modify: `src/tono_politico/speech2text/audio_fetcher/models.py`
- Modify: `src/tono_politico/speech2text/audio_fetcher/playlist.py`
- Test: `tests/test_audio_fetcher_models.py`
- Test: `tests/test_audio_fetcher_playlist.py`

Add an optional field such as:

```python
fecha_fuente: Literal["upload_date", "release_date", "missing", "invalid"] | None = None
```

Keep `fecha` as `YYYYMMDD | None`. The parser must distinguish:

- valid `upload_date` → `fecha_fuente="upload_date"`;
- valid fallback `release_date` → `fecha_fuente="release_date"`;
- no usable date → `fecha=None`, `fecha_fuente="missing"`;
- malformed date values → `fecha=None`, `fecha_fuente="invalid"`.

Do not introduce timestamp-to-date conversion in this phase.

### Task 2.3: Parse original playlist identity from yt-dlp

**Files:**
- Modify: `src/tono_politico/speech2text/audio_fetcher/playlist.py`
- Test: `tests/test_audio_fetcher_playlist.py`

Use the first valid record’s:

1. `playlist_title`;
2. fallback `playlist`;
3. fallback `"playlist_sin_nombre"`.

Capture `playlist_id` and the input URL where available. Preserve the original display name in `PlaylistInfo.nombre` and use `nombre_cache` only for filesystem paths.

Add a fixture matching the live probe:

```json
{
  "id": "71GicqtYqpQ",
  "title": "Senator Lilly Téllez speaks out against the bill...",
  "playlist": "Play-PoliTest",
  "playlist_title": "Play-PoliTest",
  "playlist_id": "PLE9Zk7g9R__M",
  "upload_date": "20260511",
  "duration": 316.0
}
```

Assert that title, playlist identity, date, date source, and duration survive normalization.

---

## Phase 3 — Thread typed playlist metadata through the pipeline

### Task 3.1: Change audio-fetcher boundaries from a bare name to `PlaylistInfo`

**Files:**
- Modify: `src/tono_politico/speech2text/audio_fetcher/service.py`
- Modify: `src/tono_politico/speech2text/speech2text/service.py` (actual path: `src/tono_politico/speech2text/service.py`)
- Modify: `src/tono_politico/execution/runner.py`
- Modify: `tests/test_audio_fetcher_service.py`
- Modify: `tests/test_speech2text_service.py`
- Modify: `tests/test_execution_runner.py`

Prefer these typed boundaries:

```python
AudioFetcherService.fetch_one(video: VideoMeta, playlist: PlaylistInfo, ...)
SpeechToTextService.ensure_perfil(playlist: PlaylistInfo, metas: list[VideoMeta])
SpeechToTextService.procesar_one(video: VideoMeta, playlist: PlaylistInfo, ...)
```

Use `playlist.cache_name` for cache paths. Never pass the human-facing name through a parameter whose only purpose is filesystem routing. Update fakes and assertions accordingly.

### Task 3.2: Preserve playlist metadata on `AudioVideo`

**Files:**
- Modify: `src/tono_politico/speech2text/audio_fetcher/models.py`
- Modify: `src/tono_politico/speech2text/audio_fetcher/service.py`
- Modify: `tests/test_audio_fetcher_models.py`
- Modify: `tests/test_audio_fetcher_service.py`

Add an optional `playlist: PlaylistInfo` field to `AudioVideo` and update `AudioVideo.from_meta(meta, audio_path, playlist=...)`. Keep a compatibility default only if required by existing isolated tests; all runtime paths must populate it.

Assert that `AudioVideo` preserves:

- video title;
- normalized date and date source;
- original playlist name;
- cache name;
- playlist URL/ID when available.

---

## Phase 4 — Persist title/date/playlist provenance in durable artifacts

### Task 4.1: Add optional transcript source metadata

**Files:**
- Modify: `src/tono_politico/speech2text/models.py`
- Modify: `src/tono_politico/speech2text/transcribe_speech/service.py`
- Modify: `src/tono_politico/speech2text/transcribe_speech/actor_clip.py`
- Modify: `tests/test_fecha_propagacion.py`
- Modify: `tests/test_actor_transcript_serializacion.py`
- Modify: `tests/test_transcribe_speech_service.py`

Add an optional source block to `ActorTranscript` while retaining `fecha` as the existing compatibility field:

```python
@dataclass(frozen=True)
class TranscriptSource:
    playlist_name: str | None = None
    playlist_id: str | None = None
    playlist_url: str | None = None
    video_title: str | None = None
    video_url: str | None = None
    upload_date: str | None = None
    date_source: str | None = None
```

`TranscribeSpeechService` must construct it from `AudioVideo`. New JSON may contain:

```json
"source": {
  "playlist_name": "Play-PoliTest",
  "video_title": "...",
  "upload_date": "20260511",
  "date_source": "upload_date"
}
```

The loader must continue accepting existing `actor_transcript.v1` files with no `source` key.

### Task 4.2: Add metadata to execution unit results

**Files:**
- Modify: `src/tono_politico/execution/models.py`
- Modify: `src/tono_politico/execution/runner.py`
- Modify: `tests/test_execution_runner.py`

Add optional non-text metadata to `UnitResult` for successful, skipped, and failed units:

- `video_title`;
- `fecha`;
- `fecha_fuente`;
- `duration`.

Update `_unit_result_to_manifest()` to serialize these fields but never serialize transcript text.

### Task 4.3: Add top-level speech2text provenance to manifest/checkpoint

**Files:**
- Modify: `src/tono_politico/execution/models.py`
- Modify: `src/tono_politico/execution/runner.py`
- Modify: `src/tono_politico/execution/artifacts.py` if schema validation is added
- Modify: `tests/test_execution_runner.py`

Persist a top-level object similar to:

```json
"speech2text": {
  "playlist": {
    "name": "Play-PoliTest",
    "cache_name": "Play-PoliTest",
    "playlist_id": "PLE9Zk7g9R__M",
    "url": "https://www.youtube.com/playlist?list=PLE9Zk7g9R__M"
  },
  "discovered_videos": 7,
  "selected_videos": 3,
  "reference_video": {
    "video_id": "su9nURIj9XQ",
    "title": "...",
    "upload_date": "...",
    "date_source": "upload_date"
  }
}
```

The checkpoint must include the same playlist provenance and per-unit metadata so a partial run remains interpretable. The reference video is recorded separately from selected output units.

Write new schemas:

- `tono-politico.execution_manifest.v2`;
- `speech2text_checkpoint.v2`.

Document compatibility behavior for old v1 artifacts.

### Task 4.4: Extend quality metrics without copying transcript text

**Files:**
- Modify: `src/tono_politico/execution/observability.py`
- Modify: `tests/test_execution_observability.py`
- Modify: `tests/test_execution_runner.py`

Add title/date/date-source to each per-video quality entry. Keep aggregate counts and the no-text operational boundary. Choose either additive `speech2text_quality.v2` compatibility or an explicit v3; record the decision in the document and tests.

---

## Phase 5 — Reference profile and resume correctness

### Task 5.1: Record reference-profile provenance and status

**Files:**
- Modify: `src/tono_politico/execution/runner.py`
- Modify: `src/tono_politico/execution/models.py`
- Modify: `tests/test_execution_runner.py`

When `ensure_perfil()` succeeds or fails, persist:

- reference video ID;
- title/date metadata if present;
- profile status (`ready`/`missing`/`failed`);
- reason code on failure.

Do not count the reference profile as a normal selected transcript unless it is explicitly selected.

### Task 5.2: Test partial resume with metadata

**Files:**
- Modify: `tests/test_execution_runner.py`
- Modify: `tests/test_execution_plan.py`

Add tests proving that:

- one valid transcript is resumed without reprocessing;
- its metadata remains in the new checkpoint/manifest;
- missing units retain title/date metadata while they process;
- a failed unit is not mistaken for a completed artifact;
- an empty transcript directory does not satisfy resume.

---

## Phase 6 — Real metadata verification and documentation

### Task 6.1: Add a metadata-only integration probe

**Files:**
- Create: `tests/test_audio_fetcher_live_metadata.py` only if the project convention supports opt-in network tests; otherwise keep this as a documented shell probe.
- Modify: `docs/component_audio_fetcher.md`

Run without downloading audio:

```bash
yt-dlp --flat-playlist \
  --extractor-args 'youtubetab:approximate_date' \
  -j --no-warnings \
  'https://www.youtube.com/playlist?list=PLE9Zk7g9R__M'
```

Verify and record:

- playlist name and ID;
- number of discovered videos;
- non-empty title rate;
- `upload_date` availability and `fecha_fuente` distribution;
- fallback/missing-date count.

The live probe must not be required for the normal unit-test gate.

### Task 6.2: Re-run a 3-video smoke and inspect artifacts

Use `config/speech2text-quick.yaml` with `max_videos: 3`. Verify directly from the new run:

```bash
python - <<'PY'
import json
from pathlib import Path

run = Path("output/<run_id>")
manifest = json.loads((run / "manifest.json").read_text())
quality = json.loads((run / "speech2text" / "quality.json").read_text())
transcripts = list((run / "speech2text" / "actor_transcripts").glob("*.json"))

assert manifest["speech2text"]["playlist"]["name"]
assert manifest["speech2text"]["reference_video"]["video_id"]
assert all("video_title" in unit for unit in manifest["units"])
assert all("source" in json.loads(path.read_text()) for path in transcripts)
assert all(video["video_title"] for video in quality["videos"])
PY
```

Report any remaining missing dates separately from parser failure; `fecha_fuente="missing"` is valid only when yt-dlp supplied no trustworthy date.

### Task 6.3: Update living documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `src/tono_politico/speech2text/requisitos.md`
- Modify: `docs/module-speech2text.md`
- Modify: `docs/component_audio_fetcher.md`
- Modify: `docs/component_speaker_timestamps.md`
- Modify: `docs/component_transcribe_speech.md`
- Modify: `docs/configuracion.md` if schema/version claims change

Document:

- `speaker_timestamps/models.py` as the canonical DTO location;
- original playlist name vs sanitized cache name;
- `upload_date`/`release_date` policy and `fecha_fuente`;
- title/date propagation into transcripts, quality, manifest, and checkpoint;
- reference-video provenance;
- compatibility rules for old artifacts;
- the observed real smoke result, including any genuinely unavailable dates.

Search for stale claims before finishing:

```bash
rg -n "adapter\.py|quality\.py|actor_transcript\.py|whisper_clip\.py|transcripcion_actor\.py|speech2text_quality\.v1|221 passed|67 tests" \
  README.md AGENTS.md docs/ src/tono_politico/speech2text/
```

---

## Verification gates

After each code task:

```bash
uv run ruff check src/ tests/ main.py
uv run ruff format --check src/ tests/ main.py
uv run pytest <focused tests> -q
```

Final focused gate:

```bash
uv run ruff check src/ tests/ main.py
uv run ruff format --check src/ tests/ main.py
uv run pytest tests/test_audio_fetcher_*.py \
  tests/test_speaker_timestamps_models.py \
  tests/test_speaker_timestamps_service.py \
  tests/test_diarizacion_matching.py \
  tests/test_perfil_desde_output.py \
  tests/test_transcribe_speech_service.py \
  tests/test_fecha_propagacion.py \
  tests/test_actor_transcript_serializacion.py \
  tests/test_execution_observability.py \
  tests/test_execution_runner.py -q
```

Repository gate, reported honestly because of the known downstream blocker:

```bash
uv run ty check
uv run pytest tests/ -m "not slow" --tb=short
bash check.sh
uv run python main.py --config config/config.yaml --validate-config
uv run python main.py --config config/config.yaml --dry-run
```

Expected unrelated blocker unless `discursive_approach` is repaired in the same workstream:

- `ty` diagnostics from legacy imports in `topics_approach`;
- collection failures in `tests/test_discursive_approach_service.py` and `tests/test_topics_approach.py`.

## Definition of done

- [x] `speaker_timestamps/models.py` exists and owns the three diarization DTOs.
- [x] `PlaylistInfo` preserves original name, cache name, URL, and playlist ID.
- [x] `VideoMeta` preserves title, normalized date, and date source.
- [x] The typed playlist object survives discover → audio → diarization/ASR orchestration.
- [x] New `ActorTranscript` artifacts contain optional source metadata while old v1 files still load.
- [x] Manifest and checkpoint contain playlist provenance, reference-video metadata, and per-unit title/date metadata.
- [x] Quality metrics contain title/date metadata without transcript text.
- [x] Partial resume preserves metadata and does not duplicate or reprocess valid units.
- [x] A fresh 3-video smoke confirms the metadata path with real artifacts.
- [x] README, AGENTS, requirements, and component docs match the implementation.
- [x] Focused Ruff/format/tests pass; global blockers remain explicitly separated.
