# TranscriSummaryzator

[Русский](README.md) · [Installation](docs/DOCKER.md) · [Plane](docs/PLANE.md) · [Architecture](ARCHITECTURE.md) · [Limitations](docs/STATUS.md)

![Local transcription → Luna Batch meeting notes → Plane Wiki and tasks](docs/assets/overview.svg)

**Meeting recordings, notes, and tasks — with links back to the source.**

TranscriSummaryzator turns meeting audio and video into transcripts with speakers and timestamps, then extracts topics, decisions, tasks, and hypotheses. Speech models run on your machine. The current summary workflow sends text to Luna through OpenRouter Batch; you can send the resulting materials to Plane.

The project focuses on Russian meetings. Recognition errors and model omissions are possible: source links and review status help you check the output.

## What you get

- **Navigable transcripts.** Utterances, timestamps, speaker separation, and matching against saved voice profiles.
- **Meeting notes with sources.** Topics, decisions, tasks, and hypotheses link to the meeting transcript. Incomplete review is shown explicitly.
- **Editable task cards.** Change titles, descriptions, owners, deadlines, and other fields without another model call. Revisions are preserved and changes appear in the notes and exports.
- **Plane integration.** Send individual tasks and hypotheses, use separate automation switches, and create a meeting Wiki page.
- **Exports and diagnostics.** Transcripts in TXT, Markdown, HTML, and JSON, plus SRT subtitles; meeting notes in Markdown, HTML, and JSON; processing logs, status, and Batch cost accounting.

## Quick start

The primary installation is the full Docker profile: upload a recording, transcribe it locally, generate Luna meeting notes, and connect Plane.

**Requires:** Linux `amd64`, an NVIDIA GPU with a compatible driver and NVIDIA Container Toolkit, Git, and Docker Compose. Allow space for the image, models, and recordings; building from source requires at least 60 GB of free space. See the [installation guide](docs/DOCKER.md).

```bash
git clone https://github.com/R1venDev/TranscriSummaryzator.git && cd TranscriSummaryzator
./scripts/docker-install.sh --speech
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). On first setup, the installer prints the `admin` login and password once. Save the password. Running the installer without an argument also selects `--speech`. By default, it pulls the `speech` and `summary` images from `ghcr.io/r1vendev/transcrisummaryzator`; `--build` builds your checkout instead.

1. Add an OpenRouter inference key under **Settings → Summarizer → OpenRouter** (`/summary-settings`; the application UI is in Russian).
2. Configure the verified workspace and access settings using the [Docker guide](docs/DOCKER.md). Saving a key alone does not authorize transcript dispatch.
3. Upload a recording through the web interface. The speech models download into the local cache on the first job. Once transcription finishes, the summary enters the Luna queue.
4. Review the sources, edit task cards, and connect Plane if needed. Summary generation uses a paid external API.

**Validated:** the full speech environment has been built for Linux, and library dependencies and imports pass their checks. GPU audio processing and recognition quality in this image still need separate acceptance testing. See [runtime provenance](docs/DOCKER_RUNTIME.md) and [validation status](docs/STATUS.md).

### Meeting notes from existing transcripts

If you already have an application `transcript.json`, use the lightweight profile without a GPU or speech models:

```bash
./scripts/docker-install.sh --summary
docker compose exec -T app python /app/docker/import-transcript.py --name 'Meeting' < transcript.json
```

Then open the imported recording and start summary generation. Importing alone does not call a model. This profile has passed Linux build, startup, administrator access, import, and restart persistence checks.

### Updates

After backing up the persistent volumes:

```bash
git pull --ff-only && ./scripts/docker-update.sh --speech
```

Use `--summary` for a lightweight installation. Updates preserve a paused processing state; add `--enable-speech` to explicitly enable it on an existing installation. Data and keys remain in persistent volumes; the [Docker guide](docs/DOCKER.md) covers backups, recovery, and running an existing image. To include your own changes, add `--build`: `./scripts/docker-update.sh --speech --build`.

<details>
<summary>Native installation and the earlier Ollama workflow</summary>

The repository retains a [native installer](install.sh) and [systemd examples](deploy/linux). `config.example.json` selects `legacy_local` with Ollama; Docker bootstrap selects `luna_batch`. The general installer recreates Python environments, so update an existing installation according to its deployment procedure.

</details>

## How it works

| Stage | Runs where | Output |
|---|---|---|
| Upload, audio extraction, speech recognition, speaker identification | Your server or computer | `transcript.json`, utterances, timestamps |
| Meeting notes and checks against the source transcript | Luna through OpenRouter Batch | Document, task cards, review reports |
| File validation, versions, and manual edits | Your server or computer | Consistent exports and revision history |
| Optional delivery to Plane | Your Plane instance | Wiki page, tasks, and proposals |

You can generate notes from an existing `transcript.json` without running speech recognition or diarization again.

### Current Luna workflow

The `luna_batch_source_first_v1` policy starts from the full source: a writer and independent extractions run in parallel, followed by audits, a global check, and, when needed, one repair with verification of the changes.

With three extraction partitions and three audit partitions, the plan uses **8–10 inference items across 4–6 Batch creation requests**. The upper limit is **12 potentially billable items per job**. A transcript that exceeds the plan is blocked; this is not a promise to handle every meeting at a fixed price.

Queue state, remote IDs, and costs persist locally. Restarts resume known Batches. An unknown submission outcome must be reconciled with the remote system before retrying. Batch failure does not switch the workflow to a synchronous API, another model, or another provider.

Code checks file integrity and references to source utterances. Model review can remain incomplete: `review_incomplete` means you should review the notes. See the [Luna workflow](docs/SUMMARY_LUNA_SOURCE_FIRST.md) and [current limitations](docs/STATUS.md).

## Plane: connect, then send

Open **Settings → Plane** (`/plane-settings`) and enter the server URL, workspace, project, and optional Wiki collection or parent page. The API key is encrypted in server storage; the interface shows only a mask. The connection check reads available Plane data.

| Setting | Default | Behavior |
|---|---|---|
| Meeting Wiki page | Enabled after connection setup | Creates a page containing the published meeting notes |
| Automatic tasks | Off | Sends tasks after summary publication when enabled |
| Automatic hypotheses | Off | Sends hypotheses as proposals for validation when enabled |

You can send individual items from the notes while automation is off. Tasks may have no assignee. Save changes to a task card before sending it. Repeated clicks use its stored Plane association. Existing generations are not backfilled automatically.

If a remote POST has an uncertain outcome, the item enters `submission_unknown`: another create is blocked until reconciliation. This prevents blind retries; it does not guarantee exactly-once execution by an external API. Existing remote content is preserved when local content changes. Wiki availability depends on your Plane API and permissions. See the [Plane guide](docs/PLANE.md).

## Data and privacy

- **Speech processing is local.** Installed speech models process audio, video, and voice profiles. Installing dependencies and downloading weights requires access to package and model sources.
- **Luna summaries are remote.** Transcript text and review materials are sent to OpenRouter and the selected OpenAI provider. Batch does not mean zero data retention; consider the services' storage and logging settings before use.
- **Plane receives sent materials.** Once configured, meeting pages and manually or automatically selected items are sent to your specified Plane server.
- **Working data is separate from source code.** Recordings, outputs, profiles, databases, logs, and real keys are excluded from Git. Docker keeps data and secrets in separate persistent volumes; recovery needs both.

Protected settings use administrator authentication, Origin checks, and responses that disable caching. Configuration examples contain placeholders only.

## Development and validation

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

Tests cover contracts, queue behavior, recovery, task editing, and publication. They do not measure summary accuracy on your meetings. Container builds, GPU operation, Plane permissions, and output quality need separate checks in the target environment. See [status and limitations](docs/STATUS.md) and the [reference benchmark](benchmark/README.md).

## Repository map

| Path | Contents |
|---|---|
| `pipeline.py`, `pipeline_core/` | Queue, processing, and HTTP interface |
| `scripts/` | Speech workers, diagnostics, utilities |
| `summary/luna_v1/` | Luna contracts, Batch stages, task cards, publication |
| `summary/` | Summarization and Plane integration |
| `dashboard.html`, `summary_*.html`, `plane_*.html` | Web interface |
| `docker/`, `compose*.yaml`, `Dockerfile` | Container installation options |
| `tests/`, `benchmark/`, `evaluation/` | Automated checks and evaluation materials |
| `macos/`, `uploader.py` | Uploading recordings from macOS |

### Documentation

Detailed technical guides currently use Russian.

- [Changes by version](CHANGELOG.md)
- [Docker: installation, imports, updates, and limitations](docs/DOCKER.md)
- [Speech Docker runtime: versions, build, and validation](docs/DOCKER_RUNTIME.md)
- [Image publication, tags, and checks](docs/DOCKER_PUBLISH.md)
- [Architecture](ARCHITECTURE.md)
- [Luna Batch: current workflow using the source transcript](docs/SUMMARY_LUNA_SOURCE_FIRST.md)
- [Keys and administrative access](docs/SUMMARY_ADMIN_KEYS.md)
- [Plane setup and delivery behavior](docs/PLANE.md)
- [Implementation status and limitations](docs/STATUS.md)
- [Previous Luna Batch contract](docs/SUMMARY_LUNA_BATCH.md)
