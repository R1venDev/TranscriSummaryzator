# syntax=docker/dockerfile:1
ARG PYTHON_IMAGE=python:3.10-slim-bookworm
FROM ${PYTHON_IMAGE} AS runtime-base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TRANSCRI_DATA_DIR=/data TRANSCRI_CONFIG_FILE=/data/config.json \
    TRANSCRI_SUMMARY_PYTHON=/opt/summary/bin/python \
    TRANSCRI_SUMMARY_MASTER_KEY_FILE=/secrets/summary-master.key \
    TRANSCRI_DASHBOARD_HOST=0.0.0.0 \
    HF_HOME=/data/work/cache/huggingface TORCH_HOME=/data/work/cache/torch \
    XDG_CACHE_HOME=/data/work/cache HOME=/data \
    PATH=/opt/summary/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg libsndfile1 libgomp1 tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 transcri \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /data transcri \
    && mkdir /app /data /secrets && chown 10001:10001 /data /secrets \
    && chmod 0700 /data /secrets
WORKDIR /app
COPY requirements-summary-credentials.txt /app/
RUN python -m venv /opt/summary \
    && /opt/summary/bin/pip install --no-cache-dir -r requirements-summary-credentials.txt
RUN ln -s /opt/summary /app/.venv-core
EXPOSE 8765
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/summary/bin/python", "/app/docker/entrypoint.py"]
CMD ["dashboard"]

# Optional Linux amd64 speech runtime. This installs packages, never model
# weights. Speech is intentionally isolated from the summary interpreter.
FROM runtime-base AS speech-dependencies
COPY docker/requirements-*-linux.txt docker/diarizen-runtime.patch /app/docker/
ARG GIGAAM_REVISION=7447938d791c4f3e643386ee22c33777004293a5
ARG DIARIZEN_REVISION=844f5555b0a98acd0931511fc641a8c5b8ba92c7
ARG NEMO_REVISION=abb8254dac2bf5a011e6069fcaa7df71c7e3b8c1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential pkg-config libffi-dev \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/vendor \
    && git init /opt/vendor/GigaAM \
    && git -C /opt/vendor/GigaAM remote add origin https://github.com/salute-developers/GigaAM.git \
    && git -C /opt/vendor/GigaAM fetch --depth 1 origin "${GIGAAM_REVISION}" \
    && git -C /opt/vendor/GigaAM checkout --detach FETCH_HEAD \
    && git init /opt/vendor/DiariZen \
    && git -C /opt/vendor/DiariZen remote add origin https://github.com/BUTSpeechFIT/DiariZen.git \
    && git -C /opt/vendor/DiariZen fetch --depth 1 origin "${DIARIZEN_REVISION}" \
    && git -C /opt/vendor/DiariZen checkout --detach FETCH_HEAD \
    && git -C /opt/vendor/DiariZen apply --check /app/docker/diarizen-runtime.patch \
    && git -C /opt/vendor/DiariZen apply /app/docker/diarizen-runtime.patch \
    && git init /opt/vendor/NeMo \
    && git -C /opt/vendor/NeMo remote add origin https://github.com/NVIDIA/NeMo.git \
    && git -C /opt/vendor/NeMo fetch --depth 1 origin "${NEMO_REVISION}" \
    && git -C /opt/vendor/NeMo checkout --detach FETCH_HEAD
RUN /opt/summary/bin/pip install --no-cache-dir uv==0.12.11
ENV UV_PYTHON_INSTALL_DIR=/opt/pythons UV_LINK_MODE=copy UV_CACHE_DIR=/root/.cache/uv
RUN uv python install --no-bin 3.11.16 3.12.14 \
    && uv venv --python 3.11.16 /app/.venv-gigaam \
    && uv venv --python 3.11.16 /app/.venv-diarizen \
    && uv venv --python 3.12.14 /app/.venv-fusion
# These are complete snapshots of installed production package versions. The
# source requirements remain unchanged for native installations. --no-deps
# preserves the snapshot; pip check below verifies that it is self-consistent.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv-gigaam/bin/python --no-deps \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      --index-strategy unsafe-best-match \
      -r docker/requirements-gigaam-linux.txt -e /opt/vendor/GigaAM
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv-diarizen/bin/python --no-deps \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      --index-strategy unsafe-best-match \
      -r docker/requirements-diarizen-linux.txt \
      -e /opt/vendor/DiariZen -e /opt/vendor/DiariZen/pyannote-audio
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /app/.venv-fusion/bin/python --no-deps \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      --index-strategy unsafe-best-match \
      -r docker/requirements-fusion-linux.txt /opt/vendor/NeMo
USER 10001:10001
RUN --network=none uv --no-cache pip check --python /app/.venv-gigaam/bin/python \
    && uv --no-cache pip check --python /app/.venv-diarizen/bin/python \
    && uv --no-cache pip check --python /app/.venv-fusion/bin/python \
    && /app/.venv-gigaam/bin/python -c 'import torch, torchaudio, gigaam, silero_vad, faster_whisper' \
    && /app/.venv-diarizen/bin/python -c 'import torch, torchaudio; from diarizen.pipelines.inference import DiariZenPipeline' \
    && /app/.venv-fusion/bin/python -c 'import torch, torchaudio; from nemo.collections.asr.models import SortformerEncLabelModel'

# Copy application source after dependencies so normal edits keep their cache.
FROM runtime-base AS summary
COPY pipeline.py *.html *.js config.example.json vocabulary.json /app/
COPY scripts/ /app/scripts/
COPY summary/ /app/summary/
COPY contracts/ /app/contracts/
COPY evaluation/ /app/evaluation/
COPY evidence/ /app/evidence/
COPY pipeline_core/ /app/pipeline_core/
COPY project_memory/ /app/project_memory/
COPY semantics/ /app/semantics/
COPY docker/ /app/docker/
ARG SOURCE_REVISION=unknown
ARG RELEASE_VERSION=development
ENV TRANSCRISUMMARY_GIT_COMMIT=${SOURCE_REVISION}
LABEL org.opencontainers.image.title="TranscriSummaryzator" \
      org.opencontainers.image.source="https://github.com/R1venDev/TranscriSummaryzator" \
      org.opencontainers.image.revision=${SOURCE_REVISION} \
      org.opencontainers.image.version=${RELEASE_VERSION}
USER 10001:10001
RUN --network=none /opt/summary/bin/python -m unittest discover -s docker -p 'test_*.py'
ENV TRANSCRI_IMAGE_VARIANT=summary
CMD ["dashboard"]

FROM speech-dependencies AS speech
COPY pipeline.py *.html *.js config.example.json vocabulary.json /app/
COPY scripts/ /app/scripts/
COPY summary/ /app/summary/
COPY contracts/ /app/contracts/
COPY evaluation/ /app/evaluation/
COPY evidence/ /app/evidence/
COPY pipeline_core/ /app/pipeline_core/
COPY project_memory/ /app/project_memory/
COPY semantics/ /app/semantics/
COPY docker/ /app/docker/
ARG SOURCE_REVISION=unknown
ARG RELEASE_VERSION=development
ENV TRANSCRISUMMARY_GIT_COMMIT=${SOURCE_REVISION}
LABEL org.opencontainers.image.title="TranscriSummaryzator" \
      org.opencontainers.image.source="https://github.com/R1venDev/TranscriSummaryzator" \
      org.opencontainers.image.revision=${SOURCE_REVISION} \
      org.opencontainers.image.version=${RELEASE_VERSION}
USER 10001:10001
RUN --network=none /opt/summary/bin/python -m unittest discover -s docker -p 'test_*.py'
ENV TRANSCRI_IMAGE_VARIANT=speech
CMD ["watch"]
