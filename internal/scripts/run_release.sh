#!/usr/bin/env bash
# Run the release container for testing.
#
# Usage:
#   ./internal/scripts/run_release.sh          # use cached image if available
#   REBUILD=1 ./internal/scripts/run_release.sh  # force image rebuild
#
# Optional .env overrides:
#   WANDB_API_KEY, NGC_CLI_API_KEY, NGC_CLI_ORG, NGC_CLI_TEAM, NGC_CLI_FORMAT_TYPE

set -euo pipefail

LOCAL_REPO_PATH="$(realpath $(pwd))"
IMAGE_TAG="${IMAGE_TAG:-maxtoki-release}"
WORKDIR="/workspace/bionemo2"

# ── Image ────────────────────────────────────────────────────────────────────

if [[ "${REBUILD:-0}" == "1" ]] || ! docker image inspect "${IMAGE_TAG}" &>/dev/null 2>&1; then
    echo "Building release image (tag: ${IMAGE_TAG})..."
    DOCKER_BUILDKIT=1 docker build \
        --target release \
        -t "${IMAGE_TAG}" \
        -f "${LOCAL_REPO_PATH}/Dockerfile" \
        "${LOCAL_REPO_PATH}"
fi

# ── GPU runtime ──────────────────────────────────────────────────────────────

DOCKER_VERSION=$(docker version --format '{{.Server.Version}}' 2>/dev/null || echo "0.0.0")
if [ "19.03.0" = "$(printf '%s\n' "19.03.0" "${DOCKER_VERSION}" | sort -V | head -1)" ]; then
    PARAM_RUNTIME="--gpus all"
else
    PARAM_RUNTIME="--runtime=nvidia"
fi

# ── Optional .env overrides ──────────────────────────────────────────────────

[ -f "${LOCAL_REPO_PATH}/.env" ] && source "${LOCAL_REPO_PATH}/.env"

WANDB_API_KEY="${WANDB_API_KEY:-}"
NGC_CLI_API_KEY="${NGC_CLI_API_KEY:-}"
NGC_CLI_ORG="${NGC_CLI_ORG:-}"
NGC_CLI_TEAM="${NGC_CLI_TEAM:-}"
NGC_CLI_FORMAT_TYPE="${NGC_CLI_FORMAT_TYPE:-ascii}"

# ── /data mount ──────────────────────────────────────────────────────────────

DATA_MOUNT=""
if [ -d "/data" ]; then
    DATA_MOUNT="-v /data:/home/ubuntu/data"
fi

# ── Run ──────────────────────────────────────────────────────────────────────

echo "Starting release container → ${WORKDIR}"
echo "Image: ${IMAGE_TAG}"
echo "--------------------------------------------------------------"

docker run \
    --rm \
    -it \
    --network host \
    ${PARAM_RUNTIME} \
    --shm-size=4g \
    -e TMPDIR=/tmp \
    -e NUMBA_CACHE_DIR=/tmp/ \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e NGC_CLI_API_KEY="${NGC_CLI_API_KEY}" \
    -e NGC_CLI_ORG="${NGC_CLI_ORG}" \
    -e NGC_CLI_TEAM="${NGC_CLI_TEAM}" \
    -e NGC_CLI_FORMAT_TYPE="${NGC_CLI_FORMAT_TYPE}" \
    -v "${HOME}/.aws:/root/.aws" \
    -v "${HOME}/.ngc:/root/.ngc" \
    -v "${HOME}/.cache:/root/.cache" \
    -v "${HOME}/.ssh:/root/.ssh:ro" \
    -v "${HOME}/.netrc:/root/.netrc:ro" \
    ${DATA_MOUNT} \
    -w "${WORKDIR}" \
    "${IMAGE_TAG}" \
    bash
