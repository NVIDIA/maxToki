#!/usr/bin/env bash
# Run a dev container that matches the devcontainer.json environment.
#
# Usage:
#   ./internal/scripts/run_dev.sh          # use cached image if available
#   REBUILD=1 ./internal/scripts/run_dev.sh  # force image rebuild
#
# Optional .env overrides (same keys as before):
#   WANDB_API_KEY, NGC_CLI_API_KEY, NGC_CLI_ORG, NGC_CLI_TEAM, NGC_CLI_FORMAT_TYPE

set -euo pipefail

LOCAL_REPO_PATH="$(realpath $(pwd))"
REPO_NAME="$(basename ${LOCAL_REPO_PATH})"
DOCKER_REPO_PATH="/workspaces/${REPO_NAME}"
IMAGE_TAG="${IMAGE_TAG:-maxtoki-dev}"

# ── Image ────────────────────────────────────────────────────────────────────

if [[ "${REBUILD:-0}" == "1" ]] || ! docker image inspect "${IMAGE_TAG}" &>/dev/null 2>&1; then
    echo "Building dev image (tag: ${IMAGE_TAG})..."
    DOCKER_BUILDKIT=1 docker build \
        --target dev \
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

# ── Host dirs (mirrors devcontainer initializeCommand) ───────────────────────

mkdir -p ~/.aws ~/.ngc ~/.cache ~/.ssh
[ ! -f ~/.netrc ] && touch ~/.netrc
[ ! -f ~/.bash_history_devcontainer ] && touch ~/.bash_history_devcontainer

# ── /data mount ──────────────────────────────────────────────────────────────

DATA_MOUNT=""
if [ -d "/data" ]; then
    DATA_MOUNT="-v /data:/home/ubuntu/data"
fi

# ── Run ──────────────────────────────────────────────────────────────────────

echo "Starting dev container → ${DOCKER_REPO_PATH}"
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
    -w "${DOCKER_REPO_PATH}" \
    -v "${LOCAL_REPO_PATH}:${DOCKER_REPO_PATH}" \
    -v "${HOME}/.aws:/home/ubuntu/.aws" \
    -v "${HOME}/.ngc:/home/ubuntu/.ngc" \
    -v "${HOME}/.cache:/home/ubuntu/.cache" \
    -v "${HOME}/.ssh:/home/ubuntu/.ssh:ro" \
    -v "${HOME}/.netrc:/home/ubuntu/.netrc:ro" \
    -v "${HOME}/.bash_history_devcontainer:/home/ubuntu/.bash_history" \
    ${DATA_MOUNT} \
    --user root \
    "${IMAGE_TAG}" \
    bash -c "usermod -u $(id -u) ubuntu && groupmod -g $(id -g) ubuntu && su - ubuntu -c 'cd ${DOCKER_REPO_PATH} && source .devcontainer/postCreateCommand.sh && exec bash'"
