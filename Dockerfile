# Build instructions:
#
#   docker build --target dev -t maxtoki-dev .
#
# Force a rebuild with no cache:
#   docker build --no-cache --target dev -t maxtoki-dev .
#
# Base image with apex and transformer engine.
# Keep versions in sync with:
#   https://gitlab-master.nvidia.com/dl/JoC/nemo-ci/-/blob/main/llm_train/Dockerfile.train
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3

FROM ${BASE_IMAGE} AS bionemo2-base

# Install core apt packages.
RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
  <<EOF
set -eo pipefail
apt-get update -qy
apt-get install -qyy \
  libsndfile1 \
  ffmpeg \
  git \
  curl \
  pre-commit \
  sudo \
  gnupg \
  unzip \
  libsqlite3-dev
apt-get upgrade -qyy \
  rsync
rm -rf /tmp/* /var/tmp/*
EOF


## BUMP and patch TE as a solution to the issues:
## 1. https://github.com/NVIDIA/bionemo-framework/issues/422
## 2. https://github.com/NVIDIA/bionemo-framework/issues/973
## Drop this when pytorch images ship the fixed commit.
ARG TE_TAG=9d4e11eaa508383e35b510dc338e58b09c30be73

COPY ./patches/te.patch /tmp/te.patch
RUN git clone --recurse-submodules https://github.com/NVIDIA/TransformerEngine.git /tmp/TransformerEngine && \
    cd /tmp/TransformerEngine && \
    git checkout --recurse-submodules ${TE_TAG} && \
    patch -p1 < /tmp/te.patch && \
    PIP_CONSTRAINT= NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr/local/mpi \
    pip --disable-pip-version-check --no-cache-dir install .

# Install AWS CLI from source rather than prebuilt binary.
# This is good for two reasons:
#  1. It is the same on both ARM and x86
#  2. When installing this way, aws-cli doesn't bring its own pypi dependencies with it,
#     which it might do via a binary install. These extra pypi packages sometimes bring
#     CVEs with them.

RUN <<EOF
set -eo pipefail
cd /tmp
git clone --depth 1 --branch 2.27.59 https://github.com/aws/aws-cli.git
cd aws-cli
pip install .
cd /
rm -rf /tmp/aws-cli
EOF

# Nemo Run installation
# Some things are pip installed in advance to avoid dependency issues during nemo_run installation
RUN pip install hatchling urllib3  # needed to install nemo-run
ARG NEMU_RUN_TAG=v0.3.0
RUN pip install nemo_run@git+https://github.com/NVIDIA/NeMo-Run.git@${NEMU_RUN_TAG} --use-deprecated=legacy-resolver

RUN mkdir -p /workspace/bionemo2/

WORKDIR /workspace

# Addressing Security Scan Vulnerabilities
RUN rm -rf /opt/pytorch/pytorch/third_party/onnx


# Use UV to install python packages from the workspace. This just installs packages into the system's python
# environment, and does not use the current uv.lock file. Note that with python 3.12, we now need to set
# UV_BREAK_SYSTEM_PACKAGES, since the pytorch base image has made the decision not to use a virtual environment and UV
# does not respect the PIP_BREAK_SYSTEM_PACKAGES environment variable set in the base dockerfile.
COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=1 \
  UV_PYTHON_DOWNLOADS=never \
  UV_SYSTEM_PYTHON=true \
  UV_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace/bionemo2

# Install 3rd-party deps and bionemo submodules.
COPY ./LICENSE /workspace/bionemo2/LICENSE
COPY ./3rdparty /workspace/bionemo2/3rdparty
COPY ./sub-packages /workspace/bionemo2/sub-packages

ARG NEMO_VERSION=2.7.2

RUN --mount=type=bind,source=./requirements-test.txt,target=/requirements-test.txt \
  --mount=type=bind,source=./requirements-cve.txt,target=/requirements-cve.txt \
  --mount=type=cache,target=/root/.cache <<EOF
set -eo pipefail
# install nvidia-resiliency-ext separately because it doesn't yet have ARM wheels
git clone https://github.com/NVIDIA/nvidia-resiliency-ext
uv pip install nvidia-resiliency-ext/
rm -rf nvidia-resiliency-ext/
# ngcsdk causes strange dependency conflicts (ngcsdk requires protobuf<4, but nemo_toolkit requires protobuf==4.24.4, deleting it from the uv pip install prevents installation conflicts)
sed -i "/ngcsdk/d" ./sub-packages/bionemo-core/pyproject.toml
uv pip install --no-build-isolation \
"nemo_toolkit[llm]==${NEMO_VERSION}" \
./3rdparty/*  \
./sub-packages/bionemo-core \
./sub-packages/bionemo-llm \
./sub-packages/bionemo-maxtoki \
./sub-packages/bionemo-testing \
scanpy \
-r /requirements-cve.txt \
-r /requirements-test.txt

# Remove llama-index: bionemo doesn't use it and it introduces CVEs.
uv pip uninstall llama-index llama-index-core llama-index-legacy 2>/dev/null || true
# Install back ngcsdk, as a WAR for the protobuf version conflict with nemo_toolkit.
uv pip install ngcsdk==3.64.3  # Temporary fix for changed filename, see https://nvidia.slack.com/archives/C074Z808N05/p1746231345981209
# Install >=0.46.1 bitsandbytes specifically because it has CUDA>12.9 support.
# TODO(trvachov) remove this once it stops conflicting with strange NeMo requirements.txt files
uv pip uninstall bitsandbytes && uv pip install bitsandbytes==0.46.1

# Addressing security scan issue - CVE vulnerability https://github.com/advisories/GHSA-g4r7-86gm-pgqc The package is a
# dependency of lm_eval from NeMo requirements_eval.txt. We also remove zstandard, another dependency of lm_eval, which
# seems to be causing issues with NGC downloads. See https://nvbugspro.nvidia.com/bug/5149698
uv pip uninstall sqlitedict zstandard

rm -rf ./3rdparty
rm -rf /tmp/*
EOF

# In the devcontainer image, we just copy over the finished `dist-packages` folder from the build image back into the
# base pytorch container. We can then set up a non-root user and uninstall the bionemo and 3rd-party packages, so that
# they can be installed in an editable fashion from the workspace directory. This lets us install all the package
# dependencies in a cached fashion, so they don't have to be built from scratch every time the devcontainer is rebuilt.
FROM ${BASE_IMAGE} AS dev

RUN --mount=type=cache,id=apt-cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,id=apt-lib,target=/var/lib/apt,sharing=locked \
  <<EOF
set -eo pipefail
apt-get update -qy
apt-get install -qyy \
  sudo
rm -rf /tmp/* /var/tmp/*
EOF

# Use a non-root user to use inside a devcontainer (with ubuntu 23 and later, we can use the default ubuntu user).
ARG USERNAME=ubuntu
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME

# Here we delete the dist-packages directory from the pytorch base image, and copy over the dist-packages directory from
# the build image. This ensures we have all the necessary dependencies installed (megatron, nemo, etc.).
RUN <<EOF
  set -eo pipefail
  rm -rf /usr/local/lib/python3.12/dist-packages
  mkdir -p /usr/local/lib/python3.12/dist-packages
  chmod 777 /usr/local/lib/python3.12/dist-packages
  chmod 777 /usr/local/bin
EOF

USER $USERNAME

COPY --from=bionemo2-base --chown=$USERNAME:$USERNAME --chmod=777 \
  /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

COPY --from=ghcr.io/astral-sh/uv:0.6.13 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
  UV_COMPILE_BYTECODE=0 \
  UV_PYTHON_DOWNLOADS=never \
  UV_SYSTEM_PYTHON=true \
  UV_BREAK_SYSTEM_PACKAGES=1

RUN --mount=type=bind,source=./requirements-dev.txt,target=/workspace/bionemo2/requirements-dev.txt \
  --mount=type=cache,target=/root/.cache <<EOF
  set -eo pipefail
  uv pip install -r /workspace/bionemo2/requirements-dev.txt
  rm -rf /tmp/*
EOF

RUN <<EOF
  set -eo pipefail
  rm -rf /usr/local/lib/python3.12/dist-packages/bionemo*
  pip uninstall -y megatron_core
EOF


# Transformer engine attention defaults
# FIXME the following result in unstable training curves even if they are faster
#  see https://github.com/NVIDIA/bionemo-framework/pull/421
#ENV NVTE_FUSED_ATTN=1 NVTE_FLASH_ATTN=0
FROM dev AS development

WORKDIR /workspace/bionemo2
COPY --from=bionemo2-base /workspace/bionemo2/ .
COPY ./internal ./internal
# because of the `rm -rf ./3rdparty` in bionemo2-base (only Megatron-LM remains)
COPY ./3rdparty ./3rdparty

USER root

RUN <<EOF
set -eo pipefail
find . -name __pycache__ -type d -print | xargs rm -rf
uv pip install --no-build-isolation --editable ./internal/infra-bionemo
for sub in ./3rdparty/* \
           ./sub-packages/bionemo-core \
           ./sub-packages/bionemo-llm \
           ./sub-packages/bionemo-maxtoki \
           ./sub-packages/bionemo-testing; do
    uv pip install --no-deps --no-build-isolation --editable $sub
done
EOF

# Since the entire repo is owned by root, switching username for development breaks things.
ARG USERNAME=ubuntu
RUN chown $USERNAME:$USERNAME -R /workspace/bionemo2/
USER $USERNAME

# The 'release' target needs to be last so that it's the default build target.
FROM bionemo2-base AS release

RUN mkdir -p /workspace/bionemo2/.cache/

COPY VERSION .
COPY ./README.md ./
COPY ./ci/scripts ./ci/scripts

# Fix a CRIT vuln: https://github.com/advisories/GHSA-vqfr-h8mv-ghfj
RUN uv pip install h11==0.16.0

RUN chmod 777 -R /workspace/bionemo2/

# Transformer engine attention defaults
# We have to declare this again because the devcontainer splits from the release image's base.
# FIXME the following results in unstable training curves even if faster.
#  See https://github.com/NVIDIA/bionemo-framework/pull/421
# ENV NVTE_FUSED_ATTN=1 NVTE_FLASH_ATTN=0
