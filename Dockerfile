# 베이스: NVIDIA CUDA 런타임 + Ubuntu (cu128과 정합)
ARG CUDA_IMAGE=nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# OS 패키지 설치 (컴파일/분산 학습 준비)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    pkg-config \
    openmpi-bin \
    libopenmpi-dev \
    libaio-dev \
    && rm -rf /var/lib/apt/lists/*

# uv 설치 (Python/패키지/venv 관리 전용)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y
ENV PATH="/root/.local/bin:${PATH}"
# Docker 레이어 캐시 친화적 파일 복사 모드
ENV UV_LINK_MODE=copy

WORKDIR /app

# 의존성 캐시를 위해 pyproject만 먼저 복사 (두 서브프로젝트 모두)
COPY src/pytorch-forecasting/pyproject.toml ./src/pytorch-forecasting/pyproject.toml
COPY src/data_with_darts/pyproject.toml ./src/data_with_darts/pyproject.toml

# 빌드 시 설치할 서브프로젝트 선택 (기본: data_with_darts)
ARG PROJECT=src/data_with_darts
WORKDIR /app/${PROJECT}

# venv 생성 및 의존성 동기화 (dev 제외, Python 3.12 사용)
# - [tool.uv.index]와 [tool.uv.sources] 설정을 활용해 PyTorch nightly(CUDA 12.8) 채널 사용
RUN uv sync --python 3.12 --no-dev

# 이제 전체 소스 복사 후, 프로젝트(에디터블 포함) 재동기화
WORKDIR /app
COPY . .
WORKDIR /app/${PROJECT}
RUN uv sync --python 3.12 --no-dev

# 컨테이너 기본 셸
ENV VIRTUAL_ENV="/app/${PROJECT}/.venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# 기본 커맨드: 셸 진입 (필요 시 uv run으로 실행)
CMD ["bash"]


