####################################################################################################
# builder: install needed dependencies
####################################################################################################

ARG PYTHON_VERSION=3.11
ARG POETRY_VERSION=1.6
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=${POETRY_VERSION} \
  POETRY_HOME="/opt/poetry" \
  POETRY_VIRTUALENVS_IN_PROJECT=true \
  POETRY_NO_INTERACTION=1 \
  PYSETUP_PATH="/opt/pysetup" \
  VENV_PATH="/opt/pysetup/.venv"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        curl \
        build-essential \
        dumb-init \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache --upgrade pip \
    && curl -sSL https://install.python-poetry.org | python3 -

####################################################################################################
# udf: used for running the udf vertices
####################################################################################################
FROM builder AS udf

ARG INSTALL_EXTRAS

WORKDIR $PYSETUP_PATH
COPY ./pyproject.toml ./poetry.lock ./

# TODO install cpu/gpu based on args/arch
RUN poetry install --without dev --no-cache --no-root --extras "${INSTALL_EXTRAS}" && \
    poetry run pip install --no-cache "torch>=2.0,<3.0" --index-url https://download.pytorch.org/whl/cpu && \
    poetry run pip install --no-cache "pytorch-lightning>=2.0<3.0" && \
    rm -rf ~/.cache/pypoetry/

COPY . /app
WORKDIR /app

ENTRYPOINT ["/usr/bin/dumb-init", "--"]

EXPOSE 5000
