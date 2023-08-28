####################################################################################################
# builder: install needed dependencies
####################################################################################################

ARG PYTHON_VERSION=3.11
ARG POETRY_VERSION=1.6
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=on \
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
        wget \
        # deps for building python deps
        build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    \
    # install dumb-init
    && wget -O /dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.5/dumb-init_1.2.5_x86_64 \
    && chmod +x /dumb-init \
    && curl -sSL https://install.python-poetry.org | python3 -

####################################################################################################
# udf: used for running the udf vertices
####################################################################################################
FROM builder AS udf

ARG INSTALL_EXTRAS

WORKDIR $PYSETUP_PATH
COPY ./pyproject.toml ./poetry.lock ./
COPY requirements ./requirements

RUN poetry install --without dev --no-cache --no-root -E numaflow --extras "${INSTALL_EXTRAS}" && \
    poetry run pip install --no-cache -r requirements/requirements-torch.txt && \
    rm -rf ~/.cache/pypoetry/

ADD . /app
WORKDIR /app

ENTRYPOINT ["/dumb-init", "--"]

EXPOSE 5000