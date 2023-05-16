####################################################################################################
# builder: install needed dependencies
####################################################################################################

FROM python:3.10-slim-bullseye AS builder

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=on \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.4.2 \
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
    && apt-get install -y git \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    \
    # install dumb-init
    && wget -O /dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.5/dumb-init_1.2.5_x86_64 \
    && chmod +x /dumb-init \
    && curl -sSL https://install.python-poetry.org | python3 -

####################################################################################################
# mlflow: used for running the mlflow server
####################################################################################################
FROM builder AS mlflow

WORKDIR $PYSETUP_PATH
COPY ./pyproject.toml ./poetry.lock ./
RUN poetry install --only mlflowserver --no-cache --no-root && \
    rm -rf ~/.cache/pypoetry/

ADD . /app
WORKDIR /app

RUN chmod +x entry.sh

ENTRYPOINT ["/dumb-init", "--"]
CMD ["/app/entry.sh"]

EXPOSE 5000

####################################################################################################
# udf: used for running the udf vertices
####################################################################################################
FROM builder AS udf

WORKDIR $PYSETUP_PATH
COPY ./pyproject.toml ./poetry.lock ./
COPY requirements ./requirements

RUN poetry install --no-cache --no-root && \
    poetry run pip install --no-cache -r requirements/requirements-torch.txt && \
    rm -rf ~/.cache/pypoetry/

ADD . /app
WORKDIR /app

RUN chmod +x entry.sh

ENTRYPOINT ["/dumb-init", "--"]
CMD ["/app/entry.sh"]

EXPOSE 5000
