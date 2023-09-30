####################################################################################################
# builder: install needed dependencies and setup virtual environment
####################################################################################################

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG POETRY_VERSION=1.6
ARG INSTALL_EXTRAS

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VERSION=${POETRY_VERSION} \
    POETRY_HOME="/opt/poetry" \
    PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir poetry==$POETRY_VERSION

WORKDIR /app
COPY poetry.lock pyproject.toml ./

RUN poetry install --without dev --no-root --extras "${INSTALL_EXTRAS}"  \
    && poetry run pip install --no-cache-dir "torch>=2.0,<3.0" --index-url https://download.pytorch.org/whl/cpu \
    && poetry run pip install --no-cache-dir "lightning[pytorch]" \
    && rm -rf $POETRY_CACHE_DIR \
    && pip cache purge \
    && apt-get purge -y --auto-remove build-essential

####################################################################################################
# runtime: used for running the udf vertices
####################################################################################################
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install --no-install-recommends -y dumb-init \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && apt-get purge -y --auto-remove


ENV VIRTUAL_ENV=/app/.venv
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . /app
WORKDIR /app

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
EXPOSE 5000
