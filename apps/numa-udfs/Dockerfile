####################################################################################################
# builder: install needed dependencies and setup virtual environment
####################################################################################################

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-bookworm AS builder

ARG POETRY_VERSION=1.6

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    POETRY_VERSION=${POETRY_VERSION} \
    POETRY_HOME="/opt/poetry" \
    PATH="$POETRY_HOME/bin:$PATH"

RUN pip install --no-cache-dir poetry==$POETRY_VERSION

WORKDIR /app

#COPY numalogic-0.11a0-py3-none-any.whl /app
#COPY numalogic_registry-0.11a0-py3-none-any.whl /app
#COPY numalogic_connectors-0.11a0-py3-none-any.whl /app

COPY poetry.lock pyproject.toml ./

RUN poetry install --without dev --no-root  \
    && rm -rf $POETRY_CACHE_DIR \
    && pip cache purge

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

COPY ../.. /app
WORKDIR /app

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
EXPOSE 5000
