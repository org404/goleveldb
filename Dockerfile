ARG PYTHON_VERSION

## build stage

FROM python:${PYTHON_VERSION}-slim AS build

ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

# Set the working directory
WORKDIR /api

# Copy only dependencies
COPY constraints.txt ./constraints.txt
COPY --from=license constraints.txt ./constraints-license.txt
COPY --from=storage constraints.txt ./constraints-storage.txt

# Install all dependencies
RUN PIP_ROOT_USER_ACTION=ignore PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir -r constraints.txt -r constraints-storage.txt -r constraints-license.txt \
    && python -c "import compileall; compileall.compile_path(maxlevels=1000, legacy=True)"

# Copy the rest of application code
COPY . .
COPY --from=storage . /storage
COPY --from=license . /license

# Install the application
RUN PIP_ROOT_USER_ACTION=ignore PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir --no-deps -e . /storage /license \
    && python -c "import compileall; compileall.compile_path(maxlevels=1000, legacy=True)" \
    && find /api/langgraph_api -type f -name '*.py' -print0 | xargs -0 rm

## final stage

FROM python:${PYTHON_VERSION}-slim AS final

ARG PYTHON_VERSION
ARG REVISION
ARG VARIANT

# Copy from build stage
COPY --from=build /api /api
COPY --from=build /storage /storage
COPY --from=build /license /license
COPY --from=build /usr/local/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages
COPY --from=build /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY --from=build /usr/local/bin/langgraph-verify-graphs /usr/local/bin/langgraph-verify-graphs

# Set working directory
WORKDIR /api

# Enable Python stack traces on segfaults https://stackoverflow.com/a/29246977
# Ensure Python does not buffer output, which is recommended when inside a container.
ENV PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=True PORT=8000 PIP_ROOT_USER_ACTION=ignore N_WORKERS=1 N_JOBS_PER_WORKER=10 LANGSMITH_LANGGRAPH_API_REVISION=$REVISION LANGSMITH_LANGGRAPH_API_VARIANT=$VARIANT

LABEL org.opencontainers.image.revision=$REVISION

HEALTHCHECK --interval=5s --timeout=2s --retries=5 CMD [ "python", "/api/healthcheck.py" ]

CMD exec uvicorn langgraph_api.server:app --log-config /api/logging.json --host 0.0.0.0 --port $PORT --no-access-log
