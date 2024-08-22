.PHONY: build release lint format test test_watch build-3.11 build-3.12 start-api-postgres start-db start-postgres

# lint commands

lint:
	poetry run ruff .
	poetry run ruff format . --diff

format:
	poetry run ruff format .
	poetry run ruff --select I --fix .

# test commands

TEST_PATH ?= "tests/integration_tests/"

REVISION ?= $(shell git rev-parse --short HEAD)

test-postgres-license-noop:
	PYTHONPATH=../license_noop:../storage_postgres:$PYTHONPATH DATABASE_URI=postgres://postgres:postgres@localhost:5433/postgres?sslmode=disable poetry run pytest -v $(TEST_PATH)

test-postgres-license-jwt:
	PYTHONPATH=../license_jwt:../storage_postgres:$PYTHONPATH DATABASE_URI=postgres://postgres:postgres@localhost:5433/postgres?sslmode=disable poetry run pytest -v $(TEST_PATH)

test-postgres-watch:
	PYTHONPATH=../license_noop:../storage_postgres:$PYTHONPATH DATABASE_URI=postgres://postgres:postgres@localhost:5433/postgres?sslmode=disable poetry run ptw .

# build commands

build-3.11-postgres:
	docker build --build-arg PYTHON_VERSION='3.11' --build-arg REVISION=$(REVISION) --build-arg VARIANT=cloud --build-context storage=../storage_postgres --build-context license=../license_noop -t langchain/langgraph-api:3.11 .

build-3.12-postgres:
	docker build --build-arg PYTHON_VERSION='3.12' --build-arg REVISION=$(REVISION) --build-arg VARIANT=cloud --build-context storage=../storage_postgres --build-context license=../license_noop -t langchain/langgraph-api:3.12 .

build-3.11-postgres-licensed:
	docker build --build-arg PYTHON_VERSION='3.11' --build-arg REVISION=$(REVISION) --build-arg VARIANT=licensed --build-context storage=../storage_postgres --build-context license=../license_jwt -t langchain/langgraph-api:3.11 .

build-3.12-postgres-licensed:
	docker build --build-arg PYTHON_VERSION='3.12' --build-arg REVISION=$(REVISION) --build-arg VARIANT=licensed --build-context storage=../storage_postgres --build-context license=../license_jwt -t langchain/langgraph-api:3.12 .

build-postgres: build-3.11-postgres build-3.12-postgres

build-postgres-licensed: build-3.11-postgres-licensed build-3.12-postgres-licensed

# dev commands

start-postgres-api:
	sleep 2 && N_JOBS_PER_WORKER=2 LANGSERVE_GRAPHS='{"agent": "./tests/graphs/agent.py:graph", "other": "./tests/graphs/other.py:make_graph"}' LANGSMITH_LANGGRAPH_API_VARIANT=test DATABASE_URI=postgres://postgres:postgres@localhost:5433/postgres?sslmode=disable MIGRATIONS_PATH=../storage_postgres/migrations PYTHONPATH=../license_noop:../storage_postgres:$PYTHONPATH poetry run uvicorn langgraph_api.server:app --reload --port 9123 --log-config logging.json --reload-exclude 'tests/integration_tests/**' --reload-include '../storage_postgres/langgraph_storage' --no-access-log

start-postgres-api-licensed:
	sleep 2 && N_JOBS_PER_WORKER=2 LANGSERVE_GRAPHS='{"agent": "./tests/graphs/agent.py:graph", "other": "./tests/graphs/other.py:make_graph"}' DATABASE_URI=postgres://postgres:postgres@localhost:5433/postgres?sslmode=disable MIGRATIONS_PATH=../storage_postgres/migrations PYTHONPATH=../license_jwt:../storage_postgres:$PYTHONPATH poetry run uvicorn langgraph_api.server:app --reload --port 9123 --log-config logging.json --reload-exclude 'tests/integration_tests/**' --reload-include '../storage_postgres/langgraph_storage' --no-access-log

start-trial:
	LANGSERVE_GRAPHS='{"agent": "./tests/graphs/agent.py:graph", "other": "./tests/graphs/other.py:make_graph"}' PYTHONPATH=../storage_trial:$PYTHONPATH poetry run uvicorn langgraph_api.trial:app --reload --port 9124 --log-config logging.json --reload-exclude 'tests/integration_tests/**'

start-postgres:
	npx concurrently --kill-others "docker compose up --remove-orphans" "make start-postgres-api"

start-postgres-license:
	npx concurrently --kill-others "docker compose up --remove-orphans" "make start-postgres-api-licensed"
