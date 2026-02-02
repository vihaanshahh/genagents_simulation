.PHONY: help build run clean test setup docker-build docker-run docker-compose-up docker-compose-down docker-shell start stop api-test

# Default target
help:
	@echo "GenAgents Simulation - Makefile Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup              - Setup local Python environment"
	@echo "  make docker-setup       - Setup Docker environment"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build       - Build Docker image"
	@echo "  make docker-run         - Run simulation in Docker"
	@echo "  make docker-compose-up  - Start services with docker-compose"
	@echo "  make docker-compose-down - Stop docker-compose services"
	@echo "  make docker-shell       - Open interactive shell in container"
	@echo "  make docker-clean       - Remove Docker images and containers"
	@echo ""
	@echo "API Server Commands:"
	@echo "  make start              - Start API server in background"
	@echo "  make stop               - Stop API server"
	@echo "  make restart            - Restart API server"
	@echo "  make status             - Check server status"
	@echo "  make api-test           - Test API endpoints"
	@echo "  make sim-test           - Test simulation with formatted output"
	@echo "  make stream-test        - Test streaming simulation with progress"
	@echo "  make list-sims          - List running simulations"
	@echo "  make cancel-sim ID=xxx  - Cancel a running simulation"
	@echo "  make logs               - View API server logs"
	@echo ""
	@echo "Local Commands:"
	@echo "  make run                - Run simulation locally"
	@echo "  make test               - Run endpoint tests"
	@echo "  make clean              - Clean temporary files"
	@echo ""
	@echo "Example Usage:"
	@echo "  make docker-build"
	@echo "  make docker-run QUESTION='Do you like AI?' OPTIONS='Yes,No' AGENTS=5"

# Variables
QUESTION ?= Do you support renewable energy?
OPTIONS ?= Yes,No,Undecided
LLM_CONFIG ?= gpt-oss-120b
AGENTS ?= 1

# Setup local environment
setup:
	@echo "Setting up local Python environment..."
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete! Activate with: source venv/bin/activate"

# Setup Docker environment
docker-setup:
	@echo "Setting up Docker environment..."
	./setup-docker.sh

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t genagents-simulation:latest .
	@echo "Build complete!"

# Run simulation in Docker
docker-run:
	@echo "Running simulation in Docker..."
	docker run --rm \
		--env-file .env \
		-v "$$(pwd)/genagents_simulation/agent_bank:/app/genagents_simulation/agent_bank:ro" \
		-v "$$(pwd)/genagents_simulation/configs:/app/genagents_simulation/configs:ro" \
		genagents-simulation:latest \
		python genagents_simulation/run.py \
		--question "$(QUESTION)" \
		--options "$(OPTIONS)" \
		--llm_config_name "$(LLM_CONFIG)" \
		--agent_count $(AGENTS)

# Start docker-compose services
docker-compose-up:
	@echo "Starting services with docker-compose..."
	docker-compose up

# Stop docker-compose services
docker-compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

# Open interactive shell in container
docker-shell:
	@echo "Opening interactive shell..."
	docker run -it --rm \
		--env-file .env \
		-v "$$(pwd)/genagents_simulation:/app/genagents_simulation" \
		genagents-simulation:latest \
		/bin/bash

# Clean Docker images and containers
docker-clean:
	@echo "Cleaning Docker resources..."
	docker-compose down -v
	docker rmi genagents-simulation:latest || true
	docker system prune -f

# Run simulation locally
run:
	@echo "Running simulation locally..."
	PYTHONPATH=$$(pwd):$$PYTHONPATH python genagents_simulation/run.py \
		--question "$(QUESTION)" \
		--options "$(OPTIONS)" \
		--llm_config_name "$(LLM_CONFIG)" \
		--agent_count $(AGENTS)

# Run endpoint tests
test:
	@echo "Running endpoint tests..."
	python test_endpoint_bedrock.py

# Clean temporary files
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	@echo "Clean complete!"

# Build and run in one command
docker-quick: docker-build docker-run

# Full Docker setup and run
docker-full: docker-setup docker-build docker-run

# Start API server
start:
	@./start.sh

# Stop API server
stop:
	@./stop.sh

# Test API
api-test:
	@./test-api.sh

# Test simulation with formatted output
sim-test:
	@./test-simulation.sh "$(QUESTION)" $(AGENTS)

# Test streaming simulation
stream-test:
	@./test-stream.sh "$(QUESTION)" $(AGENTS)

# List running simulations
list-sims:
	@./list-simulations.sh

# Alias for list-sims
list: list-sims

# Cancel a simulation
cancel-sim:
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make cancel-sim ID=<simulation_id>"; \
		echo "First run: make list-sims"; \
		exit 1; \
	fi
	@./cancel-simulation.sh $(ID)

# View logs
logs:
	@docker logs -f genagents-api

# Restart API server
restart:
	@./restart.sh

# Check status
status:
	@./status.sh
