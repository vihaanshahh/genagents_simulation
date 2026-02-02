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
	@echo "  make start              - Start API server with docker-compose"
	@echo "  make stop               - Stop API server"
	@echo "  make restart            - Restart API server"
	@echo "  make status             - Check server status and health"
	@echo "  make logs               - View API server logs"
	@echo "  make api-test           - Test API endpoints"
	@echo "  make sim-test           - Test simulation (set QUESTION and AGENTS)"
	@echo "  make stream-test        - Test streaming simulation"
	@echo "  make list               - List running containers"
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
	@echo "Starting API server with docker-compose..."
	@docker-compose up -d
	@echo "API server started! Access at http://localhost:8000"
	@echo "Admin dashboard: http://localhost:8000/admin"

# Stop API server
stop:
	@echo "Stopping API server..."
	@docker-compose down
	@echo "API server stopped."

# Restart API server
restart:
	@echo "Restarting API server..."
	@docker-compose restart
	@echo "API server restarted!"

# Check status
status:
	@echo "Container Status:"
	@docker-compose ps
	@echo ""
	@echo "Health Check:"
	@curl -f http://localhost:8000/health 2>/dev/null && echo "API is healthy!" || echo "API is not responding"

# View logs
logs:
	@docker-compose logs -f genagents-api

# Test API health endpoint
api-test:
	@echo "Testing API endpoints..."
	@echo ""
	@echo "1. Health Check:"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "Failed"
	@echo ""
	@echo "2. Models endpoint:"
	@curl -s http://localhost:8000/models | python -m json.tool || echo "Failed"
	@echo ""
	@echo "3. Metrics endpoint:"
	@curl -s http://localhost:8000/metrics | python -m json.tool || echo "Failed"

# Test simulation with formatted output
sim-test:
	@echo "Running simulation test..."
	@curl -X POST http://localhost:8000/simulate \
		-H "Content-Type: application/json" \
		-d '{"question": "$(QUESTION)", "options": ["Yes", "No", "Undecided"], "agent_count": $(AGENTS), "llm_config_name": "$(LLM_CONFIG)", "use_memory": false}' \
		| python -m json.tool

# Test streaming simulation
stream-test:
	@echo "Running streaming simulation test..."
	@echo "This will show real-time progress..."
	@curl -X POST http://localhost:8000/simulate/stream \
		-H "Content-Type: application/json" \
		-d '{"question": "$(QUESTION)", "options": ["Yes", "No", "Undecided"], "agent_count": $(AGENTS), "llm_config_name": "$(LLM_CONFIG)", "use_memory": false}'

# List running simulations
list-sims:
	@echo "Checking running containers..."
	@docker-compose ps

# Alias for list-sims
list: list-sims
