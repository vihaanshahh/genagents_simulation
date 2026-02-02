#!/bin/bash

set -e

echo "=========================================="
echo "GenAgents Simulation - AWS Deployment"
echo "=========================================="
echo ""

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-your_dockerhub_username}"
IMAGE_NAME="genagents-simulation"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Use docker-compose.yml (which should be the production config on server)
COMPOSE_FILE="docker-compose.yml"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with required environment variables."
    exit 1
fi

echo "Step 1: Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down || true
echo ""

echo "Step 2: Pulling latest Docker image..."
# If IMAGE_TAG is not a full path, construct it
if [[ ! "$IMAGE_TAG" =~ "/" ]]; then
    export IMAGE_TAG="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
fi
docker pull $IMAGE_TAG
echo ""

echo "Step 3: Starting containers..."
docker-compose -f $COMPOSE_FILE up -d
echo ""

echo "Step 4: Waiting for service to be ready..."
sleep 10

# Check if container is running
if docker ps | grep -q genagents-api; then
    echo "Container is running!"
    echo ""
    
    echo "Step 5: Checking health..."
    for i in {1..10}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            echo "Health check passed!"
            break
        else
            echo "Waiting for service... (attempt $i/10)"
            sleep 3
        fi
    done
    echo ""
    
    echo "Step 6: Cleaning up old images..."
    docker image prune -f
    echo ""
    
    echo "=========================================="
    echo "Deployment Complete!"
    echo "=========================================="
    echo ""
    echo "Service Status:"
    docker-compose -f $COMPOSE_FILE ps
    echo ""
    echo "View logs with:"
    echo "  docker-compose -f $COMPOSE_FILE logs -f"
    echo ""
    echo "API Endpoints:"
    echo "  http://localhost:8000/health"
    echo "  http://localhost:8000/admin"
    echo "  http://localhost:8000/simulate"
    echo ""
else
    echo "ERROR: Container failed to start!"
    echo ""
    echo "Checking logs..."
    docker-compose -f $COMPOSE_FILE logs --tail=50
    exit 1
fi
