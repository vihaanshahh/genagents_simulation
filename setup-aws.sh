#!/bin/bash

echo "=========================================="
echo "GenAgents Simulation - AWS Setup"
echo "=========================================="
echo ""

# This script sets up the AWS EC2 instance for deployment
# Run this once on your EC2 instance after initial setup

set -e

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this script as root"
    exit 1
fi

echo "Step 1: Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "Docker installed successfully!"
else
    echo "Docker already installed."
fi
echo ""

echo "Step 2: Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "Docker Compose installed successfully!"
else
    echo "Docker Compose already installed."
fi
echo ""

echo "Step 3: Creating deployment directory..."
DEPLOY_DIR="${HOME}/genagents_simulation"
mkdir -p $DEPLOY_DIR
cd $DEPLOY_DIR
echo "Deployment directory created at: $DEPLOY_DIR"
echo ""

echo "Step 4: Setting up .env file..."
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# AWS Credentials
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_REGION=us-east-1

# Cerebras API
CEREBRAS_API_KEY=your_key_here
CEREBRAS_API=https://api.cerebras.ai/v1
CEREBRAS_MODEL=llama-3.3-70b

# OpenAI (optional)
OPENAI_API_KEY=your_key_here

# Database (optional)
DATABASE_URL=postgresql://user:pass@host/db

# Other settings
LLM_VERS=gpt-oss-120b
DEBUG=False
PORT=8000
EOF
    echo "Created .env file. IMPORTANT: Edit this file with your actual credentials!"
    echo "  nano $DEPLOY_DIR/.env"
else
    echo ".env file already exists."
fi
echo ""

echo "Step 5: Creating docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
services:
  genagents-api:
    image: ${IMAGE_TAG:-genagents-simulation:latest}
    container_name: genagents-api
    environment:
      - AWS_BEARER_TOKEN_BEDROCK=${AWS_BEARER_TOKEN_BEDROCK}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
      - AWS_REGION=${AWS_REGION:-us-east-1}
      - BEDROCK_MODEL=${BEDROCK_MODEL:-anthropic.claude-haiku-4-5-20251001-v1:0}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_MODEL=${OPENAI_MODEL:-gpt-5-nano}
      - CEREBRAS_API_KEY=${CEREBRAS_API_KEY}
      - CEREBRAS_API=${CEREBRAS_API}
      - CEREBRAS_MODEL=${CEREBRAS_MODEL:-llama-3.3-70b}
      - LLM_VERS=${LLM_VERS:-gpt-oss-120b}
      - DEBUG=${DEBUG:-False}
      - PORT=8000
      - DATABASE_URL=${DATABASE_URL}
    ports:
      - "${PORT:-8000}:8000"
    command: ["python", "-m", "uvicorn", "genagents_simulation.api:app", "--host", "0.0.0.0", "--port", "8000"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "python /usr/local/bin/healthcheck.py || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
EOF
echo "Created docker-compose.yml"
echo ""

echo "Step 6: Creating deploy.sh script..."
cat > deploy.sh << 'DEPLOYEOF'
#!/bin/bash
set -e

echo "Starting deployment..."

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-your_dockerhub_username}"
IMAGE_NAME="genagents-simulation"
COMPOSE_FILE="docker-compose.yml"

# Get IMAGE_TAG from environment or construct it
if [ -z "$IMAGE_TAG" ]; then
    export IMAGE_TAG="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found!"
    exit 1
fi

echo "Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down || true

echo "Pulling latest Docker image: $IMAGE_TAG"
docker pull $IMAGE_TAG

echo "Starting containers..."
docker-compose -f $COMPOSE_FILE up -d

echo "Waiting for service to be ready..."
sleep 10

echo "Checking health..."
for i in {1..10}; do
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo "Health check passed!"
        break
    else
        echo "Waiting for service... (attempt $i/10)"
        sleep 3
    fi
done

echo "Cleaning up old images..."
docker image prune -f

echo "Deployment complete!"
docker-compose ps
DEPLOYEOF

chmod +x deploy.sh
echo "Created deploy.sh script"
echo ""

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your actual credentials:"
echo "   nano $DEPLOY_DIR/.env"
echo ""
echo "2. Make sure Docker group membership is active:"
echo "   newgrp docker"
echo "   (or logout and login again)"
echo ""
echo "3. Test Docker:"
echo "   docker --version"
echo "   docker-compose --version"
echo ""
echo "4. Deployment will happen automatically via GitHub Actions"
echo "   Or manually run: ./deploy.sh"
echo ""
echo "5. Access your API at:"
echo "   http://$(curl -s ifconfig.me):8000/admin"
echo ""
