# AWS Deployment Guide

This guide explains how to set up automatic deployments to your AWS EC2 instance whenever code is pushed to the `main` branch.

## Prerequisites

1. **Docker Hub Account** (or AWS ECR)
2. **AWS EC2 Instance** running with:
   - Docker installed
   - Docker Compose installed
   - SSH access configured

## Setup Instructions

### Step 1: Configure GitHub Secrets

Add these secrets to your GitHub repository (Settings > Secrets and variables > Actions):

| Secret Name | Description | Example |
|------------|-------------|---------|
| `DOCKER_USERNAME` | Your Docker Hub username | `myusername` |
| `DOCKER_PASSWORD` | Your Docker Hub password or token | `dckr_pat_xxxxx` |
| `AWS_HOST` | Your EC2 instance public IP or domain | `18.123.45.67` or `api.example.com` |
| `AWS_USER` | SSH username for EC2 | `ubuntu` (for Ubuntu AMI) or `ec2-user` (for Amazon Linux) |
| `AWS_SSH_KEY` | Private SSH key for EC2 access | Contents of your `.pem` file |
| `AWS_SSH_PORT` | SSH port (optional, defaults to 22) | `22` |
| `DEPLOY_PATH` | Path to app on EC2 (optional) | `/home/ubuntu/genagents_simulation` |

### Step 2: Prepare Your AWS EC2 Instance

SSH into your EC2 instance and run:

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Log out and log back in for group changes to take effect
exit
```

SSH back in and verify:

```bash
docker --version
docker-compose --version
```

### Step 3: Set Up Deployment Directory

```bash
# Create deployment directory
mkdir -p ~/genagents_simulation
cd ~/genagents_simulation

# Create .env file with your environment variables
nano .env
```

Add your environment variables to `.env`:

```bash
# AWS Credentials
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# Cerebras API
CEREBRAS_API_KEY=your_key
CEREBRAS_API=https://api.cerebras.ai/v1
CEREBRAS_MODEL=llama-3.3-70b

# OpenAI (optional)
OPENAI_API_KEY=your_key

# Database (optional)
DATABASE_URL=postgresql://user:pass@host/db

# Other settings
LLM_VERS=gpt-oss-120b
DEBUG=False
PORT=8000
```

### Step 4: Copy Deployment Files to AWS

From your local machine:

```bash
# Copy docker-compose.prod.yml
scp -i your-key.pem docker-compose.prod.yml ubuntu@your-ec2-ip:~/genagents_simulation/docker-compose.yml

# Copy deploy script
scp -i your-key.pem deploy.sh ubuntu@your-ec2-ip:~/genagents_simulation/

# SSH and make deploy script executable
ssh -i your-key.pem ubuntu@your-ec2-ip
chmod +x ~/genagents_simulation/deploy.sh
```

### Step 5: Configure AWS Security Group

Make sure your EC2 security group allows:
- **Port 22** (SSH) - for deployment
- **Port 8000** (API) - for application access

In AWS Console:
1. Go to EC2 > Security Groups
2. Select your instance's security group
3. Add inbound rules:
   - Type: Custom TCP, Port: 8000, Source: 0.0.0.0/0 (or restrict as needed)
   - Type: SSH, Port: 22, Source: Your IP

### Step 6: Test Manual Deployment

Before relying on GitHub Actions, test the deployment manually:

```bash
# On your EC2 instance
cd ~/genagents_simulation

# Set Docker username
export DOCKER_USERNAME=your_dockerhub_username

# Run deployment
./deploy.sh
```

### Step 7: Push to Main Branch

Now whenever you push to `main`, GitHub Actions will:
1. Build the Docker image
2. Push it to Docker Hub
3. SSH into your EC2 instance
4. Pull the latest image
5. Restart the containers

```bash
git add .
git commit -m "Setup deployment"
git push origin main
```

Monitor the deployment in GitHub Actions tab.

## Manual Deployment

You can also trigger deployment manually:

### From GitHub
1. Go to Actions tab
2. Select "Deploy to AWS"
3. Click "Run workflow"

### From AWS Instance
```bash
cd ~/genagents_simulation
./deploy.sh
```

## Monitoring

### Check Container Status
```bash
docker-compose ps
```

### View Logs
```bash
# All logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f genagents-api
```

### Check Health
```bash
curl http://localhost:8000/health
```

### Access Admin Dashboard
Open in browser: `http://your-ec2-ip:8000/admin`

## Troubleshooting

### Deployment Fails

1. **Check GitHub Actions logs** for error messages
2. **SSH into EC2** and check:
   ```bash
   docker-compose ps
   docker-compose logs --tail=50
   ```

### Container Won't Start

Check logs:
```bash
docker-compose logs genagents-api
```

Common issues:
- Missing environment variables in `.env`
- Invalid API keys
- Port 8000 already in use

### Can't Access API

1. Check if container is running:
   ```bash
   docker-compose ps
   ```

2. Check AWS Security Group allows port 8000

3. Check if service is listening:
   ```bash
   curl http://localhost:8000/health
   ```

### Out of Disk Space

Clean up old images:
```bash
docker system prune -a -f
```

## Rollback

If a deployment fails, rollback to a previous version:

```bash
# On EC2 instance
cd ~/genagents_simulation

# Pull specific version by commit SHA
export IMAGE_TAG=your_dockerhub_username/genagents-simulation:COMMIT_SHA
docker pull $IMAGE_TAG

# Restart with that version
docker-compose down
docker-compose up -d
```

## Alternative: AWS ECR

If you prefer AWS ECR over Docker Hub:

1. Create an ECR repository:
   ```bash
   aws ecr create-repository --repository-name genagents-simulation
   ```

2. Update GitHub secrets:
   - Remove `DOCKER_USERNAME` and `DOCKER_PASSWORD`
   - Add `AWS_ACCOUNT_ID` and `AWS_ECR_REGION`

3. Modify `.github/workflows/deploy.yml` to use ECR login and push

## Cost Optimization

### Auto-Stop Instance

To save costs, stop the EC2 instance when not in use:

```bash
# Stop instance
aws ec2 stop-instances --instance-ids i-xxxxxxxx

# Start instance
aws ec2 start-instances --instance-ids i-xxxxxxxx
```

Or use AWS Lambda to schedule automatic start/stop.

## Security Best Practices

1. **Use IAM roles** instead of hardcoded AWS credentials when possible
2. **Restrict security groups** to specific IPs
3. **Rotate secrets** regularly
4. **Use HTTPS** with SSL/TLS (set up nginx reverse proxy)
5. **Keep Docker images updated**
6. **Use Docker secrets** for sensitive data instead of environment variables

## Support

For issues:
1. Check logs: `docker-compose logs`
2. Review GitHub Actions output
3. Check EC2 instance metrics in AWS Console
