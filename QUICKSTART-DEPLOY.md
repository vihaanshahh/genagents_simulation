# Quick Start: Auto-Deploy to AWS

This guide will get your automatic deployments working in 15 minutes.

## What You Need

- Docker Hub account (free at hub.docker.com)
- AWS EC2 instance running
- SSH key for your EC2 instance

## 5-Minute Setup

### 1. Set Up AWS Instance (One-Time)

SSH into your EC2 instance and run:

```bash
# Copy and run the setup script
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/genagents_simulation/main/setup-aws.sh -o setup-aws.sh
chmod +x setup-aws.sh
./setup-aws.sh
```

Or manually:
```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again, then verify
docker --version
docker-compose --version
```

Create deployment directory:
```bash
mkdir -p ~/genagents_simulation
cd ~/genagents_simulation
```

Create `.env` file with your credentials:
```bash
nano .env
```

Paste this template and fill in your values:
```bash
# Required
CEREBRAS_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here

# Optional
DATABASE_URL=postgresql://user:pass@host/db
OPENAI_API_KEY=your_key_here

# Defaults (can leave as-is)
AWS_REGION=us-east-1
LLM_VERS=gpt-oss-120b
DEBUG=False
PORT=8000
```

### 2. Configure GitHub Secrets

Go to your GitHub repo → Settings → Secrets and variables → Actions

Click "New repository secret" and add these:

| Name | Value | Where to Find |
|------|-------|---------------|
| `DOCKER_USERNAME` | Your Docker Hub username | hub.docker.com |
| `DOCKER_PASSWORD` | Your Docker Hub token | hub.docker.com → Account Settings → Security → New Access Token |
| `AWS_HOST` | EC2 public IP | AWS Console → EC2 → Your instance → Public IPv4 address |
| `AWS_USER` | `ubuntu` or `ec2-user` | Depends on your AMI (Ubuntu uses `ubuntu`) |
| `AWS_SSH_KEY` | Contents of your .pem file | Copy entire file including BEGIN/END lines |

Optional secrets:
- `AWS_SSH_PORT` - Default is 22
- `DEPLOY_PATH` - Default is `/home/ubuntu/genagents_simulation`

### 3. Push to Main

```bash
git add .
git commit -m "Setup auto-deployment"
git push origin main
```

Watch the deployment in the Actions tab!

## Verify It Works

After GitHub Actions completes:

```bash
# Check if containers are running
curl http://YOUR_EC2_IP:8000/health

# Open admin dashboard in browser
http://YOUR_EC2_IP:8000/admin
```

## File Overview

| File | Purpose |
|------|---------|
| `.github/workflows/deploy.yml` | GitHub Actions workflow - auto-deploys on push to main |
| `docker-compose.prod.yml` | Production Docker Compose config |
| `deploy.sh` | Deployment script for AWS instance |
| `setup-aws.sh` | One-time AWS instance setup script |
| `DEPLOYMENT.md` | Detailed deployment documentation |

## Common Issues

### Can't Connect to EC2

**Problem:** GitHub Actions shows "Connection refused"

**Fix:** Make sure:
1. Security Group allows SSH (port 22) from GitHub's IPs (easier: allow from anywhere temporarily)
2. Your SSH key in GitHub secrets matches the EC2 key pair
3. Your EC2 instance is running

### Container Won't Start

SSH into EC2 and check:
```bash
cd ~/genagents_simulation
docker-compose logs genagents-api
```

Common causes:
- Missing environment variables in `.env`
- Invalid API keys
- Port 8000 already in use

### Can't Access API

**Problem:** Can curl health endpoint from EC2 but not from browser

**Fix:** AWS Security Group needs to allow inbound traffic on port 8000
1. Go to AWS Console → EC2 → Security Groups
2. Select your instance's security group
3. Edit inbound rules
4. Add: Type=Custom TCP, Port=8000, Source=0.0.0.0/0

## What Happens on Each Push

1. GitHub Actions triggers
2. Builds Docker image
3. Pushes to Docker Hub
4. SSHs into your EC2 instance
5. Pulls latest image
6. Restarts containers
7. Runs health check

Takes about 3-5 minutes total.

## Manual Deployment

Need to deploy without pushing to GitHub?

SSH into EC2:
```bash
cd ~/genagents_simulation
export DOCKER_USERNAME=your_dockerhub_username
export IMAGE_TAG=your_dockerhub_username/genagents-simulation:latest
./deploy.sh
```

## Monitoring

### View Logs
```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# View logs
cd ~/genagents_simulation
docker-compose logs -f
```

### Check Container Status
```bash
docker-compose ps
```

### Restart Container
```bash
docker-compose restart genagents-api
```

### Stop Everything
```bash
docker-compose down
```

## Cost Optimization

Your EC2 instance costs money even when idle. To save costs:

### Stop Instance When Not Using
```bash
# From AWS Console
EC2 → Instances → Select → Instance state → Stop

# Or with AWS CLI
aws ec2 stop-instances --instance-ids i-YOUR_INSTANCE_ID
```

### Start Instance
```bash
aws ec2 start-instances --instance-ids i-YOUR_INSTANCE_ID
```

**Note:** Public IP changes when you stop/start (unless using Elastic IP)

## Next Steps

- Set up HTTPS with Let's Encrypt and nginx
- Configure monitoring with CloudWatch
- Set up automatic backups
- Add staging environment
- Configure auto-scaling

See `DEPLOYMENT.md` for advanced topics.

## Support

Issues? Check:
1. GitHub Actions logs (Actions tab in GitHub)
2. EC2 instance logs (`docker-compose logs`)
3. AWS Security Group settings
4. `.env` file on EC2 has correct values

Still stuck? Review the detailed `DEPLOYMENT.md` guide.
