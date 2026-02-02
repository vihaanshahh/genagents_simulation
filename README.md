# GenAgents Simulation

Multi-agent simulation with **3,505 agents** powered by Cerebras AI.

## Features

- **4 Cerebras Models** - Auto-optimized for cost ($0.10-$2.75/1M tokens)
- **Auto-Batching** - Smart batch sizing based on model config
- **Cost Tracking** - Real-time breakdown per request
- **Admin Dashboard** - Track usage, costs, model split
- **Concurrent Models** - Run 2-4 models simultaneously
- **Rate Limit Safe** - 60k tokens/min managed globally

## Quick Start

### Install

**Local development:**
```bash
pip install -r requirements.txt
```

**Docker (recommended for production):**
```bash
make docker-build
```

### Configure
Create `.env`:
```bash
# Cerebras (required)
CEREBRAS_API_KEY=your_key
CEREBRAS_LLAMA31_ENABLED=true       # $0.10/$0.10 - CHEAPEST
CEREBRAS_GPT_OSS_ENABLED=true       # $0.35/$0.75 - FASTEST
CEREBRAS_QWEN_ENABLED=true          # $0.60/$1.20 - BALANCED
CEREBRAS_GLM_ENABLED=false          # $2.25/$2.75 - REASONING

# Neon DB for admin metrics (optional)
DATABASE_URL=postgresql://user:pass@host/db

# AWS Bedrock fallback (optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### Run

**Using Make (recommended):**
```bash
# Start API server
make start

# Check status
make status

# View logs
make logs

# Stop server
make stop
```

**Using Docker Compose directly:**
```bash
docker-compose up -d
```

**Local development:**
```bash
python -m genagents_simulation.api
```

### Test
```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Do you support renewable energy?",
    "options": ["Yes", "No", "Undecided"],
    "agent_count": 100,
    "llm_config_name": "llama3.1-8b",
    "use_memory": false
  }'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/simulate` | POST | Run simulation |
| `/simulate/stream` | POST | Run with real-time progress |
| `/admin` | GET | Admin dashboard (HTML) |
| `/admin/dashboard` | GET | Dashboard data (JSON) |
| `/admin/metrics?days=30` | GET | Detailed metrics |
| `/metrics` | GET | Real-time provider status |
| `/models` | GET | List available models |

## Admin Dashboard

### View in Browser

**Open:** `http://localhost:8000/admin`

Real-time dashboard with:
- Today/Week/Month stats
- Model split percentages
- Provider status
- Auto-refresh

### API Access
```bash
curl http://localhost:8000/admin/dashboard
```

Response:
```json
{
  "today": {"requests": 50, "agents": 5000, "cost": 0.25},
  "this_week": {"requests": 300, "agents": 30000, "cost": 1.50},
  "this_month": {"requests": 1200, "agents": 120000, "cost": 6.00},
  "model_split": [
    {
      "model": "llama3.1-8b",
      "cost": 4.50,
      "cost_percentage": 75.0,
      "token_percentage": 80.0,
      "request_count": 900
    },
    {
      "model": "gpt-oss-120b",
      "cost": 1.50,
      "cost_percentage": 25.0,
      "token_percentage": 20.0,
      "request_count": 300
    }
  ],
  "provider_status": {
    "cerebras_llama3.1-8b": {
      "daily_tokens_used": 450000,
      "tokens_remaining": 550000,
      "success_rate": "98.5%"
    }
  }
}
```

### View Detailed Metrics
```bash
curl http://localhost:8000/admin/metrics?days=7
```

Shows:
- Total accumulated costs
- Model usage split (which models used most)
- Daily trends
- Token consumption
- Recent requests
- Cost per agent statistics

## Model Selection

| Model | Cost (in/out) | Speed | Best For |
|-------|---------------|-------|----------|
| **llama3.1-8b** | $0.10/$0.10 | 2200/s | Cost optimization |
| **gpt-oss-120b** | $0.35/$0.75 | 3000/s | Speed + reasoning |
| **qwen-3-235b** | $0.60/$1.20 | 1400/s | Balanced |
| **zai-glm-4.7** | $2.25/$2.75 | 1000/s | Advanced reasoning |

**Default**: llama3.1-8b (cheapest)

## API Parameters

```json
{
  "question": "Your question",
  "options": ["Option1", "Option2"],
  "agent_count": 100,
  "llm_config_name": "llama3.1-8b",
  "use_memory": false,
  "filters": {
    "states": ["CA", "NY"],
    "age_min": 25,
    "age_max": 65
  }
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | required | Question to ask agents |
| `options` | array | required | Answer options |
| `agent_count` | integer | 1 | Number of agents |
| `llm_config_name` | string | llama3.1-8b | Model to use |
| `use_memory` | boolean | false | Enable agent memory (slower, richer) |
| `filters` | object | null | Demographic filters |

**Auto-optimized** (no manual config needed):
- `agents_per_api_call` - Batch size based on context window, pricing, rate limits
- `max_memories_per_agent` - Memory depth based on model pricing (10-20)

## Response Format

```json
{
  "individual_responses": [...],
  "summary": {
    "Your question": {
      "counts": {"Yes": 75, "No": 25},
      "percentages": {"Yes": "75.0%", "No": "25.0%"}
    }
  },
  "num_agents": 100,
  "execution_time_seconds": 12.5,
  "successful_responses": 98,
  "failed_responses": 2,
  "optimization_settings": {
    "agents_per_api_call": 150,
    "max_memories_per_agent": 0,
    "model": "llama3.1-8b"
  },
  "cost_breakdown": {
    "total_usd": 0.0125,
    "cost_per_agent": 0.000125,
    "estimated_cost_per_1000_agents": 0.13,
    "by_model": {"llama3.1-8b": 0.0125}
  },
  "token_usage": {
    "total_input_tokens": 50000,
    "total_output_tokens": 12500
  }
}
```

## Cost Examples

| Scenario | Model | Agents | Cost | Time |
|----------|-------|--------|------|------|
| Testing | llama3.1-8b | 10 | $0.0005 | 2s |
| Production | gpt-oss-120b | 500 | $0.14 | 30s |
| Research | qwen-3-235b | 100 | $0.20 | 1m |

**Budget calculator:**
- 1000 agents/day with llama3.1-8b = $0.05/day = **$1.50/month**
- 1000 agents/day with gpt-oss-120b = $0.28/day = **$8.40/month**

## Load Balancing

System uses priority-based distribution:

1. **llama3.1-8b** tried first (cheapest, fast)
2. **gpt-oss-120b** if llama fails (fastest)
3. **qwen-3-235b** if both fail (balanced)
4. **AWS Bedrock** fallback if all Cerebras exhausted
5. **OpenAI** last resort

**Automatic:**
- Rate limit handling (60k tokens/min shared)
- Daily token tracking (1M/day per model)
- Failover on errors/timeouts
- Cost optimization

## Neon DB Setup (Optional)

For admin metrics tracking:

```bash
# Create Neon project (via Neon CLI)
neonctl projects create --name genagents-metrics

# Get connection string
neonctl connection-string

# Add to .env
DATABASE_URL=postgresql://...
```

Schema auto-creates on first run. Tracks:
- All simulation requests
- Model usage split
- Daily cost trends
- Token consumption

## AWS Deployment

Using AWS Bedrock endpoint:

```bash
# Configure
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# Deploy
docker build -t genagents .
docker run -p 8000:8000 --env-file .env genagents
```

## Deployment

### Auto-Deploy to AWS

Push to `main` branch triggers automatic deployment to AWS EC2.

**Quick setup:**
1. See `QUICKSTART-DEPLOY.md` for 5-minute setup
2. Configure GitHub secrets (Docker Hub + AWS)
3. Push to main

**Manual deployment:**
```bash
make docker-build
make start
```

See `DEPLOYMENT.md` for detailed instructions.

## Make Commands

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make start` | Start API server |
| `make stop` | Stop API server |
| `make restart` | Restart API server |
| `make status` | Check server status |
| `make logs` | View server logs |
| `make api-test` | Test API endpoints |
| `make sim-test` | Run simulation test |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run simulation in Docker |
| `make clean` | Clean temporary files |

## License

MIT
