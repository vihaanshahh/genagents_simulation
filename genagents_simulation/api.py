#!/usr/bin/env python3
"""
FastAPI server for GenAgents simulation.
Provides REST API endpoints to run simulations.
"""
import os
import logging
import time
import uuid
import threading
from typing import List, Dict, Optional
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import json as json_lib

from genagents_simulation.run import run as run_simulation
from genagents_simulation.lib.metrics_db import get_metrics_db

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize metrics DB
metrics_db = get_metrics_db()

# Track running simulations for cancellation
running_simulations: Dict[str, Dict] = {}
simulations_lock = threading.Lock()

# Global cache for agent data
agents_cache: Optional[List[Dict]] = None
agents_cache_lock = threading.Lock()

# Cache for filter options (computed from agents_cache)
filter_options_cache: Optional[Dict] = None
filter_options_lock = threading.Lock()

# Cache for models
models_cache: Optional[List[Dict]] = None
models_cache_lock = threading.Lock()

app = FastAPI(
    title="GenAgents Simulation API",
    description="API for running multi-agent simulations with various LLM models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AgentFilters(BaseModel):
    states: Optional[List[str]] = None
    cities: Optional[List[str]] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    sex: Optional[List[str]] = None
    race: Optional[List[str]] = None
    political_views: Optional[List[str]] = None
    party_identification: Optional[List[str]] = None
    religion: Optional[List[str]] = None
    work_status: Optional[List[str]] = None
    marital_status: Optional[List[str]] = None
    education: Optional[List[str]] = None

class SimulationRequest(BaseModel):
    question: str
    options: List[str]
    llm_config_name: str = "gpt-5-nano"
    agent_count: int = 1
    max_workers: Optional[int] = None
    filters: Optional[AgentFilters] = None
    use_memory: Optional[bool] = None
    
    class Config:
        schema_extra = {
            "example": {
                "question": "Do you support renewable energy?",
                "options": ["Yes", "No", "Undecided"],
                "llm_config_name": "gpt-5-nano",
                "agent_count": 5,
                "use_memory": False,
                "filters": {
                    "states": ["CA", "NY"],
                    "age_min": 25,
                    "age_max": 65,
                    "sex": ["Male", "Female"]
                }
            }
        }

class SimulationResponse(BaseModel):
    individual_responses: List[Dict]
    summary: Dict
    num_agents: int
    execution_time_seconds: float
    successful_responses: int
    failed_responses: int
    errors: List[str]
    total_cost_usd: Optional[float] = None
    token_usage: Optional[Dict] = None
    optimization_settings: Optional[Dict] = None
    cost_breakdown: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    token_configured: bool
    models_available: int

class ModelInfo(BaseModel):
    config_name: str
    client: str
    model: str
    context_window: Optional[int] = None
    languages: Optional[List[str]] = None
    categories: Optional[List[str]] = None

# Endpoints
@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "GenAgents Simulation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "simulate": "/simulate (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Count available models
    import json
    try:
        with open("genagents_simulation/configs/llm_configs.json", "r") as f:
            configs = json.load(f)
            models_count = len(configs)
    except:
        models_count = 0
    
    return {
        "status": "healthy" if api_key else "unhealthy",
        "token_configured": bool(api_key),
        "models_available": models_count
    }

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List all available LLM models."""
    global models_cache
    
    # Return cached models if available
    if models_cache is not None:
        return models_cache
    
    with models_cache_lock:
        if models_cache is not None:
            return models_cache
            
        import json
        try:
            with open("genagents_simulation/configs/llm_configs.json", "r") as f:
                models_cache = json.load(f)
                return models_cache
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")

def compute_filter_options():
    """Compute filter options from agents cache."""
    global filter_options_cache
    
    if filter_options_cache is not None:
        return filter_options_cache
    
    with filter_options_lock:
        if filter_options_cache is not None:
            return filter_options_cache
        
        logger.info("Computing filter options from agents cache...")
        start_time = time.time()
        
        # Load agents cache first
        all_agents = load_agents_cache()
        
        states = set()
        cities = set()
        sexes = set()
        races = set()
        political_views = set()
        parties = set()
        religions = set()
        work_statuses = set()
        marital_statuses = set()
        educations = set()
        ages = []
        
        for agent_data in all_agents:
            if agent_data.get('state'): states.add(agent_data['state'])
            if agent_data.get('city'): cities.add(agent_data['city'])
            if agent_data.get('sex'): sexes.add(agent_data['sex'])
            if agent_data.get('race'): races.add(agent_data['race'])
            if agent_data.get('political_views'): political_views.add(agent_data['political_views'])
            if agent_data.get('party_identification'): parties.add(agent_data['party_identification'])
            if agent_data.get('religion'): religions.add(agent_data['religion'])
            if agent_data.get('work_status'): work_statuses.add(agent_data['work_status'])
            if agent_data.get('marital_status'): marital_statuses.add(agent_data['marital_status'])
            if agent_data.get('highest_degree_received'): educations.add(agent_data['highest_degree_received'])
            if agent_data.get('age'): ages.append(agent_data['age'])
        
        filter_options_cache = {
            "states": sorted(list(states)),
            "cities": sorted(list(cities)),
            "sexes": sorted(list(sexes)),
            "races": sorted(list(races)),
            "political_views": sorted(list(political_views)),
            "parties": sorted(list(parties)),
            "religions": sorted(list(religions)),
            "work_statuses": sorted(list(work_statuses)),
            "marital_statuses": sorted(list(marital_statuses)),
            "educations": sorted(list(educations)),
            "age_range": {"min": min(ages) if ages else 0, "max": max(ages) if ages else 100}
        }
        
        elapsed = time.time() - start_time
        logger.info(f"Computed filter options in {elapsed:.3f}s")
        
        return filter_options_cache

@app.get("/filter-options")
async def get_filter_options():
    """Get available filter options from agent population (cached)."""
    try:
        return compute_filter_options()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load filter options: {str(e)}")

def load_agents_cache():
    """Load all agent data into memory cache."""
    global agents_cache
    
    if agents_cache is not None:
        return agents_cache
    
    with agents_cache_lock:
        if agents_cache is not None:
            return agents_cache
            
        logger.info("Loading agents cache...")
        start_time = time.time()
        
        import os
        import json
        
        gss_base_path = os.path.join(os.path.dirname(__file__), "agent_bank/populations/gss_agents")
        cached_agents = []
        
        for root, dirs, files in os.walk(gss_base_path):
            if 'scratch.json' in files:
                try:
                    scratch_path = os.path.join(root, 'scratch.json')
                    with open(scratch_path, 'r') as f:
                        agent_data = json.load(f)
                        cached_agents.append(agent_data)
                except Exception as e:
                    logger.error(f"Error loading agent {root}: {str(e)}")
                    continue
        
        agents_cache = cached_agents
        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(cached_agents)} agents into cache in {elapsed:.2f}s")
        
        return agents_cache

@app.post("/filter-count")
async def get_filter_count(filters: AgentFilters):
    """Get count of agents matching the given filters (optimized with Set lookups)."""
    try:
        # Load cache if not already loaded
        all_agents = load_agents_cache()
        total_agents = len(all_agents)
        
        # Convert filters to dict and remove None/empty values
        filters_dict = filters.dict(exclude_none=True)
        filters_dict = {k: v for k, v in filters_dict.items() if v and (not isinstance(v, list) or len(v) > 0)}
        
        # If no filters, return total count immediately
        if not filters_dict:
            return {
                "total_agents": total_agents,
                "matching_agents": total_agents,
                "filters_applied": {}
            }
        
        # Convert filter lists to sets for O(1) lookups instead of O(n)
        filter_sets = {}
        for key, value in filters_dict.items():
            if isinstance(value, list):
                filter_sets[key] = set(value)
            else:
                filter_sets[key] = value
        
        # Apply filters (optimized with Set lookups)
        matching_count = 0
        for agent_data in all_agents:
            # Use early exit pattern for faster filtering
            if 'states' in filter_sets:
                if agent_data.get('state') not in filter_sets['states']:
                    continue
            
            if 'cities' in filter_sets:
                if agent_data.get('city') not in filter_sets['cities']:
                    continue
            
            if 'age_min' in filter_sets:
                if agent_data.get('age', 0) < filter_sets['age_min']:
                    continue
            
            if 'age_max' in filter_sets:
                if agent_data.get('age', 999) > filter_sets['age_max']:
                    continue
            
            if 'sex' in filter_sets:
                if agent_data.get('sex') not in filter_sets['sex']:
                    continue
            
            if 'race' in filter_sets:
                if agent_data.get('race') not in filter_sets['race']:
                    continue
            
            if 'political_views' in filter_sets:
                if agent_data.get('political_views') not in filter_sets['political_views']:
                    continue
            
            if 'party_identification' in filter_sets:
                if agent_data.get('party_identification') not in filter_sets['party_identification']:
                    continue
            
            if 'religion' in filter_sets:
                if agent_data.get('religion') not in filter_sets['religion']:
                    continue
            
            if 'work_status' in filter_sets:
                if agent_data.get('work_status') not in filter_sets['work_status']:
                    continue
            
            if 'marital_status' in filter_sets:
                if agent_data.get('marital_status') not in filter_sets['marital_status']:
                    continue
            
            if 'education' in filter_sets:
                if agent_data.get('highest_degree_received') not in filter_sets['education']:
                    continue
            
            # If we reach here, all filters passed
            matching_count += 1
        
        return {
            "total_agents": total_agents,
            "matching_agents": matching_count,
            "filters_applied": filters_dict
        }
        
    except Exception as e:
        logger.error(f"Error counting filtered agents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to count agents: {str(e)}")

@app.post("/simulate", response_model=SimulationResponse)
async def simulate(request: SimulationRequest):
    """Run a simulation with auto-optimized settings."""
    start_time = time.time()

    from genagents_simulation.simulation_engine.llm_provider import clear_request_usage, get_request_usage
    from genagents_simulation.simulation_engine.llm_pricing import compute_request_cost

    clear_request_usage()

    try:
        filters_dict = None
        if request.filters:
            filters_dict = request.filters.dict(exclude_none=True)
            filters_dict = {k: v for k, v in filters_dict.items() if v is not None and (not isinstance(v, list) or len(v) > 0)}
            if not filters_dict:
                filters_dict = None

        logger.info(f"Simulation: {request.agent_count} agents, model={request.llm_config_name}, memory={request.use_memory}")

        func_input_data = {
            request.question: request.options
        }

        result = run_simulation(
            llm_config_name=request.llm_config_name,
            agent_count=request.agent_count,
            func_name="func",
            func_input_data=func_input_data,
            max_workers=request.max_workers,
            filters=filters_dict,
            use_memory=request.use_memory
        )

        execution_time = time.time() - start_time

        successful = sum(1 for r in result["individual_responses"] if r.get("responses"))
        failed = request.agent_count - successful

        result["execution_time_seconds"] = round(execution_time, 2)
        result["successful_responses"] = successful
        result["failed_responses"] = failed
        result["errors"] = []

        usage_list = get_request_usage()
        if usage_list:
            total_input = sum(u[1] for u in usage_list)
            total_output = sum(u[2] for u in usage_list)
            by_model: Dict[str, Dict[str, int]] = {}
            cost_by_model: Dict[str, float] = {}
            
            for model, inp, out in usage_list:
                if model not in by_model:
                    by_model[model] = {"input_tokens": 0, "output_tokens": 0}
                by_model[model]["input_tokens"] += inp
                by_model[model]["output_tokens"] += out
            
            # Calculate cost per model
            from genagents_simulation.simulation_engine.llm_pricing import get_model_pricing
            for model, tokens in by_model.items():
                in_per_1k, out_per_1k = get_model_pricing(model)
                model_cost = (tokens["input_tokens"] / 1000.0) * in_per_1k + (tokens["output_tokens"] / 1000.0) * out_per_1k
                cost_by_model[model] = round(model_cost, 6)
            
            result["token_usage"] = {
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "by_model": by_model,
            }
            
            total_cost = compute_request_cost(usage_list)
            result["total_cost_usd"] = total_cost
            
            # Add detailed cost breakdown
            result["cost_breakdown"] = {
                "total_usd": total_cost,
                "by_model": cost_by_model,
                "cost_per_agent": round(total_cost / request.agent_count, 6) if request.agent_count > 0 else 0,
                "estimated_cost_per_1000_agents": round((total_cost / request.agent_count) * 1000, 2) if request.agent_count > 0 else 0
            }
            
            logger.info(f"Request cost: ${result['total_cost_usd']:.6f} (input: {total_input}, output: {total_output} tokens)")
        else:
            result["token_usage"] = None
            result["total_cost_usd"] = None
            result["cost_breakdown"] = None

        # Add filter metrics if filters were used
        if "filter_stats" in result:
            logger.info(f"Filter stats: {result['filter_stats']}")

        # Track metrics in database
        try:
            metrics_data = {
                'question': request.question,
                'agent_count': request.agent_count,
                'llm_model': request.llm_config_name,
                'use_memory': request.use_memory if request.use_memory is not None else False,
                'agents_per_batch': result.get('optimization_settings', {}).get('agents_per_api_call'),
                'max_memories': result.get('optimization_settings', {}).get('max_memories_per_agent'),
                'execution_time_seconds': execution_time,
                'successful_responses': successful,
                'failed_responses': failed,
                'total_input_tokens': result.get('token_usage', {}).get('total_input_tokens', 0) if result.get('token_usage') else 0,
                'total_output_tokens': result.get('token_usage', {}).get('total_output_tokens', 0) if result.get('token_usage') else 0,
                'total_cost_usd': result.get('total_cost_usd', 0),
                'cost_per_agent': result.get('cost_breakdown', {}).get('cost_per_agent', 0) if result.get('cost_breakdown') else 0,
                'filters': filters_dict,
                'models_used': {}
            }
            
            # Add per-model breakdown
            if result.get('token_usage') and result['token_usage'].get('by_model'):
                from genagents_simulation.simulation_engine.llm_pricing import get_model_pricing
                for model, tokens in result['token_usage']['by_model'].items():
                    in_per_1k, out_per_1k = get_model_pricing(model)
                    model_cost = (tokens['input_tokens'] / 1000.0) * in_per_1k + (tokens['output_tokens'] / 1000.0) * out_per_1k
                    metrics_data['models_used'][model] = {
                        'input_tokens': tokens['input_tokens'],
                        'output_tokens': tokens['output_tokens'],
                        'cost': model_cost
                    }
            
            metrics_db.track_request(metrics_data)
        except Exception as e:
            logger.error(f"Failed to track metrics: {str(e)}")

        logger.info(f"Simulation completed in {execution_time:.2f}s - Success: {successful}/{request.agent_count}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@app.get("/models/{config_name}")
async def get_model(config_name: str):
    """Get details about a specific model."""
    import json
    try:
        with open("genagents_simulation/configs/llm_configs.json", "r") as f:
            configs = json.load(f)
            for config in configs:
                if config.get("config_name") == config_name:
                    return config
            raise HTTPException(status_code=404, detail=f"Model '{config_name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/metrics")
async def get_provider_metrics():
    """Get real-time performance metrics for all LLM providers."""
    try:
        from genagents_simulation.simulation_engine.llm_provider import get_metrics
        metrics = get_metrics()
        return {
            "providers": metrics,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")


@app.get("/admin/metrics")
async def get_admin_metrics(days: int = 30):
    """
    Admin dashboard: Get comprehensive usage and cost metrics.
    
    - **days**: Number of days to include in metrics (default: 30)
    
    Returns detailed breakdown of:
    - Total requests, agents, and costs
    - Model usage split (which models are being used)
    - Daily trends
    - Recent requests
    - Cost per agent statistics
    """
    try:
        metrics = metrics_db.get_metrics_summary(days=days)
        
        # Calculate additional insights
        if metrics.get('overall'):
            overall = metrics['overall']
            if overall.get('total_agents') and overall.get('total_agents') > 0:
                overall['avg_cost_per_agent'] = round(overall.get('total_cost', 0) / overall['total_agents'], 6)
            if overall.get('total_requests') and overall.get('total_requests') > 0:
                overall['avg_cost_per_request'] = round(overall.get('total_cost', 0) / overall['total_requests'], 6)
                overall['avg_agents_per_request'] = round(overall['total_agents'] / overall['total_requests'], 1)
        
        # Calculate model split percentages
        if metrics.get('by_model'):
            total_cost = sum(m.get('cost', 0) or 0 for m in metrics['by_model'])
            total_tokens = sum((m.get('input_tokens', 0) or 0) + (m.get('output_tokens', 0) or 0) for m in metrics['by_model'])
            
            for model in metrics['by_model']:
                model_cost = model.get('cost', 0) or 0
                model_tokens = (model.get('input_tokens', 0) or 0) + (model.get('output_tokens', 0) or 0)
                
                model['cost_percentage'] = round((model_cost / total_cost * 100), 2) if total_cost > 0 else 0
                model['token_percentage'] = round((model_tokens / total_tokens * 100), 2) if total_tokens > 0 else 0
                model['avg_cost_per_request'] = round(model_cost / model.get('request_count', 1), 6)
        
        return {
            **metrics,
            "summary": {
                "period": f"Last {days} days",
                "database_connected": metrics_db.connection_string is not None
            }
        }
    except Exception as e:
        logger.error(f"Failed to get admin metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get admin metrics: {str(e)}")


@app.get("/admin")
async def admin_dashboard_html():
    """Admin dashboard HTML page."""
    template_path = os.path.join(os.path.dirname(__file__), "templates/admin.html")
    if os.path.exists(template_path):
        return FileResponse(template_path)
    return HTMLResponse("<h1>Admin template not found</h1>", status_code=404)


@app.get("/admin/dashboard")
async def get_admin_dashboard():
    """
    Admin dashboard API: Get simplified dashboard data.
    
    Returns:
    - Key statistics (today, this week, this month)
    - Model usage breakdown
    - Cost trends
    - Top models by usage
    """
    try:
        # Get metrics for different periods
        today = metrics_db.get_metrics_summary(days=1)
        week = metrics_db.get_metrics_summary(days=7)
        month = metrics_db.get_metrics_summary(days=30)
        
        # Get provider metrics (real-time)
        from genagents_simulation.simulation_engine.llm_provider import get_metrics as get_provider_metrics
        provider_metrics = get_provider_metrics()
        
        return {
            "today": {
                "requests": today.get('overall', {}).get('total_requests', 0) or 0,
                "agents": today.get('overall', {}).get('total_agents', 0) or 0,
                "cost": round(today.get('overall', {}).get('total_cost', 0) or 0, 4)
            },
            "this_week": {
                "requests": week.get('overall', {}).get('total_requests', 0) or 0,
                "agents": week.get('overall', {}).get('total_agents', 0) or 0,
                "cost": round(week.get('overall', {}).get('total_cost', 0) or 0, 4)
            },
            "this_month": {
                "requests": month.get('overall', {}).get('total_requests', 0) or 0,
                "agents": month.get('overall', {}).get('total_agents', 0) or 0,
                "cost": round(month.get('overall', {}).get('total_cost', 0) or 0, 4)
            },
            "model_split": month.get('by_model', []),
            "provider_status": provider_metrics,
            "recent_requests": month.get('recent_requests', [])[:5]
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")

@app.post("/simulate/stream")
async def simulate_stream(request: SimulationRequest):
    """
    Run a simulation with streaming progress updates.
    Returns Server-Sent Events (SSE) stream with real-time progress.
    Includes a simulation_id that can be used to cancel the simulation.
    """
    import queue
    
    simulation_id = str(uuid.uuid4())
    cancel_event = threading.Event()
    progress_queue = queue.Queue()
    result_container = {'result': None, 'error': None, 'cancelled': False}
    
    with simulations_lock:
        running_simulations[simulation_id] = {
            'cancel_event': cancel_event,
            'question': request.question,
            'agent_count': request.agent_count,
            'start_time': time.time(),
            'status': 'running'
        }
    
    def progress_callback(completed, total, agent_response):
        if cancel_event.is_set():
            return
        progress_queue.put({
            'type': 'progress',
            'completed': completed,
            'total': total,
            'percentage': round((completed / total) * 100, 1),
            'agent_response': agent_response
        })
    
    def run_simulation_thread():
        from genagents_simulation.simulation_engine.llm_provider import (
            clear_request_usage,
            get_request_usage,
        )
        from genagents_simulation.simulation_engine.llm_pricing import compute_request_cost

        clear_request_usage()
        try:
            start_time = time.time()
            func_input_data = {
                request.question: request.options
            }

            # Handle filters properly
            filters_dict = None
            if request.filters:
                filters_dict = request.filters.dict(exclude_none=True)
                # Remove empty lists
                filters_dict = {k: v for k, v in filters_dict.items() if v is not None and (not isinstance(v, list) or len(v) > 0)}
                if not filters_dict:
                    filters_dict = None

            logger.info(f"Stream simulation with filters: {filters_dict}")

            result = run_simulation(
                llm_config_name=request.llm_config_name,
                agent_count=request.agent_count,
                func_name="func",
                func_input_data=func_input_data,
                max_workers=request.max_workers,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                filters=filters_dict,
                use_memory=request.use_memory
            )

            if cancel_event.is_set():
                result_container['cancelled'] = True
                progress_queue.put({'type': 'cancelled'})
                return

            execution_time = time.time() - start_time
            successful = sum(1 for r in result["individual_responses"] if r.get("responses"))
            failed = request.agent_count - successful

            result["execution_time_seconds"] = round(execution_time, 2)
            result["successful_responses"] = successful
            result["failed_responses"] = failed
            result["errors"] = []

            for i, resp in enumerate(result["individual_responses"]):
                if not resp.get("responses"):
                    error_msg = resp.get("error", "No error message provided")
                    result["errors"].append(f"Agent {i+1}: {error_msg}")

            usage_list = get_request_usage()
            if usage_list:
                total_input = sum(u[1] for u in usage_list)
                total_output = sum(u[2] for u in usage_list)
                by_model = {}
                cost_by_model = {}
                
                for model, inp, out in usage_list:
                    if model not in by_model:
                        by_model[model] = {"input_tokens": 0, "output_tokens": 0}
                    by_model[model]["input_tokens"] += inp
                    by_model[model]["output_tokens"] += out
                
                # Calculate cost per model
                from genagents_simulation.simulation_engine.llm_pricing import get_model_pricing
                for model, tokens in by_model.items():
                    in_per_1k, out_per_1k = get_model_pricing(model)
                    model_cost = (tokens["input_tokens"] / 1000.0) * in_per_1k + (tokens["output_tokens"] / 1000.0) * out_per_1k
                    cost_by_model[model] = round(model_cost, 6)
                
                result["token_usage"] = {
                    "total_input_tokens": total_input,
                    "total_output_tokens": total_output,
                    "by_model": by_model,
                }
                
                total_cost = compute_request_cost(usage_list)
                result["total_cost_usd"] = total_cost
                
                # Add detailed cost breakdown
                result["cost_breakdown"] = {
                    "total_usd": total_cost,
                    "by_model": cost_by_model,
                    "cost_per_agent": round(total_cost / request.agent_count, 6) if request.agent_count > 0 else 0,
                    "estimated_cost_per_1000_agents": round((total_cost / request.agent_count) * 1000, 2) if request.agent_count > 0 else 0
                }
            else:
                result["token_usage"] = None
                result["total_cost_usd"] = None
                result["cost_breakdown"] = None

            result_container['result'] = result
            progress_queue.put({'type': 'complete'})
        except Exception as e:
            if not cancel_event.is_set():
                result_container['error'] = str(e)
                progress_queue.put({'type': 'error', 'message': str(e)})
        finally:
            with simulations_lock:
                if simulation_id in running_simulations:
                    running_simulations[simulation_id]['status'] = 'completed'
                    del running_simulations[simulation_id]
    
    def generate():
        yield f"data: {json_lib.dumps({'type': 'start', 'simulation_id': simulation_id, 'total': request.agent_count, 'question': request.question})}\n\n"
        
        thread = threading.Thread(target=run_simulation_thread)
        thread.daemon = True
        thread.start()
        
        try:
            while True:
                try:
                    event = progress_queue.get(timeout=0.5)
                    
                    if cancel_event.is_set() or event['type'] == 'cancelled':
                        yield f"data: {json_lib.dumps({'type': 'cancelled', 'message': 'Simulation cancelled'})}\n\n"
                        break
                    elif event['type'] == 'complete':
                        yield f"data: {json_lib.dumps({'type': 'complete', 'result': result_container['result']})}\n\n"
                        break
                    elif event['type'] == 'error':
                        yield f"data: {json_lib.dumps({'type': 'error', 'message': event.get('message', 'Unknown error')})}\n\n"
                        break
                    elif event['type'] == 'progress':
                        yield f"data: {json_lib.dumps(event)}\n\n"
                        progress_queue.task_done()
                        
                except queue.Empty:
                    if cancel_event.is_set():
                        yield f"data: {json_lib.dumps({'type': 'cancelled', 'message': 'Simulation cancelled'})}\n\n"
                        break
                    if not thread.is_alive():
                        if result_container['result']:
                            yield f"data: {json_lib.dumps({'type': 'complete', 'result': result_container['result']})}\n\n"
                        elif result_container['error']:
                            yield f"data: {json_lib.dumps({'type': 'error', 'message': result_container['error']})}\n\n"
                        break
                    continue
        finally:
            with simulations_lock:
                if simulation_id in running_simulations:
                    running_simulations[simulation_id]['status'] = 'stopped'
    
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/simulate/stream/{simulation_id}/cancel")
async def cancel_simulation(simulation_id: str):
    """
    Cancel a running simulation by its ID.
    Returns success if the simulation was found and cancelled.
    """
    with simulations_lock:
        if simulation_id not in running_simulations:
            raise HTTPException(status_code=404, detail=f"Simulation {simulation_id} not found or already completed")
        
        sim_info = running_simulations[simulation_id]
        if sim_info['status'] != 'running':
            raise HTTPException(status_code=400, detail=f"Simulation {simulation_id} is not running (status: {sim_info['status']})")
        
        sim_info['cancel_event'].set()
        sim_info['status'] = 'cancelling'
        logger.info(f"Cancelling simulation {simulation_id}")
        
        return {
            "status": "cancelled",
            "simulation_id": simulation_id,
            "message": "Simulation cancellation requested"
        }

@app.get("/simulate/stream/status")
async def list_running_simulations():
    """
    List all currently running simulations.
    """
    with simulations_lock:
        simulations = []
        for sim_id, sim_info in running_simulations.items():
            elapsed = time.time() - sim_info['start_time']
            simulations.append({
                "simulation_id": sim_id,
                "question": sim_info['question'],
                "agent_count": sim_info['agent_count'],
                "status": sim_info['status'],
                "elapsed_seconds": round(elapsed, 2)
            })
        return {
            "running_count": len(simulations),
            "simulations": simulations
        }

@app.on_event("startup")
async def startup_event():
    """Preload caches on startup for instant responses."""
    logger.info("Preloading caches on startup...")
    try:
        # Load agents cache in background
        import threading
        def preload():
            load_agents_cache()
            compute_filter_options()
            logger.info("Caches preloaded successfully")
        
        thread = threading.Thread(target=preload)
        thread.daemon = True
        thread.start()
    except Exception as e:
        logger.error(f"Error preloading caches: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    # Use multiple workers for better performance
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,  # Keep at 1 for shared cache
        log_level="info",
        access_log=False  # Disable access log for better performance
    )
