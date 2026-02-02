"""
Enterprise-grade LLM provider with intelligent load balancing and racing.

This module provides a unified interface for calling multiple LLM providers
(OpenAI, Cerebras, AWS Bedrock) with automatic load balancing, racing, and
failover capabilities for millisecond-fast responses.
"""
import os
import time
import asyncio
import logging
import threading
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import httpx

logger = logging.getLogger(__name__)

# Thread-local storage for token usage per API request (for cost calculation)
_request_usage = threading.local()

# Global rate limiting for Cerebras (shared across all models)
# Free tier: 60k tokens/min input, shared across all Cerebras models
_cerebras_global_lock = asyncio.Lock()
_cerebras_token_window: List[Tuple[float, int]] = []  # (timestamp, token_count)
CEREBRAS_MAX_TOKENS_PER_MINUTE = 60000


def _get_usage_list() -> list:
    if not hasattr(_request_usage, "usage_list"):
        _request_usage.usage_list = []
    return _request_usage.usage_list


def clear_request_usage() -> None:
    """Clear token usage for the current request. Call at start of /simulate."""
    _get_usage_list().clear()


def add_request_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """Record token usage for the current request (used for cost per API request)."""
    _get_usage_list().append((model, input_tokens, output_tokens))


def get_request_usage() -> List[Tuple[str, int, int]]:
    """Return list of (model, input_tokens, output_tokens) for the current request."""
    return list(_get_usage_list())


class ProviderType(Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    CEREBRAS = "cerebras"
    BEDROCK = "bedrock"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    provider_type: ProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    region: Optional[str] = None
    enabled: bool = True
    priority: int = 1
    timeout: float = 30.0
    max_requests_per_minute: int = 30  # Rate limit per minute


@dataclass
class ProviderMetrics:
    """Performance metrics for a provider."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    last_error: Optional[str] = None
    last_success_time: float = 0.0
    daily_tokens_used: int = 0
    daily_tokens_limit: int = 1000000
    last_reset_date: str = ""
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency in milliseconds."""
        if self.successful_requests == 0:
            return float('inf')
        return (self.total_latency / self.successful_requests) * 1000
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def tokens_remaining(self) -> int:
        """Calculate remaining tokens for the day."""
        return max(0, self.daily_tokens_limit - self.daily_tokens_used)
    
    @property
    def token_usage_percentage(self) -> float:
        """Calculate percentage of daily tokens used."""
        if self.daily_tokens_limit == 0:
            return 0.0
        return (self.daily_tokens_used / self.daily_tokens_limit) * 100


async def _check_cerebras_global_token_limit(estimated_input_tokens: int = 1000) -> None:
    """
    Check and enforce global Cerebras token rate limit (60k tokens/min shared across all models).
    This prevents hitting the shared input token limit when using multiple models concurrently.
    """
    global _cerebras_token_window
    
    async with _cerebras_global_lock:
        current_time = time.time()
        
        # Remove entries older than 1 minute
        _cerebras_token_window = [
            (ts, tokens) for ts, tokens in _cerebras_token_window 
            if current_time - ts < 60
        ]
        
        # Calculate current token usage in the window
        tokens_used = sum(tokens for _, tokens in _cerebras_token_window)
        
        # If adding this request would exceed limit, wait
        if tokens_used + estimated_input_tokens > CEREBRAS_MAX_TOKENS_PER_MINUTE:
            if _cerebras_token_window:
                oldest_time = _cerebras_token_window[0][0]
                wait_time = 60 - (current_time - oldest_time)
                if wait_time > 0:
                    logger.info(
                        f"Cerebras global token limit ({tokens_used}/{CEREBRAS_MAX_TOKENS_PER_MINUTE} tokens/min), "
                        f"waiting {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)
                    # Clean up after waiting
                    current_time = time.time()
                    _cerebras_token_window = [
                        (ts, tokens) for ts, tokens in _cerebras_token_window 
                        if current_time - ts < 60
                    ]
        
        # Record this request's estimated tokens
        _cerebras_token_window.append((current_time, estimated_input_tokens))


class ProviderClient:
    """Base class for LLM provider clients."""
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.metrics = ProviderMetrics()
        self._client = None
        self._async_client = None
        self._request_times: List[float] = []  # Track request timestamps
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the provider client."""
        raise NotImplementedError
    
    def _reset_daily_tokens_if_needed(self):
        """Reset daily token counter if it's a new day."""
        from datetime import datetime
        today = datetime.now().strftime('%Y-%m-%d')
        
        if self.metrics.last_reset_date != today:
            self.metrics.daily_tokens_used = 0
            self.metrics.last_reset_date = today
            logger.info(f"{self.config.provider_type.value} daily token counter reset")
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        async with self._lock:
            # Reset daily tokens if needed
            self._reset_daily_tokens_if_needed()
            
            # Check daily token limit (for free tier Cerebras)
            if self.config.provider_type == ProviderType.CEREBRAS:
                if self.metrics.daily_tokens_used >= self.metrics.daily_tokens_limit:
                    raise RuntimeError(
                        f"{self.config.provider_type.value} daily token limit reached "
                        f"({self.metrics.daily_tokens_used}/{self.metrics.daily_tokens_limit})"
                    )
            
            current_time = time.time()
            # Remove requests older than 1 minute
            self._request_times = [t for t in self._request_times if current_time - t < 60]
            
            # If at limit, wait
            if len(self._request_times) >= self.config.max_requests_per_minute:
                oldest_request = self._request_times[0]
                wait_time = 60 - (current_time - oldest_request)
                if wait_time > 0:
                    logger.info(f"{self.config.provider_type.value} rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    # Clean up again after waiting
                    current_time = time.time()
                    self._request_times = [t for t in self._request_times if current_time - t < 60]
            
            # Record this request
            self._request_times.append(current_time)
    
    async def invoke(self, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, Optional[Dict[str, int]]]:
        """Invoke the LLM with the given prompt. Returns (content, usage_dict or None)."""
        raise NotImplementedError
    
    async def cleanup(self):
        """Clean up resources."""
        if self._async_client:
            await self._async_client.aclose()


class OpenAIClient(ProviderClient):
    """OpenAI provider client."""
    
    async def initialize(self):
        """Initialize OpenAI client."""
        from openai import AsyncOpenAI
        
        self._async_client = AsyncOpenAI(
            api_key=self.config.api_key or os.getenv('OPENAI_API_KEY'),
            base_url=self.config.base_url,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=1000, max_connections=1000),
                timeout=httpx.Timeout(self.config.timeout, connect=5.0)
            )
        )
    
    async def invoke(self, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, Optional[Dict[str, int]]]:
        """Invoke OpenAI API. Returns (content, usage_dict)."""
        await self._check_rate_limit()
        
        # Build params based on model requirements
        params = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # gpt-5-nano and o1/o3 models: use max_completion_tokens and default temperature
        if "gpt-5" in self.config.model or "o1" in self.config.model or "o3" in self.config.model:
            params["max_completion_tokens"] = max_tokens
            # These models only support temperature=1 (default), so don't include it
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = temperature
        
        response = await self._async_client.chat.completions.create(**params)
        content = response.choices[0].message.content
        usage = None
        if getattr(response, "usage", None) is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
            }
        return content, usage


class CerebrasClient(ProviderClient):
    """Cerebras provider client."""
    
    async def initialize(self):
        """Initialize Cerebras client."""
        try:
            from cerebras.cloud.sdk import Cerebras
            self._client = Cerebras(
                api_key=self.config.api_key or os.getenv('CEREBRAS_API_KEY') or os.getenv('CEREBRAS_API')
            )
        except ImportError:
            logger.warning("cerebras-cloud-sdk not installed")
            self.config.enabled = False
    
    async def invoke(self, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, Optional[Dict[str, int]]]:
        """Invoke Cerebras API (sync client wrapped in async). Returns (content, usage_dict or estimate)."""
        # Estimate input tokens (4 chars per token)
        estimated_input = max(1, len(prompt) // 4)
        
        # Check global Cerebras token limit (shared across all models)
        await _check_cerebras_global_token_limit(estimated_input)
        
        # Check per-model rate limit
        await self._check_rate_limit()
        
        loop = asyncio.get_event_loop()
        
        def _call():
            response = self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            content = ""
            if hasattr(response, "choices") and len(response.choices) > 0:
                if hasattr(response.choices[0], "message"):
                    content = response.choices[0].message.content or ""
                elif hasattr(response.choices[0], "text"):
                    content = response.choices[0].text or ""
            if not content:
                content = str(response)
            usage = None
            if getattr(response, "usage", None) is not None:
                u = response.usage
                usage = {
                    "input_tokens": getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0,
                }
            if not usage or (usage["input_tokens"] == 0 and usage["output_tokens"] == 0):
                # Estimate: ~4 chars per token
                usage = {
                    "input_tokens": max(1, len(prompt) // 4),
                    "output_tokens": max(1, len(content) // 4),
                }
            return content, usage
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, _call)


class BedrockClient(ProviderClient):
    """AWS Bedrock provider client."""
    
    async def initialize(self):
        """Initialize Bedrock client."""
        import boto3
        from botocore.config import Config
        
        config = Config(
            max_pool_connections=200,
            retries={'max_attempts': 1, 'mode': 'standard'},
            connect_timeout=3,
            read_timeout=20,
            tcp_keepalive=True
        )
        
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if aws_access_key and aws_secret_key:
            self._client = boto3.client(
                'bedrock-runtime',
                region_name=self.config.region or 'us-east-1',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                config=config
            )
        else:
            self._client = boto3.client(
                'bedrock-runtime',
                region_name=self.config.region or 'us-east-1',
                config=config
            )
    
    async def invoke(self, prompt: str, max_tokens: int, temperature: float) -> Tuple[str, Optional[Dict[str, int]]]:
        """Invoke Bedrock API (sync client wrapped in async). Returns (content, usage_dict or None)."""
        await self._check_rate_limit()
        
        import json
        
        loop = asyncio.get_event_loop()
        
        def _call():
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            })
            
            response = self._client.invoke_model(modelId=self.config.model, body=body)
            result = json.loads(response["body"].read())
            content = result["content"][0]["text"]
            usage = None
            if "usage" in result:
                u = result["usage"]
                usage = {
                    "input_tokens": u.get("input_tokens", 0) or 0,
                    "output_tokens": u.get("output_tokens", 0) or 0,
                }
            return content, usage
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(executor, _call)


class LoadBalancedLLM:
    """
    Load-balanced LLM client with intelligent request distribution and re-routing.
    
    Distributes requests across providers concurrently for speed and cost efficiency.
    Failed or slow requests are automatically re-routed to backup providers.
    """
    
    def __init__(self):
        self.providers: List[ProviderClient] = []
        self._initialized = False
        self._lock = asyncio.Lock()
        self._round_robin_index = 0
    
    async def initialize(self):
        """Initialize all configured providers."""
        async with self._lock:
            if self._initialized:
                return
            
            configs = self._load_provider_configs()
            
            for config in configs:
                if not config.enabled:
                    continue
                
                try:
                    if config.provider_type == ProviderType.OPENAI:
                        client = OpenAIClient(config)
                    elif config.provider_type == ProviderType.CEREBRAS:
                        client = CerebrasClient(config)
                    elif config.provider_type == ProviderType.BEDROCK:
                        client = BedrockClient(config)
                    else:
                        logger.warning(f"Unknown provider type: {config.provider_type}")
                        continue
                    
                    await client.initialize()
                    if client.config.enabled:
                        self.providers.append(client)
                        logger.info(f"Initialized provider: {config.provider_type.value} ({config.model})")
                except Exception as e:
                    logger.warning(f"Failed to initialize {config.provider_type.value}: {str(e)[:100]}")
            
            if not self.providers:
                raise RuntimeError("No LLM providers available")
            
            self._initialized = True
            logger.info(f"Load balancer initialized with {len(self.providers)} provider(s)")
    
    def _load_provider_configs(self) -> List[ProviderConfig]:
        """
        Load provider configurations from environment.
        Priority ordering: higher number = try first (fast/cheap), lower number = fallback (reliable/expensive).
        
        Cerebras models prioritized by cost and speed:
        - llama3.1-8b: Cheapest ($0.10/$0.10), fastest (2200 tokens/sec)
        - gpt-oss-120b: Fast (3000 tokens/sec), cheap ($0.35/$0.75)
        - qwen-3-235b: Balanced (1400 tokens/sec, $0.60/$1.20)
        - zai-glm-4.7: Premium reasoning (1000 tokens/sec, $2.25/$2.75)
        """
        configs = []

        cerebras_key = os.getenv('CEREBRAS_API_KEY') or os.getenv('CEREBRAS_API')
        if cerebras_key:
            # Define all Cerebras models (enabled/disabled via env)
            cerebras_models = {
                'llama3.1-8b': {
                    'model': 'llama3.1-8b',
                    'priority': 10,
                    'timeout': 20.0,
                    'max_requests_per_minute': 30,
                    'enabled': os.getenv('CEREBRAS_LLAMA31_ENABLED', 'true').lower() == 'true'
                },
                'gpt-oss-120b': {
                    'model': 'gpt-oss-120b',
                    'priority': 9,
                    'timeout': 20.0,
                    'max_requests_per_minute': 30,
                    'enabled': os.getenv('CEREBRAS_GPT_OSS_ENABLED', 'true').lower() == 'true'
                },
                'qwen-3-235b': {
                    'model': 'qwen-3-235b-a22b-instruct-2507',
                    'priority': 8,
                    'timeout': 25.0,
                    'max_requests_per_minute': 30,
                    'enabled': os.getenv('CEREBRAS_QWEN_ENABLED', 'true').lower() == 'true'
                },
                'zai-glm-4.7': {
                    'model': 'zai-glm-4.7',
                    'priority': 7,
                    'timeout': 30.0,
                    'max_requests_per_minute': 10,
                    'enabled': os.getenv('CEREBRAS_GLM_ENABLED', 'false').lower() == 'true'
                }
            }
            
            # Add enabled Cerebras models
            for name, config in cerebras_models.items():
                if config['enabled']:
                    configs.append(ProviderConfig(
                        provider_type=ProviderType.CEREBRAS,
                        model=config['model'],
                        api_key=cerebras_key,
                        priority=config['priority'],
                        timeout=config['timeout'],
                        max_requests_per_minute=config['max_requests_per_minute']
                    ))

        # Bedrock: reliable fallback (priority 2)
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        if aws_key and aws_secret:
            configs.append(ProviderConfig(
                provider_type=ProviderType.BEDROCK,
                model=os.getenv('BEDROCK_MODEL', 'anthropic.claude-haiku-4-5-20251001-v1:0'),
                region=os.getenv('AWS_REGION', 'us-east-1'),
                priority=2,
                timeout=60.0,
                max_requests_per_minute=60
            ))

        # OpenAI: most reliable, use as last resort (priority 1)
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            configs.append(ProviderConfig(
                provider_type=ProviderType.OPENAI,
                model=os.getenv('OPENAI_MODEL', 'gpt-5-nano'),
                api_key=openai_key,
                priority=1,
                timeout=90.0,
                max_requests_per_minute=60
            ))

        return configs
    
    async def _invoke_provider(self, provider: ProviderClient, prompt: str,
                               max_tokens: int, temperature: float) -> Tuple[str, float]:
        """Invoke a single provider and track metrics."""
        start_time = time.time()
        provider.metrics.total_requests += 1

        try:
            content, usage = await provider.invoke(prompt, max_tokens, temperature)
            latency = time.time() - start_time

            provider.metrics.successful_requests += 1
            provider.metrics.total_latency += latency
            provider.metrics.last_success_time = time.time()

            if usage:
                input_tokens = usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("output_tokens", 0) or 0
                
                add_request_usage(
                    provider.config.model,
                    input_tokens,
                    output_tokens,
                )
                
                # Track daily token usage for Cerebras free tier
                if provider.config.provider_type == ProviderType.CEREBRAS:
                    provider.metrics.daily_tokens_used += (input_tokens + output_tokens)

            return content, latency
        except Exception as e:
            provider.metrics.failed_requests += 1
            provider.metrics.last_error = str(e)[:200]
            raise
    
    def _select_next_provider(self, exclude_indices: Optional[List[int]] = None) -> Optional[Tuple[ProviderClient, int]]:
        """
        Select next provider for load balancing using round-robin with health-based weighting.
        Returns (provider, index) or None if no providers available.
        """
        if not self.providers:
            return None
            
        exclude_indices = exclude_indices or []
        available_providers = [
            (p, i) for i, p in enumerate(self.providers) 
            if i not in exclude_indices
        ]
        
        if not available_providers:
            return None
        
        # Sort by priority (high to low), then by success rate
        available_providers.sort(
            key=lambda x: (x[0].config.priority, x[0].metrics.success_rate),
            reverse=True
        )
        
        # Use round-robin among top-priority providers
        self._round_robin_index = (self._round_robin_index + 1) % len(available_providers)
        return available_providers[self._round_robin_index]

    async def invoke_single(self, prompt: str, max_tokens: int = 500, 
                           temperature: float = 0.7) -> str:
        """
        Invoke LLM with automatic re-routing on failure.
        Tries providers in priority order, re-routing on timeout or failure.
        """
        if not self._initialized:
            await self.initialize()

        if not self.providers:
            raise RuntimeError("No providers available")

        tried_indices: List[int] = []
        exceptions: List[Exception] = []
        
        while len(tried_indices) < len(self.providers):
            provider_tuple = self._select_next_provider(exclude_indices=tried_indices)
            if not provider_tuple:
                break
                
            provider, idx = provider_tuple
            tried_indices.append(idx)
            
            try:
                result, latency = await asyncio.wait_for(
                    self._invoke_provider(provider, prompt, max_tokens, temperature),
                    timeout=provider.config.timeout
                )
                logger.debug(f"Response from {provider.config.provider_type.value} in {latency*1000:.1f}ms")
                return result
                
            except asyncio.TimeoutError:
                provider.metrics.failed_requests += 1
                provider.metrics.last_error = "Request timed out."
                exceptions.append(TimeoutError(f"{provider.config.provider_type.value}: timed out"))
                logger.warning(f"{provider.config.provider_type.value} timed out, re-routing to next provider")
                continue
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    logger.warning(f"{provider.config.provider_type.value} rate limited, re-routing")
                else:
                    logger.warning(f"{provider.config.provider_type.value} failed: {error_msg[:100]}, re-routing")
                exceptions.append(e)
                continue

        error_msg = f"All providers failed. Errors: {[str(e)[:100] for e in exceptions]}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    async def invoke_batch(self, prompts: List[str], max_tokens: int = 500,
                          temperature: float = 0.7) -> List[str]:
        """
        Invoke LLM for multiple prompts with intelligent load balancing.
        
        Distributes requests across all providers concurrently (round-robin).
        Failed/slow requests are automatically re-routed to backup providers.
        This maximizes throughput while minimizing token usage.
        """
        if not self._initialized:
            await self.initialize()
        
        if not self.providers:
            raise RuntimeError("No providers available")
        
        # Distribute prompts across providers using round-robin
        # Each prompt gets assigned to one provider, failed requests get re-routed
        tasks = []
        for prompt in prompts:
            task = self.invoke_single(prompt, max_tokens, temperature)
            tasks.append(task)
        
        # Execute all requests concurrently with automatic re-routing on failure
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                error_str = str(result)
                if "API key" in error_str.lower() or "invalid" in error_str.lower():
                    processed_results.append("ERROR: API key invalid or expired")
                else:
                    processed_results.append(f"ERROR: {error_str[:200]}")
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all providers."""
        metrics = {}
        for provider in self.providers:
            provider_metrics = {
                "model": provider.config.model,
                "total_requests": provider.metrics.total_requests,
                "successful_requests": provider.metrics.successful_requests,
                "failed_requests": provider.metrics.failed_requests,
                "success_rate": f"{provider.metrics.success_rate:.1f}%",
                "average_latency_ms": f"{provider.metrics.average_latency:.1f}",
                "last_error": provider.metrics.last_error
            }
            
            # Add daily token tracking for Cerebras
            if provider.config.provider_type == ProviderType.CEREBRAS:
                provider_metrics.update({
                    "daily_tokens_used": provider.metrics.daily_tokens_used,
                    "daily_tokens_limit": provider.metrics.daily_tokens_limit,
                    "tokens_remaining": provider.metrics.tokens_remaining,
                    "token_usage": f"{provider.metrics.token_usage_percentage:.1f}%"
                })
            
            key = f"{provider.config.provider_type.value}_{provider.config.model}"
            metrics[key] = provider_metrics
            
        return metrics
    
    async def cleanup(self):
        """Clean up all provider resources."""
        for provider in self.providers:
            try:
                await provider.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up provider: {str(e)}")


_global_llm: Optional[LoadBalancedLLM] = None


async def get_llm() -> LoadBalancedLLM:
    """Get or create the global load-balanced LLM instance."""
    global _global_llm
    if _global_llm is None:
        _global_llm = LoadBalancedLLM()
        await _global_llm.initialize()
    return _global_llm


def invoke_model(prompt: str, model: str = "gpt-5-nano", max_tokens: int = 500, 
                temperature: float = 0.7) -> str:
    """
    Synchronous wrapper for invoking LLM with load balancing.
    Races requests across all available providers.
    """
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                _invoke_model_async(prompt, max_tokens, temperature)
            )
            return future.result()
    except RuntimeError:
        return asyncio.run(_invoke_model_async(prompt, max_tokens, temperature))


async def _invoke_model_async(prompt: str, max_tokens: int, temperature: float) -> str:
    """Async implementation of invoke_model."""
    llm = await get_llm()
    return await llm.invoke_single(prompt, max_tokens, temperature)


def invoke_batch(requests: List[Dict], model: str = "gpt-5-nano", max_tokens: int = 500,
                temperature: float = 0.7, batch_size: Optional[int] = None) -> List[str]:
    """
    Synchronous wrapper for batch invocation with load balancing.
    Each request races across all available providers.
    """
    if not requests:
        return []
    
    prompts = [req["prompt"] for req in requests]
    
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                asyncio.run,
                _invoke_batch_async(prompts, max_tokens, temperature)
            )
            return future.result()
    except RuntimeError:
        return asyncio.run(_invoke_batch_async(prompts, max_tokens, temperature))


async def _invoke_batch_async(prompts: List[str], max_tokens: int, temperature: float) -> List[str]:
    """Async implementation of invoke_batch."""
    llm = await get_llm()
    return await llm.invoke_batch(prompts, max_tokens, temperature)


def get_metrics() -> Dict[str, Any]:
    """Get performance metrics for all providers."""
    global _global_llm
    if _global_llm is None:
        return {}
    return _global_llm.get_metrics()
