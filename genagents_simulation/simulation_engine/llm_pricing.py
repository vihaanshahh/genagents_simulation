"""
Pricing per 1K tokens (input, output) for LLM models.
Used to compute total cost per API request from token usage.
"""
from typing import Dict, List, Tuple

# Per 1K tokens: (input_price_usd, output_price_usd); stored as 1M rate / 1000
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-5-nano": (0.00005, 0.0004),
    
    # Cerebras Models (optimized for free tier)
    "qwen-3-235b-a22b-instruct-2507": (0.0006, 0.0012),
    "zai-glm-4.7": (0.00225, 0.00275),
    "gpt-oss-120b": (0.00035, 0.00075),
    "llama3.1-8b": (0.0001, 0.0001),
    
    # Fallback/legacy
    "llama-3.3-70b": (0.00085, 0.0012),
}


def get_model_pricing(model: str) -> Tuple[float, float]:
    """Return (input_per_1k, output_per_1k) for a model. Unknown models return (0, 0)."""
    # Normalize: strip provider prefix for lookup (e.g. openai/gpt-5-nano -> gpt-5-nano)
    key = model.split("/")[-1] if "/" in model else model
    return MODEL_PRICING.get(key, (0.0, 0.0))


def compute_request_cost(usage_list: List[Tuple[str, int, int]]) -> float:
    """
    Compute total cost in USD for a list of (model, input_tokens, output_tokens).
    """
    total = 0.0
    for model, input_tokens, output_tokens in usage_list:
        in_per_1k, out_per_1k = get_model_pricing(model)
        total += (input_tokens / 1000.0) * in_per_1k + (output_tokens / 1000.0) * out_per_1k
    return round(total, 6)
