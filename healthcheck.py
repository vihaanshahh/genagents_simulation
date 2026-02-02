#!/usr/bin/env python3
"""
Health check script for GenAgents Docker container.
"""
import sys
import os

def check_health():
    """Perform health checks."""
    checks = []
    
    # Check 1: Environment variables
    token = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if token:
        checks.append(("AWS Token", True, "Token is set"))
    else:
        checks.append(("AWS Token", False, "Token not found"))
    
    # Check 2: Agent bank exists
    agent_bank_path = "/app/genagents_simulation/agent_bank/populations"
    if os.path.exists(agent_bank_path):
        checks.append(("Agent Bank", True, f"Found at {agent_bank_path}"))
    else:
        checks.append(("Agent Bank", False, f"Not found at {agent_bank_path}"))
    
    # Check 3: Configs exist
    config_path = "/app/genagents_simulation/configs/llm_configs.json"
    if os.path.exists(config_path):
        checks.append(("LLM Configs", True, f"Found at {config_path}"))
    else:
        checks.append(("LLM Configs", False, f"Not found at {config_path}"))
    
    # Print results
    print("Health Check Results:")
    print("=" * 60)
    all_passed = True
    for name, passed, message in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} | {name:15} | {message}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("Status: HEALTHY")
        return 0
    else:
        print("Status: UNHEALTHY")
        return 1

if __name__ == "__main__":
    sys.exit(check_health())
