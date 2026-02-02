# Simulation Engine Knowledge

## Purpose
Provides configuration and utilities for the simulation environment.

## Key Components
- `settings.py`: Core configuration (created from example-settings.py)
- `global_methods.py`: Shared utility functions
- `gpt_structure.py`: AWS Bedrock API interaction
- `llm_json_parser.py`: Response parsing utilities

## Configuration
- Create settings.py from example-settings.py template
- Required settings:
  - AWS_BEARER_TOKEN_BEDROCK
  - KEY_OWNER
  - LLM_VERS (default: "gpt-oss-120b")

## Best Practices
- Use safe_generate for all LLM calls
- Handle API errors gracefully
- Parse JSON responses carefully to handle malformed output
