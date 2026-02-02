import math
import sys
import datetime
import random
import string
import re
import os

from numpy import dot
from numpy.linalg import norm

from genagents_simulation.simulation_engine.settings import * 
from genagents_simulation.simulation_engine.global_methods import *
from genagents_simulation.simulation_engine.gpt_structure import *
from genagents_simulation.simulation_engine.llm_json_parser import *
from genagents_simulation.simulation_engine.llm_provider import invoke_batch


def _calculate_optimal_batch_settings(llm_config, use_memory):
  """
  Auto-calculate optimal batch size and memory count based on:
  - Model context window
  - Token pricing (cost optimization)
  - Rate limits (throughput optimization)
  """
  import logging
  logger = logging.getLogger(__name__)
  
  # Get model config or use defaults
  context_window = llm_config.get('context_window', 8000) if llm_config else 8000
  model_name = llm_config.get('model', 'unknown') if llm_config else 'unknown'
  
  # Estimate tokens per agent description
  base_agent_desc_tokens = 100  # Self description + formatting
  memory_tokens_per_item = 50   # Avg tokens per memory
  
  # Calculate max memories based on cost optimization
  if use_memory:
    # For expensive models (>$1/1M input), use fewer memories
    if 'glm' in model_name.lower():
      max_memories = 10  # Premium model: minimize tokens
    elif 'qwen' in model_name.lower() or 'gpt-oss' in model_name.lower():
      max_memories = 15  # Mid-tier: balanced
    else:
      max_memories = 20  # Cheap model: use more context
  else:
    max_memories = 0  # No memory
  
  # Calculate tokens per agent
  tokens_per_agent = base_agent_desc_tokens + (max_memories * memory_tokens_per_item if use_memory else 0)
  
  # Calculate optimal batch size based on context window
  # Reserve 20% of context for question, options, and response
  usable_context = int(context_window * 0.8)
  max_agents_by_context = usable_context // tokens_per_agent
  
  # Apply rate limit constraints
  # Free tier Cerebras: 60k tokens/min input limit (shared across all models)
  # Conservative: assume we can use 30k tokens per batch to stay under limit
  if 'llama' in model_name.lower() or 'cerebras' in model_name.lower():
    max_tokens_per_batch = 30000  # Conservative for free tier
  else:
    max_tokens_per_batch = 50000  # More headroom for paid tiers
  
  max_agents_by_tokens = max_tokens_per_batch // tokens_per_agent
  
  # Take the minimum of all constraints
  optimal_batch_size = min(
    max_agents_by_context,
    max_agents_by_tokens,
    200  # Hard cap for quality
  )
  
  # Apply minimum batch size
  optimal_batch_size = max(optimal_batch_size, 20)
  
  # Check env var override
  env_batch_size = os.getenv('AGENTS_PER_API_CALL')
  if env_batch_size:
    try:
      optimal_batch_size = int(env_batch_size)
    except ValueError:
      pass
  
  env_max_memories = os.getenv('MAX_MEMORIES_PER_AGENT')
  if env_max_memories:
    try:
      max_memories = int(env_max_memories)
    except ValueError:
      pass
  
  logger.info(
    f"Auto-optimized settings for {model_name}: "
    f"batch_size={optimal_batch_size}, max_memories={max_memories} "
    f"(context_window={context_window}, use_memory={use_memory})"
  )
  
  return optimal_batch_size, max_memories


def _main_agent_desc(agent, anchor, use_memory=True, max_memories=20): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.get_self_description()}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  if use_memory:
    try:
      retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=max_memories)
      if len(retrieved) > 0:
        nodes = list(retrieved.values())[0]
        for node in nodes[:max_memories]:
          agent_desc += f"{node.content}\n"
    except Exception as e:
      import logging
      logger = logging.getLogger(__name__)
      logger.warning(f"Memory retrieval failed for agent: {str(e)[:100]}")
  
  return agent_desc


def _utterance_agent_desc(agent, anchor, use_memory=True, max_memories=20): 
  agent_desc = ""
  agent_desc += f"Self description: {agent.get_self_description()}\n==\n"
  agent_desc += f"Other observations about the subject:\n\n"

  if use_memory:
    try:
      retrieved = agent.memory_stream.retrieve([anchor], 0, n_count=max_memories)
      if len(retrieved) > 0:
        nodes = list(retrieved.values())[0]
        for node in nodes[:max_memories]:
          agent_desc += f"{node.content}\n"
    except Exception as e:
      import logging
      logger = logging.getLogger(__name__)
      logger.warning(f"Memory retrieval failed: {str(e)[:100]}")
  
  return agent_desc


def run_gpt_generate_categorical_resp(
  agent_desc, 
  questions,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, questions):
    str_questions = ""
    for key, val in questions.items(): 
      str_questions += f"Q: {key}\n"
      str_questions += f"Option: {val}\n\n"
    str_questions = str_questions.strip()
    return [agent_desc, str_questions]

  def _func_clean_up(gpt_response, prompt=""): 
    import logging
    logger = logging.getLogger(__name__)
    
    if gpt_response.startswith("ERROR:"):
      logger.error(f"LLM Error: {gpt_response}")
      return {"responses": [], "reasonings": [], "error": gpt_response}
    
    logger.info(f"GPT Response: {gpt_response[:500]}")
    responses, reasonings = extract_first_json_dict_categorical(gpt_response)
    logger.info(f"Extracted - Responses: {len(responses)}, Reasonings: {len(reasonings)}")
    
    if not responses:
      error_msg = f"Failed to extract response from LLM output"
      if len(gpt_response) < 200:
        error_msg += f": {gpt_response}"
      else:
        error_msg += f" (first 200 chars: {gpt_response[:200]})"
      logger.warning(error_msg)
      return {"responses": [], "reasonings": [], "error": error_msg}
    
    ret = {"responses": responses, "reasonings": reasonings}
    return ret

  def _get_fail_safe():
    return None

  if len(questions) > 1: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/batch_v1.txt" 
  else: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/singular_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, questions) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  return output, [output, prompt, prompt_input, fail_safe]


def categorical_resp(agent, questions): 
  anchor = " ".join(list(questions.keys()))
  use_memory = os.getenv('USE_AGENT_MEMORY', 'true').lower() == 'true'
  max_memories = int(os.getenv('MAX_MEMORIES_PER_AGENT', '20'))
  agent_desc = _main_agent_desc(agent, anchor, use_memory=use_memory, max_memories=max_memories)
  return run_gpt_generate_categorical_resp(
           agent_desc, questions, "1", LLM_VERS)[0]

def categorical_resp_batch(agents, questions_list, use_memory=None, llm_config=None):
  """Process multiple agents with load-balanced racing across providers."""
  from genagents_simulation.simulation_engine.gpt_structure import generate_prompt
  
  # Use memory parameter if provided, otherwise fallback to env var
  if use_memory is None:
    use_memory = os.getenv('USE_AGENT_MEMORY', 'false').lower() == 'true'
  
  # Auto-calculate optimal settings based on model config
  agents_per_api_call, max_memories_per_agent = _calculate_optimal_batch_settings(llm_config, use_memory)
  
  if len(questions_list) > 1:
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/batch_v1.txt"
  else:
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/singular_v1.txt"
  
  multi_agent_prompt_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/categorical_resp/multi_agent_batch_v1.txt"
  
  grouped_requests = []
  agent_groups = []
  
  for group_start in range(0, len(agents), agents_per_api_call):
    group_end = min(group_start + agents_per_api_call, len(agents))
    group_agents = agents[group_start:group_end]
    group_questions = questions_list[group_start:group_end]
    
    all_agent_descs = []
    for agent, questions in zip(group_agents, group_questions):
      anchor = " ".join(list(questions.keys()))
      agent_desc = _main_agent_desc(agent, anchor, use_memory=use_memory, max_memories=max_memories_per_agent)
      all_agent_descs.append(f"Participant {len(all_agent_descs) + 1}:\n{agent_desc}\n")
    
    combined_agent_desc = "\n---\n\n".join(all_agent_descs)
    
    str_questions = ""
    for key, val in group_questions[0].items():
      str_questions += f"Q: {key}\n"
      str_questions += f"Option: {val}\n\n"
    str_questions = str_questions.strip()
    
    prompt_input = [combined_agent_desc, str_questions]
    prompt = generate_prompt(prompt_input, multi_agent_prompt_file)
    
    grouped_requests.append({"prompt": prompt, "agent_count": len(group_agents)})
    agent_groups.append((group_agents, group_questions))
  
  batch_responses = invoke_batch(grouped_requests, model=LLM_VERS, max_tokens=8000, temperature=0.7)
  
  results = []
  for i, (gpt_response, (group_agents, group_questions)) in enumerate(zip(batch_responses, agent_groups)):
    import logging
    logger = logging.getLogger(__name__)
    
    if gpt_response.startswith("ERROR:"):
      for _ in group_agents:
        results.append({"responses": [], "reasonings": [], "error": gpt_response})
      continue
    
    try:
      import json
      parsed = json.loads(gpt_response)
      
      for idx, (agent, questions) in enumerate(zip(group_agents, group_questions)):
        agent_key = str(idx + 1)
        if agent_key in parsed:
          agent_data = parsed[agent_key]
          responses = [agent_data.get("Response", "")]
          reasonings = [agent_data.get("Reasoning", "")]
          results.append({"responses": responses, "reasonings": reasonings})
        else:
          responses, reasonings = extract_first_json_dict_categorical(gpt_response)
          if responses and len(responses) > idx:
            results.append({"responses": [responses[idx]], "reasonings": [reasonings[idx] if idx < len(reasonings) else ""]})
          else:
            results.append({"responses": [], "reasonings": [], "error": f"Agent {idx+1} not found in response"})
    except (json.JSONDecodeError, KeyError, TypeError):
      responses, reasonings = extract_first_json_dict_categorical(gpt_response)
      
      for idx, (agent, questions) in enumerate(zip(group_agents, group_questions)):
        if responses and len(responses) > idx:
          results.append({"responses": [responses[idx]], "reasonings": [reasonings[idx] if idx < len(reasonings) else ""]})
        else:
          error_msg = f"Failed to extract response for agent {idx+1}"
          results.append({"responses": [], "reasonings": [], "error": error_msg})
  
  # Return results along with optimization settings
  optimization_settings = {
    "use_memory": use_memory,
    "agents_per_api_call": agents_per_api_call,
    "max_memories_per_agent": max_memories_per_agent if use_memory else 0,
    "model": llm_config.get('model', 'unknown') if llm_config else 'unknown',
    "context_window": llm_config.get('context_window', 0) if llm_config else 0
  }
  
  return results, optimization_settings


def run_gpt_generate_numerical_resp(
  agent_desc, 
  questions, 
  float_resp,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, questions, float_resp):
    str_questions = ""
    for key, val in questions.items(): 
      str_questions += f"Q: {key}\n"
      str_questions += f"Range: {str(val)}\n\n"
    str_questions = str_questions.strip()

    if float_resp: 
      resp_type = "float"
    else: 
      resp_type = "integer"
    return [agent_desc, str_questions, resp_type]

  def _func_clean_up(gpt_response, prompt=""): 
    responses, reasonings = extract_first_json_dict_numerical(gpt_response)
    ret = {"responses": responses, "reasonings": reasonings}
    return ret

  def _get_fail_safe():
    return None

  if len(questions) > 1: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/batch_v1.txt" 
  else: 
    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/numerical_resp/singular_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, questions, float_resp) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  if float_resp: 
    output["responses"] = [float(i) for i in output["responses"]]
  else: 
    output["responses"] = [int(i) for i in output["responses"]]

  return output, [output, prompt, prompt_input, fail_safe]


def numerical_resp(agent, questions, float_resp): 
  anchor = " ".join(list(questions.keys()))
  use_memory = os.getenv('USE_AGENT_MEMORY', 'true').lower() == 'true'
  max_memories = int(os.getenv('MAX_MEMORIES_PER_AGENT', '20'))
  agent_desc = _main_agent_desc(agent, anchor, use_memory=use_memory, max_memories=max_memories)
  return run_gpt_generate_numerical_resp(
           agent_desc, questions, float_resp, "1", LLM_VERS)[0]


def run_gpt_generate_utterance(
  agent_desc, 
  str_dialogue,
  context,
  prompt_version="1",
  gpt_version="GPT4o",  
  verbose=False):

  def create_prompt_input(agent_desc, str_dialogue, context):
    return [agent_desc, context, str_dialogue]

  def _func_clean_up(gpt_response, prompt=""): 
    utterance = extract_first_json_dict(gpt_response)["utterance"]
    return utterance

  def _get_fail_safe():
    return None

  prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/utternace/utterance_v1.txt" 

  prompt_input = create_prompt_input(agent_desc, str_dialogue, context) 
  fail_safe = _get_fail_safe() 

  output, prompt, prompt_input, fail_safe = chat_safe_generate(
    prompt_input, prompt_lib_file, gpt_version, 1, fail_safe, 
    _func_clean_up, verbose)

  return output, [output, prompt, prompt_input, fail_safe]


def utterance(agent, curr_dialogue, context): 
  str_dialogue = ""
  for row in curr_dialogue:
    str_dialogue += f"[{row[0]}]: {row[1]}\n"
  str_dialogue += f"[{agent.get_fullname()}]: [Fill in]\n"

  anchor = str_dialogue
  use_memory = os.getenv('USE_AGENT_MEMORY', 'true').lower() == 'true'
  max_memories = int(os.getenv('MAX_MEMORIES_PER_AGENT', '20'))
  agent_desc = _utterance_agent_desc(agent, anchor, use_memory=use_memory, max_memories=max_memories)
  return run_gpt_generate_utterance(
           agent_desc, str_dialogue, context, "1", LLM_VERS)[0]

##  Ask function.
def run_gpt_generate_ask(
    agent_desc,
    questions,
    prompt_version="1",
    gpt_version="GPT4o",
    verbose=False):

    def create_prompt_input(agent_desc, questions):
        str_questions = ""
        i = 1
        for q in questions:
            str_questions += f"Q{i}: {q['question']}\n"
            str_questions += f"Type: {q['response-type']}\n"
            if q['response-type'] == 'categorical':
                str_questions += f"Options: {', '.join(q['response-options'])}\n"
            elif q['response-type'] in ['int', 'float']:
                str_questions += f"Range: {q['response-scale']}\n"
            elif q['response-type'] == 'open':
                char_limit = q.get('response-char-limit', 200)
                str_questions += f"Character Limit: {char_limit}\n"
            str_questions += "\n"
            i += 1
        return [agent_desc, str_questions.strip()]

    def _func_clean_up(gpt_response, prompt=""):
        responses = extract_first_json_dict(gpt_response)
        return responses

    def _get_fail_safe():
        return None

    prompt_lib_file = f"{LLM_PROMPT_DIR}/generative_agent/interaction/ask/batch_v1.txt"

    prompt_input = create_prompt_input(agent_desc, questions)
    fail_safe = _get_fail_safe()

    output, prompt, prompt_input, fail_safe = chat_safe_generate(
        prompt_input, prompt_lib_file, gpt_version, 1, fail_safe,
        _func_clean_up, verbose)

    return output, [output, prompt, prompt_input, fail_safe]



  





  




  





  





