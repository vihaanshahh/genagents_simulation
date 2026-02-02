import json
import time
import base64
from typing import List, Union
from functools import lru_cache
import hashlib

from genagents_simulation.simulation_engine.settings import *
from genagents_simulation.simulation_engine.llm_provider import invoke_model, invoke_batch


# ============================================================================
# #######################[SECTION 1: HELPER FUNCTIONS] #######################
# ============================================================================

def print_run_prompts(prompt_input: Union[str, List[str]], 
                      prompt: str, 
                      output: str) -> None:
  print (f"=== START =======================================================")
  print ("~~~ prompt_input    ----------------------------------------------")
  print (prompt_input, "\n")
  print ("~~~ prompt    ----------------------------------------------------")
  print (prompt, "\n")
  print ("~~~ output    ----------------------------------------------------")
  print (output, "\n") 
  print ("=== END ==========================================================")
  print ("\n\n\n")


def generate_prompt(prompt_input: Union[str, List[str]], 
                    prompt_lib_file: str) -> str:
  """Generate a prompt by replacing placeholders in a template file with 
     input."""
  if isinstance(prompt_input, str):
    prompt_input = [prompt_input]
  prompt_input = [str(i) for i in prompt_input]

  with open(prompt_lib_file, "r") as f:
    prompt = f.read()

  for count, input_text in enumerate(prompt_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", input_text)

  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]

  return prompt.strip()


# ============================================================================
# ####################### [SECTION 2: SAFE GENERATE] #########################
# ============================================================================

def _create_cache_key(prompt: str, model: str, max_tokens: int) -> str:
  key_str = f"{prompt[:200]}|{model}|{max_tokens}"
  return hashlib.md5(key_str.encode()).hexdigest()

_response_cache = {}
_cache_max_size = 1000

def gpt_request(prompt: str, 
                model: str = "gpt-5-nano", 
                max_tokens: int = 500,
                use_cache: bool = True,
                temperature: float = 0.7) -> str:
  if use_cache:
    cache_key = _create_cache_key(prompt, model, max_tokens)
    if cache_key in _response_cache:
      return _response_cache[cache_key]
  
  try:
    result = invoke_model(prompt, model, max_tokens, temperature)
    
    if use_cache and not result.startswith("ERROR"):
      if len(_response_cache) >= _cache_max_size:
        _response_cache.pop(next(iter(_response_cache)))
      _response_cache[cache_key] = result
    
    return result
  except Exception as e:
    return f"GENERATION ERROR: {str(e)[:100]}"


def gpt4_vision(messages: List[dict], max_tokens: int = 4000) -> str:
  """Make a request to OpenAI with vision."""
  try:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    formatted_messages = []
    for msg in messages:
      content = msg.get("content", "")
      if isinstance(content, list):
        formatted_content = []
        for item in content:
          if item.get("type") == "text":
            formatted_content.append({"type": "text", "text": item.get("text", "")})
          elif item.get("type") == "image_url":
            formatted_content.append(item)
        content = formatted_content
      formatted_messages.append({"role": msg.get("role", "user"), "content": content})
    
    response = client.chat.completions.create(
      model="gpt-5-nano",
      messages=formatted_messages,
      max_tokens=max_tokens,
      temperature=0.7
    )
    
    return response.choices[0].message.content
  except Exception as e:
    return f"GENERATION ERROR: {str(e)}"


def chat_safe_generate(prompt_input: Union[str, List[str]], 
                       prompt_lib_file: str,
                       gpt_version: str = "gpt-5-nano", 
                       repeat: int = 1,
                       fail_safe: str = "error", 
                       func_clean_up: callable = None,
                       verbose: bool = False,
                       max_tokens: int = 4000,
                       file_attachment: str = None,
                       file_type: str = None) -> tuple:
  """Generate a response using GPT models with error handling & retries."""
  if file_attachment and file_type:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    messages = [{"role": "user", "content": prompt}]

    if file_type.lower() == 'image':
      with open(file_attachment, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
      messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": "Please refer to the attached image."},
            {"type": "image_url", "image_url": 
              {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
      })
      response = gpt4_vision(messages, max_tokens)

    elif file_type.lower() == 'pdf':
      pdf_text = extract_text_from_pdf_file(file_attachment)
      pdf = f"PDF attachment in text-form:\n{pdf_text}\n\n"
      instruction = generate_prompt(prompt_input, prompt_lib_file)
      prompt = f"{pdf}"
      prompt += f"<End of the PDF attachment>\n=\nTask description:\n{instruction}"
      response = gpt_request(prompt, gpt_version, max_tokens)

  else:
    prompt = generate_prompt(prompt_input, prompt_lib_file)
    for i in range(repeat):
      response = gpt_request(prompt, model=gpt_version)
      if response != "GENERATION ERROR":
        break
      time.sleep(2**i)
    else:
      response = fail_safe

  if func_clean_up:
    response = func_clean_up(response, prompt=prompt)

  if verbose or DEBUG:
    print_run_prompts(prompt_input, prompt, response)

  return response, prompt, prompt_input, fail_safe


# ============================================================================
# #################### [SECTION 3: OTHER API FUNCTIONS] ######################
# ============================================================================

_embedding_cache = {}
_embedding_cache_max_size = 10000

def get_text_embedding(text: str, 
                       model: str = "text-embedding-3-small") -> List[float]:
  """Generate an embedding for the given text using OpenAI with caching."""
  if not isinstance(text, str) or not text.strip():
    raise ValueError("Input text must be a non-empty string.")

  text = text.replace("\n", " ").strip()
  
  cache_key = f"{model}:{text}"
  if cache_key in _embedding_cache:
    return _embedding_cache[cache_key]
  
  try:
    import os
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    response = client.embeddings.create(
      model=model,
      input=text
    )
    embedding = response.data[0].embedding
    
    if len(_embedding_cache) >= _embedding_cache_max_size:
      _embedding_cache.pop(next(iter(_embedding_cache)))
    _embedding_cache[cache_key] = embedding
    
    return embedding
  except Exception as e:
    import hashlib
    hash_obj = hashlib.sha256(text.encode())
    hash_bytes = hash_obj.digest()
    embedding = [(b / 127.5) - 1.0 for b in hash_bytes[:128]]
    while len(embedding) < 1536:
      embedding.append(0.0)
    return embedding[:1536]

def get_text_embeddings_batch(texts: List[str], 
                              model: str = "text-embedding-3-small") -> List[List[float]]:
  """Batch get embeddings for multiple texts at once (much faster)."""
  if not texts:
    return []
  
  import os
  from openai import OpenAI
  client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
  
  texts_to_embed = []
  text_indices = []
  cached_results = {}
  
  for idx, text in enumerate(texts):
    if not isinstance(text, str) or not text.strip():
      continue
    text = text.replace("\n", " ").strip()
    cache_key = f"{model}:{text}"
    
    if cache_key in _embedding_cache:
      cached_results[idx] = _embedding_cache[cache_key]
    else:
      texts_to_embed.append(text)
      text_indices.append(idx)
  
  if texts_to_embed:
    try:
      response = client.embeddings.create(
        model=model,
        input=texts_to_embed
      )
      
      for i, embedding in enumerate(response.data):
        cache_key = f"{model}:{texts_to_embed[i]}"
        if len(_embedding_cache) >= _embedding_cache_max_size:
          _embedding_cache.pop(next(iter(_embedding_cache)))
        _embedding_cache[cache_key] = embedding.embedding
        cached_results[text_indices[i]] = embedding.embedding
    except Exception as e:
      import hashlib
      for i, text in enumerate(texts_to_embed):
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        embedding = [(b / 127.5) - 1.0 for b in hash_bytes[:128]]
        while len(embedding) < 1536:
          embedding.append(0.0)
        cached_results[text_indices[i]] = embedding
  
  return [cached_results.get(i, [0.0] * 1536) for i in range(len(texts))]









