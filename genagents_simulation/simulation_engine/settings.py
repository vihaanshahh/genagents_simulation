from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fetch OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KEY_OWNER = "host"

DEBUG = False

MAX_CHUNK_SIZE = 4

LLM_VERS = "llama3.1-8b"

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"

# To do: Are the following needed in the new structure? Ideally Populations_Dir is for the user to define.
POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations"
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"
