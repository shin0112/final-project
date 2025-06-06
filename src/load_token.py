import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
os.environ["HF_HOME"] = "/data/wnslcosltimo12/hf_cache"

huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
login(token=huggingface_token)

groq_token = os.environ.get("GROQ_API_KEY")
