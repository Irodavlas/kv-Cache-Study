from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import os


load_dotenv()

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN not found in .env")


model_dir = Path("model/llama-3.2-3b")

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    token=hf_token,
)

print(f"Model saved to: {model_dir.resolve()}")