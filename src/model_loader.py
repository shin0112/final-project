
import logging
import torch
import os
from dotenv import load_dotenv
from pathlib import Path
from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)
os.environ["HF_HOME"] = "/data/wnslcosltimo12/hf_cache"
print("🔐 HF TOKEN PREFIX:", os.environ.get("HUGGINGFACE_TOKEN")[:10])
token = os.environ.get("HUGGINGFACE_TOKEN")


def load_model(model_name):
    if model_name == "KoAlpaca":
        return koAlpaca_loader()
    elif model_name == "Mistral":
        return mistral_loader()
    elif model_name == "llama3Ko":
        return llama3Ko_loader()
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


def koAlpaca_loader():
    model_name = "beomi/KoAlpaca-Polyglot-5.8B"
    logging.info("[모델 초기화] KoAlpaca-Polyglot-5.8B")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    logging.info("모델과 토크나이저 초기화가 완료되었습니다!")
    return model, tokenizer


def mistral_loader():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"

    # mistral_models_path = Path.home().joinpath(
    #     'mistral_models', '7B-Instruct-v0.3')
    # mistral_models_path.mkdir(parents=True, exist_ok=True)
    # snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=[
    #     "params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)

    # tokenizer = MistralTokenizer.from_file(
    #     f"{mistral_models_path}/tokenizer.model.v3")
    # model = Transformer.from_folder(mistral_models_path)

    logging.info("[모델 초기화] Mistral-7B-Instruct-v0.3")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    logging.info("모델과 토크나이저 초기화가 완료되었습니다!")
    return model, tokenizer


def llama3Ko_loader():
    model_name = "AIDX-ktds/ktdsbaseLM-v0.14-onbased-llama3.1"
    logging.info("[모델 초기화] llama3Ko")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

    model.eval()
    logging.info("모델과 토크나이저 초기화가 완료되었습니다!")

    return model, tokenizer
