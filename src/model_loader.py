
import logging
import torch
import os
from pathlib import Path
from huggingface_hub import snapshot_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

login(token=os.getenv("HUGGINGFACE_TOKEN"))


def koalpaca_loader():
    model_name = "beomi/KoAlpaca-Polyglot-5.8B"
    logging.info("모델과 토크나이저를 초기화 중입니다...")
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

    logging.info("모델과 토크나이저를 초기화 중입니다...")
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
