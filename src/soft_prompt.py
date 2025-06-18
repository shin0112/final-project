import os
import gc
import torch
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq
)
import load_token

# ===== 설정 =====
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
token = load_token.huggingface_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 모델 로드 =====


def load_koalpaca_model():
    model_name = "beomi/KoAlpaca-Polyglot-5.8B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "20GiB"},
        low_cpu_mem_usage=True,
    )
    model.gradient_checkpointing_enable()
    logging.info(
        f"모델 메모리 사용량: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    return model, tokenizer


# ===== 프롬프트 템플릿 =====
prompt_template = """
다음은 뉴스 기사 본문입니다. 아래 기사에서 '환경', '친환경', '탄소중립', '재활용', '지속 가능성', '인증', '자원 절약' 등과 관련된 **마케팅 표현 또는 제품 설명**을 중심으로, 주장과 근거가 담긴 **핵심 문장만 간결하게 요약**해 주세요.

[요약 조건]
- 환경 관련 주장과 그에 대한 **수치, 인증, 기관명 등 구체적 근거**가 포함된 문장은 근거로 사용하기 위해 원문 그대로 사용하길 권장합니다.
- **광고성 문장**이나 **제품 설명**도 그린워싱 판단에 중요하므로 반드시 포함해 주세요.
- 요약은 **최대 5문장 이내**로, 불필요한 내용 없이 간결하게 작성해 주세요.
- 기사 전체가 아닌, **환경 주장 평가에 필요한 핵심 문장만** 요약해 주세요.

------------------------------
[기사 본문]
{news}

------------------------------
[요약된 환경 관련 핵심 문장]
"""

# ===== 데이터셋 =====


class PromptDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=1024):
        df = pd.read_csv(csv_path)[["news", "label"]].dropna()
        self.samples = []
        tokenizer.pad_token = tokenizer.eos_token

        for _, row in df.iterrows():
            prompt = prompt_template.format(news=row["news"])
            enc = tokenizer(prompt, max_length=max_length,
                            truncation=True, padding="max_length")

            labels = enc["input_ids"].copy()
            prompt_len = len(tokenizer(prompt, truncation=True,
                             max_length=max_length)["input_ids"])
            labels[:prompt_len] = [-100] * prompt_len  # 프롬프트 부분은 학습 제외

            self.samples.append({
                "input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "labels": torch.tensor(labels)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ===== 학습 루프 =====


def train():
    torch.cuda.empty_cache()
    logging.info("GPU 메모리 초기화 완료")
    gc.collect()

    model, tokenizer = load_koalpaca_model()
    model.train()

    dataset = PromptDataset(
        "./koalpaca_soft_prompt_data.csv",
        tokenizer,
        max_length=512
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-5, weight_decay=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logging.info("모델과 데이터셋 준비 완료")

    for epoch in range(1):
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                logging.info(
                    f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    model.save_pretrained("./soft/soft_prompt_model_final")
    tokenizer.save_pretrained("./soft/soft_prompt_model_final")


# ===== 실행 =====
if __name__ == "__main__":
    logging.info("학습을 시작합니다... soft prompt fine-tuning GPU 메모리 최적화")
    train()
