import os
import torch
import logging
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

import load_token
token = load_token.token

# ===== 로깅 설정 & settings =====
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["WANDB_MODE"] = "disabled"

# PyTorch CUDA 메모리 설정 (32GB GPU 최적화)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"


# ===== 모델 로드 함수 =====
def llama3Ko_loader():
    model_name = "beomi/Llama-3-KoEn-8B-Instruct-preview"
    logging.info("[모델 초기화] llama3Ko-Instruct-8B")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 32GB에서는 bfloat16 사용 가능
        device_map='auto',
        token=token,
        max_memory={0: '28GiB'}  # 32GB 중 28GB 사용
    )

    model.eval()
    logging.info("모델과 토크나이저 초기화가 완료되었습니다!")
    return model, tokenizer


def load_koalpaca_model():
    model_name = "beomi/KoAlpaca-Polyglot-5.8B"
    logging.info(f"[모델 로딩] {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True)

    # GPU 메모리 확인 및 최적화 설정
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logging.info(f"GPU 총 메모리: {total_memory:.2f}GB")

    # 메모리에 따른 동적 할당
    if total_memory > 30:
        max_memory_gb = '28GiB'
        use_dtype = torch.bfloat16
    else:
        max_memory_gb = '20GiB'
        use_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=use_dtype,
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: max_memory_gb},
        low_cpu_mem_usage=True,
        # Flash Attention 제거 - 설치되지 않은 경우 오류 방지
    )

    logging.info(f"모델 메모리 사용량: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    return model, tokenizer


# ===== 프롬프트 템플릿 =====
compression_prompt_template = """
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


class PromptDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_length=1024):  # 원래 길이 유지
        df = pd.read_csv(csv_path)[["news", "label"]].dropna()
        self.samples = []
        tokenizer.pad_token = tokenizer.eos_token

        logging.info(f"데이터셋 크기: {len(df)}개 샘플")

        for i, (_, row) in enumerate(df.iterrows()):
            if i % 100 == 0:
                logging.info(f"데이터 전처리 진행: {i}/{len(df)}")

            prompt = compression_prompt_template.format(news=row["news"])
            input_enc = tokenizer(
                prompt,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

            input_ids = input_enc['input_ids'].squeeze()
            attention_mask = input_enc['attention_mask'].squeeze()

            # 라벨 생성 최적화
            prompt_len = len(tokenizer(prompt, truncation=True,
                             max_length=max_length)["input_ids"])
            labels = input_ids.clone()
            labels[:prompt_len] = -100  # 프롬프트 부분은 loss 계산에서 제외

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

        logging.info(f"데이터 전처리 완료: {len(self.samples)}개 샘플")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def print_memory_usage():
    """GPU 메모리 사용량 출력"""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logging.info(
        f"GPU 메모리 - 할당: {allocated:.2f}GB, 예약: {reserved:.2f}GB, 전체: {total:.2f}GB")


def main():
    # 초기 메모리 정리
    torch.cuda.empty_cache()
    print_memory_usage()

    model, tokenizer = load_koalpaca_model()

    # 메모리 최적화 설정
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    print_memory_usage()

    dataset = PromptDataset(
        csv_path="./koalpaca_soft_prompt_data.csv",
        tokenizer=tokenizer,
        max_length=1024  # 32GB에서는 원래 길이 사용 가능
    )

    # GPU 메모리에 따른 동적 학습 설정
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

    if total_memory > 30:
        # 32GB+ GPU 설정
        batch_size = 4
        accumulation_steps = 4
        use_bf16 = True
        use_fp16 = False
        pin_memory = True
        num_workers = 2
    else:
        # 24GB GPU 설정
        batch_size = 2
        accumulation_steps = 8
        use_bf16 = False
        use_fp16 = True
        pin_memory = False
        num_workers = 0

    training_args = TrainingArguments(
        output_dir="./soft_prompt_model",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accumulation_steps,
        num_train_epochs=3,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=use_fp16,
        bf16=use_bf16,
        learning_rate=3e-4,
        weight_decay=0.01,
        optim="adamw_torch",
        warmup_steps=50,
        lr_scheduler_type="cosine",
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to=None,
        ddp_find_unused_parameters=False,
        eval_strategy="no",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    logging.info(f"GPU 메모리 기반 최적화 설정으로 학습 시작 (총 메모리: {total_memory:.1f}GB)")
    print_memory_usage()

    try:
        trainer.train()
        logging.info("학습 완료!")

        # 모델 저장
        model.save_pretrained("./soft_prompt_model_final")
        tokenizer.save_pretrained("./soft_prompt_model_final")
        logging.info("모델 저장 완료")

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"CUDA 메모리 부족: {e}")
        print_memory_usage()

        # 메모리 정리 후 더 보수적인 설정으로 재시도
        torch.cuda.empty_cache()
        logging.info("메모리 정리 후 보수적 설정으로 재시도...")

        training_args.per_device_train_batch_size = 1
        training_args.gradient_accumulation_steps = 16
        training_args.fp16 = True
        training_args.bf16 = False

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        trainer.train()


# ===== 학습 시작 =====
if __name__ == "__main__":
    logging.info("학습을 시작합니다... soft prompt fine-tuning GPU 메모리 최적화")
    main()
