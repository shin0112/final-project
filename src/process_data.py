import os
import pandas as pd
import model_loader
import logging
import torch

from prompts import prompt_compression
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def compress_article(origin_file: str, processed_file: str):
    # 파일 경로 확인
    if os.path.exists(processed_file):
        logging.warning(f"'{processed_file}' 파일이 이미 존재합니다. 덮어쓰지 않습니다.")
        return

    logging.info(f"입력 파일 '{origin_file}'을(를) 불러옵니다.")
    try:
        df = pd.read_csv(origin_file)
        logging.info(f"CSV 파일 로드 성공: {df.shape[0]}개의 행, {df.shape[1]}개의 열")
    except Exception as e:
        logging.exception("CSV 파일을 불러오는 중 오류 발생")
        raise

    # 인증 마크 컬럼 추가
    if 'en_mark' not in df.columns:
        logging.error("CSV 파일에 'en_mark' 컬럼이 없습니다. 인증 마크 정보를 추가해주세요.")
        raise ValueError("CSV 파일에 'en_mark' 컬럼이 없습니다.")

    df["certification_type"] = df["en_mark"].apply(normalize_cert_type)
    logging.info("인증 마크 컬럼 추가 완료")
    logging.info(f"인증 마크 종류: {df['certification_type'].unique()}")
    logging.info(f"상위 5개 인증 마크:\n{df['certification_type'].head(5)}")

    # 모델 로드하기 (llama3ko)
    try:
        logging.info("기사 압축에 사용할 LLM 모델을 로드합니다.")
        model, tokenizer = model_loader.llama3Ko_loader()
        tokenizer.pad_token = tokenizer.eos_token
        logging.info("모델 로드 완료")
    except Exception as e:
        logging.exception("모델 로드 중 오류 발생")
        raise

    # 프롬프트 불러오기 (기사 압축 프롬프트)
    try:
        prompt_template = prompt_compression.base_prompt
        logging.info("프롬프트 로드 완료")
    except Exception as e:
        logging.exception("프롬프트 로드 중 오류 발생")
        raise

    compressed_results = []

    for idx, row in df.iterrows():
        try:
            news = row['full_text']
            if pd.isna(news):
                logging.warning(f"⚠️  [WARNING] 기사 내용이 비어있습니다. (인덱스: {idx})")
                compressed_results.append("기사 내용 없음")
                continue

            if len(news) < 700:
                compressed_results.append(news)
                continue

            logging.info("\n" + "=" * 80)
            logging.info(f"📄 [기사 {idx + 1}/{len(df)}] 시작")
            logging.info(f"📝 [기사 길이] {len(news)}자")
            logging.info(f"🔍 [원문 일부] {news[:200]}...")
            logging.info("-" * 80)

            prompt = prompt_template.format(news=news)
            messages = [
                {
                    "role": "system",
                    "content": "너는 환경 기사 요약 전문가야. 주어진 기사에서 환경 관련 마케팅 주장과 근거 문장만 추출해 5문장 이내로 요약해."
                },
                {"role": "user", "content": prompt}
            ]

            chat_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                chat_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    temperature=0.5,
                    top_p=0.8,
                    do_sample=True,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            compressed = tokenizer.decode(
                output_ids[0], skip_special_tokens=True)

            logging.info("✅ [압축 결과]")
            logging.info(compressed.strip())
            logging.info("=" * 80)

            compressed_results.append(extract_summary_only(compressed))

        except Exception as e:
            logging.exception(f"❌ [오류] {idx+1}번째 기사 압축 중 예외 발생")
            compressed_results.append("압축 실패")

    df['compressed_article'] = compressed_results

    output_file = processed_file
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"압축 결과가 '{output_file}'에 저장되었습니다. (총 {len(df)}개 기사)")
    except Exception as e:
        logging.exception("결과 저장 중 오류 발생")
        raise


def extract_summary_only(full_output: str) -> str:
    """
    전체 출력 중 '[요약된 환경 관련 핵심 문장]' 이후 텍스트만 추출
    """
    split_token = "[요약된 환경 관련 핵심 문장]assistant"
    if split_token in full_output:
        return full_output.split(split_token, 1)[-1].strip()
    return full_output.strip()  # fallback


def normalize_cert_type(text):
    if not isinstance(text, str):
        return "없음"
    if "탄소발자국" in text:
        return "탄소발자국"
    elif "에너지" in text:
        return "에너지절약"
    elif "환경표지" in text or "환경부 인증" in text:
        return "환경표지"
    elif "없음" in text:
        return "없음"
    else:
        return "기타"
