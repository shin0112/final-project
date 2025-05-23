import os
import pandas as pd
import model_loader
import logging

from prompts import prompt_compression
# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def compress_article(file_name: str):
    # 파일 경로 확인
    if not os.path.exists(file_name):
        logging.error(f"입력 파일 '{file_name}'을(를) 찾을 수 없습니다.")
        raise FileNotFoundError(f"{file_name} 파일을 찾을 수 없습니다.")

    logging.info(f"입력 파일 '{file_name}'을(를) 불러옵니다.")
    try:
        df = pd.read_csv(file_name)
        logging.info(f"CSV 파일 로드 성공: {df.shape[0]}개의 행, {df.shape[1]}개의 열")
    except Exception as e:
        logging.exception("CSV 파일을 불러오는 중 오류 발생")
        raise

    # 모델 로드하기 (llama3ko)
    try:
        logging.info("기사 압축에 사용할 LLM 모델을 로드합니다.")
        llm = model_loader.llama3Ko_loader()
        logging.info("모델 로드 완료")
    except Exception as e:
        logging.exception("모델 로드 중 오류 발생")
        raise

    # 프롬프트 불러오기 (기사 압축 프롬프트)
    try:
        prompt = prompt_compression.base_prompt
        logging.info("프롬프트 로드 완료")
    except Exception as e:
        logging.exception("프롬프트 로드 중 오류 발생")
        raise

    compressed_results = []

    for idx, row in df.iterrows():
        try:
            article = row['article']
            input_prompt = prompt.format(article=article)
            logging.info(
                f"{idx+1}/{len(df)}번째 기사 압축 중... (기사 길이: {len(article)}자)")
            compressed = llm.generate(input_prompt)
            compressed_results.append(compressed)
            logging.debug(f"{idx+1}번째 기사 압축 결과: {compressed[:50]}...")
        except Exception as e:
            logging.exception(f"{idx+1}번째 기사 압축 중 오류 발생")
            compressed_results.append("압축 실패")

    df['compressed_article'] = compressed_results

    output_file = file_name.replace('.csv', '_compressed.csv')
    try:
        df.to_csv(output_file, index=False)
        logging.info(f"압축 결과가 '{output_file}'에 저장되었습니다. (총 {len(df)}개 기사)")
    except Exception as e:
        logging.exception("결과 저장 중 오류 발생")
        raise
