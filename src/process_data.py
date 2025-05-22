import os
import prompts
import pandas as pd
import model_loader


def compress_article(file_name: str):
    # 파일 경로 확인
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"{file_name} 파일을 찾을 수 없습니다.")

    # CSV 파일 불러오기
    df = pd.read_csv(file_name)

    # 모델 로드하기 (llama3ko)
    llm = model_loader.llama3Ko_loader()

    # 프롬프트 불러오기 (기사 압축 프롬프트)
    prompt = prompts.prompt_compression.base_prompt

    # 기사 압축 결과 저장할 리스트
    compressed_results = []

    # 각 기사에 대해 압축 수행
    for idx, row in df.iterrows():
        article = row['article']  # 'article' 컬럼명에 맞게 수정
        input_prompt = prompt.format(article=article)
        compressed = llm.generate(input_prompt)
        compressed_results.append(compressed)

    # 결과를 새로운 컬럼에 저장
    df['compressed_article'] = compressed_results

    # 결과를 새 파일로 저장 (원본 파일명에 '_compressed' 추가)
    output_file = file_name.replace('.csv', '_compressed.csv')
    df.to_csv(output_file, index=False)
    print(f"압축 결과가 {output_file}에 새로 저장되었습니다.")
