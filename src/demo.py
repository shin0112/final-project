import load_token
import vectorStore
import query_processor
from save_data import save_and_evaluate_results
from prompts import prompt_v4, prompt_compression

from langchain_groq import ChatGroq


import os
import pandas as pd
import time
import logging
from pathlib import Path

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(
    log_dir, f"run_log_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

test_input_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'test_data.csv'
test_input_c_path = Path(__file__).parent.parent / 'data' / \
    'greenwashing' / 'test_data_compressed.csv'


def logging_result(results):
    for r in results:
        print("="*80)
        print(f"Article Preview: {r['article'][:200]}...")
        print(f"Context Preview: {r['context'][:200]}...")
        print(f"Answer:\n{r['answer']}")
        print("="*80)


def logging_model(model_name, embeddings_model, retriever_strategy, num_articles, prompt_version):
    logging.info(f"[실험 환경]")
    logging.info(f"  모델: {model_name}")
    logging.info(f"  쿼리 임베딩 모델: {embeddings_model}")
    logging.info(f"  검색 전략: {retriever_strategy}")
    logging.info(f"  검색 문서 수: {num_articles}")
    logging.info(f"  프롬프트 버전: {prompt_version}")


def groq_loader():
    # demo용 open ai 호출
    model_name = "llama3-8b-8192"
    model = ChatGroq(
        model=model_name,
        temperature=0.8,
        max_tokens=4096,
        api_key=load_token.groq_token
    )
    logging.info(f"모델 {model_name} 로드 완료")
    return model


def build_example_block(docs, max_tokens=500):
    """
    예시 문서들을 하나의 문자열로 연결하되, 최대 토큰 수를 넘지 않도록 제한
    """
    result = ""
    for doc in docs:
        text = doc.page_content.strip()
        if len(result + text) > max_tokens:
            break
        result += text + "\n"
    return result.strip()


def generate_answer_groq(
        model,
        query,
        context,
        ct="",
        prompt_version="v4-fewshot",
        rt_n=None,
):
    if prompt_version == "v4-oneshot":
        prompt_template = prompt_v4.prompt_oneshot
        example_docs = rt_n.vectorstore.similarity_search(query, k=1)
        example_block = build_example_block(
            example_docs, max_tokens=500)
        prompt = prompt_template.format(
            query=query,
            context=context,
            example=example_block,
            certification_type=ct
        )
    else:
        prompt_template = prompt_v4.prompt_zeroshot
        prompt = prompt_template.format(
            query=query,
            context=context,
            certification_type=ct
        )

    logging.info(f"프롬프트 구성 완료 (길이: {len(prompt)}자)")
    start_gen = time.time()
    output_text = model.invoke(prompt)  # ChatGroq 기반
    generation_time = time.time() - start_gen
    logging.info(f"응답 생성 시간: {generation_time:.2f}초")

    return output_text.content.strip(), generation_time


def run_rag_pipeline(prompt_version="v4-zeroshot"):
    logging.info("[START] groq 기반 RAG 파이프라인 실행")

    model = groq_loader()
    embeddings_model = vectorStore.KoSimCSE()
    base_retriever = vectorStore.load_or_create_faiss_rerank(embeddings_model)
    reranker = vectorStore.KoreanReranker(base_retriever)
    retriever = reranker.compression_retriever

    test_input = load_data()
    results = []

    for idx, row in test_input.iterrows():
        article = str(row.get("compressed_article", "")).strip()
        ct = row.get("ct", "없음")

        if not article:
            logging.warning(f"[{idx}] 빈 기사 내용 → 건너뜀")
            continue

        logging.info(f"[{idx + 1}/{len(test_input)}] 기사 처리 시작")
        logging.info(f"기사 내용: {article}")

        start_retrieve = time.time()
        docs = vectorStore.search_with_score_filter(
            retriever=retriever.base_retriever,
            query=article,
            min_score=0.75,
            k=5
        )
        retriever_time = time.time() - start_retrieve

        context_list = [doc.page_content for doc in docs]
        for i, doc in enumerate(docs, 1):
            logging.info(f"  문서{i}: {doc.page_content[:80]}...")

        news_retriever = None
        if prompt_version == "v4-oneshot":
            logging.info("뉴스 DB에서 예시 검색기 로드")
            news_store = vectorStore.load_or_create_faiss_news(
                embeddings_model)
            news_retriever = news_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 2})

        answer, gen_time = generate_answer_groq(
            model=model,
            query=article,
            context="\n".join(context_list),
            ct=ct,
            prompt_version=prompt_version,
            rt_n=news_retriever
        )

        logging.info(f"[답변 생성 완료] {gen_time:.2f}초 소요")
        logging.info(f"[답변 요약]: {answer[:150]}...")

        results.append({
            "article": article,
            "context": context_list,
            "answer": answer,
            "retriever_time": round(retriever_time, 3),
            "generate_time": round(gen_time, 3),
            "reason_summary": row.get("reason_summary", "")
        })

    logging_result(results)
    logging_model(
        model_name="llama3-8b-8192 (groq)",
        embeddings_model="KoSimCSE",
        retriever_strategy="rerank",
        num_articles=len(test_input),
        prompt_version=prompt_version
    )
    save_and_evaluate_results(
        results=results,
        test_input_df=test_input,
        filename=f"groq_rerank_{prompt_version}"
    )
    logging.info("[END] 전체 처리 완료")


def run_single_text(article: str, ct: str = "없음", prompt_version="v4-zeroshot"):
    logging.info("[START] 단일 기사 처리 시작")

    model = groq_loader()
    embeddings_model = vectorStore.KoSimCSE()
    base_retriever = vectorStore.load_or_create_faiss_rerank(embeddings_model)
    reranker = vectorStore.KoreanReranker(base_retriever)
    retriever = reranker.compression_retriever

    article = article.strip()
    if not article:
        logging.warning("빈 기사 내용 입력됨 → 종료")
        return ""

    logging.info(f"기사 내용: {article}")

    start_retrieve = time.time()
    docs = vectorStore.search_with_score_filter(
        retriever=retriever.base_retriever,
        query=article,
        min_score=0.75,
        k=5
    )
    retriever_time = time.time() - start_retrieve

    context_list = [doc.page_content for doc in docs]
    for i, doc in enumerate(docs, 1):
        logging.info(f"  문서{i}: {doc.page_content[:80]}...")

    news_retriever = None
    if prompt_version == "v4-oneshot":
        logging.info("뉴스 DB에서 예시 검색기 로드")
        news_store = vectorStore.load_or_create_faiss_news(embeddings_model)
        news_retriever = news_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 2})

    answer, gen_time = generate_answer_groq(
        model=model,
        query=article,
        context="\n".join(context_list),
        ct=ct,
        prompt_version=prompt_version,
        rt_n=news_retriever
    )

    logging.info(f"[단일 응답 완료] {gen_time:.2f}초 소요")
    logging.info(f"[응답 내용]: {answer}")

    return answer


def load_data(rows=None, legalize: bool = True):
    # 파일 경로 확인
    if not os.path.exists(test_input_c_path):
        logging.info(f"입력 파일 '{test_input_path}'을(를) 불러옵니다.")
        try:
            df = pd.read_csv(test_input_path)
            logging.info(f"CSV 파일 로드 성공: {df.shape[0]}개의 행, {df.shape[1]}개의 열")
            model = groq_loader()

            df['compressed_article'] = compress_article(
                df['full_text'].tolist(), model)
            df['ct'] = df['en_mark'].apply(normalize_cert_type)
            df.to_csv(test_input_c_path, index=False)

        except Exception as e:
            logging.exception("CSV 파일을 불러오는 중 오류 발생")
            raise

    # data 뽑기
    test_input = pd.read_csv(test_input_c_path, encoding='utf-8')
    negative_samples = test_input[test_input["greenwashing_level"] == 0]
    selected_negatives = negative_samples.sample(
        n=min(15, len(negative_samples)), random_state=42)

    remaining = 30 - len(selected_negatives)
    if remaining > 0:
        remaining_samples = test_input.drop(
            selected_negatives.index).sample(n=remaining, random_state=42)
        test_input = pd.concat(
            [selected_negatives, remaining_samples], ignore_index=True)
    else:
        test_input = selected_negatives.reset_index(drop=True)

    logging.info(f"[테스트 데이터 로드] {len(test_input)}개 문장 로드")

    test_input = query_processor.preprocess_articles(test_input, legalize)
    if not test_input.empty:
        logging.info(
            f"[확인] {test_input['compressed_article'].iloc[0][:1000]} {test_input['ct'].iloc[0]}")
    else:
        logging.warning("⚠ 처리된 문서가 없습니다!")

    return test_input


def compress_article(news_list, model=None):
    # 모델 로드하기 (llama3ko)
    try:
        if model is None:
            logging.info("기사 압축에 사용할 LLM 모델을 로드합니다.")
            model = groq_loader()
            logging.info("모델 로드 완료")
    except Exception as e:
        logging.exception("모델 로드 중 오류 발생")
        raise

    # 프롬프트 불러오기 (기사 압축 프롬프트)
    prompt_template = prompt_compression.base_prompt
    logging.info("프롬프트 로드 완료")
    compressed_results = []

    for news in news_list:
        if pd.isna(news):
            compressed_results.append("기사 내용 없음")
            continue

        if len(news) < 700:
            compressed_results.append(news)
            continue

        try:
            logging.info("기사 압축 시작")
            prompt = prompt_template.format(news=news)
            compressed = model.invoke(prompt)
            compressed_results.append(extract_summary_only(compressed))
        except Exception as e:
            logging.warning(f"기사 압축 실패: {e}")
            compressed_results.append("압축 실패")

    return compressed_results


def extract_summary_only(full_output: str) -> str:
    """
    전체 출력 중 '[요약된 환경 관련 핵심 문장]' 이후 텍스트만 추출
    """
    split_token = "[요약된 환경 관련 핵심 문장]assistant"
    if split_token in full_output:
        return full_output.split(split_token, 1)[-1].strip()
    return full_output.strip()


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


if __name__ == "__main__":
    run_rag_pipeline(prompt_version="v4-oneshot")
