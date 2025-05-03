import logging
import torch

import get_data
import model_loader
import query_processor
import vectorStore
from prompts import prompt_v3_cot_fewshot

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_prompt():
    # todo: langchain prompt 사용해보기
    return prompt_v3_cot_fewshot.fewShot_prompt

def logging_model(model_name, embeddings_model, retriever_strategy, num_articles, prompt_version):
    logging.info(f"[실험 환경]")
    logging.info(f"  모델: {model_name}")
    logging.info(f"  쿼리 임베딩 모델: {embeddings_model}")
    logging.info(f"  검색 전략: {retriever_strategy}")
    logging.info(f"  검색 문서 수: {num_articles}")
    logging.info(f"  프롬프트 버전: {prompt_version}")


def generate_answer(model, tokenizer, query, context):
    prompt = load_prompt().invoke({"query": query, "context": context})
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,  # 단문 자동 패딩 → 길이 맞추기
        truncation=True,  # 장문 자동 잘라내기
        max_length=4096,  # 최대 길이 설정
    ).to(model.device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            temperature=0.5,
            top_p=0.8,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text[len(prompt):].strip()


def run_experiment(model_name, search_strategy: str = "double", num_articles: int = 10):
    # 0. 초기화
    # 0.1. LLM 모델
    model, tokenizer = model_loader.load_model(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # padding 문제 방지

    # 0.2. 쿼리 및 문서 임베딩 모델
    embeddings_model = vectorStore.KoSimCSE()
    guideline_store = vectorStore.load_or_create_faiss_guideline(embeddings_model)

    if search_strategy == "double":
        law_store = vectorStore.load_or_create_faiss_law(embeddings_model)
        retriever_2nd = law_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
    else:
        retriever_2nd = None

    retriever_1st = guideline_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    return model, tokenizer, retriever_1st, retriever_2nd


def load_data():
    # get input data 10개
    test_input = get_data.load_test_data(5)
    test_input = test_input.dropna(subset=["full_text"])

    logging.info(f"[테스트 데이터 로드] {len(test_input)}개 문장 로드")

    test_input = query_processor.preprocess_articles(test_input)

    if test_input:
        logging.info(f"[확인] {test_input[0][:1000]}")
    else:
        logging.warning("⚠ 처리된 문서가 없습니다!")

    query = "이 기사 전체가 그린워싱에 해당합니까?"
    return test_input, query


def stepR_llama3Ko():
    logging.info("llama3Ko 모델을 사용한 그린워싱 판별 시작")

    # 모델 불러오기
    model, tokenizer, retriever_1st, retriever_2nd = run_experiment(
        model_name="llama3Ko",
        search_strategy="double"
    )

    results = []
    test_input, query = load_data()

    for idx, row in test_input.iterrows():
        raw_article = row['full_text']

        if not isinstance(raw_article, str):
            logging.warning(
                f"[{idx}] full_text가 str이 아님: {type(raw_article)} → 건너뜀")
            continue

        article = raw_article.strip()

        logging.info(f"[기사 처리 시작]")
        logging.info(f"[가이드라인 검색 + 쿼리 임베딩]")
        guideline = retriever_1st.invoke(article)
        context = "\n".join([doc.page_content for doc in guideline])[:1000]

        logging.info(f"[1차 검색된 가이드라인 문서]")
        for i, doc in enumerate(guideline, 1):
            logging.info(f"  [{i}] {doc.page_content}...")

        # todo: 만약에 guideline 내부에 비슷한 문장이 있으면, 법률 검색? 근데 유사도 검색하면 일단은 1개 이상은 있는 거 아닌가
        if len(guideline) > 0:
            logging.info(f"[그린워싱 가능성 존재]")
            logging.info(f"[법률 검색 + 쿼리 임베딩]")
            law = retriever_2nd.invoke(article)
            context = "\n".join([doc.page_content for doc in law])[:1000]

            logging.info(f"[2차 검색된 법률 문서]")
            for i, doc in enumerate(law, 1):
                logging.info(f"  [{i}] {doc.page_content}...")

        answer = generate_answer(model, tokenizer, query, context)
        logging.info(f"[답변 생성] {answer}")

        results.append({
            "article": article,
            "context": context,
            "answer": answer
        })

    for r in results:
        print("="*80)
        print(f"Article Preview: {r['article'][:200]}...")
        print(f"Context Preview: {r['context'][:200]}...")
        print(f"Answer:\n{r['answer']}")
        print("="*80)

    logging_model(
        model_name="llama3Ko",
        embeddings_model="KoSimCSE",
        retriever_strategy="double",
        num_articles=len(test_input),
        prompt_version="v3_cot_fewshot"
    )

if __name__ == "__main__":
    stepR_llama3Ko()
