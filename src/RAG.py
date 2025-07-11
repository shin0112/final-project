import pandas as pd
from query_processor import split_article_by_token_limit, decide_final_judgement
import logging
import torch
import sys
import re
import time

import get_data
import model_loader
import query_processor
import vectorStore
from prompts import prompt_v3_cot_fewshot, prompt_v4, prompt_v4_cot_news_oneshot
from save_data import save_and_evaluate_results

# 모델 일괄 초기화
embeddings_model = vectorStore.KoSimCSE()
base_retriever = vectorStore.load_or_create_faiss_rerank(
    embeddings_model=embeddings_model
)
reranker = vectorStore.KoreanReranker(base_retriever)

# Configure logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])


def load_prompt(version="fewshot"):
    if version == "base":
        prompt = prompt_v3_cot_fewshot.base_prompt
    elif version == "fewshot":
        prompt = prompt_v3_cot_fewshot.fewShot_prompt
    elif version == "oneshot":
        prompt = prompt_v3_cot_fewshot.one_shot_example_template
    elif version[:10] == "v4-oneshot":
        prompt = prompt_v4.prompt_oneshot
    elif version == "v4-zeroshot":
        prompt = prompt_v4.prompt_zeroshot

    logging.info(f"[프롬프트 전문] {prompt}")
    return prompt


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


def generate_answer(model, tokenizer, query, context, ct="", prompt_version="fewshot", rt_n=None):
    MAX_TOTAL_TOKENS = 2048
    MAX_CONTEXT_TOKENS = 800
    MAX_EXAMPLE_TOKENS = 500
    prompt_template = load_prompt(prompt_version)

    if prompt_version[:2] == "v4":
        # 예시 문서 가져오기 - 문서 1개
        if prompt_version[:10] == "v4-oneshot":
            example_docs = rt_n.vectorstore.similarity_search(query, k=1)
            logging.info(f"[뉴스 예시 문서] {example_docs[0].page_content[:200]}...")
            # 예시, 가이드라인 문서 블록 만들고 토큰 단위로 자르기
            example_block = query_processor.build_example_block(
                example_docs, tokenizer, max_tokens=MAX_EXAMPLE_TOKENS)
        context_block = query_processor.truncate_context(
            context, tokenizer, max_tokens=MAX_CONTEXT_TOKENS)

        if prompt_version[:10] == "v4-oneshot":
            prompt = prompt_template.format(
                query=query,
                context=context_block,
                example=example_block,
                certification_type=ct
            )
        else:
            prompt = prompt_template.format(
                query=query,
                context=context_block,
                certification_type=ct
            )

    else:
        prompt = prompt_template.format(query=query, context=context)
    # logging.info(f"[프롬프트] 설정 확인: {prompt[:800]}")
    logging.info(f"[프롬프트 전문] {prompt}")
    tokens = tokenizer.tokenize(prompt)
    if len(tokens) > MAX_TOTAL_TOKENS:
        logging.warning(
            f"Prompt is too long: {len(tokens)} tokens. Truncating to {MAX_TOTAL_TOKENS}.")
        tokens = tokens[:MAX_TOTAL_TOKENS]
        prompt = tokenizer.convert_tokens_to_string(tokens)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,  # 단문 자동 패딩 → 길이 맞추기
        truncation=True,  # 장문 자동 잘라내기
        max_length=MAX_TOTAL_TOKENS,  # 최대 길이 설정
    ).to(model.device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    start_gen = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_gen
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return output_text[len(prompt):].strip(), generation_time


def run_experiment(model_name,
                   embeddings_model,
                   embeddings_model_name="KoSimCSE",
                   search_strategy: str = "double",
                   num_articles: int = 10,
                   news_mix=False
                   ):
    # 0. 초기화
    # 0.1. LLM 모델
    model, tokenizer = model_loader.load_model(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # padding 문제 방지

    # 0.2. 쿼리 및 문서 임베딩 모델
    embeddings_model = embeddings_model
    if news_mix:
        guideline_store = vectorStore.load_or_create_faiss_guideline_and_news(
            embeddings_model)
        rt_n = None
    else:
        if search_strategy == "double":
            guideline_store = vectorStore.load_or_create_faiss_guideline(
                embeddings_model)
            news_store = vectorStore.load_or_create_faiss_guideline_example(
                embeddings_model)
            rt_n = news_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )
        else:
            guideline_store = base_retriever
            rt_n = None

    if search_strategy == "double":
        law_store = vectorStore.load_or_create_faiss_law(embeddings_model)
        rt_l = law_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )
    else:
        rt_l = None

    rt_g = guideline_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )

    # r_g = 가이드라인, r_l: 법률, r_n: 뉴스
    return model, tokenizer, rt_g, rt_l, rt_n


def run_model(model,
              tokenizer,
              rt_g,
              rt_l,
              rt_n,
              version="double",
              prompt_version="fewshot",
              news_mix=False
              ):
    results = []
    test_input = get_data.load_data()

    for idx, row in test_input.iterrows():
        raw_article = row['compressed_article']
        ct = row['ct']

        if not isinstance(raw_article, str):
            logging.warning(
                f"[{idx}] compressed_article가 str이 아님: {type(raw_article)} → 건너뜀")
            continue

        article = raw_article.strip()

        logging.info(f"[기사 처리 시작] {idx + 1} / {len(test_input)}")
        logging.info(f"[처리 기사 내용] {article}")

        logging.info(f"[가이드라인 검색 + 쿼리 임베딩]")
        start_retrieval = time.time()

        guideline = vectorStore.search_with_score_filter(
            retriever=rt_g,
            query=article,
            min_score=0.75
        )

        retriever_time = time.time() - start_retrieval
        logging.info(f"[문서 검색 완료] 소요 시간: {retriever_time:.2f}초")

        context = "\n".join([doc.page_content for doc in guideline])[:1000]

        logging.info(f"[1차 검색된 가이드라인 문서]")
        for i, doc in enumerate(guideline, 1):
            logging.info(f"  [{i}] {doc.page_content}...")

        # todo: 만약에 guideline 내부에 비슷한 문장이 있으면, 법률 검색? 근데 유사도 검색하면 일단은 1개 이상은 있는 거 아닌가
        if version == "double" and len(guideline) > 0:
            logging.info(f"[그린워싱 가능성 존재]")
            logging.info(f"[법률 검색 + 쿼리 임베딩]")
            law = vectorStore.search_with_score_filter(
                rt_l, article, min_score=0.75)
            context = "\n".join([doc.page_content for doc in law])[:1000]

            logging.info(f"[2차 검색된 법률 문서]")
            for i, doc in enumerate(law, 1):
                logging.info(f"  [{i}] {doc.page_content}...")

        if rt_n is not None:
            logging.info("[3차 검색 뉴스]")
            news = vectorStore.search_with_score_filter(
                rt_n, article, min_score=0.75)
            context = "\n".join([doc.page_content for doc in news])[:1000]

        answer, generate_time = generate_answer(
            model=model,
            tokenizer=tokenizer,
            query=article,
            context=context,
            ct=ct,
            prompt_version=prompt_version
        )
        logging.info(f"[답변 생성] {answer}")
        logging.info(f"[답변 종료]")

        results.append({
            "article": article,
            "context": context,
            "answer": answer,
            "retriever_time": round(retriever_time, 3),
            "generate_time": round(generate_time, 3),
            "reason_summary": row['reason_summary'],
        })

    logging_result(results)

    return test_input, results


def llama3Ko(version="double", prompt_version="fewshot", news_mix=False):
    logging.info(
        f"llama3Ko + {version} retriever + {prompt_version} 사용한 그린워싱 판별 시작")

    # 모델 불러오기
    model, tokenizer, rt_g, rt_l, rt_n = run_experiment(
        model_name="llama3Ko",
        embeddings_model=embeddings_model,
        embeddings_model_name="KoSimCSE",
        search_strategy=version,
        news_mix=news_mix,
    )

    test_input, results = run_model(
        model=model,
        tokenizer=tokenizer,
        rt_g=rt_g,
        rt_l=rt_l,
        rt_n=rt_n,
        version=version,
        prompt_version=prompt_version,
        news_mix=news_mix,
    )

    logging_model(
        model_name="llama3Ko",
        embeddings_model="KoSimCSE",
        retriever_strategy=version,
        num_articles=len(results),
        prompt_version=prompt_version
    )

    save_and_evaluate_results(
        results=results,
        test_input_df=test_input,
        filename=f"llama3Ko_{version}_{prompt_version}"
    )
    logging.info(
        f"llama3Ko + {version} retriever + {prompt_version} 사용한 그린워싱 판별 종료")


def rerank_llama3Ko(prompt_version="v4-zeroshot"):
    logging.info("llama3Ko 모델과 rerank retriever을 사용한 그린워싱 판별 시작")

    # 모델 불러오기
    model, tokenizer = model_loader.load_model("llama3Ko")
    tokenizer.pad_token = tokenizer.eos_token

    retriever = reranker.compression_retriever

    results = []
    test_input = get_data.load_data()

    for idx, row in test_input.iterrows():
        raw_article = row['compressed_article']
        ct = row['ct']

        article = raw_article.strip()

        logging.info(f"[기사 처리 시작] {idx + 1} / {len(test_input)}")
        logging.info(f"[처리 기사 내용] {article}")

        logging.info(f"[guideline&law 문서 검색 + 임베딩]")
        start_retrieval = time.time()

        all_docs = vectorStore.search_with_score_filter(
            retriever=retriever.base_retriever,
            query=article,
            min_score=0.75,
            k=5
        )

        retriever_time = time.time() - start_retrieval
        logging.info(f"[문서 검색 완료] 소요 시간: {retriever_time:.2f}초")

        context_list = [doc.page_content for doc in all_docs]

        logging.info(f"[검색된 문서]")
        for i, doc in enumerate(all_docs, 1):
            logging.info(f"  [{i}] {doc.page_content}...")

        rt_n = None
        if prompt_version == "v4-oneshot":
            # 유사도 검색으로 뉴스 문서 가져오기
            logging.info(f"[가이드라인 뉴스 데이터 검색]")
            news_store = vectorStore.load_or_create_faiss_guideline_example(
                embeddings_model)
            rt_n = news_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )
        elif prompt_version == "v4-oneshot-test":
            logging.info(f"[수집 뉴스 데이터 검색]")
            news_store = vectorStore.load_or_create_faiss_news_example(
                embeddings_model)
            rt_n = news_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2}
            )

        answer, generate_time = generate_answer(
            model=model,
            tokenizer=tokenizer,
            query=article,
            context=context_list,
            ct=ct,
            prompt_version=prompt_version,
            rt_n=rt_n
        )
        logging.info(f"[답변 생성] {answer}")
        logging.info(f"[답변 종료]")

        results.append({
            "article": article,
            "context": context_list,
            "answer": answer,
            "retriever_time": round(retriever_time, 3),
            "generate_time": round(generate_time, 3),
            "reason_summary": row['reason_summary'],
        })

    logging_result(results)
    logging_model(
        model_name="llama3Ko",
        embeddings_model="BgeReranker",
        retriever_strategy="rerank",
        num_articles=len(test_input),
        prompt_version=prompt_version
    )

    save_and_evaluate_results(
        results=results,
        test_input_df=test_input,
        filename=f"llama3Ko_rerank_{prompt_version}"
    )
    logging.info("llama3Ko 모델과 rerank retriever을 사용한 그린워싱 판별 종료")


def summarize_answers(answers):
    # 각 판단 요소를 모아 요약
    reasons = []
    laws = []
    suggestions = []
    for a in answers:
        match = re.search(
            r"1\. 판단:\s*(.*?)\n2\. 근거:\s*(.*?)\n3\. 법률:\s*(.*?)\n4\. 해결방안:\s*(.*)",
            a['answer'],
            re.DOTALL
        )
        if match:
            _, reason, law, suggestion = match.groups()
            reasons.append(reason.strip())
            laws.append(law.strip())
            suggestions.append(suggestion.strip())
    reason_summary = " / ".join(set(reasons))
    law_summary = " / ".join(set(laws))
    suggestion_summary = " / ".join(set(suggestions))
    return reason_summary, law_summary, suggestion_summary


if __name__ == "__main__":
    logging.info("""
                [실험 간단 설명] double 검색 v4-zeroshot 테스트 데이터 확정
                """)

    # llama3Ko(version="double", prompt_version="fewshot")
    # llama3Ko(version="double", prompt_version="base")
    # llama3Ko(version="single", prompt_version="fewshot")
    llama3Ko(version="double", prompt_version="v4-zeroshot")
    # double_llama3Ko_not_legalize()

    # rerank_llama3Ko(prompt_version="v4-oneshot-test")
    # rerank_llama3Ko(prompt_version="v4-oneshot")

    # llama3Ko_article_level(prompt_version="oneshot")
    # llama3Ko_article_level(prompt_version="base")

    # logging.info("유사도 평가 도입")
    # logging.info("[RUN] double + base (news 분리)")
    # llama3Ko(version="double", prompt_version="base", news_mix=False)

    # logging.info("[RUN] single + base (news 분리)")
    # llama3Ko(version="single", prompt_version="base", news_mix=False)

    # # 통합형 실험
    # logging.info("[RUN] double + base (news 통합)")
    # llama3Ko(version="double", prompt_version="base", news_mix=True)

    # logging.info("[RUN] single + base (news 통합)")
    # llama3Ko(version="single", prompt_version="base", news_mix=True)
