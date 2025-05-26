import pandas as pd
from query_processor import split_article_by_token_limit, decide_final_judgement
import logging
import torch
import sys
import re

import get_data
import model_loader
import query_processor
import vectorStore
from prompts import prompt_v3_cot_fewshot
from save_data import save_results

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
        return prompt_v3_cot_fewshot.base_prompt
    elif version == "fewshot":
        return prompt_v3_cot_fewshot.fewShot_prompt
    elif version == "oneshot":
        return prompt_v3_cot_fewshot.one_shot_example_template


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


def generate_answer(model, tokenizer, query, context, prompt_version="fewshot"):
    prompt_template = load_prompt(prompt_version)
    prompt = prompt_template.format(query=query, context=context)
    # logging.info(f"[프롬프트] 설정 확인: {prompt[:800]}")

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,  # 단문 자동 패딩 → 길이 맞추기
        truncation=True,  # 장문 자동 잘라내기
        max_length=tokenizer.model_max_length,  # 최대 길이 설정
    ).to(model.device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

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

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text[len(prompt):].strip()


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
        guideline_store = vectorStore.load_or_create_faiss_guideline(
            embeddings_model)
        news_store = vectorStore.load_or_create_faiss_news(embeddings_model)
        rt_n = news_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2}
        )

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

        if not isinstance(raw_article, str):
            logging.warning(
                f"[{idx}] compressed_article가 str이 아님: {type(raw_article)} → 건너뜀")
            continue

        article = raw_article.strip()

        logging.info(f"[기사 처리 시작] {idx + 1} / {len(test_input)}")
        logging.info(f"[처리 기사 내용] {article}")
        logging.info(f"[가이드라인 검색 + 쿼리 임베딩]")
        guideline = vectorStore.search_with_score_filter(
            rt_g, article, min_score=0.75)
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

        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            query=article,
            context=context,
            prompt_version=prompt_version
        )
        logging.info(f"[답변 생성] {answer}")
        logging.info(f"[답변 종료]")

        results.append({
            "article": article,
            "context": context,
            "answer": answer
        })

    logging_result(results)
    save_results(results,
                 test_input_df=test_input,
                 filename=f"llama3Ko_{version}_{prompt_version}"
                 )
    return test_input


def llama3Ko(version="double", prompt_version="fewshot", news_mix=False):
    logging.info(
        f"llama3Ko + {version} retriever + {prompt_version} 사용한 그린워싱 판별 시작")

    # 모델 불러오기
    model, tokenizer, rt_g, rt_l, rt_n = run_experiment(
        model_name="llama3Ko",
        embeddings_model=embeddings_model,
        embeddings_model_name="KoSimCSE",
        search_strategy="double",
        news_mix=news_mix,
    )

    test_input = run_model(
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
        num_articles=len(test_input),
        prompt_version=prompt_version
    )

    logging.info(
        f"llama3Ko + {version} retriever + {prompt_version} 사용한 그린워싱 판별 종료")


def double_llama3Ko_not_legalize():
    logging.info("llama3Ko 모델과 double retriever을 사용한 그린워싱 판별 시작 + 법률 용어화 안함")

    # 모델 불러오기
    model, tokenizer, rt_g, rt_l, rt_n = run_experiment(
        model_name="llama3Ko",
        embeddings_model=embeddings_model,
        embeddings_model_name="KoSimCSE",
        search_strategy="double"
    )

    results = []
    test_input = get_data.load_data(legalize=False)

    for idx, row in test_input.iterrows():
        raw_article = row['full_text']

        if not isinstance(raw_article, str):
            logging.warning(
                f"[{idx}] full_text가 str이 아님: {type(raw_article)} → 건너뜀")
            continue

        article = raw_article.strip()

        logging.info(f"[기사 처리 시작] {idx + 1} / {len(test_input)}")
        logging.info(f"[처리 기사 내용] {article}")
        logging.info(f"[가이드라인 검색 + 쿼리 임베딩]")
        guideline = rt_g.invoke(article)
        context = "\n".join([doc.page_content for doc in guideline])[:1000]

        logging.info(f"[1차 검색된 가이드라인 문서]")
        for i, doc in enumerate(guideline, 1):
            logging.info(f"  [{i}] {doc.page_content}...")

        # todo: 만약에 guideline 내부에 비슷한 문장이 있으면, 법률 검색? 근데 유사도 검색하면 일단은 1개 이상은 있는 거 아닌가
        if len(guideline) > 0:
            logging.info(f"[그린워싱 가능성 존재]")
            logging.info(f"[법률 검색 + 쿼리 임베딩]")
            law = rt_l.invoke(article)
            context = "\n".join([doc.page_content for doc in law])[:1000]

            logging.info(f"[2차 검색된 법률 문서]")
            for i, doc in enumerate(law, 1):
                logging.info(f"  [{i}] {doc.page_content}...")

        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            query=article,
            context=context,
        )
        logging.info(f"[답변 생성] {answer}")
        logging.info(f"[답변 종료]")

        results.append({
            "article": article,
            "context": context,
            "answer": answer
        })

    logging_result(results)
    logging_model(
        model_name="llama3Ko",
        embeddings_model="KoSimCSE",
        retriever_strategy="double",
        num_articles=len(test_input),
        prompt_version="v3_cot_fewshot"
    )

    logging.info("llama3Ko 모델과 double retriever을 사용한 그린워싱 판별 종료 + 법률 용어화 안함")


def rerank_llama3Ko(prompt_version="fewshot"):
    logging.info("llama3Ko 모델과 rerank retriever을 사용한 그린워싱 판별 시작")

    # 모델 불러오기
    model, tokenizer = model_loader.load_model("llama3Ko")
    tokenizer.pad_token = tokenizer.eos_token

    retriever = reranker.compression_retriever

    results = []
    test_input = get_data.load_data()

    for idx, row in test_input.iterrows():
        raw_article = row['compressed_article']

        article = raw_article.strip()

        logging.info(f"[기사 처리 시작] {idx + 1} / {len(test_input)}")
        logging.info(f"[처리 기사 내용] {article}")

        logging.info(f"[guideline&law 문서 검색 + 임베딩]")
        all_docs = retriever.invoke(article)
        context = "\n".join([doc.page_content for doc in all_docs])[:1000]

        logging.info(f"[검색된 문서]")
        for i, doc in enumerate(all_docs, 1):
            logging.info(f"  [{i}] {doc.page_content}...")

        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            query=article,
            context=context,
            prompt_version=prompt_version
        )
        logging.info(f"[답변 생성] {answer}")
        logging.info(f"[답변 종료]")

        results.append({
            "article": article,
            "context": context,
            "answer": answer
        })

    logging_result(results)
    logging_model(
        model_name="llama3Ko",
        embeddings_model="BgeReranker",
        retriever_strategy="rerank",
        num_articles=len(test_input),
        prompt_version=prompt_version
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


def llama3Ko_article_level(prompt_version="fewshot"):
    model, tokenizer = model_loader.load_model("llama3Ko")
    tokenizer.pad_token = tokenizer.eos_token
    retriever = reranker.compression_retriever

    test_input = get_data.load_data()
    final_results = []

    for idx, row in test_input.iterrows():
        raw_article = row['full_text']
        if not isinstance(raw_article, str):
            continue

        logging.info(f"[기사 처리 시작] {idx + 1} / {len(test_input)}")
        logging.info(
            f"[처리 기사 내용] {raw_article[:1000]}, 길이: {len(raw_article)}자")
        article = raw_article.strip()

        if len(article) >= 1024:
            logging.info(f"[기사 길이] {len(article)}자 → 기사 처리")
            article = query_processor.extract_relevant_sentences(article)

        chunks = split_article_by_token_limit(article, tokenizer)
        chunk_answers = []

        for c_idx, chunk in enumerate(chunks):
            logging.info(
                f"[{idx + 1}-{c_idx + 1}/{len(chunks)}] chunk 처리 중 (길이: {len(chunk)}자)")

            context_docs = retriever.invoke(chunk)
            context = "\n".join(
                [doc.page_content for doc in context_docs])[:1000]

            answer = generate_answer(
                model, tokenizer, chunk, context, prompt_version=prompt_version)
            logging.info(f"[답변 생성] {answer}")
            logging.info(f"[답변 종료]")

            chunk_answers.append({
                "chunk_idx": c_idx + 1,
                "chunk": chunk,
                "answer": answer
            })

        final_judgement = decide_final_judgement(chunk_answers)
        reason, law, suggestion = summarize_answers(chunk_answers)
        full_answer = f"1. 판단: {final_judgement}\n2. 근거: {reason}\n3. 법률: {law}\n4. 해결방안: {suggestion}"

        final_results.append({
            "article_idx": idx + 1,
            "article": article,
            "final_judgement": final_judgement,
            "reason_summary": reason,
            "law_summary": law,
            "suggestion_summary": suggestion,
            "answer": full_answer
        })

    save_results(
        final_results,
        test_input_df=test_input,
        filename=f"llama3Ko_article_level_{prompt_version}"
    )


if __name__ == "__main__":
    # llama3Ko(version="double", prompt_version="fewshot")
    # llama3Ko(version="double", prompt_version="base")
    # llama3Ko(version="single", prompt_version="fewshot")
    # llama3Ko(version="single", prompt_version="base")
    # double_llama3Ko_not_legalize()

    rerank_llama3Ko(prompt_version="oneshot")
    rerank_llama3Ko(prompt_version="base")

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
