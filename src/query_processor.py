import re
import logging
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

KEYWORDS = [
    "친환경", "지속 가능", "재활용", "탄소 중립", "인증", "에코", "그린", "지속 가능한",
    "환경 보호", "자원 절약", "친환경 소재", "친환경 제품", "지속 가능한 발전"
]

# todo: 더 자세하게 채우기
ENVIRONMENT_TERM_MAPPING = {
    "100% 친환경 소재": "전량 친환경 인증을 받은 소재",
    "탄소 배출 최소화": "온실가스 배출 저감 목표",
    "지속가능한 미래": "지속 가능한 발전 목표",
    "생분해성 플라스틱": "생분해성 인증을 받은 소재",
    "무독성": "인체에 무해한 인증 기준 만족",
    "친환경 제품": "환경 인증 기준 충족 제품",
}


def preprocess_articles(test_input, legalize=True):
    processed = []

    for idx, row in test_input.iterrows():
        # 전문 가져오기
        raw_article = row.get("compressed_article", "")
        # string인지 확인
        if not isinstance(raw_article, str) or not raw_article.strip():
            logging.warning(
                f"[{idx}] compressed_article가 str이 아님: {type(raw_article)} → 건너뜀")
            continue
        # 법률 용어화
        if legalize:
            raw_article = legalize_query([raw_article])[0]
        # 공백 제거
        processed.append({"compressed_article": raw_article})

    logging.info(f"[기사 전처리 완료] {len(processed)}개 문서 처리됨")
    return pd.DataFrame(processed)


def legalize_query(sentences: list[str]) -> list[str]:
    """
    법률 용어와 매칭하기 위한 자연어 처리
    """
    new_sentences = []
    for s in sentences:
        for keyword, legal_term in ENVIRONMENT_TERM_MAPPING.items():
            if keyword in s:
                s = s.replace(keyword, legal_term)
        new_sentences.append(s)
    return new_sentences


def naturalize_query(sentences) -> str:
    result = [f"'{s}'라는 문장은 그린워싱 가능성이 있는 표현인가요?" for s in sentences]
    logging.info(f"[자연어 처리] 다음 query 생성")
    for i, q in enumerate(result, 1):
        logging.info(f"{i}. {q}")
    return result


def extract_key_sentences(text: str, max_sentences: int = 5):
    # 문장 단위 분리
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # 키워드 점수 계산 및 정렬
    scores = [(s, sum(k in s for k in KEYWORDS)) for s in sentences]
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [sentence for sentence,
                score in scores if score > 0][:max_sentences]

    logging.info(f"[문장 추출] 다음 문장들이 추출됨. 문장 수 = {len(selected)}")
    for i, s in enumerate(selected, 1):
        logging.info(f"{i}. {s}")

    return selected


def split_article_by_token_limit(article, tokenizer, max_tokens=1024):
    sentences = re.split(r'(?<=[.!?])\s+', article.strip())
    chunks = []
    current_chunk = ""
    current_token_count = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        if current_token_count + len(tokens) > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_token_count = len(tokens)
        else:
            current_chunk += " " + sentence
            current_token_count += len(tokens)

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def decide_final_judgement(answers):
    num_yes = sum("그린워싱 있음" in a['answer'] for a in answers)
    num_no = sum("그린워싱 없음" in a['answer'] for a in answers)

    if num_yes > num_no:
        return "그린워싱 있음"
    elif num_no > num_yes:
        return "그린워싱 없음"
    else:
        return "판단 유보"


def postprocess_rerank_results(results):
    """
    RAG 결과에서 chunk별 응답을 받아 기사 단위 판단으로 종합하고 상세 판단 결과를 반환
    """
    grouped = defaultdict(list)
    for r in results:
        grouped[r['article_idx']].append(r)

    final_outputs = []
    for article_idx, chunks in grouped.items():
        final_judgement = decide_final_judgement(chunks)
        detailed = []
        for r in chunks:
            chunk_text = r["chunk"]
            match = re.search(
                r"1\. 판단:\s*(.*?)\n2\. 근거:\s*(.*?)\n3\. 법률:\s*(.*?)\n4\. 해결방안:\s*(.*)",
                r["answer"],
                re.DOTALL
            )
            if match:
                judgement, reason, law, suggestion = match.groups()
            else:
                judgement = reason = law = suggestion = "파싱 오류"
            detailed.append({
                "article_idx": article_idx,
                "chunk_idx": r["chunk_idx"],
                "chunk_text": chunk_text,
                "judgement": judgement,
                "reason": reason,
                "law": law,
                "suggestion": suggestion,
                "final_judgement": final_judgement
            })
        final_outputs.extend(detailed)

    return pd.DataFrame(final_outputs)


def extract_relevant_sentences(article):
    """
    기사에서 그린워싱 관련 문장 추출
    """
    sentences = re.split(r'(?<=[.!?])\s+|\n+', article.strip())
    filtered = [
        s for s in sentences
        if is_environment_related(s)
    ]
    return " ".join(filtered)


def is_environment_related(sentence):
    has_keyword = any(keyword in sentence for keyword in KEYWORDS)
    has_number = bool(re.search(r'\d+', sentence))
    return has_keyword or has_number


def example_to_prompt_format(doc) -> str:
    label = "그린워싱 있음" if doc.metadata["label"] == 1 else "그린워싱 없음"
    reason = doc.metadata.get("reason", "-")
    solution = doc.metadata.get("solution", "-")

    return f"""query: {doc.page_content.strip()}
answer:
1. 판단: {label}  
2. 근거: {reason}  
3. 해결방안: {solution}"""
