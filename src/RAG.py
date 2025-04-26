import os
import re
import logging
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

import get_ko_law
import model_loader

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
FAISS_PATH = Path(__file__).parent.parent / 'data' / 'faiss_index'
GUIDELINE_FAISS_PATH = FAISS_PATH / "guideline.faiss"
GENERAL_FAISS_PATH = FAISS_PATH / "law.faiss"
LAW_FILE_PATH = Path(__file__).parent.parent / 'data' / 'law_file_paths.json'
PROMPT_PATH = Path(__file__).parent / 'prompts' / 'prompt_v2_cot_fewshot.txt'

KEYWORDS = [
    "친환경", "지속 가능", "재활용", "탄소 중립", "인증", "에코", "그린", "지속 가능한",
    "환경 보호", "자원 절약", "친환경 소재", "친환경 제품", "지속 가능한 발전"
]


def load_prompt():
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()


class KoSimCSE:
    def __init__(self, model_name='BM-K/KoSimCSE-roberta', device=None):
        logging.info("KoSimCSE 임베딩 모델 로드 중입니다...")
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logging.info("KoSimCSE 모델 로드 완료!")

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            embeddings = outputs.last_hidden_state[:, 0]  # [CLS] 토큰 가져오기
            embeddings = embeddings / \
                embeddings.norm(dim=1, keepdim=True)  # normalize
        return embeddings.cpu().tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def load_or_create_faiss_guideline(embeddings_model):
    if GUIDELINE_FAISS_PATH.exists():
        logging.info("FAISS 인덱스를 로드 중입니다...")
        guideline_store = FAISS.load_local(
            GUIDELINE_FAISS_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        logging.info("FAISS 인덱스 로드가 완료되었습니다!")
    else:
        logging.info("한국 법률 데이터를 로드 중입니다...")
        ko_guideline_docs_list = get_ko_law.get_ko_guideline()
        ko_law_docs_list = get_ko_law.get_ko_law()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        logging.info("문서를 분할 중입니다...")
        ko_split_guideline_docs_list = [
            chunk for docs in ko_guideline_docs_list for chunk in text_splitter.split_documents(docs)
        ]
        ko_split_law_docs_list = [
            chunk for docs in ko_law_docs_list for chunk in text_splitter.split_documents(docs)
        ]
        logging.info("문서 분할이 완료되었습니다!")

        logging.info("guideline 문서 저장 중...")
        guideline_store = FAISS.from_documents(
            documents=ko_split_guideline_docs_list,
            embedding=embeddings_model,
            distance_strategy=DistanceStrategy.COSINE
        )
        logging.info("guideline 인덱스 생성 완료")

        logging.info("법률 문서 저장 중...")
        law_store = FAISS.from_documents(
            documents=ko_split_law_docs_list,
            embedding=embeddings_model,
            distance_strategy=DistanceStrategy.COSINE
        )
        logging.info("법률 인덱스 생성 완료")

        FAISS_PATH.mkdir(parents=True, exist_ok=True)
        guideline_store.save_local(FAISS_PATH)
        law_store.save_local(FAISS_PATH)
        logging.info("FAISS 인덱스가 저장되었습니다!")

    return guideline_store


def generate_answer(model, tokenizer, query, context):
    prompt = load_prompt().format(query=query, context=context)
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
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.8,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text[len(prompt):].strip()


def naturalize_query(sentences: list[str]) -> str:
    result = [f"'{s}'라는 문장은 그린워싱 가능성이 있는 표현인가요?" for s in sentences]
    logging.info(f"[자연어 처리] 다음 query 생성")
    for i, q in enumerate(result, 1):
        logging.info(f"{i}. {q}")
    return result


def extract_key_sentences(text: str, max_sentences: int = 5) -> list[str]:
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


def main():
    embeddings_model = KoSimCSE()
    guideline_store = load_or_create_faiss_guideline(embeddings_model)

    # 30줄 이상의 기사 input이 들어온다.
    # 문장 분리 후, 키워드를 기반으로 의미 있는 문장만 뽑아낸다.

    input = """
    이 텀블러는 100 % 친환경 소재로 제작되었습니다.
    우리는 지속가능한 미래를 위해 작은 변화부터 시작합니다.
    제조 과정에서 탄소 배출을 최소화하였습니다.
    포장재는 생분해성 플라스틱으로 제작되어 자연으로 돌아갑니다.
    이 제품은 환경부 인증을 받은 친환경 제품입니다.
    당신의 선택이 지구를 지킵니다.
    디자인뿐 아니라 환경까지 생각한 제품입니다.
    모든 재료는 무독성, 무해성 기준을 만족합니다.
    소비자의 건강과 환경을 동시에 고려했습니다.
    """

    key_sentences = extract_key_sentences(input)
    query_list = naturalize_query(key_sentences)
    query = '\n'.join(query_list)

    logging.info(f"질문: {query}")

    logging.info("가이드라인 문서를 검색 중입니다...")
    retriever_1st = guideline_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2}
    )
    guideline = retriever_1st.invoke(query)
    context = "\n".join([doc.page_content for doc in guideline])[:1000]
    # 유사도 검색된 문서 내용
    logging.info("[유사도 검색] 유사도 검색된 문서 내용")
    for i, doc in enumerate(guideline, 1):
        logging.info(f"  [{i}] {doc.page_content}...")

    # todo: 만약에 guideline 내부에 비슷한 문장이 있으면, 법률 검색? 근데 유사도 검색하면 일단은 1개 이상은 있는 거 아닌가

    model, tokenizer = model_loader.mistralai_loader()

    answer = generate_answer(model, tokenizer, query, context)
    logging.info(f"답변: {answer}")


if __name__ == "__main__":
    main()
