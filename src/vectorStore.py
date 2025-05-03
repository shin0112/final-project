import logging
import torch
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter

import get_ko_law

# Paths
FAISS_PATH = Path(__file__).parent.parent / 'data' / 'faiss_index'
GUIDELINE_FAISS_PATH = FAISS_PATH / "guideline"
LAW_FAISS_PATH = FAISS_PATH / "law"
LAW_FILE_PATH = Path(__file__).parent.parent / 'data' / 'law_file_paths.json'
# PROMPT_PATH = Path(__file__).parent / 'prompts' / 'prompt_v3_cot_fewshot.txt'

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

    def embed_documents(self, texts) -> list[list[float]]:
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
        GUIDELINE_FAISS_PATH.mkdir(parents=True, exist_ok=True)
        LAW_FAISS_PATH.mkdir(parents=True, exist_ok=True)
        guideline_store.save_local(GUIDELINE_FAISS_PATH)
        law_store.save_local(LAW_FAISS_PATH)

        logging.info("FAISS 인덱스가 저장되었습니다!")

    return guideline_store


def load_or_create_faiss_law(embeddings_model):
    logging.info("FAISS 인덱스를 로드 중입니다...")
    law_store = FAISS.load_local(
        LAW_FAISS_PATH,
        embeddings_model,
        allow_dangerous_deserialization=True
    )
    logging.info("FAISS 인덱스 로드가 완료되었습니다!")
    return law_store
