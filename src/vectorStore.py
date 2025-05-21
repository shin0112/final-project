from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
import logging
import torch
import pandas as pd
from langchain.docstore.document import Document
from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from itertools import chain

import get_ko_law

# Paths
FAISS_PATH = Path(__file__).parent.parent / 'data' / 'faiss_index'
GUIDELINE_FAISS_PATH = FAISS_PATH / "guideline"
LAW_FAISS_PATH = FAISS_PATH / "law"
RERANK_FAISS_PATH = FAISS_PATH / "rerank"
LAW_FILE_PATH = Path(__file__).parent.parent / 'config' / 'law_file_paths.json'
NEWS_FAISS_PATH = FAISS_PATH / "news"
NEWS_PATH = Path(__file__).parent.parent/'data' / \
    'greenwashing'/'greenwashing_train.csv'
GUIDELINE_NEWS_FAISS_PATH = FAISS_PATH / "guideline_news"
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

    def embed_documents(self, texts, batch_size: int = 16) -> list[list[float]]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                emb = outputs.last_hidden_state[:, 0]  # [CLS] 토큰
                embeddings.extend(emb.cpu().tolist())
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class MsMarcoDistilbert:
    def __init__(self, model_name='sentence-transformers/msmarco-distilbert-dot-v5', device=None):
        logging.info("MsMarcoDistilbert 임베딩 모델 로드 중입니다...")
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logging.info("MsMarcoDistilbert 모델 로드 완료!")

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts) -> list[list[float]]:
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = embeddings / \
                embeddings.norm(dim=1, keepdim=True)
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


def load_or_create_faiss_rerank(embeddings_model):
    if RERANK_FAISS_PATH.exists():
        logging.info("RERANK용 FAISS 인덱스를 로드 중입니다...")
        store = FAISS.load_local(
            RERANK_FAISS_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        logging.info("RERANK용 FAISS 인덱스 로드가 완료되었습니다!")
    else:
        logging.info("한국 가이드라인 및 법률 문서 로드 중입니다...")
        ko_guideline_docs_list = get_ko_law.get_ko_guideline()
        ko_law_docs_list = get_ko_law.get_ko_law()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        logging.info("문서를 분할 중입니다...")
        all_docs_split = [
            chunk for docs in (ko_guideline_docs_list + ko_law_docs_list)
            for chunk in text_splitter.split_documents(docs)
        ]
        logging.info("문서 분할 완료!")

        store = FAISS.from_documents(
            documents=all_docs_split,
            embedding=embeddings_model,
            distance_strategy=DistanceStrategy.COSINE
        )

        RERANK_FAISS_PATH.mkdir(parents=True, exist_ok=True)
        store.save_local(RERANK_FAISS_PATH)

        logging.info("RERANK용 FAISS 인덱스 저장 완료!")

    return store


def load_or_create_faiss_news(embedding_model):
    if NEWS_FAISS_PATH.exists():
        logging.info("뉴스 벡터 DB 로딩 중...")
        store = FAISS.load_local(
            NEWS_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        logging.info("뉴스 벡터 DB 로드 완료!")
        return store

    documents = get_news_data()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs_split = text_splitter.split_documents(documents)

    logging.info(f"{len(docs_split)}개의 기사 문서를 벡터화 중입니다...")
    store = FAISS.from_documents(
        documents=docs_split,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE
    )

    NEWS_FAISS_PATH.mkdir(parents=True, exist_ok=True)
    store.save_local(NEWS_FAISS_PATH)
    logging.info("뉴스 기사 벡터 DB 저장 완료!")

    return store


def get_news_data():
    logging.info("뉴스 기사 CSV 데이터 불러오기")
    df = pd.read_csv(NEWS_PATH)
    df = df[['title', 'content', 'greenwashing_level', 'full_text']].copy()
    df.rename(columns={
        'title': 'title',
        'content': 'summary',
        'greenwashing_level': 'label',
        'full_text': 'text'
    }, inplace=True)
    df['label'] = df['label'].fillna(0).astype(float)

    # 메타데이터
    df['metadata'] = df.apply(lambda row: {
        'title': row['title'],
        'summary': row['summary'],
        'label': row['label']
    }, axis=1)

    documents = [
        Document(page_content=row['text'], metadata=row['metadata'])
        for _, row in df.iterrows()
    ]

    return documents


def load_or_create_faiss_guideline_and_news(embedding_model):
    if GUIDELINE_NEWS_FAISS_PATH.exists():
        logging.info("guideline + news 통합 벡터 DB 로드 중...")
        store = FAISS.load_local(
            GUIDELINE_NEWS_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        logging.info("로드 완료")
        return store

    logging.info("guideline + news 통합 벡터 DB 생성 중...")

    # 가이드라인 & 뉴스 문서 로드
    guideline_docs = get_ko_law.get_ko_guideline()
    for doc in guideline_docs:
        for page in doc:
            page.metadata["type"] = "guideline"
    news_docs = get_news_data()

    flat_guideline_docs = list(chain.from_iterable(guideline_docs))
    combined_docs = flat_guideline_docs + news_docs

    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    all_docs = text_splitter.split_documents(combined_docs)

    store = FAISS.from_documents(
        documents=all_docs,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE
    )

    GUIDELINE_NEWS_FAISS_PATH.mkdir(parents=True, exist_ok=True)
    store.save_local(GUIDELINE_NEWS_FAISS_PATH)
    logging.info("guideline + news 통합 벡터 DB 저장 완료")

    return store


def search_with_score_filter(retriever, query, min_score=0.75, k=5):
    results = retriever.vectorstore.similarity_search_with_score(query, k=k)
    filtered = [
        doc for doc, score in results
        if score >= min_score
    ]
    return filtered


# def search_guideline_only(retriever, query, top_k=10, min_score=0.75):
#     results = retriever.vectorstore.similarity_search_with_score(
#         query, k=top_k)
#     return [
#         doc for doc, score in results
#         if doc.metadata.get("type") == "guideline" and score >= min_score
#     ]


class BgeReranker:
    def __init__(self):
        logging.info("BgeReranker 모델 로드 중입니다...")
        self.model = HuggingFaceCrossEncoder(
            model_name='BAAI/bge-reranker-v2-m3')
        self.compressor = CrossEncoderReranker(
            model=self.model,
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=load_or_create_faiss_rerank(
                KoSimCSE()
            ).as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            search_kwargs={"k": 2},
        )
        logging.info("BgeReranker 모델 로드 완료!")
