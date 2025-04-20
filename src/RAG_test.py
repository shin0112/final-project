import os
import logging
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from get_ko_law import get_ko_law

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
FAISS_PATH = Path(__file__).parent.parent / 'data' / 'faiss_index'
LAW_FILE_PATHS = Path(__file__).parent.parent / 'data' / 'law_file_paths.json'

PROMPT_TEMPLATE = """\
당신은 기업의 마케팅 문구를 분석하여 그린워싱 여부를 판단하는 전문가입니다.
다음 문장을 읽고, 반드시 아래 형식에 맞춰 답변하세요.

[분석할 문장]
"{query}"

[참고할 가이드라인 정보]
{context}

[요구사항]
- 판단, 근거, 해결방안을 각각 1~2문장으로 간결하게 작성합니다.
- 문장은 한국어로 작성합니다.
- 절대 같은 문장을 반복하지 않습니다.
- 응답 마지막에 반드시 '응답 끝'이라고 작성합니다.

[응답 양식]
1. 판단: (그린워싱 가능성 있음 / 없음)
2. 근거: (판단한 이유, 가이드라인을 바탕으로 설명)
3. 해결방안: (문장을 어떻게 수정하면 되는지 제안)

응답을 시작하세요:
"""


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


def load_or_create_faiss_index(embeddings_model):
    index_file = FAISS_PATH / "index.faiss"
    if index_file.exists():
        logging.info("FAISS 인덱스를 로드 중입니다...")
        vector_store = FAISS.load_local(
            FAISS_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        logging.info("FAISS 인덱스 로드가 완료되었습니다!")
    else:
        logging.info("한국 법률 데이터를 로드 중입니다...")
        ko_docs_list = get_ko_law()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        logging.info("문서를 분할 중입니다...")
        ko_split_docs_list = [
            chunk for docs in ko_docs_list for chunk in text_splitter.split_documents(docs)
        ]
        logging.info("문서 분할이 완료되었습니다!")

        logging.info("FAISS 인덱스를 생성 중입니다...")
        vector_store = FAISS.from_documents(
            documents=ko_split_docs_list,
            embedding=embeddings_model,
            distance_strategy=DistanceStrategy.COSINE
        )
        logging.info("FAISS 인덱스 생성이 완료되었습니다!")
        FAISS_PATH.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(FAISS_PATH)
        logging.info("FAISS 인덱스가 저장되었습니다!")
    return vector_store


def initialize_model_and_tokenizer(model_name):
    logging.info("모델과 토크나이저를 초기화 중입니다...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    logging.info("모델과 토크나이저 초기화가 완료되었습니다!")
    return model, tokenizer


def generate_answer(model, tokenizer, query, context):
    prompt = PROMPT_TEMPLATE.format(query=query, context=context)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,  # 단문 자동 패딩 → 길이 맞추기
        truncation=True,  # 장문 자동 잘라내기
    ).to(model.device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text[len(prompt):].strip()


def main():
    embeddings_model = KoSimCSE()
    vector_store = load_or_create_faiss_index(embeddings_model)

    query = "이 텀블러는 100% 친환경 소재로 제작되었습니다."
    logging.info(f"질문: {query}")

    logging.info("FAISS 인덱스에서 유사한 문서를 검색 중입니다...")
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1}
    )
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])[:1000]

    model_name = "beomi/KoAlpaca-Polyglot-5.8B"
    model, tokenizer = initialize_model_and_tokenizer(model_name)

    answer = generate_answer(model, tokenizer, query, context)
    logging.info(f"답변: {answer}")


if __name__ == "__main__":
    main()
