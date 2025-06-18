import re
import requests
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from demo import run_single_text

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
client_id = os.environ.get("PAPAGO_CLIENT_ID")
client_secret = os.environ.get("PAPAGO_CLIENT_SECRET")


class ArticleRequest(BaseModel):
    article: str
    ct: Optional[str] = "없음"


@app.post("/run")
def run_pipeline(req: ArticleRequest):
    result = run_single_text(req.article, req.ct)
    parsed = parse_answer(result)

    return {
        "raw": result,
        "data": parsed
    }


@app.post("/translate")
def translate_only(req: ArticleRequest):
    translated = postprocess_answer(req.article)
    return {
        "original": req.article,
        "translated": translated
    }


def postprocess_answer(text: str) -> str:
    if is_english(text):
        return translate_to_korean(text)
    return text


def is_english(text: str) -> bool:
    english_ratio = len(re.findall(r'[a-zA-Z]', text)) / max(1, len(text))
    return english_ratio > 0.4


def translate_to_korean(text: str) -> str:
    url = "https://papago.apigw.ntruss.com/nmt/v1/translation"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json"
    }
    data = {
        "source": "en",
        "target": "ko",
        "text": text
    }
    response = requests.post(url, headers=headers, json=data)
    try:
        result = response.json()['message']['result']['translatedText']
        print("[번역 결과]", result)
        return result
    except KeyError:
        print("[번역 실패 응답]", response.text)
        raise ValueError("Papago 번역 실패: 응답 구조 확인 필요")


def parse_answer(text: str) -> dict:
    """
    응답 텍스트를 판단 / 근거 / 법률 / 해결방안으로 파싱
    """
    result = {"judgement": "", "reason": "", "law": "", "solution": ""}

    patterns = {
        "judgement": r"\*\*(판단|Judgment):?\*\*\s*(.*?)\n",
        "reason": r"\*\*(근거|Reasoning):?\*\*\s*(.*?)(?=\*\*(법률|Law|Justification|Regulation):|\Z)",
        "law": r"\*\*(법률|Law|Regulation):?\*\*\s*(.*?)(?=\*\*(해결방안|Remedy):|\Z)",
        "solution": r"\*\*(해결방안|Remedy):?\*\*\s*(.*)"
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()

    return result
