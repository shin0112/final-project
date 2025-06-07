import re
import requests
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional

from src.demo import run_single_text

app = FastAPI()

client_id = os.environ.get("PAPAGO_CLIENT_ID")
client_secret = os.environ.get("PAPAGO_CLIENT_SECRET")


class ArticleRequest(BaseModel):
    article: str
    ct: Optional[str] = "없음"


@app.post("/run")
def run_pipeline(req: ArticleRequest):
    prompt_version = "v4-zeroshot"

    result = run_single_text(req.article, req.ct)
    parsed = parse_answer(result)

    return {
        "raw": result,
        "data": parsed
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
        "X-Naver-Client-Secret": client_secret
    }
    data = {
        "source": "en",
        "target": "ko",
        "text": text
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json()['message']['result']['translatedText']


def parse_answer(text: str) -> dict:
    """
    응답 텍스트를 판단 / 근거 / 법률 / 해결방안으로 파싱
    """
    result = {"judgement": "", "reason": "", "law": "", "solution": ""}

    patterns = {
        "judgement": r"\*\*판단:?\s*(.*?)\*\*",
        "reason": r"\*\*근거:?\*\*\s*(.*?)(?=\*\*법률:|\Z)",
        "law": r"\*\*법률:?\*\*\s*(.*?)(?=\*\*해결방안:|\Z)",
        "solution": r"\*\*해결방안:?\*\*\s*(.*)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()

    return result
