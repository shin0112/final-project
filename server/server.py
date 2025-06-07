import re
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional

from demo import run_single_text

app = FastAPI()


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
