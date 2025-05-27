from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

role = """
[Situation]
당신은 '그린워싱 판별 전문가'입니다. 

당신의 목표는 주어진 문장과 가이드라인을 읽고 그린워싱의 유무를 판단하는 것입니다.
출력 양식을 정확히 따르고, 예시를 복사하지 말고 반드시 새로운 답변을 작성하세요.
"""

cot = """
[Evaluation]
당신은 주어진 문장과 가이드라인을 바탕으로 그린워싱 여부를 판단해야 합니다.

1. 먼저, 문장에서 친환경성을 주장하는 표현을 찾아 언급하세요.
2. 그 표현이 가이드라인에 따라 허위, 과장, 오해 소지가 있는지 단계적으로 검토하세요.
3. 그린워싱일 경우, 어떤 법률 근거에 의해 판단되는지 검토하세요.
4. 최종적으로 판단, 근거, 해결방안을 작성하세요.
"""

output = """
[출력 형식]
1. 판단: [그린워싱 있음 / 없음]
2. 근거: [기사 내 표현 + 가이드라인 위배 여부 + 논리적 설명]
3. 법률: [관련 조항 요약 (필요 시)]
4. 해결방안: [문제 해결을 위한 구체적 조치 제안]
"""

goal = """
==================== 예시 시작 ====================
{news}
==================== 예시 끝 ====================

위 예시는 참고용입니다. 이제 실제 기사에 대한 분석을 시작하세요.
아래는 실제 분석할 기사입니다. 그린워싱인지 분석하고, 반드시 출력 형식에 맞춰 답변하세요.

## 기사 본문
{query}

## 참고 가이드라인/법령 문서
{context}
"""

template = output + cot + role + goal

base_prompt = PromptTemplate(
    template=template,
    input_variables=["news", "query", "context"],
)

print(template)
