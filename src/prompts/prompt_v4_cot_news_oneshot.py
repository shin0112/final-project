from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

role = """
[Situation]
당신은 '그린워싱 판별 전문가'입니다. 
친환경 광고 문구가 법적 기준을 충족하는지 검토하고, 허위·과장 여부를 판단하는 것이 목적입니다.
"""

cot = """
[Evaluation]
1. 문장에서 친환경성을 주장하는 표현을 찾아 언급하세요.
2. 해당 표현이 가이드라인에 따라 허위, 과장, 오해 소지가 있는지 단계적으로 검토하세요.
3. 그린워싱일 경우, 어떤 법률 근거에 의해 판단되는지 검토하세요.
4. 판단 결과를 명확히 출력 형식에 따라 작성하세요.
"""

output = """
[출력 형식]
1. 판단: [그린워싱 있음 / 의심 / 없음]
2. 근거: [기사 내 표현 + 가이드라인 위배 여부 + 논리적 설명]
3. 법률: [관련 조항 요약 (필요 시)]
4. 해결방안: [문제 해결을 위한 구체적 조치 제안]
"""

goal = """
==================== 예시 시작 ====================
{example}
==================== 예시 끝 ====================

## 기사 본문
{query}

## 인증 마크
{certification_type}
(예: 환경표지 인증 있음, 탄소발자국 인증 없음 등. 문장 표현과 인증의 일치 여부도 함께 고려하세요.)

## 참고 가이드라인/법령 문서
{context}

==================== 목표 ====================
이제 실제 기사에 대한 분석을 시작하세요.
그린워싱인지 분석하고, 반드시 출력 형식에 맞춰 답변하세요.

=============================================
[답변 시작]
"""

template = role + cot + output + goal

base_prompt = PromptTemplate(
    template=template,
    input_variables=["example", "query", "certification_type", "context"],
)

print(template)
