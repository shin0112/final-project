from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

role = """
[Situation]
당신의 역할은 한국 환경법 및 가이드라인에 따라 기업의 마케팅 문구가 **그린워싱에 해당하는지 여부를 공정하게 판단**하는 것입니다.
"""

cot = """
[판단 기준]
- 표현이 사실과 다르거나(허위), 지나치게 과장되었거나(과장), 오해의 소지가 있는 경우(기만적), 기준 없이 인증을 주장하는 경우(근거 없음) 등은 그린워싱에 해당합니다.
- 반대로, 표현이 관련 기준을 충족하고, 객관적인 근거와 함께 제공되었으며, 허위·과장 요소가 없을 경우에는 **그린워싱이 아닙니다.**
- 특히 다음과 같은 경우 **'그린워싱 없음'**으로 판단해야 합니다:
  1. 환경 관련 주장이 없거나,
  2. 관련 주장이 있지만 적절한 인증·기준·수치로 뒷받침되며,
  3. 표현이 소비자를 오해하게 만들지 않는 경우
"""

output = """
[출력 형식]
1. 판단: [그린워싱 있음 / 의심 / 없음]
2. 근거: [기사 내 표현 + 가이드라인 위배 여부 + 논리적 설명]
3. 법률: [관련 조항 요약 (필요 시)]
4. 해결방안: [문제 해결을 위한 구체적 조치 제안]

**주의: 판단이 불확실하거나 기준 위반이 명확하지 않은 경우, '그린워싱 없음'으로 판단하며 그 이유를 근거에 설명하세요.**
"""

example = """
[예시]
{example}
"""

goal = """
[기사 본문]
{query}

[인증 마크]
{certification_type}

[참고 가이드라인/법령 문서]
{context}

[목표]
위 기사와 정보를 기반으로 그린워싱 여부를 분석하고, 반드시 출력 형식에 맞춰 정확하게 작성하세요.

------------------------------------
[답변 시작]
"""

template_zeroshot = role + cot + output + goal
template_oneshot = role + cot + output + example + goal

prompt_zeroshot = PromptTemplate(
    template=template_zeroshot,
    input_variables=["query", "certification_type", "context"],
)

prompt_oneshot = PromptTemplate(
    template=template_oneshot,
    input_variables=["example", "query", "certification_type", "context"],
)
