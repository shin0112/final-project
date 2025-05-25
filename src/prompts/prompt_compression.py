from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate

template = """
다음은 뉴스 기사 본문입니다. 아래 기사에서 '환경', '친환경', '탄소중립', '재활용', '지속 가능성', '환경부 인증', '자원 절약' 등과 관련된 **마케팅 표현 또는 제품 설명**을 중심으로, 주장과 근거가 담긴 **핵심 문장만 간결하게 요약**해 주세요.

[요약 조건]
- 환경 관련 주장과 그에 대한 **수치, 인증, 기관명 등 구체적 근거**가 포함된 문장을 중심으로 정리해 주세요.
- **광고성 문장**이나 **제품 설명**도 그린워싱 판단에 중요하므로 반드시 포함해 주세요.
- 요약은 **최대 5문장 이내**로, 불필요한 내용 없이 간결하게 작성해 주세요.
- 기사 전체가 아닌, **환경 주장 평가에 필요한 핵심 문장만** 요약해 주세요.

------------------------------
[기사 본문]
{news}

------------------------------
[요약된 환경 관련 핵심 문장]
"""

base_prompt = PromptTemplate(
    template=template,
    input_variables=["news"],
)
