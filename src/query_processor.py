# todo: 더 자세하게 채우기
ENVIRONMENT_TERM_MAPPING = {
    "100% 친환경 소재": "전량 친환경 인증을 받은 소재",
    "탄소 배출 최소화": "온실가스 배출 저감 목표",
    "지속가능한 미래": "지속 가능한 발전 목표",
    "생분해성 플라스틱": "생분해성 인증을 받은 소재",
    "무독성": "인체에 무해한 인증 기준 만족",
    "친환경 제품": "환경 인증 기준 충족 제품",
}


def legalize_query(sentences: list[str]) -> list[str]:
    """
    법률 용어와 매칭하기 위한 자연어 처리
    """
    new_sentences = []
    for s in sentences:
        for keyword, legal_term in ENVIRONMENT_TERM_MAPPING.items():
            if keyword in s:
                s = s.replace(keyword, legal_term)
        new_sentences.append(s)
    return new_sentences
