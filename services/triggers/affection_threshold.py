"""호감도 절대값 트리거

현재 호감도가 특정 임계값 이상일 때 트리거 발동

사용 예시 (state JSON):
{
  "trigger_type": "affection_threshold",
  "conditions": {
    "affection_min": 10
  },
  "next_state": "selection"
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    호감도 절대값 체크

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: 현재 호감도가 최소값 이상인지 여부
    """
    conditions = transition.get("conditions", {})
    min_affection = conditions.get("affection_min", 10)

    username = context['username']
    service = context['service']
    current_affection = service._get_affection(username)

    return current_affection >= min_affection
