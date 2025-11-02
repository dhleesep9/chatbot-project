"""호감도 증가량 트리거

현재 턴에서 호감도가 일정량 이상 증가했을 때 트리거 발동

사용 예시 (state JSON):
{
  "trigger_type": "affection_increase",
  "conditions": {
    "affection_increase_min": 1
  },
  "next_state": "icebreak"
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    호감도 증가량 체크

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: 호감도가 최소 증가량 이상 증가했는지 여부
    """
    conditions = transition.get("conditions", {})
    min_increase = conditions.get("affection_increase_min", 1)
    affection_increased = context.get('affection_increased', 0)

    return affection_increased >= min_increase
