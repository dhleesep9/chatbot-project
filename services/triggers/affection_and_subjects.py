"""호감도 + 과목 선택 복합 트리거

호감도가 일정 수준 이상이고, 탐구과목이 일정 개수 이상 선택되었을 때 트리거 발동

사용 예시 (state JSON):
{
  "trigger_type": "affection_and_subjects",
  "conditions": {
    "affection_min": 10,
    "subjects_count": 2
  },
  "next_state": "daily_routine"
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    호감도 + 탐구과목 복합 조건 체크

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: 호감도와 과목 선택 조건을 모두 만족하는지 여부
    """
    conditions = transition.get("conditions", {})
    min_affection = conditions.get("affection_min", 10)
    subjects_count = conditions.get("subjects_count", 2)

    username = context['username']
    service = context['service']

    # 호감도 체크
    current_affection = service._get_affection(username)
    affection_met = current_affection >= min_affection

    # 선택된 과목 개수 체크
    selected_subjects = service._get_selected_subjects(username)
    subjects_met = len(selected_subjects) >= subjects_count

    return affection_met and subjects_met
