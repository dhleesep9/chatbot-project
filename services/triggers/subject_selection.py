"""탐구과목 선택 트리거

사용자가 입력한 메시지에서 탐구과목을 파싱하여 필요한 개수만큼 선택되었는지 확인
선택된 과목은 자동으로 영구 저장소에 저장됨

사용 예시 (state JSON):
{
  "trigger_type": "subject_selection",
  "conditions": {
    "required_count": 2
  },
  "next_state": "daily_routine"
}
"""

from services.subject_selection import parse_subjects_from_message, validate_subject_count


def evaluate(transition: dict, context: dict) -> bool:
    """
    탐구과목 선택 조건 체크 및 저장

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: 필요한 개수의 과목이 선택되었는지 여부
    """
    conditions = transition.get("conditions", {})
    required_count = conditions.get("required_count", 2)

    user_message = context.get('user_message', '')

    # 메시지에서 과목 추출
    found_subjects = parse_subjects_from_message(user_message)

    # 필요한 개수만큼 선택되었는지 확인
    if validate_subject_count(found_subjects, required_count):
        # 선택된 과목을 영구 저장소에 저장
        username = context['username']
        service = context['service']
        service._set_selected_subjects(username, found_subjects)

        print(f"[TRIGGER] subject_selection 트리거 발동: {found_subjects}")
        return True

    return False
