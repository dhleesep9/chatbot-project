"""사설모의고사 응시 트리거

사용자가 "사설모의고사 응시"라고 말하면 사설모의고사 응시 상태로 전이됩니다.

사용 예시 (state JSON):
{
  "trigger_type": "mock_exam",
  "conditions": {
    "input_contains": "사설모의고사 응시"
  },
  "next_state": "mock_exam",
  "transition_narration": "사설모의고사를 응시합니다."
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    사용자 메시지에 "사설모의고사 응시"가 포함되어 있는지 체크
    한 주에 한 번만 응시할 수 있도록 주차 체크 포함

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: "사설모의고사 응시"가 포함되어 있고, 이번 주에 아직 보지 않았으면 True
    """
    conditions = transition.get("conditions", {})
    input_contains = conditions.get("input_contains", "사설모의고사 응시")

    user_message = context.get('user_message', '')

    # 대소문자 구분 없이 포함 여부 체크
    user_message_lower = user_message.lower()
    input_contains_lower = input_contains.lower()

    is_contained = input_contains_lower in user_message_lower

    if not is_contained:
        return False

    # 현재 상태가 이미 mock_exam이면 false 반환 (중복 진입 방지)
    # mock_exam_feedback에서의 재응시는 별도로 처리됨
    current_state = context.get('current_state', '')
    if current_state == 'mock_exam':
        print(f"[TRIGGER] mock_exam 트리거 차단: 이미 mock_exam 상태입니다.")
        return False

    # 한 주에 한 번만 보도록 체크
    service = context.get('service')
    if service:
        username = context.get('username', '')
        current_week = service._get_current_week(username)
        last_week = service.mock_exam_last_week.get(username, -1)

        if current_week == last_week:
            print(f"[TRIGGER] mock_exam 트리거 차단: {username}이(가) 이미 {current_week}주차에 사설모의고사를 봤습니다.")
            return False

    if is_contained:
        print(f"[TRIGGER] mock_exam 트리거 발동: '{input_contains}' in '{user_message}'")

    return is_contained

