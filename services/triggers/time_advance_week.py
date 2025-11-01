"""1주 시간 진행 트리거

사용자가 "멘토링 종료"라고 말하면 시간이 1주 경과합니다.

사용 예시 (state JSON):
{
  "trigger_type": "time_advance_week",
  "conditions": {
    "input_contains": "멘토링 종료"
  },
  "next_state": "daily_routine",
  "transition_narration": "1주가 지났습니다."
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    사용자 메시지에 "멘토링 종료"가 포함되어 있는지 체크

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: "멘토링 종료"가 포함되어 있는지 여부
    """
    conditions = transition.get("conditions", {})
    input_contains = conditions.get("input_contains", "멘토링 종료")

    user_message = context.get('user_message', '')
    
    # 대소문자 구분 없이 포함 여부 체크
    user_message_lower = user_message.lower()
    input_contains_lower = input_contains.lower()

    is_contained = input_contains_lower in user_message_lower

    if is_contained:
        print(f"[TRIGGER] time_advance_week 트리거 발동: '{input_contains}' in '{user_message}'")

    return is_contained

