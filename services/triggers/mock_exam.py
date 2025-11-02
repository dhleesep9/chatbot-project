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

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: "사설모의고사 응시"가 포함되어 있는지 여부
    """
    conditions = transition.get("conditions", {})
    input_contains = conditions.get("input_contains", "사설모의고사 응시")

    user_message = context.get('user_message', '')
    
    # 대소문자 구분 없이 포함 여부 체크
    user_message_lower = user_message.lower()
    input_contains_lower = input_contains.lower()

    is_contained = input_contains_lower in user_message_lower

    if is_contained:
        print(f"[TRIGGER] mock_exam 트리거 발동: '{input_contains}' in '{user_message}'")

    return is_contained

