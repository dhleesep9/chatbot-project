"""유저 입력 포함 트리거

사용자의 메시지에 특정 문자열이 포함되어 있을 때 트리거 발동

사용 예시 (state JSON):
{
  "trigger_type": "user_input",
  "conditions": {
    "input_equals": "A"
  },
  "next_state": "Astate"
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    유저 입력 텍스트 포함 여부 체크

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: 사용자 메시지에 지정된 문자열이 포함되어 있는지 여부
    """
    conditions = transition.get("conditions", {})
    input_equals = conditions.get("input_equals", "")

    if not input_equals:
        return False

    user_message = context.get('user_message', '')

    # 대소문자 구분 없이 포함 여부 체크
    user_message_lower = user_message.lower()
    input_equals_lower = input_equals.lower()

    is_contained = input_equals_lower in user_message_lower

    if is_contained:
        print(f"[TRIGGER] user_input 트리거 발동: '{input_equals}' in '{user_message}'")

    return is_contained
