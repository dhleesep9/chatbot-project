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

또는

{
  "trigger_type": "user_input",
  "conditions": {
    "input_contains": "시험전략수립"
  },
  "next_state": "exam_strategy"
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
    input_contains = conditions.get("input_contains", "")

    # input_equals가 있으면 사용, 없으면 input_contains 사용
    if input_equals:
        target_string = input_equals
    elif input_contains:
        target_string = input_contains
    else:
        return False

    user_message = context.get('user_message', '')
    
    # 대소문자 구분 없이 포함 여부 체크
    user_message_lower = user_message.lower()
    target_string_lower = target_string.lower()
    
    is_contained = target_string_lower in user_message_lower

    if is_contained:
        print(f"[TRIGGER] user_input 트리거 발동: '{target_string}' in '{user_message}'")

    return is_contained
