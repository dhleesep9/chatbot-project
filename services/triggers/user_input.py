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

또는

{
  "trigger_type": "user_input",
  "conditions": {
    "input_contains_any": ["화이팅", "파이팅", "잘하고", "응원"]
  },
  "next_state": "6exam"
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
    input_contains_any = conditions.get("input_contains_any", [])

    user_message = context.get('user_message', '')
    user_message_lower = user_message.lower()

    # input_equals 체크 (완전 일치)
    if input_equals:
        target_string_lower = input_equals.lower()
        is_matched = user_message_lower == target_string_lower
        if is_matched:
            print(f"[TRIGGER] user_input 트리거 발동 (equals): '{input_equals}' == '{user_message}'")
        return is_matched

    # input_contains 체크 (단일 문자열 포함)
    elif input_contains:
        target_string_lower = input_contains.lower()
        is_contained = target_string_lower in user_message_lower
        if is_contained:
            print(f"[TRIGGER] user_input 트리거 발동 (contains): '{input_contains}' in '{user_message}'")
        return is_contained

    # input_contains_any 체크 (여러 문자열 중 하나라도 포함)
    elif input_contains_any:
        for keyword in input_contains_any:
            keyword_lower = keyword.lower()
            if keyword_lower in user_message_lower:
                print(f"[TRIGGER] user_input 트리거 발동 (contains_any): '{keyword}' in '{user_message}'")
                return True
        return False

    else:
        return False
