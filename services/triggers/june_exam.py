"""6월 모의고사 응시 트리거

사용자가 "6월 모의고사 응시" 또는 "6월 모의고사"라고 말하면 6exam 상태로 전이됩니다.

사용 예시 (state JSON):
{
  "trigger_type": "june_exam",
  "conditions": {
    "input_contains": "6월 모의고사"
  },
  "next_state": "6exam",
  "transition_narration": "6월 모의고사를 응시합니다."
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    사용자 메시지에 6월 모의고사 관련 키워드가 포함되어 있는지 체크

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: 6월 모의고사 관련 키워드가 포함되어 있는지 여부
    """
    conditions = transition.get("conditions", {})
    input_contains = conditions.get("input_contains", "6월 모의고사")

    user_message = context.get('user_message', '')
    
    # 공백 제거 후 비교
    user_message_clean = user_message.replace(" ", "").replace("\t", "").replace("\n", "")
    input_contains_clean = input_contains.replace(" ", "").replace("\t", "").replace("\n", "")
    
    # 대소문자 구분 없이 포함 여부 체크
    user_message_lower = user_message_clean.lower()
    input_contains_lower = input_contains_clean.lower()

    is_contained = input_contains_lower in user_message_lower

    if is_contained:
        print(f"[TRIGGER] june_exam 트리거 발동: '{input_contains}' in '{user_message}'")

    return is_contained

