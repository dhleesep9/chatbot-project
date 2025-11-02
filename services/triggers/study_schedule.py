"""학습 시간표 관리 상태 전이 트리거

사용자가 "학습 시간표 관리"라고 말하면 학습 시간표 관리 상태로 전이됩니다.

사용 예시 (state JSON):
{
  "trigger_type": "study_schedule",
  "conditions": {
    "input_contains": "학습 시간표 관리"
  },
  "next_state": "study_schedule",
  "transition_narration": "학습 시간표 관리 모드로 들어갑니다."
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    사용자 메시지에 "학습 시간표 관리" 또는 "학습시간표 관리"가 포함되어 있는지 체크
    공백을 제거하고 비교하여 공백 유무와 관계없이 매칭합니다.

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: "학습 시간표 관리" 또는 "학습시간표 관리"가 포함되어 있는지 여부
    """
    conditions = transition.get("conditions", {})
    input_contains = conditions.get("input_contains", "학습 시간표 관리")

    user_message = context.get('user_message', '')
    
    # 공백을 제거하고 비교 (공백 유무와 관계없이 매칭)
    user_message_no_space = user_message.replace(" ", "").replace("　", "")
    input_contains_no_space = input_contains.replace(" ", "").replace("　", "")
    
    # 대소문자 구분 없이 포함 여부 체크
    user_message_lower = user_message_no_space.lower()
    input_contains_lower = input_contains_no_space.lower()

    is_contained = input_contains_lower in user_message_lower

    if is_contained:
        print(f"[TRIGGER] study_schedule 트리거 발동: '{input_contains}' in '{user_message}' (공백 무시)")

    return is_contained

