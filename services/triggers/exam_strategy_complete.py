"""시험 전략 수집 완료 트리거

exam_strategy 상태에서 플레이어가 충분한 길이의 전략을 입력했을 때 트리거 발동

사용 예시 (state JSON):
{
  "trigger_type": "exam_strategy_complete",
  "conditions": {},
  "next_state": "daily_routine",
  "transition_narration": "전략 수집 완료. 일상 루틴으로 돌아갑니다."
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    exam_strategy 상태에서 전략이 입력되었는지 체크

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: 전략이 입력되었는지 여부
    """
    username = context.get('username')
    service = context.get('service')
    user_message = context.get('user_message', '')
    current_state = context.get('current_state', '')
    
    # exam_strategy 상태가 아니면 발동하지 않음
    if current_state != "exam_strategy":
        return False
    
    # 전략 저장소 확인
    exam_progress = service.exam_progress.get(username, {})
    
    # strategies가 없으면 전략 저장소가 아직 초기화되지 않은 상태
    if "strategies" not in exam_progress:
        return False
    
    # 충분히 긴 메시지가 입력되었는지 확인 (5자 이상)
    if len(user_message.strip()) <= 5:
        return False
    
    # 과목 추출 시도
    subject = service._extract_subject_from_strategy(user_message)
    
    if subject:
        is_contained = True
        print(f"[TRIGGER] exam_strategy_complete 트리거 발동: 전략 입력됨 (과목: {subject})")
    else:
        is_contained = False

    return is_contained

