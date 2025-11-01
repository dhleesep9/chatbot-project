"""고백 이벤트 트리거

날짜 기반 또는 수동으로 고백 이벤트를 시작합니다.

사용 예시 (state JSON):
{
  "trigger_type": "confession_event",
  "conditions": {
    "date_check": "2024-08-16"
  },
  "next_state": "confession",
  "transition_narration": "고백 이벤트가 발생했습니다."
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    고백 이벤트 트리거 평가
    
    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트
    
    Returns:
        bool: 고백 이벤트가 발생해야 하는 경우 True
    """
    service = context.get('service')
    if not service:
        return False
    
    username = context.get('username', '')
    current_state = context.get('current_state', '')
    
    # daily_routine 상태에서만 동작
    if current_state != 'daily_routine':
        return False
    
    # 조건 확인
    conditions = transition.get("conditions", {})
    
    # 날짜 기반 체크
    if "date_check" in conditions:
        target_date = conditions.get("date_check")
        try:
            current_date = service._get_game_date(username)
            if current_date == target_date:
                print(f"[TRIGGER] confession_event 트리거 발동: 날짜 {target_date} 도달")
                return True
        except Exception as e:
            print(f"[TRIGGER] confession_event 날짜 체크 오류: {e}")
    
    # 사용자가 수동으로 트리거하는 경우 (예: "고백 이벤트" 입력)
    user_message = context.get('user_message', '')
    if "고백" in user_message and "이벤트" in user_message:
        print(f"[TRIGGER] confession_event 트리거 발동: 수동 트리거 '{user_message}'")
        return True
    
    return False

