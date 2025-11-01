"""6월 모의고사 과목별 문제점 제시 트리거

플레이어가 질문하거나 이유를 물었을 때, 현재 처리 중인 과목이 없으면 첫 번째 과목의 문제점을 제시합니다.

사용 예시:
- 플레이어: "어떠니?", "전체적으로 어떠니?", "어떻게 됐어?"
- 서가윤: "국어에서는 [문제점]" (자동)
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    플레이어가 질문했는지 확인 (6exam_feedback 상태에서 현재 과목이 없을 때)
    
    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트 (user_message, current_state, june_exam_problems 등 포함)
    
    Returns:
        bool: 질문 키워드가 포함되어 있고, 현재 과목이 없는 경우 True
    """
    user_message = context.get('user_message', '')
    current_state = context.get('current_state', '')
    june_exam_problems = context.get('june_exam_problems', {})
    username = context.get('username', '')
    
    # 6exam_feedback 상태에서만 동작
    if current_state != '6exam_feedback':
        return False
    
    # 현재 문제점 정보 가져오기
    problem_info = june_exam_problems.get(username, {})
    current_subject = problem_info.get('current_subject')
    
    # 현재 과목이 없어야 함 (처음 질문하는 경우)
    if current_subject is not None:
        return False
    
    # 질문 키워드 확인
    reason_keywords = ["이유", "왜", "무엇", "어떻게", "뭐가", "문제", "원인", "이유가", 
                      "어떠니", "어떠냐", "어떠세요", "어떤", "어떠한", "전체적으로", 
                      "어떻", "어떠", "어땠니", "어땠", "어떠니요"]
    user_message_lower = user_message.lower()
    is_asking_reason = any(keyword in user_message_lower for keyword in reason_keywords)
    
    if is_asking_reason:
        print(f"[TRIGGER] june_exam_subject_problem 트리거 발동: '{user_message}' (질문 감지)")
    
    return is_asking_reason

