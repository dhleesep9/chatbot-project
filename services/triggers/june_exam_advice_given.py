"""6월 모의고사 조언 제시 트리거

플레이어가 조언을 제시했는지 확인합니다.

사용 예시:
- 플레이어: "수능특강을 양치기를 하면 분명 잘 볼 수 있을 거야"
- 트리거 발동 → 조언 판단 시작
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    플레이어가 조언을 제시했는지 확인 (6exam_feedback 상태에서 현재 과목이 있을 때)
    
    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트
    
    Returns:
        bool: 조언 키워드가 포함되어 있고, 현재 과목이 있는 경우 True
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
    
    # 현재 과목이 있어야 함
    if current_subject is None:
        return False
    
    # 조언 키워드 확인
    advice_keywords = ["이렇게", "이런", "조언", "팁", "방법", "해보", "시도", "추천", 
                      "제안", "도움", "알려", "가르쳐", "하면", "해야", "해야 할", 
                      "추천해", "조언해", "방법이", "팁이", "도와", "화이팅", "할 수", 
                      "잘 할 수", "될 거야", "될 거예요", "될 거", "해봐", "해보세요"]
    user_message_lower = user_message.lower()
    is_advice_given = any(keyword in user_message_lower for keyword in advice_keywords)
    
    # 메시지가 충분히 길면 조언으로 간주
    if not is_advice_given and len(user_message.strip()) > 10:
        is_advice_given = True
    
    if is_advice_given:
        print(f"[TRIGGER] june_exam_advice_given 트리거 발동: '{user_message[:50]}...' (조언 감지)")
    
    return is_advice_given

