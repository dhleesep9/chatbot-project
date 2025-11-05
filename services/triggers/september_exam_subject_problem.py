"""9월 모의고사 과목별 문제점 제시 트리거

플레이어가 질문하거나 이유를 물었을 때, 현재 처리 중인 과목이 없으면 첫 번째 과목의 문제점을 제시합니다.

사용 예시:
- 플레이어: "어떠니?", "전체적으로 어떠니?", "어떻게 됐어?"
- 서가윤: "국어에서는 [문제점]" (자동)
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    플레이어가 질문했는지 확인 (9exam_feedback 상태에서 현재 과목이 없을 때)
    
    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트 (user_message, current_state, september_exam_problems 등 포함)
    
    Returns:
        bool: 질문 키워드가 포함되어 있고, 현재 과목이 없는 경우 True
    """
    user_message = context.get('user_message', '')
    current_state = context.get('current_state', '')
    september_exam_problems = context.get('september_exam_problems', {})
    username = context.get('username', '')
    
    # 9exam_feedback 상태에서만 동작
    if current_state != '9exam_feedback':
        return False
    
    # 현재 문제점 정보 가져오기
    problem_info = september_exam_problems.get(username, {})
    current_subject = problem_info.get('current_subject')
    subjects = problem_info.get('subjects', {})
    subject_order = problem_info.get('subject_order', ["국어", "수학", "영어", "탐구1", "탐구2"])
    
    # 현재 과목이 없고, 아직 처리하지 않은 과목이 있어야 함
    if current_subject is not None:
        return False
    
    # 다음 처리할 과목이 있는지 확인
    next_subject = None
    for subject in subject_order:
        if not subjects.get(subject, {}).get("solved", False):
            next_subject = subject
            break
    
    if next_subject is None:
        return False  # 모든 과목 완료
    
    # 질문 없이도 바로 첫 번째 과목 문제점 제시 (9exam_feedback 상태로 전환되면 자동으로 시작)
    # 조언 후 다음 과목으로 넘어가는 경우도 자동으로 발동
    completed_count = problem_info.get('completed_count', 0)
    
    # 첫 번째 과목이든 조언 후든 모두 자동으로 발동
    print(f"[TRIGGER] september_exam_subject_problem 트리거 발동: '{user_message}' (자동 시작, 완료: {completed_count})")
    return True

