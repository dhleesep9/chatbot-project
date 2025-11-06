"""수능 엔딩 트리거

수능 성적과 능력치(체력, 호감도)를 기반으로 엔딩 조건 평가

사용 예시 (state JSON):
{
  "trigger_type": "exam_ending_trigger",
  "conditions": {
    "average_grade_min": 5.0,    # 평균 등급 최소값 (옵션)
    "average_grade_max": 6.0,    # 평균 등급 최대값 (옵션)
    "stamina_min": 80,            # 체력 최소값 (옵션)
    "affection_min": 80           # 호감도 최소값 (옵션)
  },
  "next_state": "ending_state_name"
}
"""


def evaluate(transition: dict, context: dict) -> bool:
    """
    수능 엔딩 조건 체크

    Args:
        transition: 트리거 정의
        context: 실행 컨텍스트

    Returns:
        bool: 모든 조건을 만족하는지 여부
    """
    conditions = transition.get("conditions", {})
    username = context['username']
    service = context['service']

    # 수능 성적 가져오기
    csat_exam_data = service.csat_exam_scores.get(username, {})
    exam_scores = csat_exam_data.get('scores')

    if not exam_scores:
        print(f"[EXAM_ENDING_TRIGGER] {username} 수능 성적이 없음")
        return False

    # 평균 등급 계산
    average_grade = service._calculate_average_grade(exam_scores)
    print(f"[EXAM_ENDING_TRIGGER] {username} 평균 등급: {average_grade}")

    # 평균 등급 조건 체크
    average_grade_min = conditions.get("average_grade_min")
    average_grade_max = conditions.get("average_grade_max")

    if average_grade_min is not None and average_grade < average_grade_min:
        print(f"[EXAM_ENDING_TRIGGER] 평균 등급 {average_grade} < 최소값 {average_grade_min}")
        return False

    if average_grade_max is not None and average_grade > average_grade_max:
        print(f"[EXAM_ENDING_TRIGGER] 평균 등급 {average_grade} > 최대값 {average_grade_max}")
        return False

    # 체력 조건 체크
    stamina_min = conditions.get("stamina_min")
    if stamina_min is not None:
        current_stamina = service._get_stamina(username)
        if current_stamina < stamina_min:
            print(f"[EXAM_ENDING_TRIGGER] 체력 {current_stamina} < 최소값 {stamina_min}")
            return False
        print(f"[EXAM_ENDING_TRIGGER] 체력 조건 만족: {current_stamina} >= {stamina_min}")

    # 호감도 조건 체크
    affection_min = conditions.get("affection_min")
    if affection_min is not None:
        current_affection = service._get_affection(username)
        if current_affection < affection_min:
            print(f"[EXAM_ENDING_TRIGGER] 호감도 {current_affection} < 최소값 {affection_min}")
            return False
        print(f"[EXAM_ENDING_TRIGGER] 호감도 조건 만족: {current_affection} >= {affection_min}")

    print(f"[EXAM_ENDING_TRIGGER] 모든 조건 만족 - {transition.get('next_state')} 엔딩")
    return True
