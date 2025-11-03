"""체력 관리 유틸리티 모듈

운동 시간에 따른 체력 증가 로직을 처리합니다.
"""


def apply_exercise_stamina_increase(stamina: int, exercise_hours: int) -> int:
    """
    운동 시간만큼 체력 증가
    
    Args:
        stamina: 현재 체력
        exercise_hours: 운동 시간
    
    Returns:
        새로운 체력 (최대 100)
    """
    return min(100, stamina + exercise_hours)


def process_schedule_for_stamina(schedule: dict, current_stamina: int, set_stamina_callback) -> None:
    """
    시간표에서 운동 시간을 추출하여 체력 증가 처리
    
    Args:
        schedule: 시간표 딕셔너리 {"국어": 4, "수학": 4, "운동": 2, ...}
        current_stamina: 현재 체력
        set_stamina_callback: 체력 설정 콜백 함수 (username, new_stamina)
    """
    if not schedule or "운동" not in schedule:
        return
    
    exercise_hours = schedule.get("운동", 0)
    if exercise_hours > 0:
        new_stamina = apply_exercise_stamina_increase(current_stamina, exercise_hours)
        # 콜백 함수는 username을 받아야 하므로 여기서는 반환만 하고
        # 실제 호출은 chatbot_service.py에서 처리
        return new_stamina
    
    return None


