"""사용자 데이터 관리 유틸리티 모듈

사용자 게임 데이터를 파일로 저장하고 로드합니다.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Callable

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def save_user_data(
    username: str,
    get_affection: Callable[[], int],
    get_game_state: Callable[[], str],
    get_abilities: Callable[[], Dict],
    get_selected_subjects: Callable[[], list],
    get_schedule: Callable[[], Dict],
    get_conversation_count: Callable[[], int],
    get_current_week: Callable[[], int],
    get_game_date: Callable[[], str],
    get_stamina: Callable[[], int],
    get_mental: Callable[[], int],
    get_mock_exam_last_week: Callable[[], int],
    get_career: Callable[[], Optional[str]],
    get_confidence: Callable[[], int]
):
    """
    사용자 게임 데이터를 JSON 파일로 저장

    Args:
        username: 사용자 이름
        get_*: 각 데이터를 가져오는 콜백 함수들
    """
    try:
        user_data = {
            "affection": get_affection(),
            "game_state": get_game_state(),
            "abilities": get_abilities(),
            "selected_subjects": get_selected_subjects(),
            "schedule": get_schedule(),
            "conversation_count": get_conversation_count(),
            "current_week": get_current_week(),
            "game_date": get_game_date(),
            "stamina": get_stamina(),
            "mental": get_mental(),
            "mock_exam_last_week": get_mock_exam_last_week(),
            "career": get_career(),
            "confidence": get_confidence()
        }

        user_file = BASE_DIR / f"data/users/{username}.json"
        user_file.parent.mkdir(parents=True, exist_ok=True)

        with open(user_file, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)

        print(f"[STORAGE] {username} 데이터 저장 완료")
    except Exception as e:
        print(f"[ERROR] {username} 데이터 저장 실패: {e}")


def load_user_data(
    username: str,
    set_affection: Callable[[int], None],
    set_game_state: Callable[[str], None],
    set_abilities: Callable[[Dict], None],
    set_selected_subjects: Callable[[list], None],
    set_schedule: Callable[[Dict], None],
    set_conversation_count: Callable[[int], None],
    set_current_week: Callable[[int], None],
    set_game_date: Callable[[str], None],
    set_stamina: Callable[[int], None],
    set_mental: Callable[[int], None],
    set_mock_exam_last_week: Callable[[int], None],
    set_career: Callable[[Optional[str]], None],
    set_confidence: Callable[[int], None]
) -> bool:
    """
    사용자 게임 데이터를 JSON 파일에서 로드

    Args:
        username: 사용자 이름
        set_*: 각 데이터를 설정하는 콜백 함수들

    Returns:
        bool: 로드 성공 여부
    """
    try:
        user_file = BASE_DIR / f"data/users/{username}.json"

        if not user_file.exists():
            print(f"[STORAGE] {username} 저장 파일 없음 (새 유저)")
            return False

        with open(user_file, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        # 데이터 로드
        set_affection(user_data.get("affection", 5))
        set_game_state(user_data.get("game_state", "start"))
        set_abilities(user_data.get("abilities", {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0}))
        set_selected_subjects(user_data.get("selected_subjects", []))
        set_schedule(user_data.get("schedule", {}))
        set_conversation_count(user_data.get("conversation_count", 0))
        set_current_week(user_data.get("current_week", 0))
        set_game_date(user_data.get("game_date", "2023-11-17"))
        set_stamina(user_data.get("stamina", 30))
        set_mental(user_data.get("mental", 40))
        set_mock_exam_last_week(user_data.get("mock_exam_last_week", -1))
        set_confidence(user_data.get("confidence", 50))

        # 진로 로드 (없으면 None)
        existing_career = user_data.get("career")
        if existing_career and set_career:
            set_career(existing_career)

        print(f"[STORAGE] {username} 데이터 로드 완료")
        return True
    except Exception as e:
        print(f"[ERROR] {username} 데이터 로드 실패: {e}")
        return False

