"""
탐구과목 선택 모듈

이 모듈은 selection 상태에서만 작동하며, 사용자가 입력한 메시지에서
탐구과목 2개를 파싱하여 저장합니다.
"""

import re

# 선택 가능한 탐구과목 목록
SUBJECT_OPTIONS = [
    "사회문화", "정치와법", "경제", "세계지리", "한국지리",
    "생활과윤리", "윤리와사상", "세계사", "동아시아사",
    "물리학1", "화학1", "지구과학1", "생명과학1",
    "물리학2", "화학2", "지구과학2", "생명과학2"
]


def parse_subjects_from_message(user_message: str) -> list:
    """
    사용자 메시지에서 탐구과목 추출

    Args:
        user_message: 사용자 입력 메시지

    Returns:
        추출된 과목 리스트 (최대 2개)
    """
    found_subjects = []

    # 메시지를 정규화 (공백 제거)
    normalized_message = user_message.replace(" ", "")

    # 각 과목이 메시지에 포함되어 있는지 확인
    for subject in SUBJECT_OPTIONS:
        # 공백 없는 과목명으로 체크
        normalized_subject = subject.replace(" ", "")

        if normalized_subject in normalized_message:
            found_subjects.append(subject)

            # 2개 찾으면 중단
            if len(found_subjects) >= 2:
                break

    return found_subjects[:2]  # 최대 2개만 반환


def validate_subject_count(subjects: list, required_count: int = 2) -> bool:
    """
    선택된 과목 개수가 요구사항을 만족하는지 확인

    Args:
        subjects: 선택된 과목 리스트
        required_count: 필요한 과목 개수 (기본값: 2)

    Returns:
        조건 만족 여부
    """
    return len(subjects) >= required_count


def get_subject_list_text() -> str:
    """
    선택 가능한 과목 목록을 텍스트로 반환

    Returns:
        과목 목록 문자열
    """
    return ", ".join(SUBJECT_OPTIONS)
