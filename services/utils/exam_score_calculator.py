"""시험 점수 계산 유틸리티 모듈

능력치를 기반으로 시험 성적을 계산합니다.
"""

import math
import random
from typing import Dict, Optional


def calculate_percentile(ability: int) -> float:
    """
    능력치를 백분위로 변환
    공식: 2 * sqrt(능력치)
    """
    if ability <= 0:
        return 0.0
    percentile = 2 * math.sqrt(ability)
    return min(100.0, percentile)  # 최대 100%


def calculate_grade_from_percentile(percentile: float) -> int:
    """
    백분위를 등급으로 변환 (수능 등급 체계)
    1등급: 96~100
    2등급: 89~95
    3등급: 77~88
    4등급: 60~76
    5등급: 40~59
    6등급: 23~39
    7등급: 11~22
    8등급: 4~10
    9등급: 1~3
    """
    if percentile >= 96:
        return 1
    elif percentile >= 89:
        return 2
    elif percentile >= 77:
        return 3
    elif percentile >= 60:
        return 4
    elif percentile >= 40:
        return 5
    elif percentile >= 23:
        return 6
    elif percentile >= 11:
        return 7
    elif percentile >= 4:
        return 8
    else:
        return 9


def calculate_exam_scores(abilities: Dict[str, int], strategy_bonus: float = 0.0) -> Dict[str, Dict]:
    """
    능력치를 기반으로 시험 성적 계산 (전략 보너스 포함)
    반환값: {"국어": {"grade": 1, "percentile": 85.5}, "수학": {"grade": 2, "percentile": 90.2}, ...}
    
    Args:
        abilities: 능력치 딕셔너리 {"국어": 100, "수학": 200, ...}
        strategy_bonus: 전략 보너스 (0.0~0.2) - 전략 품질에 따라 추가됨
    """
    scores = {}
    
    for subject, ability in abilities.items():
        # 전략 보너스 적용 (능력치에 추가)
        adjusted_ability = ability * (1.0 + strategy_bonus)
        percentile = calculate_percentile(adjusted_ability)
        grade = calculate_grade_from_percentile(percentile)
        scores[subject] = {
            "grade": grade,
            "percentile": round(percentile, 1)
        }
    
    return scores


def identify_weak_subject(exam_scores: Dict[str, Dict]) -> str:
    """
    시험 점수에서 가장 취약한 과목 식별 (등급이 가장 낮은 과목)
    """
    if not exam_scores:
        return "수학"  # 기본값
    
    # 등급이 가장 높은(숫자가 큰) 과목을 취약 과목으로 선택
    weak_subject = max(exam_scores.items(), key=lambda x: x[1]['grade'])
    return weak_subject[0]


def generate_weakness_message(subject: str, score_data: Dict) -> str:
    """
    취약 과목에 대한 취약점 메시지 생성 (과목별 다양한 예시)
    """
    weakness_examples = {
        "국어": [
            "국어에서 선택과목 시간에 시간을 다 써버려서 비문학 지문을 제대로 읽지 못했어요...",
            "국어에서 문학 작품 해석이 너무 어려웠어요. 작가의 의도를 파악하지 못했어요.",
            "국어 비문학 지문이 너무 길어서 읽는 속도가 느렸어요. 시간이 부족했어요.",
            "국어에서 고전 문학 부분을 제대로 이해하지 못했어요. 한자어가 많아서 어려웠어요.",
            "국어 화법 작문에서는 시간이 부족해서 대충 썼어요. 구조화된 글쓰기가 어려웠어요."
        ],
        "수학": [
            "수학에서 미적분 문제를 풀다가 시간이 너무 많이 걸렸어요...",
            "수학 기하 문제에서 도형을 그려도 풀이 방법이 생각이 안 났어요.",
            "수학에서 확률과 통계 부분을 완전히 틀렸어요. 경우의 수를 세는 게 헷갈렸어요.",
            "수학에서 삼각함수 문제가 너무 어려웠어요. 공식을 외웠는데 적용이 안 됐어요.",
            "수학 계산 실수가 너무 많았어요. 과정은 맞는데 답이 틀렸어요."
        ],
        "영어": [
            "영어에서 독해 지문을 읽고 문제를 풀 때 시간이 부족했어요...",
            "영어 어휘 문제에서 모르는 단어가 너무 많아서 문맥으로 유추했는데 틀렸어요.",
            "영어 문법 문제를 풀 때 시제를 헷갈려서 틀렸어요.",
            "영어에서 빈칸 채우기 문제가 어려웠어요. 문맥을 파악하지 못했어요.",
            "영어 작문 문제에서 표현이 자연스럽지 않아서 점수를 많이 깎였어요."
        ],
        "탐구1": [
            "탐구1에서 개념 문제는 알겠는데, 응용 문제가 너무 어려웠어요...",
            "탐구1에서 실험 문제를 풀 때 실험 과정을 제대로 이해하지 못했어요.",
            "탐구1에서 그래프 분석 문제가 헷갈렸어요. 데이터를 읽는 게 어려웠어요.",
            "탐구1에서 서술형 문제에서 답은 맞는데 표현이 부족해서 점수를 못 받았어요.",
            "탐구1에서 선택지가 비슷비슷해서 구분하기가 어려웠어요."
        ],
        "탐구2": [
            "탐구2에서 시간 분배가 안 되어서 마지막 문제들을 대충 풀었어요...",
            "탐구2에서 개념 연결 문제가 너무 어려웠어요. 서로 다른 개념을 연결하는 게 힘들었어요.",
            "탐구2에서 계산 문제에서 단위 변환을 실수했어요.",
            "탐구2에서 문제 해석이 어려웠어요. 문제가 뭘 요구하는지 모르겠었어요.",
            "탐구2에서 기출 문제는 풀었는데, 새로 나온 유형은 전혀 몰랐어요."
        ]
    }
    
    # 과목별 예시 가져오기
    examples = weakness_examples.get(subject, weakness_examples.get("수학", []))
    
    # 예시가 없으면 기본 메시지 생성
    if not examples or len(examples) == 0:
        return f"{subject}에서 어려운 부분이 많았어요. 특히 응용 문제가 어려웠어요."
    
    # 랜덤으로 선택
    selected_message = random.choice(examples)
    
    # 선택된 메시지가 비어있으면 기본 메시지 반환
    if not selected_message or len(selected_message.strip()) == 0:
        return f"{subject}에서 어려운 부분이 많았어요. 특히 응용 문제가 어려웠어요."
    
    return selected_message


def calculate_average_grade(exam_scores: Dict[str, Dict]) -> float:
    """
    시험 점수 딕셔너리에서 평균 등급 계산
    
    Args:
        exam_scores: {"국어": {"grade": 1, ...}, "수학": {"grade": 2, ...}, ...}
    
    Returns:
        float: 평균 등급 (소수점 첫째자리)
    """
    if not exam_scores:
        return 9.0
    
    total_grade = 0
    count = 0
    for subject, score_data in exam_scores.items():
        if 'grade' in score_data:
            total_grade += score_data['grade']
            count += 1
    
    if count == 0:
        return 9.0
    
    return round(total_grade / count, 1)


def generate_grade_reaction(exam_type: str, average_grade: float) -> str:
    """
    등급대별 시험 결과 반응 메시지 생성
    
    Args:
        exam_type: "mock_exam" (사설모의고사), "official_mock_exam" (정규모의고사), "june_exam" (6월 모의고사)
        average_grade: 평균 등급 (1.0~9.0)
    
    Returns:
        str: 등급대별 반응 메시지
    """
    # reactions는 현재 비어있으므로 기본 메시지 반환
    return "괜찮게 봤어요."


def generate_june_subject_problem(subject: str, score_data: Dict) -> str:
    """
    6월 모의고사 과목별 취약점 메시지 생성
    generate_weakness_message를 사용하여 취약점을 제시합니다.
    """
    return generate_weakness_message(subject, score_data)


def is_official_mock_exam_month(exam_month: str) -> bool:
    """
    정규모의고사 월인지 확인 (3, 4, 5, 7, 10월)
    6월, 9월, 수능(11월)은 False 반환
    """
    if not exam_month:
        return False
    try:
        month_num = int(exam_month.split("-")[1])
        return month_num in [3, 4, 5, 7, 10]
    except:
        return False

