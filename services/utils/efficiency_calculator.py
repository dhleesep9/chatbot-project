"""효율 계산 유틸리티 모듈

체력과 멘탈에 따른 능력치 증가 효율을 계산합니다.
"""


def calculate_stamina_efficiency(stamina: int) -> float:
    """
    체력에 따른 능력치 증가 효율 계산
    공식: 효율(%) = 100 + (체력 - 30)
    예시:
    - 체력 30: 100%
    - 체력 31: 101%
    - 체력 29: 99%
    - 체력 20: 90%
    - 체력 100: 170%
    """
    return 100 + (stamina - 30)


def calculate_mental_efficiency(mental: int) -> float:
    """
    멘탈에 따른 능력치 증가 효율 계산
    공식: 효율(%) = 100 + (멘탈 - 40)
    예시:
    - 멘탈 40: 100%
    - 멘탈 50: 110%
    - 멘탈 30: 90%
    - 멘탈 100: 160%
    """
    return 100 + (mental - 40)


def calculate_combined_efficiency(stamina: int, mental: int) -> float:
    """
    체력과 멘탈의 곱연산으로 최종 효율 계산
    공식: (체력 효율 * 멘탈 효율) / 100
    예시:
    - 체력 31(101%), 멘탈 50(110%): 101 * 110 / 100 = 111.1%
    - 체력 30(100%), 멘탈 40(100%): 100 * 100 / 100 = 100%
    """
    stamina_eff = calculate_stamina_efficiency(stamina)
    mental_eff = calculate_mental_efficiency(mental)
    return (stamina_eff * mental_eff) / 100.0

