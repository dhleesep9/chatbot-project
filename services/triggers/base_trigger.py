"""
트리거 함수 베이스 템플릿

커스텀 트리거를 만들려면:
1. services/triggers/ 디렉토리에 새 파일 생성 (예: my_custom_trigger.py)
2. evaluate() 함수 구현 (아래 시그니처 참고)
3. 파일명이 trigger_type과 일치해야 함 (my_custom_trigger.py -> "my_custom_trigger")
4. state JSON에서 "trigger_type": "my_custom_trigger" 사용

예시:
    # services/triggers/week_threshold.py
    def evaluate(transition: dict, context: dict) -> bool:
        conditions = transition.get("conditions", {})
        min_week = conditions.get("min_week", 1)
        current_week = context['service']._get_current_week(context['username'])
        return current_week >= min_week
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.chatbot_service import ChatbotService


def evaluate(transition: dict, context: dict) -> bool:
    """
    트리거 조건 평가 함수 (모든 트리거가 구현해야 하는 표준 인터페이스)

    Args:
        transition: state JSON의 transition 객체
            {
                "trigger_type": "트리거타입",
                "conditions": {...},  # 트리거별 조건
                "next_state": "다음상태",
                "transition_narration": "전환 나레이션"
            }

        context: 트리거 실행 컨텍스트
            {
                'username': str,              # 사용자 이름
                'user_message': str,          # 사용자 입력 메시지
                'affection_increased': int,   # 이번 턴 호감도 증가량
                'service': ChatbotService     # 챗봇 서비스 인스턴스 (헬퍼 메서드 호출용)
            }

    Returns:
        bool: 트리거 조건 만족 여부

    Example:
        def evaluate(transition: dict, context: dict) -> bool:
            conditions = transition.get("conditions", {})
            min_value = conditions.get("min_value", 0)

            # context에서 필요한 데이터 추출
            username = context['username']
            service = context['service']

            # 조건 체크
            current_value = service._get_some_value(username)
            return current_value >= min_value
    """
    raise NotImplementedError("evaluate() 함수를 구현해야 합니다")
