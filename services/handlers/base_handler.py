"""Base State Handler

모든 state handler의 기본 인터페이스를 정의합니다.
각 state는 진입 시(on_enter)와 사용자 입력 처리 시(handle) 로직을 구현할 수 있습니다.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseStateHandler(ABC):
    """State handler 기본 클래스"""

    def __init__(self, service):
        """
        Args:
            service: ChatbotService 인스턴스 (메서드 접근용)
        """
        self.service = service

    def on_enter(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        State 진입 시 호출되는 메서드

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트 (user_message, current_state 등)

        Returns:
            Optional[Dict]: 처리 결과 (reply, narration, data 등)
                           None이면 정상 LLM 호출 진행
        """
        return None

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        사용자 입력 처리

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Optional[Dict]: 처리 결과 (reply, narration, data 등)
                           None이면 정상 LLM 호출 진행
        """
        return None

    def on_exit(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        State 종료 시 호출되는 메서드

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Optional[Dict]: 처리 결과
        """
        return None
