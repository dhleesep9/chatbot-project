"""State Handler Registry

각 state별 handler를 등록하고 호출하는 레지스트리
"""

from typing import Dict, Any, Optional, Type
from services.handlers.base_handler import BaseStateHandler


class HandlerRegistry:
    """State handler 레지스트리"""

    def __init__(self):
        self._handlers: Dict[str, BaseStateHandler] = {}

    def register(self, state_name: str, handler: BaseStateHandler):
        """
        State handler 등록

        Args:
            state_name: state 이름
            handler: handler 인스턴스
        """
        self._handlers[state_name] = handler
        print(f"[HANDLER_REGISTRY] '{state_name}' handler 등록 완료")

    def get_handler(self, state_name: str) -> Optional[BaseStateHandler]:
        """
        State handler 가져오기

        Args:
            state_name: state 이름

        Returns:
            handler 인스턴스 또는 None
        """
        return self._handlers.get(state_name)

    def has_handler(self, state_name: str) -> bool:
        """
        State handler 존재 여부 확인

        Args:
            state_name: state 이름

        Returns:
            bool: handler 존재 여부
        """
        return state_name in self._handlers

    def call_on_enter(self, state_name: str, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        State 진입 시 handler 호출

        Args:
            state_name: state 이름
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Optional[Dict]: 처리 결과
        """
        handler = self.get_handler(state_name)
        if handler:
            print(f"[HANDLER] '{state_name}' on_enter 호출")
            return handler.on_enter(username, context)
        return None

    def call_handle(self, state_name: str, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        사용자 입력 처리 시 handler 호출

        Args:
            state_name: state 이름
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Optional[Dict]: 처리 결과
        """
        handler = self.get_handler(state_name)
        if handler:
            print(f"[HANDLER] '{state_name}' handle 호출")
            return handler.handle(username, user_message, context)
        return None

    def call_on_exit(self, state_name: str, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        State 종료 시 handler 호출

        Args:
            state_name: state 이름
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Optional[Dict]: 처리 결과
        """
        handler = self.get_handler(state_name)
        if handler:
            print(f"[HANDLER] '{state_name}' on_exit 호출")
            return handler.on_exit(username, context)
        return None
