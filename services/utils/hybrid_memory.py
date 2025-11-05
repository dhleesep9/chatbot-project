"""하이브리드 대화 메모리 시스템

ConversationBufferWindowMemory + ConversationSummaryMemory를 혼합하여 사용합니다.
- 최근 3-4턴은 그대로 보존 (BufferWindow)
- 오래된 대화는 1문단으로 요약 (Summary)
"""

from typing import Dict, List, Optional
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage


class HybridConversationMemory:
    """
    하이브리드 대화 메모리 클래스

    - 최근 k개의 대화 턴은 ConversationBufferWindowMemory로 보존
    - 오래된 대화는 ConversationSummaryMemory로 요약
    """

    def __init__(self, llm, window_size: int = 4, max_summary_tokens: int = 300):
        """
        Args:
            llm: LangChain LLM 인스턴스 (요약용)
            window_size: 최근 보존할 대화 턴 수 (기본 4턴)
            max_summary_tokens: 요약 최대 토큰 수 (기본 300)
        """
        self.llm = llm
        self.window_size = window_size
        self.max_summary_tokens = max_summary_tokens

        # BufferWindow Memory: 최근 k턴 보존
        self.buffer_memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True,
            memory_key="recent_history"
        )

        # Summary Memory: 오래된 대화 요약
        self.summary_memory = ConversationSummaryMemory(
            llm=llm,
            return_messages=False,
            memory_key="summary",
            max_token_limit=max_summary_tokens
        )

        # 전체 대화 기록 (내부 관리용)
        self.full_history: List[Dict[str, str]] = []

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """
        대화 컨텍스트 저장

        Args:
            inputs: {"input": "사용자 메시지"}
            outputs: {"output": "챗봇 응답"}
        """
        user_message = inputs.get("input", "")
        bot_response = outputs.get("output", "")

        # 전체 히스토리에 추가
        self.full_history.append({
            "role": "user",
            "content": user_message
        })
        self.full_history.append({
            "role": "assistant",
            "content": bot_response
        })

        # BufferWindow에 저장 (최근 k턴만 유지됨)
        self.buffer_memory.save_context(inputs, outputs)

        # SummaryMemory에도 저장 (요약용)
        self.summary_memory.save_context(inputs, outputs)

    def get_formatted_memory(self) -> str:
        """
        포맷된 메모리 문자열 반환

        Returns:
            str: "## 과거 대화 요약\n...\n\n## 최근 대화\n..."
        """
        memory_parts = []

        # 1. 오래된 대화 요약 (Summary Memory)
        try:
            summary = self.summary_memory.load_memory_variables({}).get("summary", "")

            # 요약이 있고, 전체 히스토리가 window_size*2 보다 크면 요약 포함
            # (window_size*2 = user + assistant 메시지 쌍의 개수)
            if summary and len(self.full_history) > self.window_size * 2:
                memory_parts.append(f"## 과거 대화 요약\n{summary.strip()}")
        except Exception as e:
            print(f"[WARN] Summary Memory 로드 실패: {e}")

        # 2. 최근 대화 (BufferWindow Memory)
        try:
            recent_messages = self.buffer_memory.load_memory_variables({}).get("recent_history", [])

            if recent_messages:
                recent_lines = []
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        recent_lines.append(f"사용자: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        recent_lines.append(f"챗봇: {msg.content}")

                if recent_lines:
                    memory_parts.append(f"## 최근 대화\n" + "\n".join(recent_lines))
        except Exception as e:
            print(f"[WARN] BufferWindow Memory 로드 실패: {e}")

        # 메모리가 있으면 결합, 없으면 빈 문자열
        if memory_parts:
            return "\n\n".join(memory_parts)
        return ""

    def clear(self) -> None:
        """메모리 초기화"""
        self.buffer_memory.clear()
        self.summary_memory.clear()
        self.full_history = []

    def get_conversation_count(self) -> int:
        """전체 대화 턴 수 반환 (user + assistant 쌍)"""
        return len(self.full_history) // 2


class HybridMemoryManager:
    """
    사용자별 하이브리드 메모리 관리 클래스
    """

    def __init__(self, llm, window_size: int = 4):
        """
        Args:
            llm: LangChain LLM 인스턴스
            window_size: 최근 보존할 대화 턴 수
        """
        self.llm = llm
        self.window_size = window_size
        self.memories: Dict[str, HybridConversationMemory] = {}

    def get_memory(self, username: str) -> HybridConversationMemory:
        """
        사용자별 메모리 가져오기 (없으면 생성)

        Args:
            username: 사용자 이름

        Returns:
            HybridConversationMemory: 사용자별 메모리 인스턴스
        """
        if username not in self.memories:
            self.memories[username] = HybridConversationMemory(
                llm=self.llm,
                window_size=self.window_size
            )
        return self.memories[username]

    def save_context(self, username: str, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        """
        사용자별 대화 컨텍스트 저장

        Args:
            username: 사용자 이름
            inputs: {"input": "사용자 메시지"}
            outputs: {"output": "챗봇 응답"}
        """
        memory = self.get_memory(username)
        memory.save_context(inputs, outputs)

    def get_formatted_memory(self, username: str) -> str:
        """
        사용자별 포맷된 메모리 문자열 반환

        Args:
            username: 사용자 이름

        Returns:
            str: 포맷된 메모리 문자열
        """
        memory = self.get_memory(username)
        return memory.get_formatted_memory()

    def clear_memory(self, username: str) -> None:
        """
        사용자별 메모리 초기화

        Args:
            username: 사용자 이름
        """
        if username in self.memories:
            self.memories[username].clear()

    def clear_all(self) -> None:
        """모든 사용자 메모리 초기화"""
        for memory in self.memories.values():
            memory.clear()
        self.memories = {}
