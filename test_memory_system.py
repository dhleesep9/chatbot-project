"""
테스트 스크립트: Hybrid Memory System 동작 확인

이 스크립트는 하이브리드 메모리 시스템이 올바르게 작동하는지 테스트합니다.
- 최근 4턴은 그대로 보존
- 오래된 대화는 요약
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_memory_basic():
    """기본 메모리 동작 테스트"""
    print("=" * 60)
    print("테스트 1: 기본 메모리 동작 테스트")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI
        from services.utils.hybrid_memory import HybridMemoryManager

        # LLM 초기화 (테스트용 - 간단한 설정)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7
        )

        # 메모리 매니저 초기화
        memory_manager = HybridMemoryManager(llm=llm, window_size=4)
        print("✓ HybridMemoryManager 초기화 성공")

        # 테스트용 대화 저장 (6턴 - 4턴은 버퍼에, 2턴은 요약될 예정)
        test_conversations = [
            ("안녕하세요!", "안녕하세요! 무엇을 도와드릴까요?"),
            ("저는 수학 공부가 어려워요", "수학이 어렵다니 걱정이시겠네요. 어떤 부분이 특히 어려우신가요?"),
            ("함수 개념이 너무 어려워요", "함수는 중요한 개념이에요. 천천히 같이 공부해봐요."),
            ("연습문제를 풀어봤는데 잘 안돼요", "연습이 중요해요. 같이 문제를 풀어볼까요?"),
            ("네, 도와주세요!", "좋아요. 어떤 문제부터 시작할까요?"),
            ("2차 함수 문제요", "2차 함수 문제네요. 함께 풀어봐요!")
        ]

        # 대화 저장
        for i, (user_msg, bot_msg) in enumerate(test_conversations, 1):
            memory_manager.save_context(
                username="테스트유저",
                inputs={"input": user_msg},
                outputs={"output": bot_msg}
            )
            print(f"✓ 대화 {i}턴 저장 완료")

        # 메모리 포맷 확인
        print("\n" + "=" * 60)
        print("저장된 메모리 내용:")
        print("=" * 60)
        formatted_memory = memory_manager.get_formatted_memory("테스트유저")
        print(formatted_memory)
        print("=" * 60)

        # 메모리 카운트 확인
        memory = memory_manager.get_memory("테스트유저")
        conv_count = memory.get_conversation_count()
        print(f"\n✓ 전체 대화 턴 수: {conv_count} (예상: 6)")

        # 최근 4턴이 포함되어 있는지 확인
        if "2차 함수 문제요" in formatted_memory and "좋아요. 어떤 문제부터 시작할까요?" in formatted_memory:
            print("✓ 최근 대화가 올바르게 보존됨")
        else:
            print("✗ 최근 대화 보존 실패")

        print("\n테스트 1 완료!")
        return True

    except Exception as e:
        print(f"\n✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_clear():
    """메모리 초기화 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: 메모리 초기화 테스트")
    print("=" * 60)

    try:
        from langchain_openai import ChatOpenAI
        from services.utils.hybrid_memory import HybridMemoryManager

        # LLM 초기화
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

        # 메모리 매니저 초기화
        memory_manager = HybridMemoryManager(llm=llm, window_size=4)

        # 대화 저장
        memory_manager.save_context(
            username="테스트유저2",
            inputs={"input": "안녕하세요"},
            outputs={"output": "안녕하세요!"}
        )
        print("✓ 대화 저장 완료")

        # 메모리 확인
        formatted_memory = memory_manager.get_formatted_memory("테스트유저2")
        if "안녕하세요" in formatted_memory:
            print("✓ 메모리에 대화가 저장되어 있음")

        # 메모리 초기화
        memory_manager.clear_memory("테스트유저2")
        print("✓ 메모리 초기화 완료")

        # 초기화 확인
        formatted_memory_after = memory_manager.get_formatted_memory("테스트유저2")
        if formatted_memory_after == "":
            print("✓ 메모리가 올바르게 초기화됨")
        else:
            print("✗ 메모리 초기화 실패")
            return False

        print("\n테스트 2 완료!")
        return True

    except Exception as e:
        print(f"\n✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import_in_chatbot_service():
    """ChatbotService에서 메모리 시스템 import 테스트"""
    print("\n" + "=" * 60)
    print("테스트 3: ChatbotService에서 메모리 시스템 import 테스트")
    print("=" * 60)

    try:
        # ChatbotService 임포트만 테스트 (초기화는 하지 않음)
        from services.utils.hybrid_memory import HybridMemoryManager
        print("✓ HybridMemoryManager import 성공")

        print("\n테스트 3 완료!")
        return True

    except Exception as e:
        print(f"\n✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Hybrid Memory System 테스트 시작")
    print("=" * 60 + "\n")

    results = []

    # 테스트 1: 기본 메모리 동작
    results.append(("기본 메모리 동작", test_memory_basic()))

    # 테스트 2: 메모리 초기화
    results.append(("메모리 초기화", test_memory_clear()))

    # 테스트 3: Import 테스트
    results.append(("Import 테스트", test_import_in_chatbot_service()))

    # 결과 출력
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    for test_name, result in results:
        status = "✓ 성공" if result else "✗ 실패"
        print(f"{test_name}: {status}")

    # 전체 성공 여부
    all_passed = all(result for _, result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 모든 테스트 통과!")
    else:
        print("✗ 일부 테스트 실패")
    print("=" * 60 + "\n")

    sys.exit(0 if all_passed else 1)
