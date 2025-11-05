"""
테스트 스크립트: 조건부 캐릭터 정보 포함 로직 확인

conversation_count가 1, 5, 11, 21, 31, 41, ... 일 때만 전체 캐릭터 정보가 포함되는지 확인
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# services/__init__.py의 import를 피하기 위해 직접 파일을 import
import importlib.util
spec = importlib.util.spec_from_file_location(
    "prompt_builder",
    os.path.join(os.path.dirname(__file__), "services", "utils", "prompt_builder.py")
)
prompt_builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompt_builder)

should_include_full_character_info = prompt_builder.should_include_full_character_info


def test_should_include_full_character_info():
    """조건부 포함 로직 테스트"""
    print("=" * 60)
    print("조건부 캐릭터 정보 포함 로직 테스트")
    print("=" * 60)

    # 포함되어야 하는 conversation_count
    should_include = [1, 5, 11, 21, 31, 41, 51, 61, 71, 81, 91, 101]

    # 포함되지 않아야 하는 conversation_count
    should_not_include = [0, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35]

    print("\n✅ 포함되어야 하는 conversation_count:")
    all_correct_include = True
    for count in should_include:
        result = should_include_full_character_info(count)
        status = "✓" if result else "✗"
        if not result:
            all_correct_include = False
        print(f"  {status} conversation_count={count}: {result}")

    print("\n❌ 포함되지 않아야 하는 conversation_count:")
    all_correct_exclude = True
    for count in should_not_include:
        result = should_include_full_character_info(count)
        status = "✓" if not result else "✗"
        if result:
            all_correct_exclude = False
        print(f"  {status} conversation_count={count}: {result}")

    print("\n" + "=" * 60)
    if all_correct_include and all_correct_exclude:
        print("✅ 모든 테스트 통과!")
        return True
    else:
        print("❌ 일부 테스트 실패")
        return False


def test_pattern():
    """1 + 10n 패턴 테스트"""
    print("\n" + "=" * 60)
    print("1 + 10n 패턴 테스트 (31부터 100까지)")
    print("=" * 60)

    expected_pattern = [31, 41, 51, 61, 71, 81, 91]

    print("\n예상되는 패턴:")
    print(f"  {expected_pattern}")

    actual_pattern = []
    for i in range(31, 101):
        if should_include_full_character_info(i):
            actual_pattern.append(i)

    print("\n실제 패턴:")
    print(f"  {actual_pattern}")

    if expected_pattern == actual_pattern:
        print("\n✅ 패턴 일치!")
        return True
    else:
        print("\n❌ 패턴 불일치")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("조건부 캐릭터 정보 포함 로직 테스트 시작")
    print("=" * 60 + "\n")

    results = []

    # 테스트 1: 기본 로직
    results.append(("기본 로직", test_should_include_full_character_info()))

    # 테스트 2: 패턴 확인
    results.append(("패턴 확인", test_pattern()))

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
