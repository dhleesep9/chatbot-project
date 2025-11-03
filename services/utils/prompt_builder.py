"""프롬프트 빌더 유틸리티 모듈

시스템 프롬프트와 사용자 프롬프트를 구성합니다.
"""

from typing import Optional, List, Dict


def get_affection_tone(config: Dict, affection: int) -> str:
    """
    호감도 구간에 따른 말투 지시사항 반환 (chatbot_config.json에서만 읽어옴)
    
    Args:
        config: chatbot_config.json 설정
        affection: 호감도 (0~100)
    
    Returns:
        str: 말투 지시사항
    """
    affection_config = config.get("affection_tone", {})

    # config가 없으면 경고하고 빈 문자열 반환
    if not affection_config:
        print("[WARN] chatbot_config.json에 affection_tone 설정이 없습니다.")
        return ""

    # 호감도 구간에 따라 config에서 읽어오기
    tone_config = None
    if affection <= 10:
        tone_config = affection_config.get("very_low", {})
    elif affection <= 30:
        tone_config = affection_config.get("low", {})
    elif affection <= 50:
        tone_config = affection_config.get("medium", {})
    elif affection <= 70:
        tone_config = affection_config.get("high", {})
    else:  # 70~100
        tone_config = affection_config.get("very_high", {})

    # tone 필드가 배열이면 조인, 문자열이면 그대로 반환
    tone = tone_config.get("tone", None)
    if tone is None:
        print(f"[WARN] 호감도 구간 설정이 없습니다. (affection: {affection})")
        return ""

    # 배열인 경우 \n으로 조인
    if isinstance(tone, list):
        return "\n".join(tone)
    # 문자열인 경우 그대로 반환 (하위 호환성)
    elif isinstance(tone, str):
        return tone
    else:
        print(f"[WARN] tone 필드 형식이 올바르지 않습니다. (affection: {affection})")
        return ""


def build_system_prompt(config: Optional[Dict]) -> str:
    """
    시스템 프롬프트 생성 (캐릭터 설정, 역할 지침, 대화 예시 포함)
    
    Args:
        config: chatbot_config.json 설정
    
    Returns:
        str: 시스템 프롬프트
    """
    if not config:
        return "당신은 재수생입니다."

    system_parts = []

    # 1. 기본 캐릭터 정보
    character = config.get("character", {})
    if character:
        bot_name = config.get("name", "챗봇")
        system_parts.append(f"## 캐릭터 정보")
        system_parts.append(f"당신은 '{bot_name}'입니다.")

        # 나이, 대학, 전공
        if character.get("age"):
            system_parts.append(f"- 나이: {character.get('age')}세")
        if character.get("university"):
            system_parts.append(f"- 대학/상태: {character.get('university')}")
        if character.get("major"):
            system_parts.append(f"- 전공/목표: {character.get('major')}")

        # 성격
        if character.get("personality"):
            system_parts.append(f"\n### 성격")
            system_parts.append(character.get("personality"))

        # 배경
        if character.get("background"):
            system_parts.append(f"\n### 배경")
            system_parts.append(character.get("background"))

        # 주요 고민사항
        concerns = character.get("major_concerns", [])
        if concerns:
            system_parts.append(f"\n### 주요 고민사항")
            for concern in concerns:
                system_parts.append(f"- {concern}")

        # 도움이 필요한 부분
        needs_help = character.get("needs_help_with", [])
        if needs_help:
            system_parts.append(f"\n### 도움이 필요한 부분")
            for need in needs_help:
                system_parts.append(f"- {need}")

        # 역할 지침
        role_directives = character.get("role_directives", {})
        if role_directives:
            system_parts.append(f"\n## 역할 지침")

            # 반드시 따라야 할 규칙
            must_follow = role_directives.get("must_follow_rules", [])
            if must_follow:
                system_parts.append(f"\n### ✅ 반드시 따라야 할 규칙:")
                for i, rule in enumerate(must_follow, 1):
                    system_parts.append(f"{i}. {rule}")

            # 절대 하지 말아야 할 것
            must_not = role_directives.get("must_not_do", [])
            if must_not:
                system_parts.append(f"\n### 🚫 절대 하지 말아야 할 것:")
                for i, rule in enumerate(must_not, 1):
                    system_parts.append(f"{i}. {rule}")

    # 2. 대화 예시
    dialogue_examples = config.get("dialogue_examples", {})
    if dialogue_examples:
        system_parts.append(f"\n## 대화 예시")

        # 도움 요청 시
        asking = dialogue_examples.get("asking_for_help", [])
        if asking:
            system_parts.append(f"\n### 도움을 요청할 때:")
            for example in asking:
                system_parts.append(f"- \"{example}\"")

        # 불안감 표현 시
        anxiety = dialogue_examples.get("expressing_anxiety", [])
        if anxiety:
            system_parts.append(f"\n### 불안감을 표현할 때:")
            for example in anxiety:
                system_parts.append(f"- \"{example}\"")

        # 멘토 조언에 반응할 때
        reacting = dialogue_examples.get("reacting_to_mentor_advice", [])
        if reacting:
            system_parts.append(f"\n### 멘토의 조언에 반응할 때:")
            for example in reacting:
                system_parts.append(f"- \"{example}\"")

    return "\n".join(system_parts)


def build_user_prompt(
    user_message: str,
    context: Optional[str] = None,
    username: str = "사용자",
    game_state: str = "ice_break",
    state_context: str = "",
    selected_subjects: Optional[List[str]] = None,
    schedule_set: bool = False,
    official_mock_exam_grade_info: Optional[Dict] = None,
    current_week: int = 0,
    last_mock_exam_week: int = -1
) -> str:
    """
    LLM 프롬프트 구성 (호감도 및 게임 상태 반영)
    
    Returns:
        str: 구성된 프롬프트
    """
    if selected_subjects is None:
        selected_subjects = []

    # 프롬프트 시작 (호감도 말투가 메인)
    prompt_parts = []
    
    # 게임 상태 컨텍스트 추가
    if state_context.strip():
        prompt_parts.append(state_context.strip())
    
    # 정규모의고사 피드백 상태에서 등급 정보 추가
    if game_state == "official_mock_exam_feedback" and official_mock_exam_grade_info:
        avg_grade = official_mock_exam_grade_info.get("average_grade", 9.0)
        grade_reaction = official_mock_exam_grade_info.get("grade_reaction", "")
        
        # 등급대별로 다른 응답 가이드 제공
        if avg_grade <= 2.0:
            grade_guide = "학생의 평균 등급은 1-2등급입니다. 이는 매우 우수한 성적입니다. 격려와 함께 더 높은 목표를 제시하되, 자신감을 갖도록 도와주세요."
        elif avg_grade <= 4.0:
            grade_guide = "학생의 평균 등급은 3-4등급입니다. 좋은 성적입니다. 칭찬과 함께 조금만 더 노력하면 더 좋아질 수 있다고 격려해주세요."
        elif avg_grade <= 6.0:
            grade_guide = "학생의 평균 등급은 5-6등급입니다. 아쉬운 성적입니다. 좌절하지 말고 차근차근 기본기를 다지면 개선될 수 있다고 격려해주세요."
        elif avg_grade <= 8.0:
            grade_guide = "학생의 평균 등급은 7-8등급입니다. 힘든 성적입니다. 비관하지 말고 기초부터 차근차근 시작하면 된다고 희망을 주세요."
        else:
            grade_guide = "학생의 평균 등급은 9등급입니다. 매우 어려운 성적입니다. 하지만 포기하지 말고 하나씩 배워나가면 좋아질 수 있다고 희망적인 메시지를 전달해주세요."
        
        prompt_parts.append(f"[정규모의고사 성적 정보]\n평균 등급: {avg_grade:.1f}등급\n등급대별 반응: {grade_reaction}\n\n[응답 가이드]\n{grade_guide}\n\n학생이 '결과가 좋지 않다', '성적이 나쁘다', '어떻게 해야 하죠' 등의 말을 할 때는 위 등급 정보를 고려하여 적절하게 응답하세요. 등급이 높을수록(수치가 클수록) 더 따뜻하고 격려하는 말을 해주세요.")

    # 선택과목 정보 추가 (icebreak 또는 mentoring 단계)
    if game_state in ["icebreak", "mentoring"]:
        if selected_subjects:
            subjects_text = ", ".join(selected_subjects)
            prompt_parts.append(f"[현재 선택된 탐구과목: {subjects_text}]")
            if len(selected_subjects) < 2:
                prompt_parts.append(f"(아직 {2 - len(selected_subjects)}개 더 선택할 수 있습니다.)")
        else:
            prompt_parts.append("[선택된 탐구과목: 없음]")
            prompt_parts.append("(아직 탐구과목을 선택하지 않았습니다. 자연스럽게 선택과목을 선택하도록 유도하세요.)")

    # 시간표 설정 안내 (daily_routine 단계에서는 14시간 제한 정보를 주지 않음)
    if game_state == "daily_routine":
        if not schedule_set:
            prompt_parts.append("[중요] 아직 주간 학습 시간표가 설정되지 않았습니다. 플레이어에게 '학습 시간표 관리'를 통해 시간표를 설정하도록 자연스럽게 안내하세요. 14시간 제한이나 구체적인 시간표 형식은 언급하지 마세요.")
        else:
            # 시간표가 이미 설정된 경우, 시간표에 대해 언급하지 말 것
            prompt_parts.append("[중요] 시간표는 이미 설정되어 있습니다. 시간표, 학습 시간, 시간표 관리, 시간 분배 등 시간표와 관련된 내용은 절대 언급하지 마세요. 시간표가 언급되면 자연스럽게 다른 주제로 대화를 이어가세요.")
        
        # 사설모의고사 한 주에 한 번 제한 안내
        if current_week == last_mock_exam_week and last_mock_exam_week >= 0:
            prompt_parts.append(f"[중요] 이번 주({current_week}주차)에 이미 사설모의고사를 봤습니다. 플레이어가 '사설모의고사 응시'를 요청하면, 이미 이번 주에 봤다는 것을 자연스럽게 알려주고 다음 주에 볼 수 있다고 안내하세요.")
    
    # 6exam_feedback 또는 9exam_feedback 상태에서는 절대로 여러 과목을 한 번에 말하지 않도록 지시
    if game_state == "6exam_feedback" or game_state == "9exam_feedback":
        prompt_parts.append("[중요] 절대로 여러 과목(국어, 수학, 영어, 탐구1, 탐구2)을 한 번에 말하지 마세요. 현재 대화하고 있는 과목 하나만 얘기하세요. 예를 들어, 국어에 대해 얘기하고 있다면 국어만 언급하고 수학, 영어, 탐구 등을 함께 말하지 마세요.")

    # 프롬프트 조립
    sys_prompt = "\n\n".join(prompt_parts)

    prompt = sys_prompt.strip() + "\n\n"
    if context:
        prompt += "[참고 정보]\n" + context.strip() + "\n\n"
    prompt += f"{username}: {user_message.strip()}"
    return prompt

