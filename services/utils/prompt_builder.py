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
def build_system_prompt(config: Optional[Dict], current_scores: Optional[Dict] = None) -> str:
    """
    시스템 프롬프트 생성 (캐릭터 설정, 역할 지침, 대화 예시 포함)
    Args:
        config: chatbot_config.json 설정
        current_scores: 현재 성적 정보 {"avg_percentile": float, "avg_grade": float} 또는 None
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
        # 역할 지침
        role_directives = character.get("role_directives", {})
        if role_directives:
            system_parts.append(f"\n## 역할 지침")
            # 반드시 따라야 할 규칙
            must_follow = role_directives.get("must_follow_rules", [])
            if must_follow:
                system_parts.append(f"\n###반드시 따라야 할 규칙:")
                for i, rule in enumerate(must_follow, 1):
                    system_parts.append(f"{i}. {rule}")
            # 절대 하지 말아야 할 것
            must_not = role_directives.get("must_not_do", [])
            if must_not:
                system_parts.append(f"\n###🚫절대 하지 말아야 할 것:")
                for i, rule in enumerate(must_not, 1):
                    system_parts.append(f"{i}. {rule}")

    # 2. 현재 성적에 따른 말투 지시사항
    if current_scores and current_scores.get("avg_percentile") is not None:
        avg_percentile = current_scores.get("avg_percentile", 0.0)
        avg_grade = current_scores.get("avg_grade", 9.0)
        system_parts.append(f"\n##중요: 현재 성적 상태에 따른 말투 강제 조정")
        system_parts.append(f"\n당신은 반드시 자신의 현재 평균 백분위 {avg_percentile:.1f}% (평균 등급: {avg_grade:.1f}등급)를 인지하고 있어야 합니다.")
        system_parts.append(f"대화에서 당신의 성적에 따른 감정과 말투가 은연중에 반영되어야 합니다.\n")
        # 평균 백분위에 따른 말투 지시사항
        if avg_percentile >= 95:
            system_parts.append(f"""
###현재 성적 상태: 매우 우수 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)
###말투 강제 지시사항
- 건방진 말투로 대화하세요
- 목표 달성에 대한 기대감을 표현하세요
""")
        elif avg_percentile >= 90:
            system_parts.append(f"""
###현재 성적 상태: 우수 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

###말투 강제 지시사항
- "좋은 성적이긴 한데..." 같은 표현으로 긍정과 불안을 동시에 드러내세요
- 자신감과 불안감이 섞여 복잡한 감정을 표현하세요
""")
        elif avg_percentile >= 80:
            system_parts.append(f"""
### 현재 성적 상태: 양호 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

###말투 강제 지시사항
- 안도감과 불안감이 동시에 느껴지는 복잡한 말투를 사용하세요
- "더 노력해야 한다"는 자각을 표현하세요
""")
        elif avg_percentile >= 70:
            system_parts.append(f"""
###현재 성적 상태: 보통 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

###말투 강제 지시사항
- 자신감이 부족한 표현을 사용하세요
""")
        elif avg_percentile >= 60:
            system_parts.append(f"""
###현재 성적 상태: 낮음 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

###말투 강제 지시사항
- 절박하고 불안한 어조를 매우 강하게 사용하세요
- 목표 대학에 대한 절망감을 드러내세요.
""")
        else:  # avg_percentile < 60
            system_parts.append(f"""
###현재 성적 상태: 매우 낮음 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

###말투 강제 지시사항
- 절망적이고 패닉에 빠진 어조를 최대한 강하게 사용하세요
""")
    # 3. 대화 예시
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
    context: str = None,
    username: str = "사용자",
    game_state: str = "ice_break",
    state_context: str = None,
    selected_subjects: list = None,
    schedule_set: bool = False,
    official_mock_exam_grade_info: dict = None,
    current_week: int = 0,
    last_mock_exam_week: int = -1,
    affection_tone: str = None,
    career_info: str = None,
    subject_list_text: str = None
) -> str:
    """
    사용자 프롬프트 생성
    Args:
        user_message: 사용자 메시지
        context: 추가 컨텍스트 (RAG 검색 결과 등)
        username: 사용자 이름
        game_state: 현재 게임 상태
        state_context: 게임 상태별 컨텍스트
        selected_subjects: 선택한 과목 리스트
        schedule_set: 시간표 설정 여부
        official_mock_exam_grade_info: 공식 모의고사 성적 정보
        current_week: 현재 주차
        last_mock_exam_week: 마지막 사설모의고사 주차
        affection_tone: 호감도 말투 (선택적)
        career_info: 진로 정보 (선택적)
        subject_list_text: 선택과목 목록 텍스트 (선택적)
    Returns:
        str: 사용자 프롬프트
    """
    # 모든 지시사항을 동등한 레벨로 수집 (우선순위 없음)
    all_instructions = []
    
    # 사용자 메시지가 단순 응원 메시지인지 확인
    simple_encouragement_keywords = ["화이팅", "파이팅", "힘내", "응원", "화이팅해", "파이팅해", "힘내요", "응원해", 
                                    "화이팅!", "파이팅!", "힘내!", "응원!", "좋아", "잘하고", "잘해", "수고", "잘할 수"]
    user_message_lower = user_message.lower().strip()
    is_simple_encouragement = any(keyword in user_message_lower for keyword in simple_encouragement_keywords) and len(user_message.strip()) <= 10
    
    # 선택과목 목록 요청 시 추가 (icebreak/mentoring 상태에서만)
    if subject_list_text and game_state in ["icebreak", "mentoring"]:
        all_instructions.append(f"[선택과목 목록]\n{subject_list_text}\n\n사용자가 위 목록 중에서 선택과목을 고를 수 있도록 안내하세요. (최대 2개)")
    
    # affection_tone 추가
    if affection_tone and affection_tone.strip():
        all_instructions.append(affection_tone.strip())
    
    # career_info는 단순 응원 메시지가 아닐 때만 추가 (진로 질문이 있을 때만)
    if career_info and career_info.strip():
        # 단순 응원 메시지가 아니고, 진로 관련 키워드가 있을 때만 추가
        career_keywords = ["진로", "목표", "꿈", "되고 싶", "되려고", "하려고", "희망", "장래", "미래", "직업"]
        has_career_keyword = any(keyword in user_message_lower for keyword in career_keywords)
        
        if not is_simple_encouragement and has_career_keyword:
            all_instructions.append(career_info.strip())
        elif not is_simple_encouragement and len(user_message.strip()) > 20:
            # 긴 메시지이고 단순 응원이 아닐 때는 추가 (진로 얘기 가능성)
            all_instructions.append(career_info.strip())
    
    if state_context:
        all_instructions.append(state_context)
    
    # 단순 응원 메시지일 때는 진로 얘기하지 않도록 명시
    if is_simple_encouragement:
        all_instructions.append("[중요] 플레이어가 단순히 응원 메시지('화이팅', '파이팅', '힘내' 등)를 보냈습니다. 진로, 목표, 꿈, 직업 등에 대해 얘기하지 마세요. 간단히 감사하거나 고마워하는 반응만 하세요.")
    
    prompt_parts = all_instructions
    # 사설모의고사 한 주에 한 번 제한 안내
    if current_week == last_mock_exam_week and last_mock_exam_week >= 0:
        prompt_parts.append(f"이번 주({current_week}주차)에 이미 사설모의고사를 봤습니다. 플레이어가 '사설모의고사 응시'를 요청하면, 이미 이번 주에 봤다는 것을 알려주고 다음 주에 볼 수 있다고 안내하세요.")
    # 6exam_feedback 또는 9exam_feedback 상태에서는 절대로 여러 과목을 한 번에 말하지 않도록 지시
    if game_state == "6exam_feedback" or game_state == "9exam_feedback":
        prompt_parts.append("절대로 여러 과목(국어, 수학, 영어, 탐구1, 탐구2)을 한 번에 말하지 마세요. 현재 대화하고 있는 과목 하나만 얘기하세요. 예를 들어, 국어에 대해 얘기하고 있다면 국어만 언급하고 수학, 영어, 탐구 등을 함께 말하지 마세요.")
    
    # 프롬프트 조립
    sys_prompt = "\n\n".join(prompt_parts)

    prompt = sys_prompt.strip() + "\n\n"
    if context:
        prompt += "[참고 정보]\n" + context.strip() + "\n\n"
    prompt += f"{username}: {user_message.strip()}"
    return prompt

