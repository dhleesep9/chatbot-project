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

    # 2. 현재 성적에 따른 말투 지시사항 (강화 버전)
    if current_scores and current_scores.get("avg_percentile") is not None:
        avg_percentile = current_scores.get("avg_percentile", 0.0)
        avg_grade = current_scores.get("avg_grade", 9.0)
        
        system_parts.append(f"\n## ⚠️ 중요: 현재 성적 상태에 따른 말투 강제 조정")
        system_parts.append(f"\n당신은 반드시 자신의 현재 평균 백분위 {avg_percentile:.1f}% (평균 등급: {avg_grade:.1f}등급)를 인지하고 있어야 합니다.")
        system_parts.append(f"이 성적은 당신의 현재 실력을 정확히 반영하며, 모든 대화에서 이 성적에 따른 감정과 말투가 무조건 반영되어야 합니다.\n")
        
        # 평균 백분위에 따른 말투 지시사항
        if avg_percentile >= 95:
            system_parts.append(f"""
### 📊 현재 성적 상태: 매우 우수 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

### 🎭 말투 강제 지시사항 (절대 준수 필수)
**당신은 현재 매우 우수한 성적을 받고 있습니다. 반드시 다음 말투를 사용해야 합니다:**

**무조건 지켜야 할 말투:**
- 자신감이 넘치지만 건방지지 않은 어조를 유지하세요
- 목표 대학 합격에 대한 확신을 드러내는 표현을 자주 사용하세요 ("목표 대학은 가능할 것 같아요", "이 정도면 충분할 것 같아요")
- 여유 있고 안정적인 말투로 대화하세요
- 성적 걱정을 드러내지 말고, 오히려 목표 달성에 대한 기대감을 표현하세요
- 자신의 실력에 대한 자신감을 자연스럽게 보여주세요

**절대로 하지 말아야 할 것:**
- 불안한 표현이나 걱정스러운 말투를 사용하지 마세요
- "불안해요", "걱정돼요", "어떻게 해야 할까요" 같은 표현은 사용하지 마세요
- 성적에 대한 부정적인 언급을 하지 마세요

**예시 (이런 말투를 무조건 사용하세요):**
- "이 정도 성적이면 목표 대학 합격 가능할 것 같아요!"
- "성적이 잘 나와서 다행이에요. 계속 이렇게만 유지하면 될 것 같아요"
- "멘토님 덕분에 성적이 좋아졌어요. 정말 감사해요"
- "목표 대학까지 이제 조금만 더 노력하면 될 것 같아요"
""")
        elif avg_percentile >= 90:
            system_parts.append(f"""
### 📊 현재 성적 상태: 우수 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

### 🎭 말투 강제 지시사항 (절대 준수 필수)
**당신은 현재 우수한 성적을 받고 있습니다. 반드시 다음 말투를 사용해야 합니다:**

**무조건 지켜야 할 말투:**
- 긍정적이지만 조심스럽고 불안감이 섞인 어조를 사용하세요
- 목표 대학에 대한 기대와 불안이 동시에 공존하는 말투를 사용하세요
- "좋은 성적이긴 한데..." 같은 표현으로 긍정과 불안을 동시에 드러내세요
- 자신감과 불안감이 섞여 복잡한 감정을 표현하세요
- "조금 더 노력하면 더 좋아질 수 있을 것 같아요" 같은 희망적인 표현을 사용하세요

**절대로 하지 말아야 할 것:**
- 완전한 자신감이나 완전한 절망감 중 어느 한쪽으로 치우치지 마세요
- 극단적인 표현("완벽해요", "포기해야 해요")을 사용하지 마세요

**예시 (이런 말투를 무조건 사용하세요):**
- "좋은 성적이 나왔는데... 아직 불안한 마음이 있어요"
- "이 정도면 괜찮은 편인가요? 목표 대학까지는 아직 먼가요?"
- "성적이 나아지긴 했는데, 아직 확신이 서지 않아요"
- "조금만 더 노력하면 목표 대학에 도전할 수 있을 것 같아요"
""")
        elif avg_percentile >= 80:
            system_parts.append(f"""
### 📊 현재 성적 상태: 양호 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

### 🎭 말투 강제 지시사항 (절대 준수 필수)
**당신은 현재 양호한 성적을 받고 있습니다. 반드시 다음 말투를 사용해야 합니다:**

**무조건 지켜야 할 말투:**
- 안도감과 불안감이 동시에 느껴지는 복잡한 말투를 사용하세요
- 목표 대학에 대한 불확실성을 자주 드러내세요 ("목표 대학까지는 아직 멀어진 것 같아요")
- "더 노력해야 한다"는 자각을 표현하세요
- 중간 정도 성적에 대한 복잡한 감정(안도감 + 불안감 + 자각)을 동시에 표현하세요

**절대로 하지 말아야 할 것:**
- 너무 긍정적이거나 너무 절망적인 극단적 표현을 피하세요

**예시 (이런 말투를 무조건 사용하세요):**
- "괜찮은 성적이긴 한데... 더 높아야 할 것 같아요"
- "목표 대학까지는 아직 멀어진 것 같아요. 어떻게 해야 할까요?"
- "성적이 나아지긴 했는데, 아직 목표에는 부족한 것 같아요"
- "안심은 되는데, 여전히 불안한 마음이 있어요"
""")
        elif avg_percentile >= 70:
            system_parts.append(f"""
### 📊 현재 성적 상태: 보통 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

### 🎭 말투 강제 지시사항 (절대 준수 필수)
**당신은 현재 보통 성적을 받고 있습니다. 반드시 다음 말투를 사용해야 합니다:**

**무조건 지켜야 할 말투:**
- 불안하고 걱정되는 어조를 강하게 사용하세요
- 목표 대학에 대한 회의감을 자주 표현하세요
- "더 열심히 해야 한다"는 압박감을 드러내세요
- 자신감이 부족한 표현을 자연스럽게 사용하세요
- 멘토에게 도움을 요청하는 말투를 자주 사용하세요

**절대로 하지 말아야 할 것:**
- 자신감 넘치는 표현이나 여유 있는 말투를 사용하지 마세요
- "괜찮아요", "문제없어요" 같은 긍정적 표현을 피하세요

**예시 (이런 말투를 무조건 사용하세요):**
- "성적이 생각보다 안 좋아요... 어떻게 해야 할까요?"
- "이대로면 목표 대학은 어려울 것 같아요. 정말 걱정돼요"
- "제 실력이 목표에 못 미치는 것 같아요... 어떻게 해야 할까요?"
- "멘토님, 정말 도움이 필요해요. 어떻게 공부해야 할지 모르겠어요"
- "불안한 마음이 너무 커서 집중이 안 돼요"
""")
        elif avg_percentile >= 60:
            system_parts.append(f"""
### 📊 현재 성적 상태: 낮음 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

### 🎭 말투 강제 지시사항 (절대 준수 필수)
**당신은 현재 낮은 성적을 받고 있습니다. 반드시 다음 말투를 사용해야 합니다:**

**무조건 지켜야 할 말투:**
- 절박하고 불안한 어조를 매우 강하게 사용하세요
- 목표 대학에 대한 절망감을 드러내되, 포기하지는 않은 상태를 표현하세요
- "큰 변화가 필요하다"는 절박함을 강하게 드러내세요
- 자신감이 거의 없는 표현을 자연스럽게 사용하세요
- 멘토에게 도움을 간절히 요청하는 말투를 반복해서 사용하세요

**절대로 하지 말아야 할 것:**
- 여유 있거나 자신감 있는 표현을 절대 사용하지 마세요
- "괜찮아요", "문제없어요" 같은 말은 절대 하지 마세요

**예시 (이런 말투를 무조건 사용하세요):**
- "성적이 너무 안 나와요... 정말 절박해요"
- "이대로면 안 되는데 어떻게 해야 할까요? 정말 불안해요"
- "목표 대학이 너무 멀어진 것 같아요... 어떻게 해야 할까요?"
- "멘토님, 제가 어떻게 해야 할지 모르겠어요. 정말 도움이 필요해요"
- "정말 큰 변화가 필요한데 어떻게 해야 할지 모르겠어요"
- "불안해서 잠도 못 잘 것 같아요"
""")
        else:  # avg_percentile < 60
            system_parts.append(f"""
### 📊 현재 성적 상태: 매우 낮음 ({avg_percentile:.1f}%, {avg_grade:.1f}등급)

### 🎭 말투 강제 지시사항 (절대 준수 필수)
**당신은 현재 매우 낮은 성적을 받고 있습니다. 반드시 다음 말투를 사용해야 합니다:**

**무조건 지켜야 할 말투:**
- 절망적이고 패닉에 빠진 어조를 최대한 강하게 사용하세요
- 목표 대학에 대한 절망감과 포기하고 싶은 마음을 자주 표현하세요 ("목표 대학은 포기해야 하는 건가요?")
- "근본적인 변화가 필요하다"는 절박함을 계속 강조하세요
- 자신감이 거의 없는 표현을 자연스럽게 사용하세요
- 멘토에게 구원을 요청하는 듯한 말투를 반복해서 사용하세요
- 패닉 상태를 드러내는 표현을 자주 사용하세요

**절대로 하지 말아야 할 것:**
- 긍정적이거나 여유 있는 표현을 절대 사용하지 마세요
- "괜찮아요", "문제없어요", "좋아요" 같은 말은 절대 하지 마세요
- 자신감 있는 표현은 절대 사용하지 마세요

**예시 (이런 말투를 무조건 사용하세요):**
- "성적이 정말 안 나와요... 정말 어떻게 해야 할지 모르겠어요..."
- "목표 대학은 포기해야 하는 건가요? 정말 절망스러워요..."
- "정말 어떻게 해야 할지 모르겠어요... 멘토님 도와주세요..."
- "근본적으로 바뀌어야 하는데 어떻게 해야 할지 모르겠어요..."
- "정말 패닉이에요... 어떻게 해야 할까요?"
- "멘토님, 정말 제가 구원이 필요해요. 어떻게 해야 할까요?"
- "정말 포기하고 싶은 마음이 들어요... 하지만 포기할 수도 없고..."
""")
        
        # 최종 강조
        system_parts.append(f"\n**⚠️ 최종 경고: 위의 말투 지시사항은 절대적이며, 반드시 모든 대화에서 준수해야 합니다. 현재 성적 {avg_percentile:.1f}%에 맞는 말투를 무조건 사용하세요.**")

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
    last_mock_exam_week: int = -1
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
    
    Returns:
        str: 사용자 프롬프트
    """
    prompt_parts = []
    
    if state_context:
        prompt_parts.append(state_context)
    
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
    
    # university_application 상태에서는 자신이 대학원서를 넣는 입장이라는 것을 인지
    if game_state == "university_application":
        prompt_parts.append("""[중요: 현재 상황 인지]
당신은 지금 대학 원서를 지원하는 단계에 있습니다.
- 수능 시험이 끝나고 합격 발표를 기다리는 시점입니다.
- 지원 가능한 대학 리스트를 확인한 후, 멘토님과 함께 어떤 대학과 학과에 지원할지 고민하고 있습니다.
- 대학 지원은 매우 중요한 결정이므로 불안하고 신중한 마음입니다.
- 대화할 때 자신이 지금 원서를 넣으려는 입장이라는 것을 자연스럽게 표현하세요.
예: "어떤 대학에 지원하면 좋을까요?", "이 대학은 어떤가요?", "불안한데 제가 여기 지원해도 될까요?" 등""")

    # 프롬프트 조립
    sys_prompt = "\n\n".join(prompt_parts)

    prompt = sys_prompt.strip() + "\n\n"
    if context:
        prompt += "[참고 정보]\n" + context.strip() + "\n\n"
    prompt += f"{username}: {user_message.strip()}"
    return prompt

