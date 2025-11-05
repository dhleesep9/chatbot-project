"""Mock Exam Feedback State Handler

mock_exam_feedback, official_mock_exam_feedback 상태에서의 로직을 처리합니다.
- 취약점 정보 관리
- 조언 품질 평가 (LLM 사용)
- 능력치/멘탈/호감도 변경
- daily_routine 전이
"""

from typing import Dict, Any, Optional
from services.handlers.base_handler import BaseStateHandler


class MockExamFeedbackHandlerBase(BaseStateHandler):
    """mock_exam_feedback, official_mock_exam_feedback의 공통 로직을 처리하는 base class"""

    # 서브클래스에서 정의해야 할 속성
    EXAM_NAME = None  # "mock_exam_feedback" or "official_mock_exam_feedback"
    EXAM_DISPLAY_NAME = None  # "사설모의고사" or "정규모의고사"
    WEAKNESS_STORAGE_ATTR = None  # "mock_exam_weakness" or "official_mock_exam_weakness"
    RETEST_KEYWORD = None  # "사설모의고사 응시" (mock only, official은 None)
    ABILITY_SCALE = None  # 능력치 스케일 팩터 (6월: 10, 9월: 20)

    def on_enter(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        mock_exam_feedback 상태 진입 시 처리
        첫 번째 과목의 문제점을 표시

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        print(f"[{self.EXAM_NAME.upper()}] {username}이(가) {self.EXAM_NAME} 상태에 진입했습니다.")

        # 피드백 정보 가져오기
        weakness_storage = getattr(self.service, self.WEAKNESS_STORAGE_ATTR)
        weakness_info = weakness_storage.get(username, {})

        subject_problems = weakness_info.get("subject_problems", {})
        subject_order = weakness_info.get("subject_order", [])
        current_index = weakness_info.get("current_index", 0)

        if not subject_problems or not subject_order:
            print(f"[{self.EXAM_NAME.upper()}_WARN] 과목별 문제점 정보가 없습니다.")
            return {
                'skip_llm': False,
                'reply': None,
                'narration': f"{self.EXAM_DISPLAY_NAME} 성적표를 확인하고 조언을 주세요."
            }

        # 첫 번째 과목 문제점 표시
        if current_index < len(subject_order):
            current_subject = subject_order[current_index]
            problem_message = subject_problems.get(current_subject, "")

            # state 정보 가져오기
            state_info = self.service._get_state_info(self.EXAM_NAME)
            state_name = state_info.get("name", self.EXAM_DISPLAY_NAME) if state_info else self.EXAM_DISPLAY_NAME

            # 진행 상황 표시
            progress_info = f"({current_index + 1}/{len(subject_order)} 과목)"

            print(f"[{self.EXAM_NAME.upper()}] {current_subject} 문제점 표시: {problem_message}")

            return {
                'skip_llm': True,
                'reply': f"[{state_name}] {progress_info} {current_subject} 과목:\n{problem_message}\n\n이 과목에 대한 조언을 해주세요.",
                'narration': None
            }

        # 모든 과목 완료 (이 경우는 on_enter에서는 발생하지 않음)
        return {
            'skip_llm': True,
            'reply': "모든 과목에 대한 피드백이 완료되었습니다.",
            'narration': None,
            'transition_to': 'daily_routine'
        }

    def _parse_subject_from_weakness_message(self, user_message: str) -> Optional[str]:
        """
        사용자 메시지에서 과목명 추출
        예: "국어은 이랬어요: ...", "수학에서 ...", "영어는 ...", "탐구1은 ..."
        
        Returns:
            과목명 (국어, 수학, 영어, 탐구1, 탐구2) 또는 None
        """
        import re
        user_message_lower = user_message.lower()
        
        # 가능한 과목명 목록 (순서 중요: 더 긴 과목명부터 확인)
        subjects = ["탐구2", "탐구1", "국어", "수학", "영어"]
        
        # 각 과목명이 메시지에 포함되어 있는지 확인
        for subject in subjects:
            subject_lower = subject.lower()
            # 과목명이 메시지 시작 부분에 있는지 확인
            if user_message_lower.startswith(subject_lower):
                return subject
            # 특정 패턴으로 나타나는지 확인: "과목명은", "과목명는", "과목명에서" 등
            pattern = rf"{re.escape(subject)}(은|는|에서|이|가|이랬어요|이렇게|이런|이건|이것은|이것이)"
            if re.search(pattern, user_message, re.IGNORECASE):
                return subject
        
        return None

    def _get_advice_score_range(self) -> tuple:
        """
        조언 점수 범위 반환 (서브클래스에서 오버라이드 필요)
        
        Returns:
            (min_score, max_score) 튜플
        """
        return (0, 20)  # 기본값

    def _judge_advice_quality_with_range(self, username: str, advice: str, weak_subject: str, weakness_message: str) -> int:
        """
        LLM을 사용하여 조언 품질을 범위에 맞게 평가 (서브클래스에서 범위 오버라이드)
        
        Args:
            username: 사용자 이름
            advice: 조언 내용
            weak_subject: 취약 과목
            weakness_message: 취약점 메시지
        
        Returns:
            평가 점수 (범위는 _get_advice_score_range에서 정의)
        """
        min_score, max_score = self._get_advice_score_range()
        
        # 부정적 키워드 체크
        negative_keywords = [
            "망해", "망하", "포기", "포기해", "그만둬", "그만", "안돼", "못해", 
            "별로", "좋지않", "좋지 않", "안좋", "안 좋", "나쁘", "싫", "미워",
            "에휴", "아이고", "제발", "짜증", "답답", "한심", "바보", "멍청",
            "쓸모없", "쓸모 없", "소용없", "소용 없", "시작도", "시작도 못해",
            "이딴", "저딴", "이런", "저런", "그냥", "망했", "망했어", "망해라",
            "좆같", "지랄", "죽어", "죽어라", "꺼져", "시발", "개같", "병신"
        ]
        
        advice_lower = advice.lower()
        for keyword in negative_keywords:
            if keyword in advice_lower:
                print(f"[ADVICE_JUDGE] 부정적 키워드 직접 감지: '{keyword}' in '{advice}' → {min_score}점")
                return min_score
        
        if not self.service.client:
            # LLM이 없으면 중간값 반환
            import random
            return random.randint((min_score + max_score) // 4, (min_score + max_score) * 3 // 4)
        
        # 프롬프트 구성 - 각 모의고사 타입에 맞는 교육 전문가 역할 정의
        exam_type = "사설모의고사" if "mock" in self.EXAM_NAME.lower() else "정규모의고사"
        system_prompt = f"당신은 {exam_type} 전문 교육 전문가입니다. 학생을 격려하고 도와주는 멘토의 조언이 취약점을 해결하는 데 얼마나 적절한지 {min_score}~{max_score} 사이의 점수로 평가하세요. 취약점과 조언의 연관성, 구체성, 실행 가능성을 종합적으로 고려하세요. **중요: 반드시 {min_score}부터 {max_score} 사이의 점수만 사용하세요.**"
        
        user_prompt = f"""플레이어(멘토)가 재수생에게 다음과 같은 조언을 했습니다:
{advice}

학생의 취약점:
과목: {weak_subject}
내용: {weakness_message}

이 조언이 취약점을 해결하는 데 얼마나 적절한지 {min_score}~{max_score} 사이의 정수 점수로 평가해주세요.

**중요: 반드시 {min_score}부터 {max_score} 사이의 점수만 사용하세요. 이 범위를 벗어난 점수는 사용하지 마세요. 절대 0~20 같은 작은 범위를 사용하지 마세요.**

평가 기준:
- {max_score}점: 매우 적절함 (취약점과 직접 관련, 구체적이고 실행 가능한 조언)
- {int((min_score + max_score) * 0.875)}점: 적절함 (취약점과 관련, 실용적인 조언)
- {int((min_score + max_score) * 0.75)}점: 보통 (일반적인 조언, 취약점과 약간 관련)
- {int((min_score + max_score) * 0.625)}점: 부적절함 (취약점과 관련 없거나 추상적인 조언)
- {min_score}점: 매우 부적절함 (부정적이거나 해로운 조언)

점수만 숫자로 답변해주세요 (예: {(min_score + max_score) // 2})."""
        
        try:
            print(f"[ADVICE_JUDGE] {exam_type} 전문가 LLM 호출 시작 - 조언: '{advice[:50]}...', 취약점: {weak_subject}")
            print(f"[ADVICE_JUDGE] 평가 범위: {min_score}~{max_score}")
            print(f"[ADVICE_JUDGE] System prompt: {system_prompt[:150]}...")
            
            response = self.service.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=15
            )
            
            judgment_text = response.choices[0].message.content.strip()
            print(f"[ADVICE_JUDGE] {exam_type} 전문가 LLM 원본 응답: {judgment_text}")
            
            # 점수 파싱 (범위에 맞게)
            import re
            # 범위 내의 모든 숫자 찾기
            numbers = re.findall(r'\d+', judgment_text)
            if numbers:
                score = int(numbers[0])
                # 범위 제한만 적용 (스케일 없음)
                score = max(min_score, min(max_score, score))
                print(f"[ADVICE_JUDGE] {exam_type} 전문가 파싱된 점수: {score}점 ({min_score}~{max_score} 범위)")
                return score
            else:
                # 파싱 실패 시 중간값 반환
                mid_score = (min_score + max_score) // 2
                print(f"[ADVICE_JUDGE] {exam_type} 전문가 점수 파싱 실패, 중간값 반환: {mid_score}점")
                return mid_score
        except Exception as e:
            print(f"[ERROR][ADVICE_JUDGE] {exam_type} 전문가 LLM 호출 실패: {e}")
            import traceback
            print(f"[ERROR][ADVICE_JUDGE] 스택 트레이스:\n{traceback.format_exc()}")
            # 에러 시 중간값 반환
            return (min_score + max_score) // 2

    def _calculate_ability_increase(self, username: str, subject: str, advice_score: int) -> float:
        """
        능력치 증가량 계산 (서브클래스에서 오버라이드 필요)
        
        Args:
            username: 사용자 이름
            subject: 과목명
            advice_score: 조언 점수 (0~20)
        
        Returns:
            최종 능력치 증가량
        """
        # 기본 구현: 스케일 팩터 없이 계산
        stamina = self.service._get_stamina(username)
        mental = self.service._get_mental(username)
        efficiency = self.service._calculate_combined_efficiency(stamina, mental) / 100.0
        base_increase = float(advice_score) * efficiency
        increased = self.service._apply_ability_multipliers(username, subject, base_increase)
        return increased

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        모의고사 피드백 로직 처리 - 순차적으로 모든 과목에 대해 조언 처리

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        current_state = context.get('current_state')

        # [1] 재응시 확인 (mock_exam_feedback만 해당)
        if self.RETEST_KEYWORD and self.RETEST_KEYWORD in user_message:
            # 한 주에 한 번만 보도록 체크
            current_week = self.service._get_current_week(username)
            last_week = self.service.mock_exam_last_week.get(username, -1)

            if current_week == last_week and last_week >= 0:
                print(f"[{self.EXAM_NAME.upper()}] 재응시 차단: {username}이(가) 이미 {current_week}주차에 사설모의고사를 봤습니다.")
                return {
                    'skip_llm': True,
                    'reply': f"이번 주({current_week}주차)에는 이미 사설모의고사를 봤어요. 다음 주에 다시 볼 수 있어요.",
                    'narration': None,
                    'transition_to': None  # 전이하지 않음
                }

            print(f"[{self.EXAM_NAME.upper()}] {self.RETEST_KEYWORD} 감지 - mock_display로 전환")
            return {
                'skip_llm': True,
                'reply': None,
                'narration': None,
                'transition_to': 'mock_display',
                'retest': True
            }

        # [2] 피드백 정보 가져오기
        weakness_storage = getattr(self.service, self.WEAKNESS_STORAGE_ATTR)
        weakness_info = weakness_storage.get(username, {})

        subject_problems = weakness_info.get("subject_problems", {})
        subject_order = weakness_info.get("subject_order", [])
        current_index = weakness_info.get("current_index", 0)
        completed_subjects = weakness_info.get("completed_subjects", [])

        if not subject_problems or not subject_order:
            print(f"[{self.EXAM_NAME.upper()}] 과목별 문제점 정보가 없습니다.")
            return None

        # 현재 처리 중인 과목
        if current_index >= len(subject_order):
            print(f"[{self.EXAM_NAME.upper()}] 모든 과목 완료")
            return {
                'skip_llm': True,
                'reply': "모든 과목에 대한 피드백이 완료되었습니다!",
                'narration': "일상 루틴으로 돌아갑니다.",
                'transition_to': 'daily_routine'
            }

        current_subject = subject_order[current_index]
        current_problem_message = subject_problems.get(current_subject, "")

        # [3] 조언 감지 확인
        advice_given = self.service._check_if_advice_given(user_message)
        print(f"[{self.EXAM_NAME.upper()}] 조언 감지: {advice_given}, message='{user_message}'")

        if not advice_given:
            # 조언이 감지되지 않으면 LLM 호출
            return None

        # [4] 조언 품질 판단
        advice_score = self._judge_advice_quality_with_range(username, user_message, current_subject, current_problem_message)
        score_range = self._get_advice_score_range()
        print(f"[{self.EXAM_NAME.upper()}] {current_subject} 조언 점수: {advice_score}점 ({score_range[0]}~{score_range[1]})")

        # [5] 능력치/멘탈/호감도 변경
        min_score, max_score = score_range
        score_range_size = max_score - min_score

        # 점수를 0~20 범위로 정규화
        if score_range_size > 0:
            normalized_score = ((advice_score - min_score) / score_range_size) * 20
        else:
            normalized_score = 10

        # 정규화된 점수에 따라 호감도와 멘탈 변화 결정
        if normalized_score >= 11:
            affection_change = 2
            mental_change = 5
            advice_quality_text = "좋은"
        elif normalized_score >= 6:
            affection_change = 1
            mental_change = 2
            advice_quality_text = "보통"
        else:
            affection_change = -2
            mental_change = -2
            advice_quality_text = "부적절한"

        current_affection = self.service._get_affection(username)
        new_affection = max(0, min(100, current_affection + affection_change))
        self.service._set_affection(username, new_affection)

        current_mental = self.service._get_mental(username)
        new_mental = max(0, min(100, current_mental + mental_change))
        self.service._set_mental(username, new_mental)

        # 능력치 증가
        abilities = self.service._get_abilities(username)
        increased_amount = 0.0
        if current_subject in abilities:
            increased_amount = self._calculate_ability_increase(username, current_subject, advice_score)
            abilities[current_subject] = min(2500, abilities[current_subject] + increased_amount)
            self.service._set_abilities(username, abilities)

        print(f"[{self.EXAM_NAME.upper()}] {current_subject} 조언 평가 완료: 점수={advice_score}, 호감도 {affection_change:+.0f}, 멘탈 {mental_change:+.0f}, 능력치 +{increased_amount:.2f}")

        # [6] 다음 과목으로 진행 또는 완료
        # 현재 과목을 완료 목록에 추가
        completed_subjects.append(current_subject)
        current_index += 1

        # 정보 업데이트
        weakness_info['current_index'] = current_index
        weakness_info['completed_subjects'] = completed_subjects
        weakness_storage[username] = weakness_info

        # 다음 과목이 있는지 확인
        if current_index < len(subject_order):
            # 다음 과목으로 진행
            next_subject = subject_order[current_index]
            next_problem_message = subject_problems.get(next_subject, "")
            progress_info = f"({current_index + 1}/{len(subject_order)} 과목)"

            # state 정보 가져오기
            state_info = self.service._get_state_info(self.EXAM_NAME)
            state_name = state_info.get("name", self.EXAM_DISPLAY_NAME) if state_info else self.EXAM_DISPLAY_NAME

            feedback_message = f"{advice_quality_text} 조언이었어요! {current_subject} 능력치 +{increased_amount:.0f}, 호감도 {affection_change:+.0f}, 멘탈 {mental_change:+.0f}\n\n"
            next_subject_message = f"{progress_info} {next_subject} 과목:\n{next_problem_message}\n\n이 과목에 대한 조언을 해주세요."

            return {
                'skip_llm': True,
                'reply': f"[{state_name}] {feedback_message}{next_subject_message}",
                'narration': None
            }
        else:
            # 모든 과목 완료
            print(f"[{self.EXAM_NAME.upper()}] 모든 과목 피드백 완료")

            # 피드백 정보 삭제
            if username in weakness_storage:
                del weakness_storage[username]

            return {
                'skip_llm': False,  # LLM으로 마무리 메시지 생성
                'reply': None,
                'narration': "모든 과목에 대한 피드백이 완료되었습니다.\n\n일상 루틴으로 돌아갑니다.",
                'transition_to': 'daily_routine',
                'advice_user_input': user_message,
                'advice_score': advice_score
            }


class MockExamFeedbackHandler(MockExamFeedbackHandlerBase):
    """mock_exam_feedback state handler (6월 모의고사)"""
    EXAM_NAME = "mock_exam_feedback"
    EXAM_DISPLAY_NAME = "사설모의고사"
    WEAKNESS_STORAGE_ATTR = "mock_exam_weakness"
    RETEST_KEYWORD = "사설모의고사 응시"
    # ABILITY_SCALE은 더 이상 사용하지 않음 (LLM이 직접 0~20 범위로 평가)

    def _get_advice_score_range(self) -> tuple:
        """사설모의고사: 0~20 범위로 평가"""
        return (0, 20)

    def _calculate_ability_increase(self, username: str, subject: str, advice_score: int) -> float:
        """
        사설모의고사 능력치 증가 계산
        advice_score는 이미 0~20 범위로 평가됨
        """
        stamina = self.service._get_stamina(username)
        mental = self.service._get_mental(username)
        efficiency = self.service._calculate_combined_efficiency(stamina, mental) / 100.0
        
        # advice_score는 이미 0~20 범위이므로 efficiency만 적용
        base_increase = float(advice_score) * efficiency
        
        # 디버그 로그
        print(f"[{self.EXAM_NAME.upper()}] 능력치 계산: 클래스={type(self).__name__}, advice_score={advice_score} (0~20), efficiency={efficiency:.4f}, base_increase={base_increase:.2f}")
        
        # 배율 적용 (진로-과목 + 시험 전략)
        increased = self.service._apply_ability_multipliers(username, subject, base_increase)
        print(f"[{self.EXAM_NAME.upper()}] 배율 적용 후: increased={increased:.2f}")
        
        return increased


class OfficialMockExamFeedbackHandler(MockExamFeedbackHandlerBase):
    """official_mock_exam_feedback state handler (9월 모의고사)"""
    EXAM_NAME = "official_mock_exam_feedback"
    EXAM_DISPLAY_NAME = "정규모의고사"
    WEAKNESS_STORAGE_ATTR = "official_mock_exam_weakness"
    RETEST_KEYWORD = None  # 정규모의고사는 재응시 없음
    # ABILITY_SCALE은 더 이상 사용하지 않음 (LLM이 직접 0~20 범위로 평가)

    def _get_advice_score_range(self) -> tuple:
        """정규모의고사: 0~20 범위로 평가"""
        return (0, 20)

    def _calculate_ability_increase(self, username: str, subject: str, advice_score: int) -> float:
        """
        정규모의고사 능력치 증가 계산
        advice_score는 이미 0~20 범위로 평가됨
        """
        stamina = self.service._get_stamina(username)
        mental = self.service._get_mental(username)
        efficiency = self.service._calculate_combined_efficiency(stamina, mental) / 100.0
        
        # advice_score는 이미 0~20 범위이므로 efficiency만 적용
        base_increase = float(advice_score) * efficiency
        
        # 디버그 로그
        print(f"[{self.EXAM_NAME.upper()}] 능력치 계산: 클래스={type(self).__name__}, advice_score={advice_score} (0~20), efficiency={efficiency:.4f}, base_increase={base_increase:.2f}")
        
        # 배율 적용 (진로-과목 + 시험 전략)
        increased = self.service._apply_ability_multipliers(username, subject, base_increase)
        print(f"[{self.EXAM_NAME.upper()}] 배율 적용 후: increased={increased:.2f}")
        
        return increased
