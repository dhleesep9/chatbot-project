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

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        모의고사 피드백 로직 처리

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
            
            print(f"[{self.EXAM_NAME.upper()}] {self.RETEST_KEYWORD} 감지 - mock_exam으로 전환")
            return {
                'skip_llm': True,
                'reply': None,
                'narration': None,
                'transition_to': 'mock_exam',
                'retest': True
            }

        # [2] 취약점 정보 가져오기
        weakness_storage = getattr(self.service, self.WEAKNESS_STORAGE_ATTR)
        weakness_info = weakness_storage.get(username, {})
        current_weak_subject = weakness_info.get("subject")
        current_weakness_message = weakness_info.get("message")

        # 취약점 정보가 없으면 처리하지 않음
        if not current_weak_subject or not current_weakness_message:
            print(f"[{self.EXAM_NAME.upper()}] 취약점 정보가 없습니다.")
            # 조언을 주었는지 확인 (있으면 일상 루틴으로)
            if self.service._check_if_advice_given(user_message):
                print(f"[{self.EXAM_NAME.upper()}] 취약점 정보 없지만 조언 감지 - daily_routine으로 전환")
                return {
                    'skip_llm': True,
                    'reply': "조언 감사합니다!",
                    'narration': "일상 루틴으로 돌아갑니다.",
                    'transition_to': 'daily_routine'
                }
            return None

        # [3] 조언 감지 확인
        advice_given = self.service._check_if_advice_given(user_message)
        print(f"[{self.EXAM_NAME.upper()}] 조언 감지: {advice_given}, message='{user_message}'")

        if not advice_given:
            # 조언이 감지되지 않으면 LLM 호출
            return None

        # [4] 조언 품질 판단 (0~20 점수)
        advice_score = self.service._judge_advice_quality(username, user_message, current_weak_subject, current_weakness_message)
        print(f"[{self.EXAM_NAME.upper()}] 조언 점수: {advice_score}점 (0~20)")

        # [5] 능력치/멘탈/호감도 변경 (점수에 따라)
        narration = None
        
        # 점수에 따라 호감도와 멘탈 변화 결정
        if advice_score >= 11:
            # 적절한 조언 (11~20점): 호감도 +2, 멘탈 +5
            affection_change = 2
            mental_change = 5
            advice_quality_text = "좋은"
        elif advice_score >= 6:
            # 보통 조언 (6~10점): 호감도 +1, 멘탈 +2
            affection_change = 1
            mental_change = 2
            advice_quality_text = "보통"
        else:
            # 부적절한 조언 (0~5점): 호감도 -2, 멘탈 -2
            affection_change = -2
            mental_change = -2
            advice_quality_text = "부적절한"
        
        current_affection = self.service._get_affection(username)
        new_affection = max(0, min(100, current_affection + affection_change))
        self.service._set_affection(username, new_affection)

        current_mental = self.service._get_mental(username)
        new_mental = max(0, min(100, current_mental + mental_change))
        self.service._set_mental(username, new_mental)

        # 능력치 증가: 점수에 따라 0~20 (효율과 배율 적용)
        abilities = self.service._get_abilities(username)
        increased_amount = 0.0
        if current_weak_subject in abilities:
            # 체력과 멘탈 효율 적용
            stamina = self.service._get_stamina(username)
            mental = self.service._get_mental(username)
            efficiency = self.service._calculate_combined_efficiency(stamina, mental) / 100.0
            
            # 점수를 그대로 base_increase로 사용 (0~20)
            base_increase = float(advice_score) * efficiency
            # 배율 적용 (진로-과목 + 시험 전략)
            increased = self.service._apply_ability_multipliers(username, current_weak_subject, base_increase)
            increased_amount = increased
            abilities[current_weak_subject] = min(2500, abilities[current_weak_subject] + increased)
            self.service._set_abilities(username, abilities)

        if advice_score >= 6:
            narration = f"{advice_quality_text} 조언이었어요! {current_weak_subject} 능력치가 {increased_amount:.0f} 증가, 호감도 {affection_change:+.0f}, 멘탈 {mental_change:+.0f} 변화했습니다.\n\n일상 루틴으로 돌아갑니다."
        else:
            narration = f"{advice_quality_text} 조언이었어요. 호감도 {affection_change:+.0f}, 멘탈 {mental_change:+.0f} 변화했습니다.\n\n일상 루틴으로 돌아갑니다."
        
        print(f"[{self.EXAM_NAME.upper()}] 조언 점수: {advice_score}점 - 호감도 {affection_change:+.0f}, 멘탈 {mental_change:+.0f}, {current_weak_subject} +{increased_amount:.2f}")

        # [6] 취약점 정보 삭제
        if username in weakness_storage:
            del weakness_storage[username]

        # [7] 조언에 대한 반응 생성 (나중에 LLM으로 생성)
        return {
            'skip_llm': False,  # LLM으로 반응 생성
            'reply': None,  # LLM이 생성
            'narration': narration,
            'transition_to': 'daily_routine',
            'advice_user_input': user_message,
            'advice_score': advice_score
        }


class MockExamFeedbackHandler(MockExamFeedbackHandlerBase):
    """mock_exam_feedback state handler"""
    EXAM_NAME = "mock_exam_feedback"
    EXAM_DISPLAY_NAME = "사설모의고사"
    WEAKNESS_STORAGE_ATTR = "mock_exam_weakness"
    RETEST_KEYWORD = "사설모의고사 응시"


class OfficialMockExamFeedbackHandler(MockExamFeedbackHandlerBase):
    """official_mock_exam_feedback state handler"""
    EXAM_NAME = "official_mock_exam_feedback"
    EXAM_DISPLAY_NAME = "정규모의고사"
    WEAKNESS_STORAGE_ATTR = "official_mock_exam_weakness"
    RETEST_KEYWORD = None  # 정규모의고사는 재응시 없음
