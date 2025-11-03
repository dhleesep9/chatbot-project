"""Exam Feedback State Handler

6exam_feedback, 9exam_feedback 상태에서의 로직을 처리합니다.
- 과목별 문제점 추적
- 조언 품질 평가 (LLM 사용)
- 능력치/멘탈/호감도 변경
- 5과목 완료 시 daily_routine 전이
"""

from typing import Dict, Any, Optional
from services.handlers.base_handler import BaseStateHandler


class ExamFeedbackHandlerBase(BaseStateHandler):
    """6exam_feedback, 9exam_feedback의 공통 로직을 처리하는 base class"""

    # 서브클래스에서 정의해야 할 속성
    EXAM_NAME = None  # "6exam_feedback" or "9exam_feedback"
    EXAM_DISPLAY_NAME = None  # "6월 모의고사" or "9월 모의고사"
    PROBLEM_STORAGE_ATTR = None  # "june_exam_problems" or "september_exam_problems"
    SUBJECT_PROBLEM_TRIGGER = None  # "june_exam_subject_problem" or "september_exam_subject_problem"
    ADVICE_GIVEN_TRIGGER = None  # "june_exam_advice_given" or "september_exam_advice_given"
    PROBLEM_GENERATOR_METHOD = None  # "_generate_june_subject_problem" or "_generate_september_subject_problem"

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
        problem_storage = getattr(self.service, self.PROBLEM_STORAGE_ATTR)
        problem_info = problem_storage.get(username, {})

        # 문제점 정보 초기화 (처음 진입 시)
        if not problem_info:
            exam_scores = self.service._calculate_mock_exam_scores(username)
            problem_info = {
                "scores": exam_scores,
                "subjects": {
                    "국어": {"problem": None, "solved": False},
                    "수학": {"problem": None, "solved": False},
                    "영어": {"problem": None, "solved": False},
                    "탐구1": {"problem": None, "solved": False},
                    "탐구2": {"problem": None, "solved": False}
                },
                "current_subject": None,
                "completed_count": 0,
                "subject_order": ["국어", "수학", "영어", "탐구1", "탐구2"]
            }
            problem_storage[username] = problem_info

        subjects = problem_info.get("subjects", {})
        current_subject = problem_info.get("current_subject")
        completed_count = problem_info.get("completed_count", 0)
        subject_order = problem_info.get("subject_order", ["국어", "수학", "영어", "탐구1", "탐구2"])

        # 다음 대화할 과목 찾기
        next_subject = None
        for subject in subject_order:
            if not subjects.get(subject, {}).get("solved", False):
                next_subject = subject
                break

        # 트리거 컨텍스트 구성
        trigger_context = {
            'username': username,
            'user_message': user_message,
            'current_state': context.get('new_state'),
            self.PROBLEM_STORAGE_ATTR: problem_storage,
            'service': self.service
        }

        subject_problem_reply = None
        advice_reply = None
        narration = None
        transition_to = None
        skip_llm = False

        # [1] 문제점 제시 확인 (트리거 사용)
        subject_problem_trigger = {
            "trigger_type": self.SUBJECT_PROBLEM_TRIGGER,
            "conditions": {}
        }

        if self.service.trigger_registry.evaluate_trigger(self.SUBJECT_PROBLEM_TRIGGER, subject_problem_trigger, trigger_context) and next_subject:
            # 첫 번째 과목의 문제점 생성
            subject_scores = problem_info.get("scores", {}).get(next_subject, {})
            problem_generator = getattr(self.service, self.PROBLEM_GENERATOR_METHOD)
            subject_problem = problem_generator(next_subject, subject_scores)

            # 현재 과목 설정 및 문제점 저장
            subjects[next_subject]["problem"] = subject_problem
            problem_info["current_subject"] = next_subject
            problem_info["subjects"] = subjects
            problem_storage[username] = problem_info

            subject_problem_reply = f"{next_subject}은 이랬어요: {subject_problem}"
            print(f"[{self.EXAM_NAME.upper()}] {next_subject} 과목 문제점: {subject_problem}")

        # [2] 조언 제시 확인 (트리거 사용)
        advice_given_trigger = {
            "trigger_type": self.ADVICE_GIVEN_TRIGGER,
            "conditions": {}
        }

        if self.service.trigger_registry.evaluate_trigger(self.ADVICE_GIVEN_TRIGGER, advice_given_trigger, trigger_context):
            # 현재 과목이 있고 아직 해결되지 않은 경우에만 처리
            if current_subject and not subjects.get(current_subject, {}).get("solved", False):
                skip_llm = True  # 조언 처리 시 LLM 호출 건너뛰기

                # LLM으로 해결방안 적절성 판단
                current_problem = subjects.get(current_subject, {}).get("problem", "")
                is_solution_good = self.service._judge_advice_quality(username, user_message, current_subject, current_problem)

                # 조언에 대한 서가윤의 반응 생성 (LLM 사용)
                advice_reply = self._generate_advice_reaction(user_message, current_subject, current_problem, is_solution_good)

                # 능력치/멘탈/호감도 변경
                if is_solution_good:
                    # 적절한 조언: 해당과목 +100, 멘탈 +5, 호감도 +2 (배율 적용)
                    abilities = self.service._get_abilities(username)
                    increased_amount = 100.0
                    if current_subject in abilities:
                        # 체력과 멘탈 효율 적용
                        stamina = self.service._get_stamina(username)
                        mental = self.service._get_mental(username)
                        efficiency = self.service._calculate_combined_efficiency(stamina, mental) / 100.0
                        
                        base_increase = 100.0 * efficiency
                        # 배율 적용 (진로-과목 + 시험 전략)
                        increased_amount = self.service._apply_ability_multipliers(username, current_subject, base_increase)
                        abilities[current_subject] = min(2500, abilities[current_subject] + increased_amount)
                        self.service._set_abilities(username, abilities)

                    current_mental = self.service._get_mental(username)
                    new_mental = min(100, current_mental + 5)
                    self.service._set_mental(username, new_mental)

                    current_affection = self.service._get_affection(username)
                    new_affection = min(100, current_affection + 2)
                    self.service._set_affection(username, new_affection)

                    narration = f"적절한 조언이였습니다 {current_subject}과목 능력치 +{increased_amount:.0f} 멘탈 +5 호감도 +2"
                    print(f"[{self.EXAM_NAME.upper()}] {current_subject} 해결방안 적절함 - 능력치 +{increased_amount:.2f}, 멘탈 +5")
                else:
                    # 부적절한 조언: 호감도 -2, 멘탈 -5
                    current_affection = self.service._get_affection(username)
                    new_affection = max(0, current_affection - 2)
                    self.service._set_affection(username, new_affection)

                    current_mental = self.service._get_mental(username)
                    new_mental = max(0, current_mental - 5)
                    self.service._set_mental(username, new_mental)

                    narration = f"적절하지 않은 조언이였습니다. 호감도 -2, 멘탈 -5"
                    print(f"[{self.EXAM_NAME.upper()}] {current_subject} 해결방안 부적절함 - 호감도 -2, 멘탈 -5")

                # 현재 과목 완료 처리
                subjects[current_subject]["solved"] = True
                completed_count += 1
                problem_info["completed_count"] = completed_count
                problem_info["subjects"] = subjects
                problem_info["current_subject"] = None
                problem_storage[username] = problem_info

                # 다음 과목 찾기
                next_subject_after = None
                for subject in subject_order:
                    if not subjects.get(subject, {}).get("solved", False):
                        next_subject_after = subject
                        break

                # 다음 과목이 있으면 문제점 제시 (같은 턴에 표시)
                if next_subject_after and completed_count < 5:
                    subject_scores = problem_info.get("scores", {}).get(next_subject_after, {})
                    problem_generator = getattr(self.service, self.PROBLEM_GENERATOR_METHOD)
                    subject_problem = problem_generator(next_subject_after, subject_scores)

                    # 현재 과목 설정 및 문제점 저장
                    subjects[next_subject_after]["problem"] = subject_problem
                    problem_info["current_subject"] = next_subject_after
                    problem_info["subjects"] = subjects
                    problem_storage[username] = problem_info

                    subject_problem_reply = f"{next_subject_after}은 이랬어요: {subject_problem}"
                    print(f"[{self.EXAM_NAME.upper()}] {next_subject_after} 과목 문제점 (조언 후): {subject_problem}")

                # 모든 과목 완료 확인
                if completed_count >= 5:
                    transition_to = "daily_routine"
                    if narration:
                        narration = f"{narration}\n\n모든 과목의 문제점을 해결했습니다. 일상 루틴으로 돌아갑니다."
                    else:
                        narration = "모든 과목의 문제점을 해결했습니다. 일상 루틴으로 돌아갑니다."

                    # 문제점 정보 초기화
                    if username in problem_storage:
                        del problem_storage[username]

                    print(f"[{self.EXAM_NAME.upper()}] 모든 과목 완료 - daily_routine으로 전이")

        return {
            'skip_llm': skip_llm,
            'reply': advice_reply,
            'narration': narration,
            'transition_to': transition_to,
            'subject_problem_reply': subject_problem_reply
        }

    def _generate_advice_reaction(self, user_message: str, subject: str, problem: str, is_good: bool) -> str:
        """
        조언에 대한 서가윤의 반응 생성 (LLM 사용)

        Args:
            user_message: 사용자의 조언 메시지
            subject: 과목명
            problem: 문제점
            is_good: 조언 적절성 여부

        Returns:
            str: 서가윤의 반응 메시지
        """
        try:
            if self.service.client:
                if is_good:
                    reaction_prompt = f"""당신은 서가윤입니다. 멘토(플레이어)가 {subject} 과목의 문제점('{problem}')에 대해 다음과 같은 조언을 했습니다:

"{user_message}"

이 조언이 적절하고 도움이 되는 조언입니다. 서가윤의 캐릭터에 맞게 이 조언에 대한 반응을 자연스럽게 한 문장으로 표현하세요. 감사하고 고마워하는 긍정적인 반응을 보이세요."""
                else:
                    reaction_prompt = f"""당신은 서가윤입니다. 멘토(플레이어)가 {subject} 과목의 문제점('{problem}')에 대해 다음과 같은 조언을 했습니다:

"{user_message}"

이 조언이 적절하지 못한 조언입니다. 서가윤의 캐릭터에 맞게 이 조언에 대한 반응을 자연스럽게 한 문장으로 표현하세요. 당황하거나 어색해하는 반응을 보이되, "잘모르겠네요", "음... 그렇군요" 같은 자연스러운 표현을 사용하세요."""

                response = self.service.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.service._build_system_prompt()},
                        {"role": "user", "content": reaction_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=100
                )
                reaction = response.choices[0].message.content.strip()
                print(f"[{self.EXAM_NAME.upper()}_ADVICE] 조언 반응 생성: {reaction}")
                return reaction
            else:
                return "감사해요! 도움이 될 것 같아요." if is_good else "잘모르겠네요..."
        except Exception as e:
            print(f"[ERROR] 조언 반응 생성 실패: {e}")
            return "감사해요! 도움이 될 것 같아요." if is_good else "잘모르겠네요..."


class JuneExamFeedbackHandler(ExamFeedbackHandlerBase):
    """6exam_feedback state handler"""
    EXAM_NAME = "6exam_feedback"
    EXAM_DISPLAY_NAME = "6월 모의고사 피드백"
    PROBLEM_STORAGE_ATTR = "june_exam_problems"
    SUBJECT_PROBLEM_TRIGGER = "june_exam_subject_problem"
    ADVICE_GIVEN_TRIGGER = "june_exam_advice_given"
    PROBLEM_GENERATOR_METHOD = "_generate_june_subject_problem"


class SeptemberExamFeedbackHandler(ExamFeedbackHandlerBase):
    """9exam_feedback state handler"""
    EXAM_NAME = "9exam_feedback"
    EXAM_DISPLAY_NAME = "9월 모의고사 피드백"
    PROBLEM_STORAGE_ATTR = "september_exam_problems"
    SUBJECT_PROBLEM_TRIGGER = "september_exam_subject_problem"
    ADVICE_GIVEN_TRIGGER = "september_exam_advice_given"
    PROBLEM_GENERATOR_METHOD = "_generate_september_subject_problem"
