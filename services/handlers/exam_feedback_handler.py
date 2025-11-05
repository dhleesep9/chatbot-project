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
        exam_type = "사설모의고사" if "6" in self.EXAM_NAME else "정규모의고사"
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
        if not problem_info or not problem_info.get("scores"):
            # 6exam 핸들러에서 이미 성적을 저장했을 수 있으므로 먼저 확인
            if problem_info and problem_info.get("scores"):
                exam_scores = problem_info["scores"]
            else:
                # 성적이 없으면 다시 계산
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
            print(f"[{self.EXAM_NAME.upper()}] 문제점 정보 초기화 완료: 성적={exam_scores}")

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

        # [1] 문제점 제시 확인 (트리거 사용 또는 자동 시작)
        subject_problem_trigger = {
            "trigger_type": self.SUBJECT_PROBLEM_TRIGGER,
            "conditions": {}
        }

        # 다음 과목이 있고, 현재 과목이 없으면 자동으로 첫 번째 과목 문제점 제시
        # (트리거가 발동하거나, 처음 진입 시 자동으로 시작)
        should_show_problem = (
            next_subject and 
            current_subject is None and 
            not subjects.get(next_subject, {}).get("solved", False)
        )
        
        if should_show_problem:
            # 트리거 확인 또는 자동 시작
            trigger_fired = self.service.trigger_registry.evaluate_trigger(
                self.SUBJECT_PROBLEM_TRIGGER, subject_problem_trigger, trigger_context
            )
            
            # 트리거가 발동했거나, 처음 진입 시(current_subject가 None이고 완료된 과목이 없으면) 자동으로 시작
            if trigger_fired or (completed_count == 0 and current_subject is None):
                # 첫 번째 과목의 문제점 생성
                subject_scores = problem_info.get("scores", {}).get(next_subject, {})
                if not subject_scores:
                    print(f"[{self.EXAM_NAME.upper()}] 경고: {next_subject} 과목의 성적 정보가 없습니다. 성적 재계산 중...")
                    exam_scores = self.service._calculate_mock_exam_scores(username)
                    problem_info["scores"] = exam_scores
                    subject_scores = exam_scores.get(next_subject, {})
                    problem_storage[username] = problem_info
                
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

                # LLM으로 해결방안 적절성 판단 (각 핸들러에서 정의한 범위로 직접 평가)
                current_problem = subjects.get(current_subject, {}).get("problem", "")
                advice_score = self._judge_advice_quality_with_range(username, user_message, current_subject, current_problem)
                score_range = self._get_advice_score_range()
                print(f"[{self.EXAM_NAME.upper()}] 조언 점수: {advice_score}점 ({score_range[0]}~{score_range[1]})")

                # 조언에 대한 서가윤의 반응 생성 (LLM 사용)
                # 점수 범위에 따라 기준점 계산 (비율로 변환하여 판단)
                score_range = self._get_advice_score_range()
                min_score, max_score = score_range
                score_range_size = max_score - min_score
                
                # 점수를 0~20 범위로 정규화하여 기존 기준 적용
                if score_range_size > 0:
                    normalized_score_for_reaction = ((advice_score - min_score) / score_range_size) * 20
                else:
                    normalized_score_for_reaction = 10  # 기본값
                
                # 정규화된 점수가 11 이상이면 True (적절한 조언)
                is_good_for_reaction = normalized_score_for_reaction >= 11
                advice_reply = self._generate_advice_reaction(username, user_message, current_subject, current_problem, is_good_for_reaction)

                # 능력치/멘탈/호감도 변경 (점수에 따라)
                # 점수 범위에 따라 기준점 계산 (비율로 변환)
                score_range = self._get_advice_score_range()
                min_score, max_score = score_range
                score_range_size = max_score - min_score
                
                # 점수를 0~20 범위로 정규화하여 기존 기준 적용
                if score_range_size > 0:
                    normalized_score = ((advice_score - min_score) / score_range_size) * 20
                else:
                    normalized_score = 10  # 기본값
                
                # 정규화된 점수에 따라 호감도와 멘탈 변화 결정
                if normalized_score >= 11:
                    # 적절한 조언: 호감도 +2, 멘탈 +5
                    affection_change = 2
                    mental_change = 5
                    advice_quality_text = "적절한"
                elif normalized_score >= 6:
                    # 보통 조언: 호감도 +1, 멘탈 +2
                    affection_change = 1
                    mental_change = 2
                    advice_quality_text = "보통"
                else:
                    # 부적절한 조언: 호감도 -2, 멘탈 -5
                    affection_change = -2
                    mental_change = -5
                    advice_quality_text = "적절하지 않은"
                
                current_affection = self.service._get_affection(username)
                new_affection = max(0, min(100, current_affection + affection_change))
                self.service._set_affection(username, new_affection)

                current_mental = self.service._get_mental(username)
                new_mental = max(0, min(100, current_mental + mental_change))
                self.service._set_mental(username, new_mental)

                # 능력치 증가: 각 핸들러에서 정의한 범위로 직접 평가된 점수 사용
                abilities = self.service._get_abilities(username)
                increased_amount = 0.0
                if current_subject in abilities:
                    # 체력과 멘탈 효율 적용
                    stamina = self.service._get_stamina(username)
                    mental = self.service._get_mental(username)
                    efficiency = self.service._calculate_combined_efficiency(stamina, mental) / 100.0
                    
                    # advice_score는 이미 범위로 평가되었으므로 efficiency만 적용
                    base_increase = float(advice_score) * efficiency
                    # 배율 적용 (진로-과목 + 시험 전략)
                    increased_amount = self.service._apply_ability_multipliers(username, current_subject, base_increase)
                    abilities[current_subject] = min(2500, abilities[current_subject] + increased_amount)
                    self.service._set_abilities(username, abilities)

                # 정규화된 점수로 조언 품질 판단
                if normalized_score >= 6:
                    narration = f"{advice_quality_text} 조언이였습니다 {current_subject}과목 능력치 +{increased_amount:.0f} 멘탈 {mental_change:+.0f} 호감도 {affection_change:+.0f}"
                else:
                    narration = f"{advice_quality_text} 조언이였습니다. 호감도 {affection_change:+.0f}, 멘탈 {mental_change:+.0f}"
                
                print(f"[{self.EXAM_NAME.upper()}] {current_subject} 조언 점수: {advice_score}점 - 호감도 {affection_change:+.0f}, 멘탈 {mental_change:+.0f}, 능력치 +{increased_amount:.2f}")

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

    def _generate_advice_reaction(self, username: str, user_message: str, subject: str, problem: str, is_good: bool) -> str:
        """
        조언에 대한 서가윤의 반응 생성 (LLM 사용)

        Args:
            username: 사용자 이름
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
                        {"role": "system", "content": self.service._build_system_prompt(username)},
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

    def _get_advice_score_range(self) -> tuple:
        """6월 모의고사: 100~200 범위로 평가"""
        return (100, 200)


class SeptemberExamFeedbackHandler(ExamFeedbackHandlerBase):
    """9exam_feedback state handler"""
    EXAM_NAME = "9exam_feedback"
    EXAM_DISPLAY_NAME = "9월 모의고사 피드백"
    PROBLEM_STORAGE_ATTR = "september_exam_problems"
    SUBJECT_PROBLEM_TRIGGER = "september_exam_subject_problem"
    ADVICE_GIVEN_TRIGGER = "september_exam_advice_given"
    PROBLEM_GENERATOR_METHOD = "_generate_september_subject_problem"

    def _get_advice_score_range(self) -> tuple:
        """9월 모의고사: 300~400 범위로 평가"""
        return (300, 400)
