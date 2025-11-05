"""Official Exam State Handler

6exam_pre/9exam_pre/11exam_pre와 6exam/9exam/11exam state에서의 로직을 처리합니다.
- _pre state: 시험 전 응원 메시지 표시
- exam state: 성적 발표
"""

from typing import Dict, Any, Optional
from services.handlers.base_handler import BaseStateHandler


class ExamPreHandlerBase(BaseStateHandler):
    """6exam_pre, 9exam_pre, 11exam_pre의 공통 로직을 처리하는 base class"""

    # 서브클래스에서 정의해야 할 속성
    EXAM_NAME = None  # "6exam_pre" or "9exam_pre" or "11exam_pre"
    EXAM_DISPLAY_NAME = None  # "6월 평가원 모의고사" or "9월 평가원 모의고사" or "수능"

    def on_enter(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        시험 전 응원 메시지 표시

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        raise NotImplementedError("Subclass must implement on_enter")


class SixExamPreHandler(ExamPreHandlerBase):
    """6exam_pre state handler"""
    EXAM_NAME = "6exam_pre"
    EXAM_DISPLAY_NAME = "6월 평가원 모의고사"

    def on_enter(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        6월 모의고사 전 응원 메시지 표시

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        # 호감도 가져오기
        affection = self.service._get_affection(username)

        # 호감도에 따른 메시지 선택 (배열로 분리)
        if affection >= 50:
            # \n으로 구분하여 별도의 말뭉치로 출력
            fixed_reply = [
                "쌤 ... 너무 떨려용..",
                "재수 시작하고 제대로 치는 첫 평가원 모의고사에요...",
                "저 잘 할 수 있겠죠 ..??  응원해주세요 ㅠㅠ"
            ]
        else:
            fixed_reply = ["썜 .. 잘하고 올게요.. ㅠ"]

        print(f"[{self.EXAM_NAME.upper()}] {username}의 호감도: {affection} - 메시지 개수: {len(fixed_reply)}")

        # narration은 state JSON에 정의되어 있으므로 여기서는 None
        return {
            'skip_llm': True,  # LLM 호출 건너뛰기
            'fixed_reply': fixed_reply,  # 배열로 반환
            'narration': None,  # state JSON의 narration 사용
            'transition_to': None
        }


class NineExamPreHandler(ExamPreHandlerBase):
    """9exam_pre state handler"""
    EXAM_NAME = "9exam_pre"
    EXAM_DISPLAY_NAME = "9월 평가원 모의고사"

    def on_enter(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        9월 모의고사 전 응원 메시지 표시

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        # 고정 메시지 (배열로 분리)
        fixed_reply = [
            "쌤, 이번엔 진짜 잡을 거예요.",
            "9평은… 절대 안 망할꺼에요!!!!"
        ]

        print(f"[{self.EXAM_NAME.upper()}] {username} - 메시지 개수: {len(fixed_reply)}")

        # narration은 state JSON에 정의되어 있으므로 여기서는 None
        return {
            'skip_llm': True,  # LLM 호출 건너뛰기
            'fixed_reply': fixed_reply,  # 배열로 반환
            'narration': None,  # state JSON의 narration 사용
            'transition_to': None
        }


class ElevenExamPreHandler(ExamPreHandlerBase):
    """11exam_pre state handler"""
    EXAM_NAME = "11exam_pre"
    EXAM_DISPLAY_NAME = "수능"

    def on_enter(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        수능 전 응원 메시지 표시

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        # 호감도와 멘탈 가져오기
        affection = self.service._get_affection(username)
        mental = self.service._get_mental(username)

        # 우선순위대로 메시지 선택 (배열로 분리)
        if mental < 30:
            # 멘탈 30 미만 (최우선)
            fixed_reply = [
                "선생님 ㅠㅠㅠㅠ 저 잘 볼 수 있겠죠??",
                "너무 불안하고 떨려요 ... 아는 것도 다 실수 할 거 같아요 ...."
            ]
        elif affection < 20:
            # 호감도 20 미만
            fixed_reply = ["....."]
        elif affection < 50:
            # 호감도 50 미만
            fixed_reply = ["선생님 저 잘 보고 올게요 .."]
        else:
            # 호감도 50 이상
            fixed_reply = ["쌤,  최선을 다해서 잘 보고 올게요...! 저 위해서 꼭 기도해주셔야 해요 !!"]

        print(f"[{self.EXAM_NAME.upper()}] {username}의 호감도: {affection}, 멘탈: {mental} - 메시지 개수: {len(fixed_reply)}")

        # narration은 state JSON에 정의되어 있으므로 여기서는 None
        return {
            'skip_llm': True,  # LLM 호출 건너뛰기
            'fixed_reply': fixed_reply,  # 배열로 반환
            'narration': None,  # state JSON의 narration 사용
            'transition_to': None
        }


class OfficialExamHandlerBase(BaseStateHandler):
    """6exam, 9exam의 공통 로직을 처리하는 base class"""

    # 서브클래스에서 정의해야 할 속성
    EXAM_NAME = None  # "6exam" or "9exam"
    EXAM_DISPLAY_NAME = None  # "6월 모의고사" or "9월 모의고사"
    FEEDBACK_STATE = None  # "6exam_feedback" or "9exam_feedback"
    PROBLEM_STORAGE_ATTR = None  # "june_exam_problems" or "september_exam_problems"

    QUESTION_KEYWORDS = [
        "어땠니", "어떠니", "어떠니요", "어땠", "어떤지", "어떠냐",
        "어떠세요", "어떠", "어떻", "어떠한지", "어떠했니", "어떠했어",
        "결과", "성적", "어땠어", "어떠했어요", "어떠했니요"
    ]

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        모의고사 성적 발표 로직 처리

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        # 질문 키워드 확인
        user_message_lower = user_message.lower()
        is_asking = any(keyword in user_message_lower for keyword in self.QUESTION_KEYWORDS)

        # 문제점 추적 시스템 확인
        problem_storage = getattr(self.service, self.PROBLEM_STORAGE_ATTR)
        problem_info = problem_storage.get(username, {})
        scores_already_shown = problem_info and problem_info.get("scores")

        # 질문이 들어왔거나 성적이 아직 발표되지 않은 경우 성적 발표
        if is_asking or not scores_already_shown:
            # 성적 계산 (전략 보너스 없음)
            exam_scores = self.service._calculate_mock_exam_scores(username)
            
            # 성적이 제대로 계산되었는지 확인
            if not exam_scores or len(exam_scores) == 0:
                print(f"[{self.EXAM_NAME.upper()}] 경고: 성적 계산 결과가 비어있습니다. 재계산 시도...")
                exam_scores = self.service._calculate_mock_exam_scores(username)
            
            print(f"[{self.EXAM_NAME.upper()}] 계산된 성적: {exam_scores}")

            # 성적표 나레이션 생성 (한 번만)
            narration = None
            if not scores_already_shown:
                score_parts = []
                for subject in ["국어", "수학", "영어", "탐구1", "탐구2"]:
                    if subject in exam_scores:
                        score_data = exam_scores[subject]
                        score_parts.append(f"{subject} {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                    else:
                        print(f"[{self.EXAM_NAME.upper()}] 경고: {subject} 과목의 성적이 없습니다.")

                # 성적표가 비어있지 않은 경우에만 나레이션 생성
                if score_parts:
                    # 나레이션에는 성적 발표 안내와 성적표 포함
                    narration = f"{self.EXAM_DISPLAY_NAME} 성적이 발표되었습니다.\n" + " ".join(score_parts)
                    print(f"[{self.EXAM_NAME.upper()}] 생성된 나레이션: {narration}")
                else:
                    print(f"[{self.EXAM_NAME.upper()}] 오류: 성적표가 비어있습니다!")

                # 문제점 추적 시스템 초기화
                problem_storage[username] = {
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

                print(f"[{self.EXAM_NAME.upper()}] {username}의 {self.EXAM_DISPLAY_NAME} 성적 발표 완료")

            return {
                'skip_llm': False,  # LLM 호출 진행
                'reply': None,  # LLM이 생성
                'narration': narration,
                'transition_to': self.FEEDBACK_STATE,
                'data': {
                    'exam_scores': exam_scores
                }
            }
        else:
            # 질문이 아닌 경우: LLM 호출 건너뛰기 (성적 발표 전까지 대기)
            state_info = self.service._get_state_info(self.EXAM_NAME)
            state_name = state_info.get("name", self.EXAM_NAME)
            reply = f"[{state_name}] {self.EXAM_DISPLAY_NAME}가 끝났어요. 시험 결과를 보고 싶으시면 '어땠니?'라고 물어봐주세요."
            print(f"[{self.EXAM_NAME.upper()}] {username}의 메시지가 질문이 아님 - LLM 호출 건너뛰기 (질문 대기)")

            return {
                'skip_llm': True,  # LLM 호출 건너뛰기
                'reply': reply,
                'narration': None,
                'transition_to': None
            }


class JuneExamHandler(OfficialExamHandlerBase):
    """6exam state handler"""
    EXAM_NAME = "6exam"
    EXAM_DISPLAY_NAME = "6월 모의고사"
    FEEDBACK_STATE = "6exam_feedback"
    PROBLEM_STORAGE_ATTR = "june_exam_problems"


class SeptemberExamHandler(OfficialExamHandlerBase):
    """9exam state handler"""
    EXAM_NAME = "9exam"
    EXAM_DISPLAY_NAME = "9월 모의고사"
    FEEDBACK_STATE = "9exam_feedback"
    PROBLEM_STORAGE_ATTR = "september_exam_problems"


class CSATExamHandler(OfficialExamHandlerBase):
    """11exam state handler (수능) - 피드백 없이 성적 발표만"""
    EXAM_NAME = "11exam"
    EXAM_DISPLAY_NAME = "수능"
    FEEDBACK_STATE = None  # 피드백 없음
    PROBLEM_STORAGE_ATTR = "csat_exam_scores"

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        수능 성적 발표 로직 처리 (피드백 없음)

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        # 대학지원하기 키워드 확인 - transition이 처리하도록 None 반환
        if "대학지원하기" in user_message or "대학 지원하기" in user_message:
            print(f"[{self.EXAM_NAME.upper()}] 대학지원하기 감지 - transition으로 처리하도록 None 반환")
            return None  # transition이 처리하도록 None 반환

        # 지원 가능 대학 보기 키워드 확인
        if "지원가능대학" in user_message or "지원 가능 대학" in user_message or "합격 가능 대학" in user_message:
            print(f"[{self.EXAM_NAME.upper()}] 지원 가능 대학 확인 요청 감지: {user_message}")
            return self._handle_university_check(username, user_message)

        # 성적 정보 확인
        score_storage = getattr(self.service, self.PROBLEM_STORAGE_ATTR, {})
        scores_already_shown = score_storage.get(username, {}).get("scores")

        # 성적이 아직 발표되지 않은 경우 성적 발표
        if not scores_already_shown:
            # 성적 계산 (전략 보너스 없음)
            exam_scores = self.service._calculate_mock_exam_scores(username)

            # 성적표 나레이션 생성
            score_parts = []
            for subject in ["국어", "수학", "영어", "탐구1", "탐구2"]:
                if subject in exam_scores:
                    score_data = exam_scores[subject]
                    score_parts.append(f"{subject} {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")

            # 나레이션: 성적표만 (수능 끝 안내는 transition_narration에서 처리)
            narration = " ".join(score_parts)

            # 성적 정보 저장
            if not hasattr(self.service, self.PROBLEM_STORAGE_ATTR):
                setattr(self.service, self.PROBLEM_STORAGE_ATTR, {})
            score_storage = getattr(self.service, self.PROBLEM_STORAGE_ATTR)
            score_storage[username] = {"scores": exam_scores}

            print(f"[{self.EXAM_NAME.upper()}] {username}의 {self.EXAM_DISPLAY_NAME} 성적 발표 완료")

            # 엔딩 조건 체크 (우선순위: 5급 공채 > 유학)
            transition_to = self._check_public_agent_ending(username, exam_scores)
            if not transition_to:
                transition_to = self._check_world_await_ending(username, exam_scores)

            return {
                'skip_llm': False,  # LLM 호출 진행
                'reply': None,  # LLM이 생성
                'narration': narration,
                'transition_to': transition_to,  # 조건 만족 시 public_agent로 전이
                'data': {
                    'exam_scores': exam_scores
                }
            }
        # 성적이 이미 발표된 경우 (재확인)
        elif scores_already_shown:
            # 이미 발표된 성적을 다시 보여줌
            exam_scores = scores_already_shown

            score_parts = []
            for subject in ["국어", "수학", "영어", "탐구1", "탐구2"]:
                if subject in exam_scores:
                    score_data = exam_scores[subject]
                    score_parts.append(f"{subject} {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")

            narration = f"{self.EXAM_DISPLAY_NAME} 성적: " + " ".join(score_parts)

            return {
                'skip_llm': False,  # LLM 호출 진행
                'reply': None,  # LLM이 생성
                'narration': narration,
                'transition_to': None,
                'data': {
                    'exam_scores': exam_scores
                }
            }
        else:
            # 질문이 아닌 경우: LLM 호출 건너뛰기
            state_info = self.service._get_state_info(self.EXAM_NAME)
            state_name = state_info.get("name", self.EXAM_NAME) if state_info else self.EXAM_NAME
            reply = f"[{state_name}] {self.EXAM_DISPLAY_NAME}가 끝났어요. 시험 결과를 보고 싶으시면 '어땠니?'라고 물어봐주세요."
            print(f"[{self.EXAM_NAME.upper()}] {username}의 메시지가 질문이 아님 - LLM 호출 건너뛰기 (질문 대기)")
            
            return {
                'skip_llm': True,  # LLM 호출 건너뛰기
                'reply': reply,
                'narration': None,
                'transition_to': None  # 상태 전이 없음
            }
    
    def _handle_university_check(self, username: str, user_message: str) -> Dict[str, Any]:
        """
        지원 가능 대학 확인 로직
        
        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            
        Returns:
            Dict: 처리 결과
        """
        # 성적 정보 가져오기
        score_storage = getattr(self.service, self.PROBLEM_STORAGE_ATTR, {})
        exam_scores = score_storage.get(username, {}).get("scores") if isinstance(score_storage.get(username), dict) else None
        
        if not exam_scores:
            # 성적이 없는 경우 다시 계산
            exam_scores = self.service._calculate_mock_exam_scores(username)
            score_storage = getattr(self.service, self.PROBLEM_STORAGE_ATTR)
            score_storage[username] = {"scores": exam_scores}
            setattr(self.service, self.PROBLEM_STORAGE_ATTR, score_storage)
        
        # 평균 백분위 계산
        percentiles = [exam_scores[subject]['percentile'] for subject in ["국어", "수학", "영어", "탐구1", "탐구2"] if subject in exam_scores]
        avg_percentile = sum(percentiles) / len(percentiles) if percentiles else 0.0
        
        # 대학 정보 로드
        universities = self.service._get_university_admissions_info()
        
        # 지원 가능/불가 대학 분류
        eligible_universities = []
        ineligible_universities = []
        
        for uni in universities:
            is_eligible = avg_percentile >= uni.get('cutoff_percentile', 0)
            if is_eligible:
                eligible_universities.append(uni)
            else:
                ineligible_universities.append(uni)
        
        # 백분위 순으로 정렬 (높은 순)
        eligible_universities.sort(key=lambda x: x.get('cutoff_percentile', 0), reverse=True)
        ineligible_universities.sort(key=lambda x: x.get('cutoff_percentile', 0), reverse=True)
        
        # 결과 메시지 생성 (narration으로 전체 리스트 표시)
        narration = f"평균 백분위: {avg_percentile:.1f}%\n\n"
        
        if eligible_universities:
            narration += f"📋 [지원 가능 대학/학과]\n"
            for uni in eligible_universities:
                narration += f"\n✅ {uni['university']} {uni['department']} (커트라인: {uni['cutoff_percentile']}%)"
        else:
            narration += "📋 [지원 가능 대학/학과]\n없음"
        
        narration += "\n\n"
        
        if ineligible_universities:
            narration += f"📋 [지원 불가 대학/학과]\n"
            for uni in ineligible_universities:
                narration += f"\n❌ {uni['university']} {uni['department']} (커트라인: {uni['cutoff_percentile']}%)"
        else:
            narration += "📋 [지원 불가 대학/학과]\n없음"
        
        print(f"[{self.EXAM_NAME.upper()}] {username}의 지원 가능 대학 확인 - 평균 백분위: {avg_percentile:.1f}%")
        
        return {
            'skip_llm': True,
            'reply': None,
            'narration': narration,
            'transition_to': None
        }

    def _check_public_agent_ending(self, username: str, exam_scores: Dict[str, Any]) -> Optional[str]:
        """
        5급 공채 엔딩 조건 체크

        조건:
        - 영어 3등급 이하
        - 수학 3등급 이하
        - 국어 1등급
        - (탐구1 + 탐구2) / 2 < 2.0 (평균이 2등급보다 좋음)

        Args:
            username: 사용자 이름
            exam_scores: 수능 성적

        Returns:
            Optional[str]: 조건 만족 시 'public_agent', 아니면 None
        """
        try:
            # 각 과목 등급 가져오기
            korean_grade = exam_scores.get('국어', {}).get('grade', 9)
            math_grade = exam_scores.get('수학', {}).get('grade', 9)
            english_grade = exam_scores.get('영어', {}).get('grade', 9)
            tamgu1_grade = exam_scores.get('탐구1', {}).get('grade', 9)
            tamgu2_grade = exam_scores.get('탐구2', {}).get('grade', 9)

            # 탐구 평균 계산
            tamgu_avg = (tamgu1_grade + tamgu2_grade) / 2.0

            # 조건 체크
            is_english_ok = english_grade >= 3  # 3등급 이하 (3, 4, 5, ...)
            is_math_ok = math_grade >= 3  # 3등급 이하
            is_korean_ok = korean_grade == 1  # 1등급
            is_tamgu_ok = tamgu_avg < 2.0  # 평균이 2.0보다 작음 (1~2등급 사이)

            print(f"[11EXAM] {username}의 5급 공채 조건 체크:")
            print(f"  - 국어 {korean_grade}등급 (1등급 필요): {is_korean_ok}")
            print(f"  - 수학 {math_grade}등급 (3등급 이하): {is_math_ok}")
            print(f"  - 영어 {english_grade}등급 (3등급 이하): {is_english_ok}")
            print(f"  - 탐구 평균 {tamgu_avg:.1f}등급 (2등급 미만): {is_tamgu_ok}")

            # 모든 조건 만족 시
            if is_korean_ok and is_math_ok and is_english_ok and is_tamgu_ok:
                print(f"[11EXAM] {username}의 5급 공채 엔딩 조건 만족! public_agent로 전이")
                return 'public_agent'
            else:
                print(f"[11EXAM] {username}의 5급 공채 엔딩 조건 미충족")
                return None

        except Exception as e:
            print(f"[11EXAM] 5급 공채 조건 체크 중 오류 발생: {e}")
            return None

    def _check_world_await_ending(self, username: str, exam_scores: Dict[str, Any]) -> Optional[str]:
        """
        유학 엔딩 조건 체크

        조건:
        - 수능 평균 4등급 이하
        - 자신감 80점 이상

        Args:
            username: 사용자 이름
            exam_scores: 수능 성적

        Returns:
            Optional[str]: 조건 만족 시 'world_await', 아니면 None
        """
        try:
            # 각 과목 등급 가져오기
            korean_grade = exam_scores.get('국어', {}).get('grade', 9)
            math_grade = exam_scores.get('수학', {}).get('grade', 9)
            english_grade = exam_scores.get('영어', {}).get('grade', 9)
            tamgu1_grade = exam_scores.get('탐구1', {}).get('grade', 9)
            tamgu2_grade = exam_scores.get('탐구2', {}).get('grade', 9)

            # 평균 등급 계산
            avg_grade = (korean_grade + math_grade + english_grade + tamgu1_grade + tamgu2_grade) / 5.0

            # 자신감 가져오기
            confidence = self.service._get_confidence(username)

            # 조건 체크
            is_low_grade = avg_grade >= 4.0  # 평균 4등급 이하 (4, 5, 6...)
            is_high_confidence = confidence >= 80  # 자신감 80 이상

            print(f"[11EXAM] {username}의 유학 엔딩 조건 체크:")
            print(f"  - 평균 등급: {avg_grade:.2f} (4등급 이하 필요): {is_low_grade}")
            print(f"  - 자신감: {confidence} (80 이상 필요): {is_high_confidence}")

            # 모든 조건 만족 시
            if is_low_grade and is_high_confidence:
                print(f"[11EXAM] {username}의 유학 엔딩 조건 만족! world_await로 전이")
                return 'world_await'
            else:
                print(f"[11EXAM] {username}의 유학 엔딩 조건 미충족")
                return None

        except Exception as e:
            print(f"[11EXAM] 유학 엔딩 조건 체크 중 오류 발생: {e}")
            return None
