"""Mock Exam State Handler

사설모의고사 응시 state에서의 로직을 처리합니다.
- 성적표 생성
- 취약 과목 식별
- 취약점 메시지 생성
- mock_exam_feedback 상태로 자동 전이
"""

from typing import Dict, Any, Optional
from services.handlers.base_handler import BaseStateHandler


class MockExamHandler(BaseStateHandler):
    """mock_exam state handler"""

    def on_enter(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        mock_exam state 진입 시 자동으로 성적표 생성 및 피드백 전이

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        # 현재 주차 기록 (한 주에 한 번만 보도록)
        current_week = self.service._get_current_week(username)
        self.service.mock_exam_last_week[username] = current_week
        print(f"[MOCK_EXAM] {username}의 사설모의고사 응시 주차 기록: {current_week}주차")

        # 사설모의고사 응시 후 체력과 멘탈 감소
        current_stamina = self.service._get_stamina(username)
        current_mental = self.service._get_mental(username)
        new_stamina = max(0, current_stamina - 10)
        new_mental = max(0, current_mental - 10)
        self.service._set_stamina(username, new_stamina)
        self.service._set_mental(username, new_mental)
        print(f"[MOCK_EXAM] {username}의 체력 {current_stamina} → {new_stamina} (-10), 멘탈 {current_mental} → {new_mental} (-10)")

        # 사설모의고사 응시 - 성적표 생성
        mock_exam_scores = self.service._calculate_mock_exam_scores(username)

        # 성적표 나레이션 생성
        score_lines = []
        for subject, score_data in mock_exam_scores.items():
            score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")

        # 평균 등급 계산 및 반응 생성
        average_grade = self.service._calculate_average_grade(mock_exam_scores)
        grade_reaction = self.service._generate_grade_reaction("mock_exam", average_grade)

        # 나레이션에는 성적표만 포함
        mock_exam_narration = "사설모의고사 성적표가 발표되었습니다:\n" + "\n".join(score_lines)

        # 각 과목별 문제점 생성
        subject_problems = {}
        for subject, score_data in mock_exam_scores.items():
            problem_message = self.service._generate_weakness_message(subject, score_data)
            if not problem_message or len(problem_message.strip()) == 0:
                problem_message = f"{subject}에서 어려운 부분이 많았어요. 특히 응용 문제가 어려웠어요."
            subject_problems[subject] = problem_message

        # 과목 순서 정의 (고정 순서)
        subject_order = list(mock_exam_scores.keys())

        # 피드백 처리를 위한 정보 저장
        self.service.mock_exam_weakness[username] = {
            "scores": mock_exam_scores,
            "subject_problems": subject_problems,
            "subject_order": subject_order,
            "current_index": 0,  # 현재 처리 중인 과목 인덱스
            "completed_subjects": []  # 완료된 과목 목록
        }

        print(f"[MOCK_EXAM] {username}의 사설모의고사 성적표 생성 완료.")
        print(f"[MOCK_EXAM] 과목별 문제점 생성 완료: {list(subject_problems.keys())}")

        # state 정보 가져오기
        state_info = self.service._get_state_info("mock_exam")
        state_name = state_info.get("name", "사설모의고사 응시") if state_info else "사설모의고사 응시"

        # mock_exam에서는 성적표 narration만 표시하고,
        # reply는 mock_exam_feedback의 on_enter에서 표시
        return {
            'skip_llm': True,  # LLM 호출 건너뛰기
            'reply': None,  # mock_exam_feedback의 on_enter에서 표시
            'narration': mock_exam_narration,
            'transition_to': 'mock_exam_feedback',
            'data': {
                'mock_exam_scores': mock_exam_scores,
                'grade_reaction': grade_reaction
            }
        }

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        mock_exam state에서 사용자 입력 처리
        (transition_to가 작동하지 않아 mock_exam 상태에 머물러 있는 경우를 대비)

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        print(f"[MOCK_EXAM] {username}이(가) mock_exam 상태에 있습니다. mock_exam_feedback으로 전환합니다.")

        # 취약점 정보가 이미 저장되어 있는지 확인
        if username not in self.service.mock_exam_weakness:
            # 취약점 정보가 없으면 다시 생성 (on_enter가 호출되지 않았을 경우)
            print(f"[MOCK_EXAM] 취약점 정보가 없습니다. 다시 생성합니다.")

            # 사설모의고사 응시 - 성적표 생성
            mock_exam_scores = self.service._calculate_mock_exam_scores(username)
            weak_subject = self.service._identify_weak_subject(mock_exam_scores)
            weakness_message = self.service._generate_weakness_message(weak_subject, mock_exam_scores.get(weak_subject, {}))

            if not weakness_message or len(weakness_message.strip()) == 0:
                weakness_message = f"{weak_subject}에서 어려운 부분이 많았어요. 특히 응용 문제가 어려웠어요."

            # 성적표 나레이션 생성
            score_lines = []
            for subject, score_data in mock_exam_scores.items():
                score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")

            mock_exam_narration = "사설모의고사 성적표가 발표되었습니다:\n" + "\n".join(score_lines)

            # 취약점 정보 저장
            self.service.mock_exam_weakness[username] = {
                "subject": weak_subject,
                "message": weakness_message,
                "scores": mock_exam_scores  # 성적표도 함께 저장
            }

            print(f"[MOCK_EXAM] 성적표 재생성 완료. 취약 과목: {weak_subject}")
        else:
            # 취약점 정보가 있으면 기존 정보 사용
            weakness_info = self.service.mock_exam_weakness[username]
            weakness_message = weakness_info.get("message", "")

            # 성적표 재생성 (저장된 scores가 없으므로 다시 계산)
            mock_exam_scores = self.service._calculate_mock_exam_scores(username)
            score_lines = []
            for subject, score_data in mock_exam_scores.items():
                score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")

            mock_exam_narration = "사설모의고사 성적표가 발표되었습니다:\n" + "\n".join(score_lines)

            print(f"[MOCK_EXAM] 기존 취약점 정보 사용")

        # state 정보 가져오기
        state_info = self.service._get_state_info("mock_exam")
        state_name = state_info.get("name", "사설모의고사 응시") if state_info else "사설모의고사 응시"

        # mock_exam_feedback으로 전환
        return {
            'skip_llm': True,
            'reply': f"[{state_name}] {weakness_message}",
            'narration': mock_exam_narration,
            'transition_to': 'mock_exam_feedback'
        }
