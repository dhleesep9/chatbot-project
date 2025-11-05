"""Mock Display State Handler

사설모의고사 성적 발표 state에서의 로직을 처리합니다.
- 성적표 생성 및 출력
- 각 과목별 문제점 생성
- mock_exam 상태로 자동 전이
"""

from typing import Dict, Any, Optional
from services.handlers.base_handler import BaseStateHandler


class MockDisplayHandler(BaseStateHandler):
    """mock_display state handler"""

    def on_enter(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        mock_display state 진입 시 성적표 생성 및 출력

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        # 현재 주차 기록 (한 주에 한 번만 보도록)
        current_week = self.service._get_current_week(username)
        self.service.mock_exam_last_week[username] = current_week
        print(f"[MOCK_DISPLAY] {username}의 사설모의고사 응시 주차 기록: {current_week}주차")

        # 사설모의고사 응시 후 체력과 멘탈 감소
        current_stamina = self.service._get_stamina(username)
        current_mental = self.service._get_mental(username)
        new_stamina = max(0, current_stamina - 10)
        new_mental = max(0, current_mental - 10)
        self.service._set_stamina(username, new_stamina)
        self.service._set_mental(username, new_mental)
        print(f"[MOCK_DISPLAY] {username}의 체력 {current_stamina} → {new_stamina} (-10), 멘탈 {current_mental} → {new_mental} (-10)")

        # 사설모의고사 응시 - 성적표 생성
        mock_exam_scores = self.service._calculate_mock_exam_scores(username)

        # 성적표 나레이션 생성
        score_lines = []
        for subject, score_data in mock_exam_scores.items():
            score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")

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

        print(f"[MOCK_DISPLAY] {username}의 사설모의고사 성적표 생성 완료.")
        print(f"[MOCK_DISPLAY] 과목별 문제점 생성 완료: {list(subject_problems.keys())}")

        # mock_display에서는 성적표 narration만 표시하고,
        # 바로 mock_exam으로 자동 전이
        return {
            'skip_llm': True,  # LLM 호출 건너뛰기
            'reply': None,  # 성적표는 narration으로만 표시
            'narration': mock_exam_narration,
            'transition_to': 'mock_exam'  # 바로 mock_exam으로 전이
        }

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        mock_display state에서 사용자 입력 처리
        (일반적으로 자동 전이되므로 호출되지 않음)

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        print(f"[MOCK_DISPLAY] {username}이(가) mock_display 상태에서 입력했습니다. mock_exam으로 전환합니다.")

        # 바로 mock_exam으로 전환
        return {
            'skip_llm': True,
            'reply': None,
            'narration': None,
            'transition_to': 'mock_exam'
        }
