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
        # 사설모의고사 응시 - 성적표 생성
        mock_exam_scores = self.service._calculate_mock_exam_scores(username)
        weak_subject = self.service._identify_weak_subject(mock_exam_scores)
        weakness_message = self.service._generate_weakness_message(weak_subject, mock_exam_scores.get(weak_subject, {}))

        # 취약점 메시지 검증 (비어있으면 기본 메시지 사용)
        if not weakness_message or len(weakness_message.strip()) == 0:
            weakness_message = f"{weak_subject}에서 어려운 부분이 많았어요. 특히 응용 문제가 어려웠어요."
            print(f"[MOCK_EXAM_WARN] 취약점 메시지가 비어있어 기본 메시지 사용: {weakness_message}")

        # 성적표 나레이션 생성
        score_lines = []
        for subject, score_data in mock_exam_scores.items():
            score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")

        # 평균 등급 계산 및 반응 생성
        average_grade = self.service._calculate_average_grade(mock_exam_scores)
        grade_reaction = self.service._generate_grade_reaction("mock_exam", average_grade)

        # 나레이션에는 성적표만 포함
        mock_exam_narration = "사설모의고사 성적표가 발표되었습니다:\n" + "\n".join(score_lines)

        # 취약점 정보 저장 (피드백에서 사용)
        self.service.mock_exam_weakness[username] = {
            "subject": weak_subject,
            "message": weakness_message
        }

        print(f"[MOCK_EXAM] {username}의 사설모의고사 성적표 생성 완료. 취약 과목: {weak_subject}")
        print(f"[MOCK_EXAM] 취약점 메시지 즉시 표시: {weakness_message}")

        # state 정보 가져오기
        state_info = self.service._get_state_info("mock_exam")
        state_name = state_info.get("name", "사설모의고사 응시") if state_info else "사설모의고사 응시"

        return {
            'skip_llm': True,  # LLM 호출 건너뛰기
            'reply': f"[{state_name}] {weakness_message}",
            'narration': mock_exam_narration,
            'transition_to': 'mock_exam_feedback',
            'data': {
                'mock_exam_scores': mock_exam_scores,
                'weak_subject': weak_subject,
                'weakness_message': weakness_message,
                'grade_reaction': grade_reaction
            }
        }
