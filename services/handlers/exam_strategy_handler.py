"""Exam Strategy State Handler

시험 전략 수립 state에서의 로직을 처리합니다.
- 과목별 전략 수집
- 전략 품질 평가 (LLM 사용)
- 보너스 배수 계산 및 저장
"""

from typing import Dict, Any, Optional
from services.handlers.base_handler import BaseStateHandler


class ExamStrategyHandler(BaseStateHandler):
    """exam_strategy state handler"""

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        시험 전략 수립 로직 처리

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과 {
                'skip_llm': bool,  # LLM 호출 건너뛰기 여부
                'reply': str,  # 서가윤의 응답 (LLM으로 생성 예정이면 None)
                'narration': str,  # 나레이션 메시지
                'user_input': str  # LLM에 전달할 사용자 입력 (전략 텍스트)
            }
        """
        # 전략 저장소 초기화
        if username not in self.service.exam_progress:
            self.service.exam_progress[username] = {
                "strategies": {},  # 과목별 전략 저장 {"국어": {"strategy": "...", "quality": "VERY_GOOD"}, ...}
                "subjects_completed": []
            }

        # 전략 수집 확인 (충분히 긴 메시지면 전략으로 간주)
        if len(user_message.strip()) > 5:
            strategy = user_message.strip()
            # 과목 추출
            subject = self.service._extract_subject_from_strategy(strategy)

            if subject:
                # 과목별 전략 품질 평가
                strategy_quality = self.service._judge_exam_strategy_quality(username, strategy)

                # 과목별 전략 저장
                self.service.exam_progress[username]["strategies"][subject] = {
                    "strategy": strategy,
                    "quality": strategy_quality
                }

                # 전략 품질에 따른 피드백 (보너스 배수 표시)
                bonus_multiplier = {"VERY_GOOD": 1.5, "GOOD": 1.05, "POOR": 1.0}.get(strategy_quality, 1.05)
                quality_message = {
                    "VERY_GOOD": f"정교한 전략입니다! {subject}과목 보너스 {bonus_multiplier}배",
                    "GOOD": f"좋은 전략입니다. {subject}과목 보너스 {bonus_multiplier}배",
                    "POOR": f"기본 전략입니다. {subject}과목 보너스 {bonus_multiplier}배"
                }.get(strategy_quality, f"좋은 전략입니다. {subject}과목 보너스 {bonus_multiplier}배")

                print(f"[EXAM_STRATEGY] {subject} 과목 전략 수립 완료 - 품질: {strategy_quality}, 보너스: {bonus_multiplier}배")

                # 서가윤의 응답은 LLM으로 생성 (자연스러운 반응)
                return {
                    'skip_llm': False,  # LLM 호출 필요
                    'reply': None,  # LLM이 생성
                    'narration': quality_message,
                    'user_input': strategy  # LLM에 전달할 전략 텍스트
                }
            else:
                # 과목을 찾지 못한 경우
                print(f"[EXAM_STRATEGY] 과목을 찾지 못했습니다.")
                return {
                    'skip_llm': True,  # LLM 호출 건너뛰기
                    'reply': "과목을 명시해주세요. 예시: '국어의 경우 비문학 3점짜리는 최대한 마지막에 풀어라'",
                    'narration': None,
                    'user_input': None
                }
        else:
            # 전략이 아직 없는 경우: 전략 수집 요청
            print(f"[EXAM_STRATEGY] 전략 수집 대기 중 - 메시지가 짧음 ({len(user_message.strip())}자)")
            return {
                'skip_llm': True,  # LLM 호출 건너뛰기
                'reply': "과목별 시험 전략을 알려주세요. 예시: '국어의 경우 비문학 3점짜리는 최대한 마지막에 풀어라'",
                'narration': None,
                'user_input': None
            }
