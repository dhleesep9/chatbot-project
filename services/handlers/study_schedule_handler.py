"""Study Schedule State Handler

학습 시간표 관리 state에서의 로직을 처리합니다.
- 시간표 파싱
- 총 시간 검증 (14시간 이하)
- 시간표 저장
- daily_routine 상태로 자동 전이
"""

from typing import Dict, Any, Optional
from services.handlers.base_handler import BaseStateHandler


class StudyScheduleHandler(BaseStateHandler):
    """study_schedule state handler"""

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        학습 시간표 관리 로직 처리

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과 {
                'schedule_updated': bool,  # 시간표 업데이트 여부
                'schedule': dict,  # 업데이트된 시간표
                'transition_to': str,  # 전이할 state (성공 시 "daily_routine")
                'narration': str,  # 나레이션 메시지
                'error': str  # 에러 메시지 (있는 경우)
            }
        """
        # 시간표 파싱
        parsed_schedule = self.service._parse_schedule_from_message(user_message, username)

        if parsed_schedule:
            total_hours = sum(parsed_schedule.values())
            if total_hours <= 14:
                # 시간표 설정
                self.service._set_schedule(username, parsed_schedule)
                print(f"[SCHEDULE] {username}의 시간표가 설정되었습니다: {parsed_schedule}")

                return {
                    'schedule_updated': True,
                    'schedule': parsed_schedule,
                    'transition_to': 'daily_routine',
                    'narration': "시간표 설정을 완료했습니다. 일상 루틴으로 돌아갑니다.",
                    'error': None
                }
            else:
                # 총 시간 초과
                print(f"[SCHEDULE] 총 시간이 14시간을 초과합니다: {total_hours}시간")
                return {
                    'schedule_updated': False,
                    'schedule': None,
                    'transition_to': None,
                    'narration': f"총 시간이 14시간을 초과합니다. ({total_hours}시간) 14시간 이하로 다시 설정해주세요.",
                    'error': 'TOTAL_HOURS_EXCEEDED'
                }
        else:
            # 시간표 파싱 실패 (유효한 시간표 형식이 아님)
            # LLM으로 정상 처리
            return None
