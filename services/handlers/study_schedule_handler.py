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
        print(f"[STUDY_SCHEDULE_HANDLER] 핸들러 호출됨 - 사용자: {username}, 메시지: {user_message}")
        parsed_schedule = self.service._parse_schedule_from_message(user_message, username)
        print(f"[STUDY_SCHEDULE_HANDLER] 시간표 파싱 결과: {parsed_schedule}")

        if parsed_schedule and len(parsed_schedule) > 0:
            total_hours = sum(parsed_schedule.values())
            if total_hours <= 14:
                # 시간표 설정
                self.service._set_schedule(username, parsed_schedule)
                print(f"[SCHEDULE] {username}의 시간표가 설정되었습니다: {parsed_schedule}")

                # 진로-과목 매칭 확인 및 나레이션 구성
                narration_parts = ["시간표 설정을 완료했습니다. 일상 루틴으로 돌아갑니다."]
                
                # 진로와 선택과목 가져오기
                career = self.service._get_career(username)
                selected_subjects = self.service._get_selected_subjects(username)
                
                print(f"[SCHEDULE_NARRATION] 진로: {career}, 선택과목: {selected_subjects}, 파싱된 시간표: {parsed_schedule}")
                
                if career and len(selected_subjects) > 0:
                    from services.utils.career_manager import get_career_subject_bonus_multiplier
                    
                    # 탐구1, 탐구2에 대해 진로-과목 매칭 확인
                    efficiency_messages = []
                    for i, subject_key in enumerate(["탐구1", "탐구2"]):
                        print(f"[SCHEDULE_NARRATION] 체크 중: {subject_key}, 시간표에 있음: {subject_key in parsed_schedule}, 시간: {parsed_schedule.get(subject_key, 0)}")
                        if subject_key in parsed_schedule and parsed_schedule[subject_key] > 0:
                            if i < len(selected_subjects):
                                actual_subject = selected_subjects[i]
                                print(f"[SCHEDULE_NARRATION] 실제 선택과목: {actual_subject} (탐구{i+1})")
                                multiplier = get_career_subject_bonus_multiplier(career, actual_subject)
                                print(f"[SCHEDULE_NARRATION] 배율: {multiplier} (진로: {career}, 과목: {actual_subject})")
                                if multiplier > 1.0:
                                    # 배율을 더 읽기 쉽게 표시
                                    efficiency_messages.append(f"'{career}' 진로와 '{actual_subject}' 탐구과목이 시너지를 발휘합니다! 탐구과목 효율이 {multiplier}배 상승합니다. ⭐")
                                    print(f"[SCHEDULE_NARRATION] 효율 메시지 생성: 진로 '{career}'와 과목 '{actual_subject}' 매칭 성공!")
                            else:
                                print(f"[SCHEDULE_NARRATION] 경고: {subject_key}는 시간표에 있지만 선택과목이 없습니다 (인덱스 {i})")
                        else:
                            print(f"[SCHEDULE_NARRATION] {subject_key}는 시간표에 없거나 시간이 0입니다")
                    
                    if efficiency_messages:
                        narration_parts.append("")
                        narration_parts.extend(efficiency_messages)
                        print(f"[SCHEDULE_NARRATION] 효율 메시지 추가됨: {efficiency_messages}")
                    else:
                        print(f"[SCHEDULE_NARRATION] 효율 메시지가 없습니다. 진로: {career}, 선택과목: {selected_subjects}, 시간표: {parsed_schedule}")
                
                narration = "\n".join(narration_parts)
                print(f"[SCHEDULE_NARRATION] 최종 나레이션: {narration}")

                # 진로-과목 시너지 정보를 LLM 프롬프트에 전달하기 위한 정보 구성
                career_synergy_info = None
                if efficiency_messages and career and len(selected_subjects) > 0:
                    synergy_info_parts = []
                    for i, subject_key in enumerate(["탐구1", "탐구2"]):
                        if subject_key in parsed_schedule and parsed_schedule[subject_key] > 0:
                            if i < len(selected_subjects):
                                actual_subject = selected_subjects[i]
                                multiplier = get_career_subject_bonus_multiplier(career, actual_subject)
                                if multiplier > 1.0:
                                    synergy_info_parts.append(f"'{career}' 진로와 '{actual_subject}' 탐구과목이 시너지를 발휘합니다! 탐구과목 효율이 {multiplier}배 상승합니다.")
                    if synergy_info_parts:
                        career_synergy_info = "\n".join(synergy_info_parts)

                return {
                    'schedule_updated': True,
                    'schedule': parsed_schedule,
                    'transition_to': 'daily_routine',
                    'narration': narration,
                    'career_synergy_info': career_synergy_info,  # LLM 프롬프트에 추가할 정보
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
            print(f"[STUDY_SCHEDULE_HANDLER] 시간표 파싱 실패 - LLM으로 처리합니다. 메시지: {user_message}")
            # LLM으로 정상 처리
            return None
