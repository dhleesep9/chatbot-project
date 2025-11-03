"""Subject Selection State Handler

탐구과목 선택 state에서의 로직을 처리합니다.
- 사용자 메시지에서 탐구과목 파싱
- 2개 선택 여부 확인
- 선택된 과목 저장
- daily_routine 상태로 자동 전이
"""

from typing import Dict, Any, Optional
from services.handlers.base_handler import BaseStateHandler


# 선택 가능한 탐구과목 목록
SUBJECT_OPTIONS = [
    "사회문화", "정치와법", "경제", "세계지리", "한국지리",
    "생활과윤리", "윤리와사상", "세계사", "동아시아사",
    "물리학1", "화학1", "지구과학1", "생명과학1",
    "물리학2", "화학2", "지구과학2", "생명과학2"
]


class SubjectSelectionHandler(BaseStateHandler):
    """selection state handler"""

    def _parse_subjects_from_message(self, user_message: str) -> list:
        """
        사용자 메시지에서 탐구과목 추출

        Args:
            user_message: 사용자 입력 메시지

        Returns:
            추출된 과목 리스트 (최대 2개)
        """
        found_subjects = []

        # 메시지를 정규화 (공백 제거)
        normalized_message = user_message.replace(" ", "")

        # 각 과목이 메시지에 포함되어 있는지 확인
        for subject in SUBJECT_OPTIONS:
            # 공백 없는 과목명으로 체크
            normalized_subject = subject.replace(" ", "")

            if normalized_subject in normalized_message:
                found_subjects.append(subject)

                # 2개 찾으면 중단
                if len(found_subjects) >= 2:
                    break

        return found_subjects[:2]  # 최대 2개만 반환

    def _get_subject_list_text(self) -> str:
        """
        선택 가능한 과목 목록을 텍스트로 반환

        Returns:
            과목 목록 문자열
        """
        return ", ".join(SUBJECT_OPTIONS)

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        탐구과목 선택 로직 처리

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과 {
                'subjects_selected': bool,  # 과목 선택 완료 여부
                'subjects': list,  # 선택된 과목 목록
                'transition_to': str,  # 전이할 state (성공 시 "daily_routine")
                'narration': str,  # 나레이션 메시지
                'error': str  # 에러 메시지 (있는 경우)
            }
        """
        # 메시지에서 과목 추출
        found_subjects = self._parse_subjects_from_message(user_message)

        # 2개 선택되었는지 확인
        if len(found_subjects) >= 2:
            # 선택된 과목을 영구 저장소에 저장
            self.service._set_selected_subjects(username, found_subjects)
            print(f"[SELECTION] {username}의 탐구과목 선택 완료: {found_subjects}")

            # 진로와의 시너지 확인
            career = self.service._get_career(username)
            synergy_messages = []
            
            if career:
                from services.utils.career_manager import get_career_subject_bonus_multiplier
                
                for subject in found_subjects:
                    multiplier = get_career_subject_bonus_multiplier(career, subject)
                    if multiplier > 1.0:
                        synergy_messages.append(f"'{career}' 진로와 '{subject}' 탐구과목이 시너지를 발휘합니다! 탐구과목 배율 {multiplier}배")
                        print(f"[CAREER_SYNERGY] {username}의 '{career}' 진로와 '{subject}' 과목 시너지 발생! (배율: {multiplier}배)")

            # 나레이션 구성
            narration_parts = [f"탐구과목 선택이 완료되었습니다! ({', '.join(found_subjects)})"]
            
            if synergy_messages:
                narration_parts.extend(synergy_messages)
            
            narration = "\n".join(narration_parts)

            return {
                'subjects_selected': True,
                'subjects': found_subjects,
                'transition_to': 'daily_routine',
                'narration': narration,
                'error': None
            }
        else:
            # 과목 선택 실패 (2개 미만)
            if len(found_subjects) == 1:
                print(f"[SELECTION] 1개만 선택됨: {found_subjects[0]}")
                return {
                    'subjects_selected': False,
                    'subjects': found_subjects,
                    'transition_to': None,
                    'narration': f"{found_subjects[0]}을(를) 선택하셨습니다. 1개를 더 선택해주세요.",
                    'error': 'INSUFFICIENT_SUBJECTS'
                }
            else:
                # 과목을 하나도 찾지 못한 경우 - LLM으로 정상 처리
                print(f"[SELECTION] 유효한 과목을 찾지 못함")
                return None
