"""University Application State Handler

대학 지원 state에서의 로직을 처리합니다.
- 지원 가능 대학 리스트 표시
- 대학 지원 처리
- 엔딩 처리
"""

from typing import Dict, Any, Optional
from services.handlers.base_handler import BaseStateHandler
import re


class UniversityApplicationHandler(BaseStateHandler):
    """university_application state handler"""

    def on_enter(self, username: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        university_application state 진입 시 지원 가능 대학 리스트 표시

        Args:
            username: 사용자 이름
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과
        """
        # 성적 정보 가져오기
        score_storage = getattr(self.service, 'csat_exam_scores', {})
        exam_scores = score_storage.get(username, {}).get("scores")
        
        if not exam_scores:
            # 성적이 없는 경우 다시 계산
            exam_scores = self.service._calculate_mock_exam_scores(username)
            if not hasattr(self.service, 'csat_exam_scores'):
                self.service.csat_exam_scores = {}
            self.service.csat_exam_scores[username] = {"scores": exam_scores}
        
        # 평균 백분위 계산
        percentiles = [exam_scores[subject]['percentile'] for subject in ["국어", "수학", "영어", "탐구1", "탐구2"] if subject in exam_scores]
        avg_percentile = sum(percentiles) / len(percentiles) if percentiles else 0.0
        
        # 대학 정보 로드
        universities = self.service._get_university_admissions_info()
        
        # 지원 가능 대학만 필터링
        eligible_universities = []
        for uni in universities:
            if avg_percentile >= uni.get('cutoff_percentile', 0):
                eligible_universities.append(uni)
        
        # 백분위 순으로 정렬 (높은 순)
        eligible_universities.sort(key=lambda x: x.get('cutoff_percentile', 0), reverse=True)
        
        # 지원 가능 대학 리스트 메시지 생성
        narration = f"평균 백분위: {avg_percentile:.1f}%\n\n"
        narration += "📋 [지원 가능 대학/학과]\n"
        
        if eligible_universities:
            for uni in eligible_universities:
                narration += f"\n✅ {uni['university']} {uni['department']} (커트라인: {uni['cutoff_percentile']}%)"
        else:
            narration += "\n없음"
        
        narration += "\n\n지원하고 싶은 대학과 학과를 '대학명 학과명' 형식으로 입력해주세요."
        narration += "\n예: '서울대학교 컴퓨터공학과', '연세대학교 의학과'"
        
        print(f"[UNIVERSITY_APPLICATION] {username}의 지원 가능 대학 리스트 표시 - 평균 백분위: {avg_percentile:.1f}%")
        
        # 지원 가능 대학 정보 저장
        if not hasattr(self.service, 'university_application_info'):
            self.service.university_application_info = {}
        self.service.university_application_info[username] = {
            'eligible_universities': eligible_universities,
            'avg_percentile': avg_percentile,
            'exam_scores': exam_scores
        }
        
        return {
            'skip_llm': True,  # LLM 호출 건너뛰기
            'reply': None,
            'narration': narration,
            'transition_to': None
        }

    def handle(self, username: str, user_message: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        대학 지원 로직 처리

        Args:
            username: 사용자 이름
            user_message: 사용자 입력 메시지
            context: 실행 컨텍스트

        Returns:
            Dict: 처리 결과 (엔딩 포함)
        """
        # 저장된 지원 가능 대학 정보 가져오기
        application_info = getattr(self.service, 'university_application_info', {}).get(username, {})
        eligible_universities = application_info.get('eligible_universities', [])
        
        if not eligible_universities:
            # 정보가 없으면 다시 로드
            return self.on_enter(username, context)
        
        # 대학 지원 관련 키워드 확인 (대학명/학과명이 있는지 확인)
        # 대학명과 학과명 추출 (예: "서울대학교 컴퓨터공학과")
        # 더 유연한 패턴: "대학교" 또는 "대학"으로 끝나는 학교명, "과" 또는 "학과"로 끝나는 학과명
        # 전체 메시지에서 대학명과 학과명을 추출하되, 중간에 다른 텍스트가 있어도 추출 가능하도록
        university_pattern = r'([가-힣]+(?:대학교|대학))'
        department_pattern = r'([가-힣]+(?:과|학과|전공|계열|학부))'
        
        # 메시지에서 대학명과 학과명 추출 시도
        university_match = re.search(university_pattern, user_message)
        department_match = re.search(department_pattern, user_message)
        
        print(f"[UNIVERSITY_APPLICATION] 추출 시도 - university_match: {university_match}, department_match: {department_match}")
        print(f"[UNIVERSITY_APPLICATION] user_message: '{user_message}'")
        
        # 대학 지원 관련 키워드가 없으면 일반 대화로 처리 (None 반환하여 LLM 호출)
        support_keywords = ["지원", "합격", "입학", "대학", "학과"]
        has_support_keyword = any(keyword in user_message for keyword in support_keywords)
        
        # 대학명/학과명도 없고 대학 지원 관련 키워드도 없으면 일반 대화로 처리
        if not university_match and not department_match and not has_support_keyword:
            print(f"[UNIVERSITY_APPLICATION] 일반 대화로 처리 (대학 지원 관련 키워드 없음)")
            return None  # None 반환 시 LLM이 일반 대화 처리
        
        # 대학명과 학과명이 모두 추출되었을 때만 합격 처리
        if university_match and department_match:
            applied_university = university_match.group(1).strip()
            applied_department = department_match.group(1).strip()
            
            print(f"[UNIVERSITY_APPLICATION] 추출된 대학: '{applied_university}', 학과: '{applied_department}'")
            
            # 전체 대학 목록 가져오기 (지원 가능/불가 모두 포함)
            all_universities = self.service._get_university_admissions_info()
            
            # 지원 가능 대학 목록에서 확인
            matched_uni = None
            for uni in eligible_universities:
                if applied_university in uni['university'] or uni['university'] in applied_university:
                    if applied_department in uni['department'] or uni['department'] in applied_department:
                        matched_uni = uni
                        break
            
            # 전체 대학 목록에서도 확인 (지원 가능 목록에 없는 경우)
            if not matched_uni:
                for uni in all_universities:
                    if applied_university in uni['university'] or uni['university'] in applied_university:
                        if applied_department in uni['department'] or uni['department'] in applied_department:
                            matched_uni = uni
                            break
            
            # 목록에 없는 대학/학과인 경우 게임 종료하지 않고 다시 입력 요청
            if not matched_uni:
                print(f"[UNIVERSITY_APPLICATION] 유효하지 않은 대학/학과 입력: {applied_university} {applied_department}")
                return {
                    'skip_llm': True,
                    'reply': f"멘토님, '{applied_university} {applied_department}'는 지원 가능한 대학 목록에 없는 것 같아요. 다시 확인하고 입력해주시겠어요?",
                    'narration': f"⚠️ '{applied_university} {applied_department}'는 지원 가능한 대학 목록에 없는 학과입니다.\n\n지원하고 싶은 대학과 학과를 '대학명 학과명' 형식으로 다시 입력해주세요.",
                    'transition_to': None,  # 현재 상태 유지
                    'game_ended': False  # 게임 종료하지 않음
                }
            
            # 사용자가 입력한 대학과 학과 사용 (목록에 존재하는 경우)
            final_university = matched_uni['university']
            final_department = matched_uni['department']
            
            print(f"[UNIVERSITY_APPLICATION] {username}의 대학 지원: {final_university} {final_department}")
            
            # 성적 정보
            exam_scores = application_info.get('exam_scores', {})
            score_text = " ".join([f"{subject} {data['grade']}등급" for subject, data in exam_scores.items()])
            
            # 서가윤의 호감도 가져오기
            affection = self.service._get_affection(username)
            
            # LLM을 통해 서가윤의 캐릭터에 맞는 합격 엔딩 메시지 생성
            # 서가윤이 직접 합격 결과를 확인하고 인지하는 상황으로 설정
            ending_prompt = f"""멘토님, 합격 발표를 확인했어요... 잠깐, 이게... 이게 정말...?

합격 확인 결과:
- 대학: {final_university}
- 학과: {final_department}
- 이름: 서가윤

서가윤이 지금 합격 발표 페이지를 보고 있고, 자신의 이름과 함께 "{final_university} {final_department}" 합격 내역을 확인하고 있습니다.

서가윤의 성격:
- 불안하고 감정 기복이 심하지만, 진심 어린 지지와 격려를 받으면 다시 용기를 얻는 성격
- 멘토에 대한 신뢰는 아직 완전하지 않아 방어적이지만, 동시에 진심으로 의지하고 싶어함
- 원래 목표는 서강대학교였지만, 지금 {final_university} {final_department}에 합격한 사실을 확인하고 있습니다

현재 호감도: {affection}/100

서가윤의 반응을 자연스럽게 표현해주세요:
1. 합격 발표를 확인하는 순간의 반응 (놀람, 믿기지 않음)
2. 자신의 이름과 "{final_university} {final_department}" 합격 내역을 직접 확인하며 반복하는 모습
3. "나... 정말 {final_university} {final_department}에 합격한 거예요?" 같은 식으로 자신의 합격을 확인하고 인지하는 과정
4. 합격 사실을 깨달은 후 기쁨과 안도감 표현
5. 멘토에게 감사하는 마음
6. 호감도에 따라 감정 표현의 차이 (낮으면 조금 어색하거나, 높으면 더 진심 어린 감사)

서가윤의 말투로, 3-4문장으로 자연스럽게 응답해주세요. 반드시 "{final_university} {final_department}"를 직접 언급하며 자신이 합격했다는 것을 확인하고 인지하는 모습을 보여주세요."""
            
            try:
                # LLM 호출하여 엔딩 메시지 생성 (ChatbotService의 client 사용)
                if not self.service.client:
                    raise ValueError("OpenAI Client가 초기화되지 않았습니다.")
                
                response = self.service.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": self.service._build_system_prompt()},
                        {"role": "user", "content": ending_prompt}
                    ],
                    temperature=0.9,
                    max_tokens=200
                )
                
                seogayoon_reply = response.choices[0].message.content.strip()
                print(f"[UNIVERSITY_APPLICATION] LLM이 생성한 서가윤의 엔딩 메시지: {seogayoon_reply}")
                
            except Exception as e:
                print(f"[UNIVERSITY_APPLICATION] LLM 호출 실패, 기본 메시지 사용: {e}")
                # LLM 호출 실패 시 기본 메시지 (합격을 인지하는 과정)
                seogayoon_reply = f"멘토님... 잠깐만요... 이게... 제 이름이... 서가윤... {final_university} {final_department}... 나... 정말 {final_university} {final_department}에 합격한 거예요? 정말 믿기지가 않아요...! 멘토 덕분에 여기까지 올 수 있었어요. 정말 고마워요...!"
            
            # 엔딩 나레이션 생성 (합격 발표 확인 장면)
            narration = f"📋 합격 발표 확인\n\n"
            narration += f"서가윤이 합격 발표 페이지를 확인하고 있습니다...\n\n"
            narration += f"🎓 합격 내역\n"
            narration += f"대학: {final_university}\n"
            narration += f"학과: {final_department}\n"
            narration += f"이름: 서가윤\n\n"
            narration += f"수능 성적: {score_text}\n\n"
            narration += f"🎉 축하합니다! 서가윤이 {final_university} {final_department}에 합격했습니다!\n\n"
            narration += f"수고하셨습니다. 게임을 완료하셨습니다."
            
            # reply가 None이 아닌지 확인하고, 없으면 기본 메시지 사용
            if not seogayoon_reply:
                seogayoon_reply = f"멘토님... 잠깐만요... 이게... 제 이름이... 서가윤... {final_university} {final_department}... 나... 정말 {final_university} {final_department}에 합격한 거예요? 정말 믿기지가 않아요...! 멘토 덕분에 여기까지 올 수 있었어요. 정말 고마워요...!"
            
            print(f"[UNIVERSITY_APPLICATION] 최종 reply: '{seogayoon_reply}'")
            print(f"[UNIVERSITY_APPLICATION] 최종 narration 길이: {len(narration)}")
            
            return {
                'skip_llm': True,  # LLM은 이미 호출했으므로 skip
                'reply': seogayoon_reply,  # 서가윤의 합격 반응 (반드시 포함)
                'narration': narration,
                'transition_to': None,
                'game_ended': True  # 엔딩 플래그
            }
        else:
            # 대학 지원 관련 키워드는 있지만 형식이 맞지 않는 경우 - 안내 메시지
            print(f"[UNIVERSITY_APPLICATION] 대학명 또는 학과명 추출 실패 (대학 지원 관련 키워드는 있음)")
            print(f"[UNIVERSITY_APPLICATION] 추출된 대학명: {university_match.group(1) if university_match else '없음'}")
            print(f"[UNIVERSITY_APPLICATION] 추출된 학과명: {department_match.group(1) if department_match else '없음'}")
            
            # 대학명이나 학과명 중 하나라도 추출되지 않은 경우 안내 메시지
            missing_info = []
            if not university_match:
                missing_info.append("대학명")
            if not department_match:
                missing_info.append("학과명")
            
            guidance_msg = f"지원하고 싶은 {', '.join(missing_info)}을 '대학명 학과명' 형식으로 명확하게 입력해주세요.\n"
            guidance_msg += f"예: '서울대학교 컴퓨터공학과', '연세대학교 의학과', '서강대학교 전자공학과'"
            
            # 안내 메시지는 narration으로 표시하고, 서가윤의 응답은 LLM으로 생성
            return {
                'skip_llm': False,  # LLM 호출하여 서가윤이 안내 메시지에 대한 반응 생성
                'reply': None,  # LLM이 생성
                'narration': guidance_msg,
                'transition_to': None
            }

