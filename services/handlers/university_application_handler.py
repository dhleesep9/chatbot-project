"""University Application State Handler

대학 지원 state에서의 로직을 처리합니다.
- 지원 가능 대학 리스트 표시
- 가군/나군/다군으로 원서 접수
- 합격 확률 계산 및 합격 처리
- 합격한 대학 중 선택하여 입학
- 엔딩 처리
"""

from typing import Dict, Any, Optional, List
from services.handlers.base_handler import BaseStateHandler
import re
import random


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
        
        # 군별로 지원 가능 대학 필터링 및 분류
        eligible_by_group = {
            '가군': [],
            '나군': [],
            '다군': []
        }
        
        for uni in universities:
            if avg_percentile >= uni.get('cutoff_percentile', 0):
                group = uni.get('group', '가군')  # 기본값은 가군
                if group in eligible_by_group:
                    eligible_by_group[group].append(uni)
        
        # 각 군별로 백분위 순으로 정렬 (높은 순)
        for group in eligible_by_group:
            eligible_by_group[group].sort(key=lambda x: x.get('cutoff_percentile', 0), reverse=True)
        
        # 전체 지원 가능 대학 리스트 (필터링용)
        eligible_universities = []
        for group_universities in eligible_by_group.values():
            eligible_universities.extend(group_universities)
        
        # 지원 가능 대학 리스트 메시지 생성 (군별로 구분)
        narration = f"평균 백분위: {avg_percentile:.1f}%\n\n"
        narration += "📋 [지원 가능 대학/학과]\n"
        narration += "="*50 + "\n"
        
        group_emoji = {"가군": "🔵", "나군": "🟡", "다군": "🟢"}
        
        has_eligible = False
        for group in ['가군', '나군', '다군']:
            group_universities = eligible_by_group[group]
            if group_universities:
                has_eligible = True
                emoji = group_emoji.get(group, "📋")
                narration += f"\n{emoji} {group}:\n"
                narration += "─"*50 + "\n"
                
                # 합격가능/소신/도전으로 분류
                # diff = 학생 백분위 - 커트라인
                # diff가 음수면 학생이 낮음, 양수면 학생이 높음
                confident = []  # 합격가능: 학생이 높거나 0.5% 이내 낮음
                moderate = []   # 소신: 학생이 0.5%~2% 낮음 (커트라인이 높음)
                challenge = []  # 도전: 학생이 2% 이상 낮음 (커트라인이 높음)
                
                for uni in group_universities:
                    cutoff = uni.get('cutoff_percentile', 0)
                    diff = avg_percentile - cutoff
                    
                    # 학생 백분위가 커트라인보다 높거나 같으면 무조건 합격가능
                    if diff >= 0:
                        confident.append(uni)
                    elif diff >= -0.5:  # 학생이 0.5% 이내로 낮음 → 합격가능
                        confident.append(uni)
                    elif diff >= -2.0:  # 학생이 0.5%~2% 낮음 → 소신
                        moderate.append(uni)
                    else:  # 학생이 2% 이상 낮음 → 도전
                        challenge.append(uni)
                
                # 합격가능 (🟢)
                if confident:
                    narration += "\n  🟢 합격가능:\n"
                    for uni in confident:
                        narration += f"    🟢 {uni['university']} {uni['department']} (커트라인: {uni['cutoff_percentile']}%)\n"
                
                # 소신 (🟡)
                if moderate:
                    narration += "\n  🟡 소신:\n"
                    for uni in moderate:
                        narration += f"    🟡 {uni['university']} {uni['department']} (커트라인: {uni['cutoff_percentile']}%)\n"
                
                # 도전 (🔴)
                if challenge:
                    narration += "\n  🔴 도전:\n"
                    for uni in challenge:
                        narration += f"    🔴 {uni['university']} {uni['department']} (커트라인: {uni['cutoff_percentile']}%)\n"
                
                narration += "\n"
        
        if not has_eligible:
            # 지원 가능한 대학이 없으면 3su_ending으로 전이
            print(f"[UNIVERSITY_APPLICATION] {username}의 지원 가능 대학 없음 - 3su_ending으로 전이 (평균 백분위: {avg_percentile:.1f}%)")

            # 3su_ending state의 fixed_reply 가져오기
            ending_state_info = self.service._get_state_info('3su_ending')
            ending_reply = None
            if ending_state_info:
                ending_reply = ending_state_info.get('fixed_reply')

            # fixed_reply가 없으면 기본 메시지 사용
            if not ending_reply:
                ending_reply = "선생님 .... 저 이번에도 시험 망쳤어요 ... \n저번보다는 잘 봤는데 그래도 아쉬워서  ㅠㅠㅠㅠㅠ \n한 번 더하려구요.."

            narration = f"평균 백분위: {avg_percentile:.1f}%\n\n"
            narration += "📋 [지원 가능 대학/학과]\n"
            narration += "="*50 + "\n"
            narration += "\n지원 가능한 대학이 없습니다.\n"
            narration += "\n수능 성적이 기대에 미치지 못했습니다. 하지만 서가윤은 포기하지 않기로 했습니다. 다시 한 번, 더 높은 목표를 향해..."

            return {
                'skip_llm': True,  # LLM 호출 건너뛰기
                'reply': ending_reply,
                'narration': narration,
                'transition_to': '3su_ending',
                'game_ended': True  # 게임 종료 플래그
            }
        
        narration += "\n\n지원하고 싶은 대학과 학과를 '대학명 학과명' 형식으로 입력해주세요."
        narration += "\n예: '서울대학교 컴퓨터공학과', '연세대학교 의학과'"
        
        print(f"[UNIVERSITY_APPLICATION] {username}의 지원 가능 대학 리스트 표시 - 평균 백분위: {avg_percentile:.1f}%")
        
        # 지원 가능 대학 정보 저장
        if not hasattr(self.service, 'university_application_info'):
            self.service.university_application_info = {}
        self.service.university_application_info[username] = {
            'eligible_universities': eligible_universities,
            'avg_percentile': avg_percentile,
            'exam_scores': exam_scores,
            'applications': {  # 가군/나군/다군별 지원 정보
                '가군': [],
                '나군': [],
                '다군': []
            },
            'admission_results': {  # 합격 결과
                '가군': [],
                '나군': [],
                '다군': []
            },
            'current_group': None  # 현재 선택 중인 군
        }
        
        narration += "\n\n" + "="*50
        narration += "\n🎓 대학 원서 접수 안내 🎓"
        narration += "\n" + "="*50
        narration += "\n\n이제 가군, 나군, 다군으로 나눠서 원서를 접수할 수 있습니다."
        narration += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        narration += "\n📌 원서 접수 방법:"
        narration += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        narration += "\n"
        narration += "\n💡 각 군별로 하나의 대학만 지원할 수 있습니다!"
        narration += "\n   예: '서강대학교 경영학과', '연세대학교 의학과'"
        narration += "\n"
        narration += "\n💡 여러 군을 한 번에 입력할 수 있습니다 (각 군당 하나씩):"
        narration += "\n   예: '가군 연세대학교 경제학과 나군 연세대학교 의학과 다군 서울시립대학교 경제학과'"
        narration += "\n"
        narration += "\n또는 특정 군만 보고 싶다면:"
        narration += "\n  🔵 가군 원서 넣기"
        narration += "\n  🟡 나군 원서 넣기"
        narration += "\n  🟢 다군 원서 넣기"
        narration += "\n  또는 여러 군을 동시에: '가군 나군 원서 넣기', '모든 군 원서 넣기'"
        narration += "\n"
        narration += "\n모든 원서를 넣으셨다면 '원서 접수 완료'라고 입력해주세요!"
        narration += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        return {
            'skip_llm': True,  # LLM 호출 건너뛰기
            'reply': None,
            'narration': narration,
            'transition_to': None
        }

    def _calculate_admission_probability(self, student_percentile: float, cutoff_percentile: float) -> float:
        """
        누적백분위 차이에 따른 합격 확률 계산
        
        Args:
            student_percentile: 학생의 평균 백분위
            cutoff_percentile: 대학의 커트라인 백분위
        
        Returns:
            float: 합격 확률 (0.0 ~ 1.0)
        """
        percentile_diff = abs(student_percentile - cutoff_percentile)
        
        if percentile_diff <= 0.5:
            return 0.5  # 50% 확률
        else:
            return 0.01  # 1% 확률
    
    def _check_admission(self, student_percentile: float, cutoff_percentile: float) -> bool:
        """
        합격 여부 확인 (확률 기반)
        
        Args:
            student_percentile: 학생의 평균 백분위
            cutoff_percentile: 대학의 커트라인 백분위
        
        Returns:
            bool: 합격 여부
        """
        # 학생 백분위가 커트라인보다 높거나 같으면 무조건 합격
        if student_percentile >= cutoff_percentile:
            return True
        
        # 학생 백분위가 커트라인보다 낮은 경우에만 확률 기반 계산
        probability = self._calculate_admission_probability(student_percentile, cutoff_percentile)
        return random.random() < probability

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
        avg_percentile = application_info.get('avg_percentile', 0.0)
        applications = application_info.get('applications', {'가군': [], '나군': [], '다군': []})
        admission_results = application_info.get('admission_results', {'가군': [], '나군': [], '다군': []})
        current_group = application_info.get('current_group')
        
        if not eligible_universities:
            # 정보가 없으면 다시 로드
            return self.on_enter(username, context)
        
        # 1단계: 가군/나군/다군 선택 확인 (여러 군 동시 선택 가능)
        group_keywords = {'가군': ['가군'], '나군': ['나군'], '다군': ['다군']}
        selected_groups = []
        
        # "모든 군" 또는 "전체" 키워드 확인
        if any(keyword in user_message for keyword in ['모든 군', '전체', '가군 나군 다군', '가나다']):
            selected_groups = ['가군', '나군', '다군']
        else:
            # 개별 군 선택 확인
            for group, keywords in group_keywords.items():
                if any(keyword in user_message for keyword in keywords):
                    if group not in selected_groups:
                        selected_groups.append(group)
        
        # 가군/나군/다군 선택 시 (여러 군 동시 표시)
        if selected_groups:
            # 선택된 군들을 active_groups로 저장 (모든 군을 동시에 접수할 수 있도록)
            application_info['active_groups'] = selected_groups
            # current_group은 첫 번째 선택된 군으로 설정 (하위 호환성)
            application_info['current_group'] = selected_groups[0] if selected_groups else None
            
            group_emoji = {"가군": "🔵", "나군": "🟡", "다군": "🟢"}
            
            narration = "="*50 + "\n"
            if len(selected_groups) == 1:
                emoji = group_emoji.get(selected_groups[0], "📋")
                narration += f"{emoji} {selected_groups[0]} 원서 접수 {emoji}\n"
            else:
                narration += "📋 여러 군 원서 접수\n"
            narration += "="*50 + "\n\n"
            
            # 각 선택된 군별로 현황 표시
            for selected_group in selected_groups:
                current_applications = applications.get(selected_group, [])
                group_eligible = []
                for uni in eligible_universities:
                    if uni.get('group', '가군') == selected_group:
                        group_eligible.append(uni)
                
                emoji = group_emoji.get(selected_group, "📋")
                narration += f"{emoji} {selected_group} 현황:\n"
                narration += "─"*50 + "\n"
                
                if current_applications:
                    narration += f"  📝 현재 지원한 대학 ({len(current_applications)}개):\n"
                    for app in current_applications:
                        narration += f"    ✅ {app['university']} {app['department']}\n"
                    narration += "\n"
                else:
                    narration += f"  📝 아직 지원한 대학이 없습니다.\n\n"
                
                if group_eligible:
                    narration += f"  📋 지원 가능 대학 ({len(group_eligible)}개):\n"
                    narration += "─"*50 + "\n"
                    
                    # 평균 백분위 가져오기
                    student_percentile = avg_percentile
                    
                    # 합격가능/소신/도전으로 분류
                    # diff = 학생 백분위 - 커트라인
                    # diff가 음수면 학생이 낮음, 양수면 학생이 높음
                    confident = []  # 합격가능: 학생이 높거나 0.5% 이내 낮음
                    moderate = []   # 소신: 학생이 0.5%~2% 낮음 (커트라인이 높음)
                    challenge = []  # 도전: 학생이 2% 이상 낮음 (커트라인이 높음)
                    
                    for uni in group_eligible:
                        cutoff = uni.get('cutoff_percentile', 0)
                        diff = student_percentile - cutoff
                        
                        # 학생 백분위가 커트라인보다 높거나 같으면 무조건 합격가능
                        if diff >= 0:
                            confident.append(uni)
                        elif diff >= -0.5:  # 학생이 0.5% 이내로 낮음 → 합격가능
                            confident.append(uni)
                        elif diff >= -2.0:  # 학생이 0.5%~2% 낮음 → 소신
                            moderate.append(uni)
                        else:  # 학생이 2% 이상 낮음 → 도전
                            challenge.append(uni)
                    
                    # 합격가능 (🟢)
                    if confident:
                        narration += "\n  🟢 합격가능:\n"
                        for uni in confident:
                            narration += f"    🟢 {uni['university']} {uni['department']} (커트라인: {uni['cutoff_percentile']}%)\n"
                    
                    # 소신 (🟡)
                    if moderate:
                        narration += "\n  🟡 소신:\n"
                        for uni in moderate:
                            narration += f"    🟡 {uni['university']} {uni['department']} (커트라인: {uni['cutoff_percentile']}%)\n"
                    
                    # 도전 (🔴)
                    if challenge:
                        narration += "\n  🔴 도전:\n"
                        for uni in challenge:
                            narration += f"    🔴 {uni['university']} {uni['department']} (커트라인: {uni['cutoff_percentile']}%)\n"
                    
                    narration += "\n"
                
                narration += "\n"
            
            narration += "─"*50 + "\n"
            narration += "📌 지원 방법:\n"
            narration += "─"*50 + "\n"
            narration += "⚠️ 각 군별로 하나의 대학만 지원할 수 있습니다!\n\n"
            narration += "지원하고 싶은 대학과 학과를 '대학명 학과명' 형식으로 입력해주세요.\n"
            narration += "예: '서강대학교 경영학과', '연세대학교 의학과'\n"
            narration += "대학을 선택하면 자동으로 해당 군에 추가됩니다.\n"
            narration += "이미 지원한 군에는 추가로 지원할 수 없습니다.\n\n"
            narration += "모든 원서를 넣으셨다면 '원서 접수 완료'라고 입력해주세요.\n"
            narration += "─"*50 + "\n"
            
            # 지원 정보 업데이트
            if not hasattr(self.service, 'university_application_info'):
                self.service.university_application_info = {}
            self.service.university_application_info[username] = application_info
            
            groups_text = ", ".join(selected_groups) if len(selected_groups) > 1 else selected_groups[0]
            return {
                'skip_llm': True,
                'reply': f"네, {groups_text} 원서 접수를 시작할게요.",
                'narration': narration,
                'transition_to': None
            }
        
        # 2단계: 원서 접수 완료 확인
        if '원서 접수 완료' in user_message or '접수 완료' in user_message:
            # 지원한 대학이 있는지 확인 (모든 군 확인)
            has_applications = (
                len(applications.get('가군', [])) > 0 or 
                len(applications.get('나군', [])) > 0 or 
                len(applications.get('다군', [])) > 0
            )
            
            if not has_applications:
                return {
                    'skip_llm': True,
                    'reply': "아직 지원한 대학이 없어요. 먼저 대학을 선택해주세요.",
                    'narration': "지원할 대학을 먼저 선택해주세요.\n예: '서강대학교 경영학과', '연세대학교 의학과'",
                    'transition_to': None
                }
            
            # 합격 발표 처리
            all_admissions = []
            for group in ['가군', '나군', '다군']:
                group_applications = applications.get(group, [])
                group_results = []
                
                for app in group_applications:
                    matched_uni = None
                    for uni in eligible_universities:
                        if uni['university'] == app['university'] and uni['department'] == app['department']:
                            matched_uni = uni
                            break
                    
                    if matched_uni:
                        cutoff = matched_uni.get('cutoff_percentile', 0)
                        is_admitted = self._check_admission(avg_percentile, cutoff)
                        
                        result = {
                            'university': app['university'],
                            'department': app['department'],
                            'cutoff_percentile': cutoff,
                            'student_percentile': avg_percentile,
                            'admitted': is_admitted,
                            'group': group
                        }
                        
                        group_results.append(result)
                        if is_admitted:
                            all_admissions.append(result)
                
                admission_results[group] = group_results
            
            application_info['admission_results'] = admission_results
            self.service.university_application_info[username] = application_info
            
            # 합격 결과 표시
            narration = "="*50 + "\n"
            narration += "🎓 합격 발표 결과 🎓\n"
            narration += "="*50 + "\n\n"
            
            group_emoji = {"가군": "🔵", "나군": "🟡", "다군": "🟢"}
            
            for group in ['가군', '나군', '다군']:
                group_results = admission_results.get(group, [])
                if group_results:
                    emoji = group_emoji.get(group, "📋")
                    narration += "─"*50 + "\n"
                    narration += f"{emoji} {group} 결과 {emoji}\n"
                    narration += "─"*50 + "\n"
                    for result in group_results:
                        status = "✅ 합격" if result['admitted'] else "❌ 불합격"
                        narration += f"  {status} - {result['university']} {result['department']}\n"
                        narration += f"    (학생 백분위: {result['student_percentile']:.1f}%, 커트라인: {result['cutoff_percentile']}%)\n"
                    narration += "\n"
            
            if all_admissions:
                narration += "="*50 + "\n"
                narration += "🎉 축하합니다! 합격한 대학이 있습니다! 🎉\n"
                narration += "="*50 + "\n\n"
                narration += "합격한 대학 중 하나를 선택하여 입학하세요.\n\n"
                narration += "─"*50 + "\n"
                narration += "📋 합격한 대학 목록:\n"
                narration += "─"*50 + "\n"
                for i, adm in enumerate(all_admissions, 1):
                    group_emoji_symbol = group_emoji.get(adm['group'], "📋")
                    narration += f"  {i}. {adm['university']} {adm['department']} ({group_emoji_symbol} {adm['group']})\n"
                narration += "\n"
                narration += "─"*50 + "\n"
                narration += "입학하고 싶은 대학과 학과를 입력해주세요.\n"
                narration += "─"*50 + "\n"
            else:
                narration += "\n안타깝게도 모든 대학에 불합격했습니다.\n"
                narration += "게임이 종료됩니다."
                
                return {
                    'skip_llm': True,
                    'reply': "모든 대학에 불합격했어요... 정말 안타깝네요.",
                    'narration': narration,
                    'transition_to': None,
                    'game_ended': True
                }
            
            return {
                'skip_llm': True,
                'reply': "합격 발표 결과를 확인했어요...",
                'narration': narration,
                'transition_to': None
            }
        
        # 3단계: 여러 군 동시 입력 처리 ("가군 연세대학교 경제학과 나군 연세대학교 의학과" 형식)
        # 먼저 여러 군이 동시에 입력되었는지 확인
        # 패턴: (가군|나군|다군) + 공백 + 대학명 + 공백 + 학과명
        multi_group_pattern = r'(가군|나군|다군)\s+([가-힣]+(?:대학교|대학))\s+([가-힣]+(?:과|학과|전공|계열|학부))'
        multi_group_matches = re.findall(multi_group_pattern, user_message)
        
        # 공백이 없는 경우도 처리 (예: "가군연세대학교경제학과")
        if not multi_group_matches or len(multi_group_matches) <= 1:
            multi_group_pattern_no_space = r'(가군|나군|다군)([가-힣]+(?:대학교|대학))([가-힣]+(?:과|학과|전공|계열|학부))'
            multi_group_matches = re.findall(multi_group_pattern_no_space, user_message)
        
        if multi_group_matches and len(multi_group_matches) > 1:
            # 여러 군 동시 입력 처리
            group_emoji = {"가군": "🔵", "나군": "🟡", "다군": "🟢"}
            processed_applications = []
            failed_applications = []
            
            for group, university, department in multi_group_matches:
                # 대학 매칭
                matched_uni = None
                for uni in eligible_universities:
                    if uni.get('group', '가군') == group:
                        if university in uni['university'] or uni['university'] in university:
                            if department in uni['department'] or uni['department'] in department:
                                matched_uni = uni
                                break
                
                if not matched_uni:
                    failed_applications.append({
                        'group': group,
                        'university': university,
                        'department': department,
                        'reason': '지원 가능한 대학 목록에 없음'
                    })
                    continue
                
                # 각 군당 하나만 허용 확인
                group_applications = applications.get(group, [])
                is_duplicate = False
                
                # 이미 해당 군에 지원한 대학이 있는지 확인 (각 군당 하나만 허용)
                if len(group_applications) > 0:
                    existing_uni = group_applications[0]
                    failed_applications.append({
                        'group': group,
                        'university': university,
                        'department': department,
                        'reason': f"이미 {group}에 '{existing_uni['university']} {existing_uni['department']}'를 지원했습니다. 각 군당 하나의 대학만 지원할 수 있습니다."
                    })
                    is_duplicate = True
                else:
                    # 정확히 동일한 대학/학과는 중복 확인
                    for app in group_applications:
                        if app['university'] == matched_uni['university'] and app['department'] == matched_uni['department']:
                            failed_applications.append({
                                'group': group,
                                'university': university,
                                'department': department,
                                'reason': '이미 지원함'
                            })
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    # 원서 접수
                    new_application = {
                        'university': matched_uni['university'],
                        'department': matched_uni['department'],
                        'cutoff_percentile': matched_uni.get('cutoff_percentile', 0)
                    }
                    group_applications.append(new_application)
                    applications[group] = group_applications
                    
                    processed_applications.append({
                        'group': group,
                        'university': matched_uni['university'],
                        'department': matched_uni['department']
                    })
                    
                    # active_groups에 해당 군이 없으면 추가
                    active_groups = application_info.get('active_groups', [])
                    if group not in active_groups:
                        active_groups.append(group)
                        application_info['active_groups'] = active_groups
            
            # 결과 업데이트
            application_info['applications'] = applications
            if processed_applications:
                application_info['current_group'] = processed_applications[0]['group']
            self.service.university_application_info[username] = application_info
            
            # 결과 메시지 생성
            narration = "="*50 + "\n"
            narration += "📋 여러 군 원서 접수 완료\n"
            narration += "="*50 + "\n\n"
            
            if processed_applications:
                narration += "✅ 성공적으로 지원한 대학:\n"
                narration += "─"*50 + "\n"
                for app in processed_applications:
                    emoji = group_emoji.get(app['group'], "📋")
                    narration += f"{emoji} {app['group']}: {app['university']} {app['department']}\n"
                narration += "\n"
            
            if failed_applications:
                narration += "⚠️ 지원 실패한 대학:\n"
                narration += "─"*50 + "\n"
                for app in failed_applications:
                    emoji = group_emoji.get(app['group'], "📋")
                    reason_msg = "존재하지 않는 대학/학과입니다" if app['reason'] == '지원 가능한 대학 목록에 없음' else "이미 지원한 대학입니다"
                    narration += f"{emoji} {app['group']}: {app['university']} {app['department']} - {reason_msg}\n"
                narration += "\n"
                narration += "💡 올바른 입력 형식:\n"
                narration += "   - '대학명 학과명' 형식으로 정확히 입력해주세요\n"
                narration += "   - 여러 군 동시 입력: '가군 대학명 학과명 나군 대학명 학과명'\n"
                narration += "   - 대학명과 학과명은 정확하게 입력해야 합니다\n"
                narration += "   예: '서강대학교 경영학과', '연세대학교 의학과'\n\n"
            
            # 전체 지원 현황 표시
            narration += "─"*50 + "\n"
            narration += "📝 전체 지원 현황:\n"
            narration += "─"*50 + "\n"
            for group in ['가군', '나군', '다군']:
                group_apps = applications.get(group, [])
                if group_apps:
                    group_emoji_symbol = group_emoji.get(group, "📋")
                    narration += f"\n{group_emoji_symbol} {group} ({len(group_apps)}개):\n"
                    for i, app in enumerate(group_apps, 1):
                        narration += f"  {i}. {app['university']} {app['department']}\n"
            
            narration += "\n"
            narration += "─"*50 + "\n"
            narration += "📌 다음 단계:\n"
            narration += "─"*50 + "\n"
            narration += "추가로 지원할 대학이 있으면 대학명과 학과명을 입력해주세요.\n"
            narration += "모든 원서를 넣으셨다면 '원서 접수 완료'라고 입력해주세요.\n"
            narration += "─"*50 + "\n"
            
            success_count = len(processed_applications)
            if success_count > 0:
                reply = f"네, {success_count}개 대학에 지원했어요."
            else:
                reply = "지원에 실패한 대학이 있어요."
            
            return {
                'skip_llm': True,
                'reply': reply,
                'narration': narration,
                'transition_to': None
            }
        
        # 대학명과 학과명 패턴 추출 (입학 선택과 원서 접수 모두에서 사용)
        university_pattern = r'([가-힣]+(?:대학교|대학))'
        department_pattern = r'([가-힣]+(?:과|학과|전공|계열|학부))'
        
        university_match = re.search(university_pattern, user_message)
        department_match = re.search(department_pattern, user_message)
        
        # 4단계: 합격한 대학 중 입학 선택 (가장 먼저 확인 - 원서 접수보다 우선)
        # 합격 결과가 있고, 대학명과 학과명이 입력되면 입학 처리
        if university_match and department_match and any(admission_results.values()):
            applied_university = university_match.group(1).strip()
            applied_department = department_match.group(1).strip()
            
            # 합격한 대학인지 확인
            matched_admission = None
            for group_results in admission_results.values():
                for result in group_results:
                    if result.get('admitted', False):
                        if applied_university in result['university'] or result['university'] in applied_university:
                            if applied_department in result['department'] or result['department'] in applied_department:
                                matched_admission = result
                                break
                if matched_admission:
                    break
            
            if matched_admission:
                # 합격한 대학에 입학
                final_university = matched_admission['university']
                final_department = matched_admission['department']
                
                print(f"[UNIVERSITY_APPLICATION] {username}의 입학 선택: {final_university} {final_department}")
                
                # 성적 정보
                exam_scores = application_info.get('exam_scores', {})
                score_text = " ".join([f"{subject} {data['grade']}등급" for subject, data in exam_scores.items()])
                
                # 서가윤의 호감도 가져오기
                affection = self.service._get_affection(username)

                # 서강대학교 입학 확인
                is_sogang = '서강대학교' in final_university or '서강대' in final_university

                # 엔딩 state info 가져오기
                if is_sogang:
                    if affection >= 80:
                        # 캠퍼스 커플 엔딩
                        ending_state = 'campus_couple'
                        ending_image = '/static/images/chatbot/end/서강대2.png'
                    else:
                        # 서강대 입학 엔딩
                        ending_state = 'sogang'
                        ending_image = '/static/images/chatbot/end/서강대.png'
                else:
                    # 일반 대학 입학 엔딩 (fixed_reply 없음)
                    ending_state = None
                    ending_image = None

                # 엔딩 state의 fixed_reply 가져오기
                seogayoon_reply = None
                if ending_state:
                    ending_state_info = self.service._get_state_info(ending_state)
                    if ending_state_info:
                        seogayoon_reply = ending_state_info.get('fixed_reply')

                # fixed_reply가 없으면 기본 메시지 사용
                if not seogayoon_reply:
                    seogayoon_reply = f"멘토님... 정말 고마워요. 제가 {final_university} {final_department}에 합격하고 입학할 수 있게 된 건 전부 멘토님 덕분이에요. 멘토님이 옆에 있어줘서 힘들 때도 포기하지 않고 여기까지 올 수 있었어요. 정말 감사드려요...! 앞으로도 멘토님과 함께라면 자신있게 새로운 시작을 할 수 있을 것 같아요!"

                print(f"[UNIVERSITY_APPLICATION] {username}의 입학 엔딩 - fixed_reply 사용: '{seogayoon_reply[:50]}...'")

                # 엔딩 나레이션 생성
                narration = f"📋 입학 선택 완료\n\n"
                narration += f"서가윤이 {final_university} {final_department}에 입학할 것을 선택했습니다.\n\n"
                narration += f"🎓 입학 내역\n"
                narration += f"대학: {final_university}\n"
                narration += f"학과: {final_department}\n"
                narration += f"이름: 서가윤\n\n"
                narration += f"수능 성적: {score_text}\n\n"
                narration += f"🎉 축하합니다! 서가윤이 {final_university} {final_department}에 입학합니다!\n\n"
                narration += f"수고하셨습니다. 게임을 완료하셨습니다."

                if is_sogang:
                    # 서강대학교 입학 - 호감도에 따라 엔딩 결정
                    if affection >= 80:
                        # 캠퍼스 커플 엔딩
                        print(f"[UNIVERSITY_APPLICATION] {username}의 서강대 캠퍼스 커플 엔딩 (호감도: {affection})")
                        return {
                            'skip_llm': True,
                            'reply': seogayoon_reply,
                            'narration': narration,
                            'transition_to': 'campus_couple',
                            'game_ended': True
                        }
                    else:
                        # 서강대 입학 엔딩
                        print(f"[UNIVERSITY_APPLICATION] {username}의 서강대 입학 엔딩 (호감도: {affection})")
                        return {
                            'skip_llm': True,
                            'reply': seogayoon_reply,
                            'narration': narration,
                            'transition_to': 'sogang',
                            'game_ended': True
                        }
                else:
                    # 일반 대학 입학 엔딩
                    return {
                        'skip_llm': True,
                        'reply': seogayoon_reply,
                        'narration': narration,
                        'transition_to': None,
                        'game_ended': True
                    }
            else:
                # 합격하지 않은 대학 선택 시
                return {
                    'skip_llm': True,
                    'reply': f"'{applied_university} {applied_department}'는 합격한 대학이 아니에요. 합격한 대학 중에서 선택해주세요.",
                    'narration': f"⚠️ '{applied_university} {applied_department}'는 합격하지 않은 대학입니다.\n\n합격한 대학 중에서 선택해주세요.",
                    'transition_to': None
                }
        
        # 단일 대학 입력 처리 (입학 선택이 아닐 때만 - 원서 접수 단계)
        if university_match and department_match:
            applied_university = university_match.group(1).strip()
            applied_department = department_match.group(1).strip()
            
            # 대학 매칭 (모든 군에서 검색 - 대학이 속한 군을 자동 판별)
            matched_uni = None
            for uni in eligible_universities:
                if applied_university in uni['university'] or uni['university'] in applied_university:
                    if applied_department in uni['department'] or uni['department'] in applied_department:
                        matched_uni = uni
                        break
            
            if not matched_uni:
                # 비슷한 대학명이나 학과명이 있는지 확인
                similar_universities = []
                similar_departments = []
                
                for uni in eligible_universities:
                    # 대학명이 부분적으로 일치하는지 확인
                    if applied_university in uni['university'] or uni['university'] in applied_university:
                        similar_universities.append(uni)
                    # 학과명이 부분적으로 일치하는지 확인
                    if applied_department in uni['department'] or uni['department'] in applied_department:
                        similar_departments.append(uni)
                
                narration = "="*50 + "\n"
                narration += "⚠️ 입력 오류\n"
                narration += "="*50 + "\n\n"
                narration += f"'{applied_university} {applied_department}'는 지원 가능한 대학 목록에 존재하지 않습니다.\n\n"
                
                if similar_universities or similar_departments:
                    narration += "💡 비슷한 대학/학과를 찾았습니다:\n"
                    narration += "─"*50 + "\n"
                    if similar_universities:
                        narration += "비슷한 대학명:\n"
                        for uni in similar_universities[:5]:  # 최대 5개만
                            group_emoji = {"가군": "🔵", "나군": "🟡", "다군": "🟢"}
                            emoji = group_emoji.get(uni.get('group', '가군'), "📋")
                            narration += f"  {emoji} {uni['university']} {uni['department']} ({uni.get('group', '가군')})\n"
                    if similar_departments:
                        narration += "\n비슷한 학과명:\n"
                        for uni in similar_departments[:5]:  # 최대 5개만
                            group_emoji = {"가군": "🔵", "나군": "🟡", "다군": "🟢"}
                            emoji = group_emoji.get(uni.get('group', '가군'), "📋")
                            narration += f"  {emoji} {uni['university']} {uni['department']} ({uni.get('group', '가군')})\n"
                    narration += "\n"
                
                narration += "─"*50 + "\n"
                narration += "📌 올바른 입력 형식:\n"
                narration += "─"*50 + "\n"
                narration += "1. '대학명 학과명' 형식으로 정확히 입력해주세요\n"
                narration += "   예: '서강대학교 경영학과', '연세대학교 의학과'\n\n"
                narration += "2. 여러 군 동시 입력 가능:\n"
                narration += "   예: '가군 서강대학교 경영학과 나군 연세대학교 의학과'\n\n"
                narration += "3. 대학명과 학과명은 정확히 입력해야 합니다\n"
                narration += "   - '서강대' ❌ → '서강대학교' ✅\n"
                narration += "   - '경영' ❌ → '경영학과' ✅\n\n"
                
                return {
                    'skip_llm': True,
                    'reply': f"'{applied_university} {applied_department}'는 지원 가능한 대학 목록에 없어요. 올바른 형식으로 다시 입력해주세요.",
                    'narration': narration,
                    'transition_to': None
                }
            
            # 대학이 속한 군 자동 판별
            matched_group = matched_uni.get('group', '가군')
            
            # 각 군당 하나만 허용 확인
            group_applications = applications.get(matched_group, [])
            if len(group_applications) > 0:
                existing_uni = group_applications[0]
                return {
                    'skip_llm': True,
                    'reply': f"이미 {matched_group}에 '{existing_uni['university']} {existing_uni['department']}'를 지원했어요. 각 군당 하나의 대학만 지원할 수 있어요.",
                    'narration': f"⚠️ {matched_group}에는 이미 '{existing_uni['university']} {existing_uni['department']}'를 지원했습니다.\n\n각 군별로 하나의 대학만 지원할 수 있습니다.\n다른 군을 선택하시거나 기존 지원을 변경하려면 먼저 취소해야 합니다.",
                    'transition_to': None
                }
            
            # 원서 접수 (해당 군에 자동 추가)
            new_application = {
                'university': matched_uni['university'],
                'department': matched_uni['department'],
                'cutoff_percentile': matched_uni.get('cutoff_percentile', 0)
            }
            group_applications.append(new_application)
            applications[matched_group] = group_applications
            
            # active_groups에 해당 군이 없으면 추가
            active_groups = application_info.get('active_groups', [])
            if matched_group not in active_groups:
                active_groups.append(matched_group)
                application_info['active_groups'] = active_groups
            
            application_info['applications'] = applications
            application_info['current_group'] = matched_group  # 하위 호환성
            self.service.university_application_info[username] = application_info
            
            # 군별 이모지
            group_emoji = {"가군": "🔵", "나군": "🟡", "다군": "🟢"}
            emoji = group_emoji.get(matched_group, "📋")
            
            narration = "="*50 + "\n"
            narration += f"{emoji} {matched_group} 원서 접수 완료 {emoji}\n"
            narration += "="*50 + "\n\n"
            narration += f"✅ {matched_uni['university']} {matched_uni['department']} ({matched_group})에 지원했습니다!\n\n"
            
            # 모든 군의 지원 현황 표시
            narration += "─"*50 + "\n"
            narration += "📝 전체 지원 현황:\n"
            narration += "─"*50 + "\n"
            for group in ['가군', '나군', '다군']:
                group_apps = applications.get(group, [])
                if group_apps:
                    group_emoji_symbol = group_emoji.get(group, "📋")
                    narration += f"\n{group_emoji_symbol} {group} ({len(group_apps)}개):\n"
                    for i, app in enumerate(group_apps, 1):
                        narration += f"  {i}. {app['university']} {app['department']}\n"
            
            narration += "\n"
            narration += "─"*50 + "\n"
            narration += "📌 다음 단계:\n"
            narration += "─"*50 + "\n"
            narration += "추가로 지원할 대학이 있으면 대학명과 학과명을 입력해주세요.\n"
            narration += "모든 원서를 넣으셨다면 '원서 접수 완료'라고 입력해주세요.\n"
            narration += "─"*50 + "\n"
            
            return {
                'skip_llm': True,
                'reply': f"네, {matched_uni['university']} {matched_uni['department']}에 지원했어요.",
                'narration': narration,
                'transition_to': None
            }
        
        # 일반 대화 처리 (대학 지원 관련 키워드 없으면 LLM 처리)
        support_keywords = ["지원", "합격", "입학", "대학", "학과", "원서"]
        has_support_keyword = any(keyword in user_message for keyword in support_keywords)
        
        if not has_support_keyword:
            return None  # LLM이 일반 대화 처리
        
        # 대학명이나 학과명이 부분적으로만 추출된 경우 안내
        if (university_match and not department_match) or (not university_match and department_match):
            missing_info = "학과명" if not department_match else "대학명"
            
            narration = "="*50 + "\n"
            narration += "⚠️ 입력 형식 오류\n"
            narration += "="*50 + "\n\n"
            narration += f"입력하신 내용에서 {missing_info}을 찾을 수 없습니다.\n\n"
            
            if university_match:
                narration += f"✅ 찾은 대학명: {university_match.group(1)}\n"
            if department_match:
                narration += f"✅ 찾은 학과명: {department_match.group(1)}\n"
            
            narration += "\n─"*50 + "\n"
            narration += "📌 올바른 입력 형식:\n"
            narration += "─"*50 + "\n"
            narration += "1. '대학명 학과명' 형식으로 정확히 입력해주세요\n"
            narration += "   ✅ 올바른 예시:\n"
            narration += "      - '서강대학교 경영학과'\n"
            narration += "      - '연세대학교 의학과'\n"
            narration += "      - '서울대학교 컴퓨터공학과'\n\n"
            narration += "2. 여러 군 동시 입력도 가능합니다:\n"
            narration += "   ✅ 예: '가군 서강대학교 경영학과 나군 연세대학교 의학과'\n\n"
            narration += "3. 주의사항:\n"
            narration += "   - 대학명은 '대학교' 또는 '대학'으로 끝나야 합니다\n"
            narration += "   - 학과명은 '과', '학과', '전공', '계열', '학부'로 끝나야 합니다\n"
            narration += "   - 대학명과 학과명 사이에 공백이 필요합니다\n"
            
            return {
                'skip_llm': True,
                'reply': f"입력 형식이 올바르지 않아요. '{missing_info}'을 포함해서 '대학명 학과명' 형식으로 다시 입력해주세요.",
                'narration': narration,
                'transition_to': None
            }
        
        return None  # 기본값: LLM이 처리

