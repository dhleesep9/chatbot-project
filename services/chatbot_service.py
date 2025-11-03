"""
🎯 챗봇 서비스 - 구현 파일 (ver 2025-11-02)

이 파일은 챗봇의 핵심 AI 로직을 담당합니다.
아래 아키텍처를 참고하여 직접 설계하고 구현하세요.

📐 시스템 아키텍처:

┌─────────────────────────────────────────────────────────┐
│ 1. 초기화 단계 (ChatbotService.__init__)                  │
├─────────────────────────────────────────────────────────┤
│  - OpenAI Client 생성                                    │
│  - ChromaDB 연결 (벡터 데이터베이스)                       │
│  - LangChain Memory 초기화 (대화 기록 관리)               │
│  - Config 파일 로드                                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 2. RAG 파이프라인 (generate_response 내부)               │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  사용자 질문 "학식 추천해줘"                              │
│       ↓                                                  │
│  [_create_embedding()]                                   │
│       ↓                                                  │
│  질문 벡터: [0.12, -0.34, ..., 0.78]  (3072차원)        │
│       ↓                                                  │
│  [_search_similar()]  ← ChromaDB 검색                    │
│       ↓                                                  │
│  검색 결과: "학식은 곤자가가 맛있어" (유사도: 0.87)        │
│       ↓                                                  │
│  [_build_prompt()]                                       │
│       ↓                                                  │
│  최종 프롬프트 = 시스템 설정 + RAG 컨텍스트 + 질문        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 3. LLM 응답 생성                                         │
├─────────────────────────────────────────────────────────┤
│  OpenAI GPT-4 API 호출                                   │
│       ↓                                                  │
│  "학식은 곤자가에서 먹는 게 제일 좋아! 돈까스가 인기야"    │
│       ↓                                                  │
│  [선택: 이미지 검색]                                      │
│       ↓                                                  │
│  응답 반환: {reply: "...", image: "..."}                 │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 메모리 저장 (LangChain Memory)                        │
├─────────────────────────────────────────────────────────┤
│  대화 기록에 질문-응답 저장                               │
│  다음 대화에서 컨텍스트로 활용                            │
└─────────────────────────────────────────────────────────┘


💡 핵심 구현 과제:

1. **Embedding 생성**
   - OpenAI API를 사용하여 텍스트를 벡터로 변환
   - 모델: text-embedding-3-large (3072차원)

2. **RAG 검색 알고리즘** ⭐ 가장 중요!
   - ChromaDB에서 유사 벡터 검색
   - 유사도 계산: similarity = 1 / (1 + distance)
   - threshold 이상인 문서만 선택

3. **LLM 프롬프트 설계**
   - 시스템 프롬프트 (캐릭터 설정)
   - RAG 컨텍스트 통합
   - 대화 기록 포함

4. **대화 메모리 관리**
   - LangChain의 ConversationSummaryBufferMemory 사용
   - 대화가 길어지면 자동으로 요약


📚 참고 문서:
- ARCHITECTURE.md: 시스템 아키텍처 상세 설명
- IMPLEMENTATION_GUIDE.md: 단계별 구현 가이드
- README.md: 프로젝트 개요


⚠️ 주의사항:
- 이 파일의 구조는 가이드일 뿐입니다
- 자유롭게 재설계하고 확장할 수 있습니다
- 단, generate_response() 함수 시그니처는 유지해야 합니다
  (app.py에서 호출하기 때문)
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import json

# 환경변수 로드
load_dotenv()

# 프로젝트 루트 경로
BASE_DIR = Path(__file__).resolve().parent.parent

# import openai  # linter: 실제 사용은 동적 import, 명시적 작성 (실제 사용은 __init__ 내부)
# import chromadb  # linter: 실제 사용은 동적 import, 명시적 작성 (실제 사용은 _init_chromadb 내부)
# from langchain.memory import ConversationSummaryBufferMemory  # linter: 실제 사용은 동적 import, 명시적 작성 (실제 사용은 __init__ 내부)


class ChatbotService:
    """
    챗봇 서비스 클래스
    
    이 클래스는 챗봇의 모든 AI 로직을 캡슐화합니다.
    
    주요 책임:
    1. OpenAI API 관리
    2. ChromaDB 벡터 검색
    3. LangChain 메모리 관리
    4. 응답 생성 파이프라인
    
    직접 구현해야 할 메서드:
    - __init__: 모든 구성 요소 초기화
    - _load_config: 설정 파일 로드
    - _init_chromadb: 벡터 데이터베이스 초기화
    - _create_embedding: 텍스트 → 벡터 변환
    - _search_similar: RAG 검색 수행 (핵심!)
    - _build_prompt: 프롬프트 구성
    - generate_response: 최종 응답 생성 (모든 로직 통합)
    """
    
    def __init__(self):
        print("[ChatbotService] 초기화 중... ")

        # 1. Config 로드
        self.config = self._load_config()
        print("[ChatbotService] config loaded. name:", self.config.get('name', ''))

        # 1.5. States 로드 (별도 JSON 파일들)
        self.states = self._load_states()
        print(f"[ChatbotService] states loaded: {list(self.states.keys())}")

        # 1.6. Debug Commands 로드 (별도 JSON 파일)
        self.debug_commands = self._load_debug_commands()
        print(f"[ChatbotService] debug commands loaded: {len(self.debug_commands.get('commands', []))} commands")

        # 1.6.5. University Admissions 정보 로드
        self.university_admissions = self._load_university_admissions()
        print(f"[ChatbotService] university admissions loaded: {len(self.university_admissions)} universities")

        # 1.7. Trigger Registry 초기화 (자동으로 모든 트리거 로드)
        from services.triggers.trigger_registry import TriggerRegistry
        self.trigger_registry = TriggerRegistry()
        print(f"[ChatbotService] trigger registry loaded: {self.trigger_registry.list_triggers()}")

        # 1.8. Handler Registry 초기화 및 handler 등록
        from services.handlers.handler_registry import HandlerRegistry
        from services.handlers.exam_strategy_handler import ExamStrategyHandler
        from services.handlers.study_schedule_handler import StudyScheduleHandler
        from services.handlers.mock_exam_handler import MockExamHandler
        from services.handlers.official_exam_handler import JuneExamHandler, SeptemberExamHandler, CSATExamHandler
        from services.handlers.subject_selection_handler import SubjectSelectionHandler
        from services.handlers.exam_feedback_handler import JuneExamFeedbackHandler, SeptemberExamFeedbackHandler
        from services.handlers.mock_exam_feedback_handler import MockExamFeedbackHandler, OfficialMockExamFeedbackHandler
        from services.handlers.university_application_handler import UniversityApplicationHandler

        self.handler_registry = HandlerRegistry()
        self.handler_registry.register('exam_strategy', ExamStrategyHandler(self))
        self.handler_registry.register('study_schedule', StudyScheduleHandler(self))
        self.handler_registry.register('mock_exam', MockExamHandler(self))
        self.handler_registry.register('6exam', JuneExamHandler(self))
        self.handler_registry.register('9exam', SeptemberExamHandler(self))
        self.handler_registry.register('11exam', CSATExamHandler(self))
        self.handler_registry.register('selection', SubjectSelectionHandler(self))
        self.handler_registry.register('6exam_feedback', JuneExamFeedbackHandler(self))
        self.handler_registry.register('9exam_feedback', SeptemberExamFeedbackHandler(self))
        self.handler_registry.register('mock_exam_feedback', MockExamFeedbackHandler(self))
        self.handler_registry.register('official_mock_exam_feedback', OfficialMockExamFeedbackHandler(self))
        self.handler_registry.register('university_application', UniversityApplicationHandler(self))
        print(f"[ChatbotService] handler registry loaded: exam_strategy, study_schedule, mock_exam, 6exam, 9exam, 11exam, selection, 6exam_feedback, 9exam_feedback, mock_exam_feedback, official_mock_exam_feedback, university_application")

        # 2. OpenAI Client 초기화
        try:
            import openai
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 필요합니다.")
            self.client = OpenAI(api_key=api_key)
            print("[ChatbotService] OpenAI Client 초기화 완료")
        except Exception as e:
            print(f"[ERROR][ChatbotService] OpenAI Client 초기화 실패: {e}")
            self.client = None

        # 3. ChromaDB 초기화
        try:
            self.collection = self._init_chromadb()
            print("[ChatbotService] ChromaDB 컬렉션 연결 성공")
        except Exception as e:
            print(f"[ERROR][ChatbotService] ChromaDB 초기화 실패: {e}")
            self.collection = None

        # 4. LangChain Memory (optional, 실제 사용시 확장)
        try:
            from langchain.memory import ConversationSummaryBufferMemory
            self.memory = None  # 추후 필요시 ConversationSummaryBufferMemory로 초기화
            print("[ChatbotService] LangChain Memory 준비 (미사용)")
        except Exception as e:
            print(f"[WARN][ChatbotService] LangChain Memory 사용 불가: {e}")
            self.memory = None

        # 5. 호감도 저장 (username을 키로 하는 딕셔너리)
        self.affections = {}  # {username: affection_value}
        print("[ChatbotService] 호감도 시스템 초기화 완료")

        # 5.5. 능력치 저장 (username을 키로 하는 딕셔너리)
        # 능력치: 국어, 수학, 영어, 탐구1, 탐구2 (0~100)
        self.abilities = {}  # {username: {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0}}
        print("[ChatbotService] 능력치 시스템 초기화 완료")

        # 6. 게임 상태 저장 (username을 키로 하는 딕셔너리)
        # 상태 종류: "ice_break", "mentoring"
        self.game_states = {}  # {username: game_state}
        print("[ChatbotService] 게임 상태 시스템 초기화 완료")

        # 7. 선택과목 목록 및 저장
        self.subject_options = [
            "사회문화", "정치와법", "경제", "세계지리", "한국지리",
            "생활과윤리", "윤리와사상", "세계사", "동아시아사",
            "물리학1", "화학1", "지구과학1", "생명과학1",
            "물리학2", "화학2", "지구과학2", "생명과학2"
        ]
        self.selected_subjects = {}  # {username: [subject1, subject2, ...]} (최대 2개)
        print("[ChatbotService] 선택과목 시스템 초기화 완료")

        # 8. 시간표 저장
        self.schedules = {}  # {username: {"국어": 4, "수학": 4, "영어": 4, "탐구1": 1, "탐구2": 1}}
        print("[ChatbotService] 시간표 시스템 초기화 완료")

        # 9. 체력 저장 (기본값 30)
        self.staminas = {}  # {username: stamina_value}
        print("[ChatbotService] 체력 시스템 초기화 완료")

        # 9.5. 멘탈 저장 (기본값 40)
        self.mentals = {}  # {username: mental_value}
        print("[ChatbotService] 멘탈 시스템 초기화 완료")
        
        # 9.9. 대학 지원 정보 저장
        self.university_application_info = {}  # {username: {"eligible_universities": [...], "avg_percentile": float, "exam_scores": {...}}}
        print("[ChatbotService] 대학 지원 정보 저장 시스템 초기화 완료")

        # 9.6. 사설모의고사 취약점 정보 저장 (피드백용)
        self.mock_exam_weakness = {}  # {username: {"subject": str, "message": str}}
        print("[ChatbotService] 사설모의고사 취약점 저장 시스템 초기화 완료")
        
        # 9.6.5. 사설모의고사 응시 주차 추적 (한 주에 한 번만 보도록)
        self.mock_exam_last_week = {}  # {username: last_week_number}
        print("[ChatbotService] 사설모의고사 응시 주차 추적 시스템 초기화 완료")
        
        # 9.7. 정규모의고사 취약점 정보 저장 (피드백용)
        self.official_mock_exam_weakness = {}  # {username: {"subject": str, "message": str}}
        print("[ChatbotService] 정규모의고사 취약점 저장 시스템 초기화 완료")
        
        # 9.8. 6월 모의고사 문제점 추적 시스템
        # {username: {"scores": {...}, "subjects": {"국어": {"problem": str, "solved": bool}, ...}, "current_subject": str, "completed_count": int}}
        self.june_exam_problems = {}
        print("[ChatbotService] 6월 모의고사 문제점 추적 시스템 초기화 완료")
        
        # 9.9. 9월 모의고사 문제점 추적 시스템
        # {username: {"scores": {...}, "subjects": {"국어": {"problem": str, "solved": bool}, ...}, "current_subject": str, "completed_count": int}}
        self.september_exam_problems = {}
        print("[ChatbotService] 9월 모의고사 문제점 추적 시스템 초기화 완료")
        
        # 9.10. 수능 성적 저장 시스템
        self.csat_exam_scores = {}
        print("[ChatbotService] 수능 성적 저장 시스템 초기화 완료")
        
        # 9. 대화 횟수 추적 (daily_routine 상태에서만)
        self.conversation_counts = {}  # {username: count}
        print("[ChatbotService] 대화 횟수 시스템 초기화 완료")

        # 10. 현재 주(week) 추적
        self.current_weeks = {}  # {username: week_number}
        print("[ChatbotService] 주(week) 추적 시스템 초기화 완료")

        # 11. 게임 날짜 저장
        self.game_dates = {}  # {username: "2023-11-17"}
        print("[ChatbotService] 게임 날짜 시스템 초기화 완료")

        # 12. 진로 저장
        self.careers = {}  # {username: career_name}
        print("[ChatbotService] 진로 시스템 초기화 완료")

        # 13. 시험 진행 정보 저장 (전략 + 학생 시점 진행)
        # {username: {"strategy": str, "strategy_quality": str, "current_subject": str, "subject_order": list, "subjects_completed": list}}
        self.exam_progress = {}
        print("[ChatbotService] 시험 진행 시스템 초기화 완료")

        print("[ChatbotService] 초기화 완료")
    
    
    def _load_config(self):
        """
        설정 파일 로드
        """
        config_path = BASE_DIR / "config/chatbot_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"챗봇 설정 파일이 존재하지 않습니다: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        return config

    def _load_states(self):
        """
        별도 JSON 파일들에서 state 정보 로드
        """
        states = {}
        state_machine = self.config.get("state_machine", {})
        states_directory = state_machine.get("states_directory", "config/states")
        available_states = state_machine.get("available_states", [])

        for state_name in available_states:
            state_file = BASE_DIR / f"{states_directory}/{state_name}.json"
            try:
                with open(state_file, encoding="utf-8") as f:
                    state_info = json.load(f)
                    states[state_name] = state_info
                    print(f"[STATE_LOADER] {state_name}.json 로드 성공")
            except FileNotFoundError:
                print(f"[WARN] State 파일 없음: {state_file}")
            except Exception as e:
                print(f"[ERROR] State 파일 로드 실패 ({state_name}): {e}")

        return states

    def _load_debug_commands(self):
        """
        디버그 명령어 설정 파일 로드
        """
        debug_commands_file = BASE_DIR / "config/debug_commands.json"
        try:
            with open(debug_commands_file, encoding="utf-8") as f:
                debug_commands = json.load(f)
                print(f"[DEBUG_LOADER] debug_commands.json 로드 성공")
                return debug_commands
        except FileNotFoundError:
            print(f"[WARN] Debug commands 파일 없음: {debug_commands_file}")
            return {"enabled": False, "commands": []}
        except Exception as e:
            print(f"[ERROR] Debug commands 파일 로드 실패: {e}")
            return {"enabled": False, "commands": []}

    def _load_university_admissions(self):
        """
        대학 입학 정보 로드
        """
        university_file = BASE_DIR / "config/university_admissions.json"
        try:
            with open(university_file, encoding="utf-8") as f:
                universities = json.load(f)
                print(f"[UNIVERSITY_LOADER] university_admissions.json 로드 성공")
                return universities
        except FileNotFoundError:
            print(f"[WARN] University admissions 파일 없음: {university_file}")
            return []
        except Exception as e:
            print(f"[ERROR] University admissions 파일 로드 실패: {e}")
            return []
    
    def _get_university_admissions_info(self):
        """
        대학 입학 정보 반환
        """
        return self.university_admissions
    
    def _get_state_info(self, state_name: str) -> dict:
        """
        State 정보 반환
        """
        return self.states.get(state_name, {})
    
    def _handle_debug_command(self, user_message: str, username: str, current_state: str, current_affection: int) -> dict:
        """
        디버그 명령어 처리 (config/debug_commands.json 기반)

        Returns:
            dict: 응답 딕셔너리 또는 None (매칭되는 명령어가 없을 경우)
        """
        if not self.debug_commands.get("enabled", False):
            return None

        user_message_clean = user_message.strip()

        for command in self.debug_commands.get("commands", []):
            if not command.get("enabled", True):
                continue

            if user_message_clean == command.get("trigger"):
                # required_state 확인
                required_state = command.get("required_state")
                if required_state and current_state != required_state:
                    error_message = command.get("error_message", "이 명령어는 특정 상태에서만 사용할 수 있습니다.")
                    return {
                        'reply': error_message,
                        'image': None,
                        'affection': current_affection,
                        'game_state': current_state,
                        'selected_subjects': self._get_selected_subjects(username),
                        'narration': None,
                        'abilities': self._get_abilities(username),
                        'schedule': self._get_schedule(username),
                        'current_date': self._get_game_date(username),
                        'stamina': self._get_stamina(username),
                        'mental': self._get_mental(username)
                    }

                # action 실행
                action = command.get("action")
                parameters = command.get("parameters", {})

                if action == "skip_weeks":
                    return self._debug_skip_weeks(username, current_affection, current_state, parameters, command)
                elif action == "increase_affection":
                    return self._debug_increase_affection(username, current_affection, current_state, parameters, command)
                elif action == "set_max_abilities":
                    return self._debug_set_max_abilities(username, current_affection, current_state, parameters, command)

        return None

    def _debug_skip_weeks(self, username: str, current_affection: int, current_state: str, parameters: dict, command: dict) -> dict:
        """1주스킵, 4주스킵 명령어 처리"""
        weeks = parameters.get("weeks", 1)
        current_schedule = self._get_schedule(username)

        # weeks만큼 반복
        narration_parts = []
        for week_num in range(weeks):
            if current_schedule:
                self._apply_schedule_to_abilities(username)

            self._increment_week(username)
            current_week = self._get_current_week(username)
            self._reset_conversation_count(username)

            current_date = self._get_game_date(username)
            new_date = self._add_days_to_date(current_date, 7)
            self._set_game_date(username, new_date)
            
            # 1주 경과 시 체력 -1
            current_stamina = self._get_stamina(username)
            new_stamina = max(0, current_stamina - 1)
            self._set_stamina(username, new_stamina)
            print(f"[STAMINA] {username}의 체력이 {current_stamina}에서 {new_stamina}로 변경되었습니다. (1주 경과로 -1)")

            # 시험 체크
            exam_month = self._check_exam_in_period(current_date, new_date)
            if exam_month:
                exam_scores = self._calculate_exam_scores(username, exam_month)
                exam_name = "수능" if exam_month.endswith("-11") else f"{exam_month[-2:]}월 모의고사"
                narration_parts.append(f"{exam_name} 성적이 발표되었습니다.")

        # 마지막 주 번호와 날짜
        final_week = self._get_current_week(username)
        final_date = self._get_game_date(username)

        # 성공 메시지
        success_message = command.get("success_message", "").replace("{week}", str(final_week))

        # 시험 결과 추가
        if narration_parts:
            success_message += "\n\n" + "\n".join(narration_parts)

        # 호감도에 따른 공부하러 가는 메시지 생성
        study_message = self._get_study_message_by_affection(current_affection)

        return {
            'reply': study_message,
            'image': None,
            'affection': current_affection,
            'game_state': current_state,
            'selected_subjects': self._get_selected_subjects(username),
            'narration': success_message,
            'abilities': self._get_abilities(username),
            'schedule': self._get_schedule(username),
            'current_date': final_date,
            'stamina': self._get_stamina(username),
            'mental': self._get_mental(username)
        }

    def _debug_increase_affection(self, username: str, current_affection: int, current_state: str, parameters: dict, command: dict) -> dict:
        """호감도5올리기 명령어 처리"""
        amount = parameters.get("amount", 5)
        new_affection = min(100, current_affection + amount)
        self._set_affection(username, new_affection)
        print(f"[DEBUG] 호감도 증가: {current_affection} -> {new_affection}")

        success_message = command.get("success_message", "")
        success_message = success_message.replace("{old_affection}", str(current_affection))
        success_message = success_message.replace("{new_affection}", str(new_affection))

        return {
            'reply': success_message,
            'image': None,
            'affection': new_affection,
            'game_state': current_state,
            'selected_subjects': self._get_selected_subjects(username),
            'narration': None,
            'abilities': self._get_abilities(username),
            'schedule': self._get_schedule(username),
            'current_date': self._get_game_date(username),
            'stamina': self._get_stamina(username),
            'mental': self._get_mental(username)
        }

    def _debug_set_max_abilities(self, username: str, current_affection: int, current_state: str, parameters: dict, command: dict) -> dict:
        """만점 명령어 처리"""
        value = parameters.get("value", 2500)
        max_abilities = {
            "국어": value,
            "수학": value,
            "영어": value,
            "탐구1": value,
            "탐구2": value
        }
        self._set_abilities(username, max_abilities)
        print(f"[DEBUG] 모든 능력치를 {value}으로 설정했습니다.")

        success_message = command.get("success_message", "")

        return {
            'reply': success_message,
            'image': None,
            'affection': current_affection,
            'game_state': current_state,
            'selected_subjects': self._get_selected_subjects(username),
            'narration': None,
            'abilities': max_abilities,
            'schedule': self._get_schedule(username),
            'current_date': self._get_game_date(username),
            'stamina': self._get_stamina(username),
            'mental': self._get_mental(username)
        }


    def _init_chromadb(self):
        """
        ChromaDB 초기화 및 rag_collection 반환
        """
        import chromadb
        db_path = BASE_DIR / "static/data/chatbot/chardb_embedding"
        if not db_path.exists():
            raise FileNotFoundError(f"ChromaDB 데이터 경로가 존재하지 않습니다: {db_path}")
        client = chromadb.PersistentClient(path=str(db_path))
        collection = client.get_collection(name="rag_collection")
        return collection
    
    
    def _create_embedding(self, text: str) -> list:
        """
        텍스트를 임베딩 벡터로 변환
        """
        if not self.client:
            raise RuntimeError("OpenAI Client가 초기화되지 않았습니다.")
        try:
            response = self.client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[ERROR] 임베딩 생성 실패: {e}")
            raise
    
    
    def _search_similar(self, query: str, threshold: float = 0.45, top_k: int = 5):
        """
        RAG 검색: 유사한 문서 찾기
        """
        if not self.collection:
            print("[WARN][RAG] ChromaDB 컬렉션이 연결되지 않았음.")
            return (None, None, None)

        if not self.client:
            print("[WARN][RAG] OpenAI Client가 연결되지 않았음.")
            return (None, None, None)

        try:
            # 1. 쿼리 임베딩 생성
            query_embedding = self._create_embedding(query)
            
            # 2. 벡터 DB 검색
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "distances", "metadatas"]
                )
            except Exception as e:
                print(f"[WARN][RAG] 벡터 DB 검색 실패: {e}")
                return (None, None, None)
            
            docs = results.get("documents", [[]])[0] if results.get("documents") else []
            dists = results.get("distances", [[]])[0] if results.get("distances") else []
            metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []

            # 3 & 4. 유사도 계산/최상위 문서 결정
            best_doc, best_sim, best_meta = None, -1, None
            for doc, dist, meta in zip(docs, dists, metas):
                similarity = 1 / (1 + dist)
                if similarity >= threshold and similarity > best_sim:
                    best_doc, best_sim, best_meta = doc, similarity, meta
            if best_doc is not None:
                return (best_doc, best_sim, best_meta)
            return (None, None, None)
        except Exception as e:
            print(f"[WARN][RAG] 임베딩 생성 실패: {e}")
            return (None, None, None)
    
    
    def _get_affection(self, username: str) -> int:
        """
        사용자의 현재 호감도 반환 (없으면 기본값 5)
        """
        return self.affections.get(username, 5)
    
    def _get_study_message_by_affection(self, affection: int) -> str:
        """
        호감도에 따라 공부하러 가는 메시지를 반환
        """
        if affection < 10:
            return "저... 이제 공부하러 가볼게요..."
        elif affection < 30:
            return "선생님, 이제 공부하러 가볼게요."
        elif affection < 50:
            return "선생님, 저는 이제 공부하러 가볼게요!"
        elif affection < 70:
            return "선생님, 저 이제 공부하러 가볼게요. 오늘도 열심히 할게요!"
        else:
            return "선생님, 저 이제 공부하러 가볼게요! 선생님 덕분에 공부가 즐거워요!"
    
    def _set_affection(self, username: str, affection: int):
        """
        사용자의 호감도 설정 (0~100 범위로 제한)
        """
        self.affections[username] = max(0, min(100, affection))
        self._save_user_data(username)  # 변경사항 저장

    def _save_user_data(self, username: str):
        """사용자 게임 데이터를 JSON 파일로 저장"""
        from services.utils.user_data_manager import save_user_data
        save_user_data(
            username,
            lambda: self._get_affection(username),
            lambda: self._get_game_state(username),
            lambda: self._get_abilities(username),
            lambda: self._get_selected_subjects(username),
            lambda: self._get_schedule(username),
            lambda: self._get_conversation_count(username),
            lambda: self._get_current_week(username),
            lambda: self._get_game_date(username),
            lambda: self._get_stamina(username),
            lambda: self._get_mental(username),
            lambda: self.mock_exam_last_week.get(username, -1),
            lambda: self._get_career(username)
        )

    def _load_user_data(self, username: str):
        """사용자 게임 데이터를 JSON 파일에서 로드"""
        from services.utils.user_data_manager import load_user_data
        load_user_data(
            username,
            lambda v: self._set_affection(username, v),
            lambda v: self._set_game_state(username, v),
            lambda v: self._set_abilities(username, v),
            lambda v: self._set_selected_subjects(username, v),
            lambda v: self._set_schedule(username, v),
            lambda v: self.conversation_counts.__setitem__(username, v),
            lambda v: self.current_weeks.__setitem__(username, v),
            lambda v: self._set_game_date(username, v),
            lambda v: self._set_stamina(username, v),
            lambda v: self._set_mental(username, v),
            lambda v: self.mock_exam_last_week.__setitem__(username, v),
            lambda v: self._set_career(username, v) if v else None
        )

    def _get_abilities(self, username: str) -> dict:
        """
        사용자의 현재 능력치 반환 (없으면 기본값)
        """
        default_abilities = {
            "국어": 0,
            "수학": 0,
            "영어": 0,
            "탐구1": 0,
            "탐구2": 0
        }
        return self.abilities.get(username, default_abilities)
    
    def _set_abilities(self, username: str, abilities: dict):
        """
        사용자의 능력치 설정 (0~2500 범위로 제한)
        """
        # 각 능력치를 0~2500 범위로 제한
        normalized = {}
        for key, value in abilities.items():
            normalized[key] = max(0, min(2500, value))
        self.abilities[username] = normalized
        self._save_user_data(username)  # 변경사항 저장
    
    def _get_stamina(self, username: str) -> int:
        """
        사용자의 현재 체력 반환 (없으면 기본값 30)
        """
        return self.staminas.get(username, 30)
    
    def _set_stamina(self, username: str, stamina: int):
        """
        사용자의 체력 설정 (0~100 범위)
        """
        self.staminas[username] = max(0, min(100, stamina))  # 체력은 0~100
        self._save_user_data(username)  # 변경사항 저장
    
    def _get_mental(self, username: str) -> int:
        """
        사용자의 현재 멘탈 반환 (없으면 기본값 40)
        """
        return self.mentals.get(username, 40)
    
    def _set_mental(self, username: str, mental: int):
        """
        사용자의 멘탈 설정 (0~100 범위)
        """
        self.mentals[username] = max(0, min(100, mental))  # 멘탈은 0~100
        self._save_user_data(username)  # 변경사항 저장
    
    def _calculate_stamina_efficiency(self, stamina: int) -> float:
        """체력에 따른 능력치 증가 효율 계산"""
        from services.utils.efficiency_calculator import calculate_stamina_efficiency
        return calculate_stamina_efficiency(stamina)
    
    def _calculate_mental_efficiency(self, mental: int) -> float:
        """멘탈에 따른 능력치 증가 효율 계산"""
        from services.utils.efficiency_calculator import calculate_mental_efficiency
        return calculate_mental_efficiency(mental)
    
    def _calculate_combined_efficiency(self, stamina: int, mental: int) -> float:
        """체력과 멘탈의 곱연산으로 최종 효율 계산"""
        from services.utils.efficiency_calculator import calculate_combined_efficiency
        return calculate_combined_efficiency(stamina, mental)
    
    def _get_game_state(self, username: str) -> str:
        """
        사용자의 현재 게임 상태 반환 (없으면 "start")
        """
        return self.game_states.get(username, "start")
    
    def _set_game_state(self, username: str, state: str):
        """
        사용자의 게임 상태 설정
        """
        # 로드된 states에서 유효한 상태 목록 가져오기
        valid_states = list(self.states.keys())

        if state in valid_states:
            self.game_states[username] = state
            state_info = self._get_state_info(state)
            state_name = state_info.get("name", state)
            print(f"[GAME_STATE] {username}의 상태가 {state}({state_name})로 변경되었습니다.")
            self._save_user_data(username)  # 변경사항 저장
        else:
            print(f"[WARN] 잘못된 게임 상태: {state}. 유효한 상태: {valid_states}")
    
    def _process_handler_result(self, handler_result: dict, narration: str) -> tuple:
        """
        핸들러 결과 처리 헬퍼 함수
        narration 병합 및 state 전이 처리
        
        Returns:
            (updated_narration, transition_to, state_changed)
        """
        if not handler_result:
            return narration, None, False
        
        # narration 병합
        if handler_result.get('narration'):
            if not narration:
                narration = handler_result['narration']
            else:
                narration = f"{narration}\n\n{handler_result['narration']}"
        
        # state 전이 처리
        transition_to = None
        state_changed = False
        if handler_result.get('transition_to'):
            transition_to = handler_result['transition_to']
            state_changed = True
            # 대상 상태의 narration도 추가
            target_state_info = self._get_state_info(transition_to)
            if target_state_info and target_state_info.get('narration'):
                if not narration:
                    narration = target_state_info['narration']
                else:
                    narration = f"{narration}\n\n{target_state_info['narration']}"
        
        return narration, transition_to, state_changed
    
    def _evaluate_transition_condition(self, username: str, transition: dict, affection_increased: int, user_message: str = "") -> bool:
        """
        전이 조건 평가 (트리거 레지스트리 기반)

        Args:
            username: 사용자 이름
            transition: 전이 정보 딕셔너리
            affection_increased: 이번 턴 호감도 증가량
            user_message: 사용자 입력 메시지

        Returns:
            조건 만족 여부
        """
        trigger_type = transition.get("trigger_type")
        print(f"[TRIGGER_EVAL_START] Starting evaluation for trigger_type: '{trigger_type}'")
        
        # 트리거가 등록되어 있는지 확인
        available_triggers = self.trigger_registry.list_triggers()
        print(f"[TRIGGER_EVAL] Available triggers: {available_triggers}")
        has_trigger = self.trigger_registry.has_trigger(trigger_type)
        print(f"[TRIGGER_EVAL] Has trigger '{trigger_type}': {has_trigger}")
        
        if not has_trigger:
            print(f"[WARN] Trigger type '{trigger_type}' not found in registry. Available triggers: {available_triggers}")
            return False

        # 트리거 실행 컨텍스트 구성
        context = {
            'username': username,
            'user_message': user_message,
            'affection_increased': affection_increased,
            'current_state': self._get_game_state(username),
            'june_exam_problems': getattr(self, 'june_exam_problems', {}),
            'september_exam_problems': getattr(self, 'september_exam_problems', {}),
            'service': self  # 트리거가 서비스 메서드에 접근할 수 있도록
        }
        
        print(f"[TRIGGER_EVAL] Evaluating trigger '{trigger_type}' with user_message: '{user_message}'")
        print(f"[TRIGGER_EVAL] Context: username={username}, current_state={context['current_state']}")

        # 트리거 레지스트리를 통해 동적으로 트리거 실행
        try:
            result = self.trigger_registry.evaluate_trigger(trigger_type, transition, context)
            print(f"[TRIGGER_EVAL] Trigger '{trigger_type}' result: {result}")
        except Exception as e:
            print(f"[ERROR] Trigger evaluation exception: {e}")
            import traceback
            traceback.print_exc()
            result = False
        
        return result

    def _check_state_transition(self, username: str, new_affection: int, affection_increased: int = 0, user_message: str = "") -> tuple:
        """
        상태 전환 조건 체크 및 전환 (state machine 기반)

        Args:
            username: 사용자 이름
            new_affection: 새로운 호감도
            affection_increased: 이번 턴 호감도 증가량
            user_message: 사용자 입력 메시지

        Returns:
            (전환 발생 여부, 전환 나레이션)
        """
        current_state = self._get_game_state(username)
        print(f"[STATE_CHECK] Current state: {current_state}, user_message: '{user_message}'")

        # Global transitions 체크 (현재 state에 무관하게 항상 확인)
        current_mental = self.mentals.get(username, 40)
        current_stamina = self.stamina.get(username, 30)
        print(f"[GLOBAL_TRANSITION_CHECK] Mental: {current_mental}, Stamina: {current_stamina}, Affection: {new_affection}")

        # 체력이 0 이하일 경우 -> broken_body 엔딩
        if current_stamina <= 0:
            print(f"[GLOBAL_TRANSITION] Stamina <= 0, transitioning to broken_body")
            self._set_game_state(username, "broken_body")
            next_state_info = self._get_state_info("broken_body")
            state_narration = next_state_info.get("narration")
            return (True, state_narration)

        # 멘탈이 0 이하일 경우 -> mental_explode 엔딩
        if current_mental <= 0:
            print(f"[GLOBAL_TRANSITION] Mental <= 0, transitioning to mental_explode")
            self._set_game_state(username, "mental_explode")
            next_state_info = self._get_state_info("mental_explode")
            state_narration = next_state_info.get("narration")
            return (True, state_narration)

        # 호감도가 100 이상일 경우 -> love_attack 엔딩
        if new_affection >= 100:
            print(f"[GLOBAL_TRANSITION] Affection >= 100, transitioning to love_attack")
            self._set_game_state(username, "love_attack")
            next_state_info = self._get_state_info("love_attack")
            state_narration = next_state_info.get("narration")
            return (True, state_narration)

        # 현재 상태 정보 가져오기 (별도 JSON에서 로드)
        state_info = self._get_state_info(current_state)
        transitions = state_info.get("transitions", [])
        print(f"[STATE_CHECK] Found {len(transitions)} transitions for {current_state}")

        # 각 전이 조건 확인
        for transition in transitions:
            trigger_type = transition.get('trigger_type')
            next_state = transition.get('next_state')
            print(f"[STATE_CHECK] Checking transition: {trigger_type} -> {next_state}")
            print(f"[STATE_CHECK] Transition details: {transition}")
            print(f"[STATE_CHECK] About to call _evaluate_transition_condition with username={username}, affection_increased={affection_increased}, user_message='{user_message}'")
            
            try:
                print(f"[STATE_CHECK] Calling _evaluate_transition_condition...")
                result = self._evaluate_transition_condition(username, transition, affection_increased, user_message)
                print(f"[STATE_CHECK] Transition evaluation result: {result} for trigger_type '{trigger_type}', next_state: '{next_state}'")
            except Exception as e:
                print(f"[ERROR] Transition evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                result = False
            
            if result:
                transition_narration = transition.get("transition_narration")

                # 상태 전이 실행
                self._set_game_state(username, next_state)
                print(f"[STATE_TRANSITION] {current_state} → {next_state}")

                # state의 narration도 함께 반환
                next_state_info = self._get_state_info(next_state)
                state_narration = next_state_info.get("narration")

                # transition_narration과 state_narration 합치기
                combined_narration = None
                if transition_narration and state_narration:
                    combined_narration = f"{transition_narration}\n\n{state_narration}"
                elif transition_narration:
                    combined_narration = transition_narration
                elif state_narration:
                    combined_narration = state_narration

                return (True, combined_narration)

        return (False, None)
    
    def _get_selected_subjects(self, username: str) -> list:
        """
        사용자가 선택한 선택과목 목록 반환
        """
        return self.selected_subjects.get(username, [])
    
    def _set_selected_subjects(self, username: str, subjects: list):
        """
        사용자의 선택과목 설정 (최대 2개)
        """
        # 최대 2개까지만 저장
        self.selected_subjects[username] = subjects[:2]
        self._save_user_data(username)  # 변경사항 저장
    
    def _parse_subject_from_message(self, user_message: str) -> list:
        """
        사용자 메시지에서 선택과목명 추출 (여러 개 가능)
        반환값: 선택과목명 리스트 (예: ["물리학1", "화학1"])
        주의: "탐구1", "탐구2" 같은 키워드는 선택과목으로 인식하지 않음
        """
        import re
        user_message_original = user_message.strip()
        user_lower = user_message.lower().strip()
        found_subjects = []
        matched_positions = set()  # 이미 매칭된 위치 추적
        
        # 먼저 전체 메시지에서 정확한 과목명이 포함되어 있는지 확인 (최우선)
        for subject in self.subject_options:
            subject_lower = subject.lower()
            # 정확한 과목명이 메시지에 포함되어 있는 경우
            if subject in user_message_original or subject_lower in user_lower:
                if subject not in found_subjects:
                    found_subjects.append(subject)
                    # 매칭된 위치 기록
                    pos = user_lower.find(subject_lower)
                    if pos >= 0:
                        matched_positions.add((pos, pos + len(subject_lower)))
        
        # 쉼표, "과", "랑", "와", 공백 등으로 구분된 단어들로 분리
        # "물리1 화학1", "물리1과 화학1", "물리1, 화학1" 등 처리
        separators = r'[,，\s\n과와랑과]+'
        possible_phrases = re.split(separators, user_message_original)
        
        # 각 단어/구에서 선택과목 찾기
        for phrase in possible_phrases:
            phrase = phrase.strip()
            if not phrase or len(phrase) < 2:
                continue
            
            # "탐구1", "탐구2" 키워드 제외
            if re.match(r'^탐구\s*[12]$', phrase, re.IGNORECASE):
                continue
            
            # 이미 정확히 매칭된 과목은 스킵
            phrase_lower = phrase.lower()
            phrase_pos = user_lower.find(phrase_lower)
            if phrase_pos >= 0:
                is_overlap = False
                for start, end in matched_positions:
                    if not (phrase_pos + len(phrase_lower) <= start or phrase_pos >= end):
                        is_overlap = True
                        break
                if is_overlap:
                    continue
            
            # 과목 옵션과 매칭 시도
            for subject in self.subject_options:
                if subject in found_subjects:
                    continue
                    
                subject_lower = subject.lower()
                
                # 정확한 일치 (가장 높은 우선순위)
                if phrase_lower == subject_lower or phrase == subject:
                    found_subjects.append(subject)
                    break
                
                # "물리학1" vs "물리1" 같은 변형 허용
                # 숫자가 일치하고 앞부분이 유사한 경우
                subject_num_match = re.search(r'\d+', subject)
                phrase_num_match = re.search(r'\d+', phrase)
                
                if subject_num_match and phrase_num_match:
                    # 숫자가 일치하는 경우
                    if subject_num_match.group() == phrase_num_match.group():
                        # 앞부분이 유사한지 확인
                        subject_prefix = subject[:subject_num_match.start()].lower().replace("학", "").replace("과", "")
                        phrase_prefix = phrase[:phrase_num_match.start()].lower()
                        
                        # "물리" vs "물리", "화학" vs "화학" 등
                        # 단어 단위로 비교하여 더 정확한 매칭
                        subject_words = re.findall(r'\w+', subject_prefix)
                        phrase_words = re.findall(r'\w+', phrase_prefix)
                        
                        # 공통 단어가 있거나, 한쪽이 다른 쪽에 포함되는 경우
                        has_common = bool(set(subject_words) & set(phrase_words))
                        is_subset = bool(set(subject_words).issubset(set(phrase_words)) or set(phrase_words).issubset(set(subject_words)))
                        
                        if (has_common or is_subset) and len(subject_prefix) >= 1 and len(phrase_prefix) >= 1:
                            found_subjects.append(subject)
                            break
        
        print(f"[SUBJECT_PARSE] '{user_message}' -> {found_subjects}")
        return found_subjects
    
    def _get_subject_list_text(self) -> str:
        """
        선택과목 목록을 텍스트로 반환
        """
        subjects_text = ""
        for i, subject in enumerate(self.subject_options, 1):
            subjects_text += f"{i}. {subject}"
            if i % 3 == 0:
                subjects_text += "\n"
            elif i < len(self.subject_options):
                subjects_text += " | "
        return subjects_text
    
    def _parse_schedule_from_message(self, user_message: str, username: str) -> dict:
        """
        사용자 메시지에서 시간표 파싱
        예: "수학4시간 국어4시간 영어4시간 탐구1 1시간 탐구2 1시간"
        반환값: {"국어": 4, "수학": 4, ...} 또는 None
        """
        import re
        
        schedule = {}
        total_hours = 0
        
        # 사용자의 선택과목 확인
        selected_subjects = self._get_selected_subjects(username)
        
        # 우선순위 기반 패턴: 더 구체적인 패턴을 먼저 매칭
        # 1. "탐구1" 또는 "탐구2" 같은 명시적 표현 우선
        # 2. 선택과목 이름 직접 언급
        # 3. 국어, 수학, 영어 기본 과목
        
        user_message_original = user_message
        user_message_lower = user_message.lower()
        
        # 위치 정보를 저장하여 중복 매칭 방지
        matched_positions = set()
        
        # 패턴 1: 탐구1, 탐구2 명시적 표현 (가장 높은 우선순위)
        for idx in range(2):
            subject_key = f"탐구{idx+1}"
            # "탐구1 4시간", "탐구1 4시간", "탐구1 4" 등 다양한 패턴
            patterns = [
                rf"탐구\s*{idx+1}\s*(\d+)\s*시간",
                rf"탐구\s*{idx+1}\s*(\d+)시간",
                rf"탐구\s*{idx+1}\s*(\d+)",
            ]
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, user_message, re.IGNORECASE))
                for match in matches:
                    start, end = match.span()
                    # 이미 다른 패턴에 매칭된 위치인지 확인
                    if not any(start <= pos <= end for pos in matched_positions):
                        hours = int(match.group(1))
                        if subject_key not in schedule:
                            schedule[subject_key] = 0
                        schedule[subject_key] += hours
                        total_hours += hours
                        matched_positions.update(range(start, end))
                        break
        
        # 패턴 2: 선택과목 이름 직접 언급 (탐구1/탐구2가 아닌 경우에만)
        if len(selected_subjects) > 0:
            # 탐구1에 해당하는 선택과목
            subject1_name = selected_subjects[0]
            patterns = [
                rf"{re.escape(subject1_name)}\s*(\d+)\s*시간",
                rf"{re.escape(subject1_name)}\s*(\d+)시간",
                rf"{re.escape(subject1_name)}\s*(\d+)",
            ]
            for pattern in patterns:
                matches = list(re.finditer(pattern, user_message, re.IGNORECASE))
                for match in matches:
                    start, end = match.span()
                    if not any(start <= pos <= end for pos in matched_positions):
                        # 탐구1로 이미 설정되지 않은 경우에만
                        if "탐구1" not in schedule:
                            hours = int(match.group(1))
                            schedule["탐구1"] = hours
                            total_hours += hours
                            matched_positions.update(range(start, end))
                            break
        
        if len(selected_subjects) > 1:
            # 탐구2에 해당하는 선택과목
            subject2_name = selected_subjects[1]
            patterns = [
                rf"{re.escape(subject2_name)}\s*(\d+)\s*시간",
                rf"{re.escape(subject2_name)}\s*(\d+)시간",
                rf"{re.escape(subject2_name)}\s*(\d+)",
            ]
            for pattern in patterns:
                matches = list(re.finditer(pattern, user_message, re.IGNORECASE))
                for match in matches:
                    start, end = match.span()
                    if not any(start <= pos <= end for pos in matched_positions):
                        # 탐구2로 이미 설정되지 않은 경우에만
                        if "탐구2" not in schedule:
                            hours = int(match.group(1))
                            schedule["탐구2"] = hours
                            total_hours += hours
                            matched_positions.update(range(start, end))
                            break
        
        # 패턴 3: 국어, 수학, 영어 기본 과목
        basic_subjects = {
            "국어": [r"국어\s*(\d+)\s*시간", r"국어\s*(\d+)시간", r"국어\s*(\d+)"],
            "수학": [r"수학\s*(\d+)\s*시간", r"수학\s*(\d+)시간", r"수학\s*(\d+)"],
            "영어": [r"영어\s*(\d+)\s*시간", r"영어\s*(\d+)시간", r"영어\s*(\d+)"],
            "운동": [r"운동\s*(\d+)\s*시간", r"운동\s*(\d+)시간", r"운동\s*(\d+)"],
        }
        
        for subject_key, patterns in basic_subjects.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, user_message, re.IGNORECASE))
                for match in matches:
                    start, end = match.span()
                    if not any(start <= pos <= end for pos in matched_positions):
                        hours = int(match.group(1))
                        if subject_key not in schedule:
                            schedule[subject_key] = 0
                        schedule[subject_key] += hours
                        total_hours += hours
                        matched_positions.update(range(start, end))
                        break
        
        # 총 시간이 14시간을 초과하면 None 반환
        if total_hours > 14:
            print(f"[SCHEDULE] 파싱 결과 총 시간이 14시간 초과: {schedule}, 총 {total_hours}시간")
            return None
        
        # 빈 딕셔너리면 None 반환
        if not schedule:
            print(f"[SCHEDULE] 파싱 결과가 비어있음: {user_message}")
            return None
        
        print(f"[SCHEDULE] 파싱 성공: {schedule}, 총 {total_hours}시간")
        return schedule
    
    def _get_schedule(self, username: str) -> dict:
        """
        사용자의 현재 시간표 반환
        """
        return self.schedules.get(username, {})
    
    def _set_schedule(self, username: str, schedule: dict):
        """
        사용자의 시간표 설정 (총 14시간 제한)
        """
        total_hours = sum(schedule.values())
        if total_hours > 14:
            # 비율로 축소
            scale = 14 / total_hours
            schedule = {k: int(v * scale) for k, v in schedule.items()}

        self.schedules[username] = schedule
        self._save_user_data(username)  # 변경사항 저장
    
    def _get_conversation_count(self, username: str) -> int:
        """
        사용자의 대화 횟수 반환 (daily_routine 상태에서만 카운트)
        """
        return self.conversation_counts.get(username, 0)
    
    def _increment_conversation_count(self, username: str):
        """
        사용자의 대화 횟수 증가
        """
        self.conversation_counts[username] = self.conversation_counts.get(username, 0) + 1
        self._save_user_data(username)  # 변경사항 저장
    
    def _reset_conversation_count(self, username: str):
        """
        사용자의 대화 횟수 초기화
        """
        self.conversation_counts[username] = 0
        self._save_user_data(username)  # 변경사항 저장
    
    def _get_current_week(self, username: str) -> int:
        """
        사용자의 현재 주(week) 반환
        """
        return self.current_weeks.get(username, 0)
    
    def _increment_week(self, username: str):
        """
        사용자의 주(week) 증가
        """
        self.current_weeks[username] = self.current_weeks.get(username, 0) + 1
        self._save_user_data(username)  # 변경사항 저장
    
    def _get_game_date(self, username: str) -> str:
        """
        사용자의 게임 날짜 반환 (기본값: "2023-11-17")
        """
        return self.game_dates.get(username, "2023-11-17")
    
    def _get_career(self, username: str) -> str:
        """
        사용자의 진로 반환 (없으면 None)
        """
        return self.careers.get(username)
    
    def _set_career(self, username: str, career: str):
        """
        사용자의 진로 설정
        """
        self.careers[username] = career
        self._save_user_data(username)  # 변경사항 저장
    
    def _set_game_date(self, username: str, date_str: str):
        """
        사용자의 게임 날짜 설정
        """
        self.game_dates[username] = date_str
        self._save_user_data(username)  # 변경사항 저장
    
    def _add_days_to_date(self, date_str: str, days: int) -> str:
        """
        날짜에 일수 추가 (YYYY-MM-DD 형식)
        """
        from datetime import datetime, timedelta
        date = datetime.strptime(date_str, "%Y-%m-%d")
        new_date = date + timedelta(days=days)
        return new_date.strftime("%Y-%m-%d")
    
    def _get_strategy_multiplier(self, username: str, subject: str) -> float:
        """
        시험 전략 배율 가져오기
        
        Args:
            username: 사용자 이름
            subject: 과목명 (국어, 수학, 영어, 탐구1, 탐구2)
        
        Returns:
            배율 (VERY_GOOD: 1.5, GOOD: 1.05, POOR: 1.0, 전략 없음: 1.0)
        """
        if username not in self.exam_progress:
            return 1.0
        
        strategies = self.exam_progress[username].get("strategies", {})
        if subject not in strategies:
            return 1.0
        
        strategy_quality = strategies[subject].get("quality", "POOR")
        multiplier_map = {
            "VERY_GOOD": 1.5,
            "GOOD": 1.05,
            "POOR": 1.0
        }
        return multiplier_map.get(strategy_quality, 1.0)
    
    def _apply_ability_multipliers(self, username: str, subject: str, base_increase: float) -> float:
        """
        능력치 증가에 배율을 적용하는 공통 함수
        - 진로-과목 배율 (1.2배)
        - 시험 전략 배율 (1.0, 1.05, 1.5배)
        
        Args:
            username: 사용자 이름
            subject: 과목명 (국어, 수학, 영어, 탐구1, 탐구2)
            base_increase: 기본 증가량
        
        Returns:
            최종 증가량 (배율 적용 후)
        """
        final_increase = base_increase
        multipliers_applied = []
        
        # 1. 진로-과목 배율 적용
        career = self._get_career(username)
        selected_subjects = self._get_selected_subjects(username)
        
        # 탐구1, 탐구2를 실제 선택과목으로 매핑
        actual_subject = subject
        if subject == "탐구1" and len(selected_subjects) > 0:
            actual_subject = selected_subjects[0]
        elif subject == "탐구2" and len(selected_subjects) > 1:
            actual_subject = selected_subjects[1]
        
        # 진로와 관련된 선택과목인지 확인
        if career and actual_subject in selected_subjects:
            from services.utils.career_manager import get_career_subject_bonus_multiplier
            career_multiplier = get_career_subject_bonus_multiplier(career, actual_subject)
            if career_multiplier > 1.0:
                final_increase = final_increase * career_multiplier
                multipliers_applied.append(f"진로-과목 {career_multiplier}배")
        
        # 2. 시험 전략 배율 적용
        strategy_multiplier = self._get_strategy_multiplier(username, subject)
        if strategy_multiplier > 1.0:
            final_increase = final_increase * strategy_multiplier
            multipliers_applied.append(f"시험전략 {strategy_multiplier}배")
        
        # 로그 출력
        if multipliers_applied:
            print(f"[ABILITY_MULTIPLIER] {username}의 '{subject}' 과목: 기본 {base_increase} → 최종 {final_increase:.2f} ({', '.join(multipliers_applied)} 적용)")
        
        return final_increase
    
    def _apply_schedule_to_abilities(self, username: str, mentoring_end_bonus: float = 1.0):
        """
        시간표에 따라 능력치 증가
        시간당 +1 증가 (체력에 따른 효율 적용)
        
        Args:
            username: 사용자 이름
            mentoring_end_bonus: 멘토링 종료 시 추가 배율 (기본값: 1.0, 멘토링 종료 시: 10.0)
        """
        schedule = self._get_schedule(username)
        if not schedule:
            return
        
        abilities = self._get_abilities(username)
        stamina = self._get_stamina(username)
        mental = self._get_mental(username)
        efficiency = self._calculate_combined_efficiency(stamina, mental) / 100.0  # 효율을 배율로 변환 (1.0 = 100%)
        
        # 운동 시간 처리 (체력 증가) - 정확히 운동 시간만큼 +1씩 증가
        exercise_hours = schedule.get("운동", 0)
        if exercise_hours > 0:
            # 현재 체력을 직접 가져와서 운동 시간만큼 더하기 (정확히 +exercise_hours)
            current_stamina = self._get_stamina(username)
            new_stamina = min(100, current_stamina + exercise_hours)  # 정확히 운동 시간만큼 증가
            self._set_stamina(username, new_stamina)
            print(f"[STAMINA] {username}의 체력이 {current_stamina}에서 {new_stamina}로 증가했습니다. (운동 {exercise_hours}시간, +{exercise_hours})")
            stamina = new_stamina  # 이후 능력치 계산에 업데이트된 체력 사용
        
        for subject, hours in schedule.items():
            if subject in abilities:
                # 체력과 멘탈의 곱연산 효율 적용: 시간 * 효율
                base_increase = hours * efficiency
                
                # 배율 적용 (진로-과목 + 시험 전략)
                increased = self._apply_ability_multipliers(username, subject, base_increase)
                
                # 멘토링 종료 보너스 배율 적용
                increased = increased * mentoring_end_bonus
                
                abilities[subject] = min(2500, abilities[subject] + increased)  # 최대 2500
            # 운동은 이미 위에서 처리했으므로 여기서는 스킵
        
        if mentoring_end_bonus > 1.0:
            print(f"[MENTORING_END_BONUS] 멘토링 종료 보너스 {mentoring_end_bonus}배 적용")
        
        self._set_abilities(username, abilities)
    
    def _advance_one_week(self, username: str, mentoring_end: bool = False) -> dict:
        """
        1주일을 진행시키는 통합 메서드
        시간표에 따라 능력치를 증가시키고, 날짜와 주차를 업데이트합니다.
        
        Args:
            username: 사용자 이름
            mentoring_end: 멘토링 종료 여부 (멘토링 종료 시 능력치 10배 증가)
        
        Returns:
            dict: 시험 결과 정보 (시험이 있었으면 포함)
        """
        current_schedule = self._get_schedule(username)
        current_date = self._get_game_date(username)
        
        # 시간표에 따라 능력치 증가
        if current_schedule:
            # 멘토링 종료 시 10배 보너스 적용
            mentoring_end_bonus = 10.0 if mentoring_end else 1.0
            self._apply_schedule_to_abilities(username, mentoring_end_bonus=mentoring_end_bonus)
            print(f"[WEEK] {username}의 1주일이 경과했습니다. 능력치가 증가했습니다.")
            print(f"[ABILITIES] 현재 능력치: {self._get_abilities(username)}")
        
        # 주차 증가
        self._increment_week(username)
        current_week = self._get_current_week(username)
        
        # 1주 경과 시 체력 -1
        current_stamina = self._get_stamina(username)
        new_stamina = max(0, current_stamina - 1)
        self._set_stamina(username, new_stamina)
        print(f"[STAMINA] {username}의 체력이 {current_stamina}에서 {new_stamina}로 변경되었습니다. (1주 경과로 -1)")
        
        # 날짜 7일 증가
        new_date = self._add_days_to_date(current_date, 7)
        self._set_game_date(username, new_date)
        
        # 대화 횟수 초기화 (1주 경과 후 리셋)
        self._reset_conversation_count(username)
        
        # 시험 체크
        exam_month = self._check_exam_in_period(current_date, new_date)
        exam_result = None
        
        if exam_month:
            # 시험 성적 계산
            exam_scores = self._calculate_exam_scores(username, exam_month)
            exam_name = "수능" if exam_month.endswith("-11") else f"{exam_month[-2:]}월 모의고사"
            
            subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
            exam_scores_text = f"{exam_name} 성적이 발표되었습니다:\n"
            
            score_lines = []
            for subject in subjects:
                if subject in exam_scores:
                    score_data = exam_scores[subject]
                    score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
            
            exam_scores_text += "\n".join(score_lines)
            
            # 정규모의고사인 경우 등급대별 반응 계산 (나레이션에는 포함하지 않고 reply로 표시)
            if self._is_official_mock_exam_month(exam_month):
                average_grade = self._calculate_average_grade(exam_scores)
                grade_reaction = self._generate_grade_reaction("official_mock_exam", average_grade)
                # exam_result에 grade_reaction 저장 (나중에 reply로 사용)
                exam_result = {
                    "name": exam_name,
                    "scores": exam_scores,
                    "text": exam_scores_text,
                    "grade_reaction": grade_reaction
                }
            else:
                exam_result = {
                    "name": exam_name,
                    "scores": exam_scores,
                    "text": exam_scores_text
                }
        
        return {
            "week": current_week,
            "date": new_date,
            "exam": exam_result
        }
    
    def _calculate_percentile(self, ability: int) -> float:
        """능력치를 백분위로 변환"""
        from services.utils.exam_score_calculator import calculate_percentile
        return calculate_percentile(ability)
    
    def _calculate_grade_from_percentile(self, percentile: float) -> int:
        """백분위를 등급으로 변환 (수능 등급 체계)"""
        from services.utils.exam_score_calculator import calculate_grade_from_percentile
        return calculate_grade_from_percentile(percentile)
    
    def _get_current_exam_month(self, date_str: str) -> str:
        """
        현재 날짜가 정확히 시험일인지 확인 (시험일 당일만 반환)
        반환값: "2024-03", "2024-04", ... "2024-11" (수능), 또는 None
        
        시험일:
        - 3월 모의고사: 2024-03-07
        - 4월 모의고사: 2024-04-04
        - 5월 모의고사: 2024-05-09
        - 6월 모의고사: 2024-06-06
        - 7월 모의고사: 2024-07-11
        - 9월 모의고사: 2024-09-05
        - 10월 모의고사: 2024-10-17
        - 수능: 2024-11-14
        
        시험일 당일만 반환 (전후 범위 제거)
        """
        from datetime import datetime
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            year = date.year
            
            # 시험일 정의
            exam_dates = {
                (year, 3, 7): "03",   # 3월 모의고사
                (year, 4, 4): "04",   # 4월 모의고사
                (year, 5, 9): "05",   # 5월 모의고사
                (year, 6, 6): "06",   # 6월 모의고사
                (year, 7, 11): "07",  # 7월 모의고사
                (year, 9, 5): "09",   # 9월 모의고사
                (year, 10, 17): "10", # 10월 모의고사
                (year, 11, 14): "11", # 수능
            }
            
            # 정확히 시험일인 경우에만 반환
            exam_key = (date.year, date.month, date.day)
            if exam_key in exam_dates:
                month_str = exam_dates[exam_key]
                return f"{year}-{month_str}"
            
            return None
        except Exception as e:
            print(f"[EXAM] 날짜 파싱 오류: {e}")
            return None
    
    def _check_exam_in_period(self, start_date: str, end_date: str) -> str:
        """
        주어진 기간(시작일부터 종료일까지) 동안 시험이 있었는지 확인
        반환값: 시험 월 (예: "2024-03") 또는 None
        
        시험은 시험일 당일에만 발생하므로, 기간 내에 시험일이 포함되어 있는지만 확인
        """
        from datetime import datetime
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            year = start.year
            
            # 시험일 정의
            exam_dates = [
                (year, 3, 7),   # 3월 모의고사
                (year, 4, 4),   # 4월 모의고사
                (year, 5, 9),   # 5월 모의고사
                (year, 6, 6),   # 6월 모의고사
                (year, 7, 11),  # 7월 모의고사
                (year, 9, 5),   # 9월 모의고사
                (year, 10, 17), # 10월 모의고사
                (year, 11, 14), # 수능
            ]
            
            # 기간 내에 시험일이 포함되어 있는지 확인
            for exam_year, exam_month, exam_day in exam_dates:
                exam_date = datetime(exam_year, exam_month, exam_day)
                if start <= exam_date <= end:
                    month_str = f"{exam_month:02d}"
                    print(f"[EXAM] 기간 내 시험 발견: {exam_date.strftime('%Y-%m-%d')} ({year}-{month_str})")
                    return f"{year}-{month_str}"
            
            print(f"[EXAM] 기간 내 시험 없음: {start_date} ~ {end_date}")
            return None
        except Exception as e:
            print(f"[EXAM] 기간 체크 오류: {e}")
            return None
    
    def _calculate_exam_scores(self, username: str, exam_month: str, strategy_bonus: float = 0.0) -> dict:
        """능력치를 기반으로 시험 성적 계산 (전략 보너스 포함)"""
        from services.utils.exam_score_calculator import calculate_exam_scores
        abilities = self._get_abilities(username)
        scores = calculate_exam_scores(abilities, strategy_bonus)
        if strategy_bonus > 0:
            print(f"[EXAM] {username}의 {exam_month} 시험 성적 계산 (전략 보너스: +{strategy_bonus*100:.1f}%): {scores}")
        else:
            print(f"[EXAM] {username}의 {exam_month} 시험 성적 계산: {scores}")
        return scores
    
    def _calculate_mock_exam_scores(self, username: str, strategy_bonus: float = 0.0) -> dict:
        """사설모의고사 성적 계산 (능력치 기반, 전략 보너스 포함)"""
        from services.utils.exam_score_calculator import calculate_exam_scores
        abilities = self._get_abilities(username)
        scores = calculate_exam_scores(abilities, strategy_bonus)
        if strategy_bonus > 0:
            print(f"[MOCK_EXAM] {username}의 사설모의고사 성적 계산 (전략 보너스: +{strategy_bonus*100:.1f}%): {scores}")
        else:
            print(f"[MOCK_EXAM] {username}의 사설모의고사 성적 계산: {scores}")
        return scores
    
    def _is_official_mock_exam_month(self, exam_month: str) -> bool:
        """정규모의고사 월인지 확인 (3, 4, 5, 7, 10월)"""
        from services.utils.exam_score_calculator import is_official_mock_exam_month
        return is_official_mock_exam_month(exam_month)
    
    def _identify_weak_subject(self, exam_scores: dict) -> str:
        """시험 점수에서 가장 취약한 과목 식별 (등급이 가장 낮은 과목)"""
        from services.utils.exam_score_calculator import identify_weak_subject
        return identify_weak_subject(exam_scores)
    
    def _generate_weakness_message(self, subject: str, score_data: dict) -> str:
        """취약 과목에 대한 취약점 메시지 생성 (과목별 다양한 예시)"""
        from services.utils.exam_score_calculator import generate_weakness_message
        return generate_weakness_message(subject, score_data)
    
    def _calculate_average_grade(self, exam_scores: dict) -> float:
        """시험 점수 딕셔너리에서 평균 등급 계산"""
        from services.utils.exam_score_calculator import calculate_average_grade
        return calculate_average_grade(exam_scores)
    
    def _generate_grade_reaction(self, exam_type: str, average_grade: float) -> str:
        """등급대별 시험 결과 반응 메시지 생성"""
        from services.utils.exam_score_calculator import generate_grade_reaction
        return generate_grade_reaction(exam_type, average_grade)
    
    def _generate_june_subject_problem(self, subject: str, score_data: dict) -> str:
        """6월 모의고사 과목별 취약점 메시지 생성"""
        from services.utils.exam_score_calculator import generate_june_subject_problem
        return generate_june_subject_problem(subject, score_data)
    
    def _check_if_advice_given(self, user_message: str) -> bool:
        """
        사용자 메시지에 조언이 포함되어 있는지 확인
        """
        advice_keywords = ["이렇게", "이런", "조언", "팁", "방법", "해보", "시도", "추천", "제안", "도움", "알려", "가르쳐", 
                          "괜찮아", "괜찮", "잘할", "할수", "할 수", "할수있", "할 수 있", "가능", "노력", "열심히", 
                          "화이팅", "힘내", "응원", "충분", "다시", "연습"]
        user_lower = user_message.lower()
        
        for keyword in advice_keywords:
            if keyword in user_lower:
                print(f"[ADVICE_CHECK] 키워드 감지: '{keyword}' in '{user_message}'")
                return True
        
        # 메시지가 충분히 길면 조언으로 간주 (10자 이상)
        if len(user_message.strip()) > 10:
            print(f"[ADVICE_CHECK] 길이 기반 조언 감지: {len(user_message.strip())}자")
            return True
        
        print(f"[ADVICE_CHECK] 조언 미감지: '{user_message}' (길이: {len(user_message.strip())}자)")
        return False
    
    def _extract_subject_from_strategy(self, strategy: str) -> str:
        """
        전략 메시지에서 과목을 추출
        예: "국어의 경우 비문학 3점짜리는 최대한 마지막에 풀어라" -> "국어"
        """
        # 과목 키워드 정의 (주요 키워드만)
        subject_keywords = {
            "국어": ["국어", "언어영역", "언매", "화작", "독서", "문학", "비문학"],
            "수학": ["수학", "미적", "기하", "확통", "확률과통계"],
            "영어": ["영어", "영어영역", "독해", "문법"],
            "탐구1": ["탐구1", "사회문화", "생활과윤리", "윤사", "한국지리", "세계지리", "동아시아사", "세계사", "경제", "정치와법"],
            "탐구2": ["탐구2", "물리학", "물리", "화학", "생명과학", "생물", "지구과학"]
        }
        
        strategy_lower = strategy.lower()
        
        # 각 과목별 키워드 매칭 (키워드만으로 판단)
        for subject, keywords in subject_keywords.items():
            for keyword in keywords:
                if keyword in strategy_lower:
                    print(f"[STRATEGY_SUBJECT] 추출된 과목: {subject} (키워드: {keyword})")
                    return subject
        
        # 과목이 명시되지 않았으면 None 반환
        print(f"[STRATEGY_SUBJECT] 과목을 찾지 못했습니다. 전략: {strategy[:50]}...")
        return None
    
    def _judge_exam_strategy_quality(self, username: str, strategy: str) -> str:
        """
        LLM을 사용하여 플레이어의 시험 전략을 평가 (VERY_GOOD, GOOD, POOR)
        chatbot_config.json에서 프롬프트 설정을 로드합니다.
        긴 전략일수록 VERY_GOOD을 받을 확률이 증가합니다.
        """
        try:
            if not self.client:
                # LLM이 없으면 기본적으로 GOOD으로 판단
                import random
                return random.choice(["VERY_GOOD", "GOOD", "POOR"])
            
            # chatbot_config.json에서 판단 설정 로드
            judgment_config = self.config.get("exam_strategy_judgment", {})
            system_prompt = judgment_config.get(
                "system_prompt", 
                "당신은 입시 전문가입니다. 수능 및 모의고사에서 학생이 제시한 전략이 정교하고 효과적인지, 단순하고 효과가 낮은지 판단하세요."
            )
            user_prompt_template = judgment_config.get(
                "user_prompt_template",
                "수능/모의고사 시험 전략을 평가하세요.\n\n플레이어(멘토)가 다음 전략을 제시했습니다:\n{strategy}\n\n이 전략이 다음과 같은 기준에 부합하는지 판단하세요:\n1. 구체적이고 실행 가능한가?\n2. 과목별 특성을 고려했는가?\n3. 시험 시간 관리를 고려했는가?\n4. 실전 상황을 고려한 정교한 전략인가?\n\n모든 기준을 만족하는 정교한 전략이면 \"VERY_GOOD\", 2~3개를 만족하는 보통 전략이면 \"GOOD\", 0~1개만 만족하는 단순한 전략이면 \"POOR\"만 답변해주세요."
            )
            temperature = judgment_config.get("temperature", 0.3)
            max_tokens = judgment_config.get("max_tokens", 20)
            positive_keywords = judgment_config.get("positive_keywords", ["VERY_GOOD", "GOOD", "정교", "구체적", "효과적", "실행가능"])
            negative_keywords = judgment_config.get("negative_keywords", ["POOR", "단순", "효과없", "구체적이지"])
            
            # 프롬프트 템플릿에 변수 치환
            try:
                if "{strategy}" in user_prompt_template:
                    judgment_prompt = user_prompt_template.format(strategy=strategy)
                else:
                    judgment_prompt = f"{user_prompt_template}\n\n전략: {strategy}"
            except KeyError as e:
                print(f"[WARN] Prompt template format error: {e}. Using strategy directly.")
                judgment_prompt = user_prompt_template.replace("{strategy}", strategy) if "{strategy}" in user_prompt_template else f"{user_prompt_template}\n\n전략: {strategy}"

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": judgment_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            judgment = response.choices[0].message.content.strip().upper()
            
            print(f"[STRATEGY_JUDGE] LLM 원본 응답: {response.choices[0].message.content.strip()}")
            
            # 키워드 기반 판단
            judgment_upper = judgment.upper()
            
            if "VERY_GOOD" in judgment_upper:
                quality = "VERY_GOOD"
                print(f"[STRATEGY_JUDGE] VERY_GOOD 판단")
            elif "POOR" in judgment_upper:
                quality = "POOR"
                print(f"[STRATEGY_JUDGE] POOR 판단")
            else:
                quality = "GOOD"
                print(f"[STRATEGY_JUDGE] GOOD 판단 (기본값 또는 키워드 없음)")
            
            # 전략 길이에 따른 보정 (긴 전략일수록 VERY_GOOD 확률 증가)
            strategy_length = len(strategy.strip())
            import random
            if quality == "GOOD" and strategy_length >= 50:
                # 50자 이상이고 GOOD이면 30% 확률로 VERY_GOOD으로 승격
                if random.random() < 0.3:
                    quality = "VERY_GOOD"
                    print(f"[STRATEGY_JUDGE] 길이 기반 승격: {strategy_length}자 → VERY_GOOD")
            elif quality != "VERY_GOOD" and strategy_length >= 100:
                # 100자 이상이고 VERY_GOOD이 아니면 50% 확률로 VERY_GOOD으로 승격
                if random.random() < 0.5:
                    quality = "VERY_GOOD"
                    print(f"[STRATEGY_JUDGE] 길이 기반 승격: {strategy_length}자 → VERY_GOOD")
            elif strategy_length >= 150:
                # 150자 이상이면 무조건 VERY_GOOD으로 승격
                if quality != "VERY_GOOD":
                    quality = "VERY_GOOD"
                    print(f"[STRATEGY_JUDGE] 길이 기반 강제 승격: {strategy_length}자 → VERY_GOOD")
            
            print(f"[STRATEGY_JUDGE] 최종 판단 결과: {quality} (judgment: '{judgment}', length: {strategy_length}자)")
            return quality
            
        except Exception as e:
            print(f"[ERROR] 시험 전략 판단 실패: {e}")
            import traceback
            traceback.print_exc()
            # 기본값으로 GOOD 반환
            return "GOOD"
    
    def _generate_student_thought(self, subject: str, ability: float, stamina: int, mental: int, strategy_quality: str) -> str:
        """
        과목별 학생의 주관적 평가 메시지 생성
        능력치, 체력, 멘탈, 전략 품질을 고려하여 주관적 판단 생성
        
        Args:
            subject: 과목명
            ability: 능력치 (0~2500)
            stamina: 체력 (0~100)
            mental: 멘탈 (0~100)
            strategy_quality: 전략 품질 (VERY_GOOD, GOOD, POOR)
        
        Returns:
            str: 학생의 주관적 평가 메시지 (예: "1교시 잘본것 같다.")
        """
        import random
        
        # 등급 계산 (능력치 -> 등급)
        percentile = self._calculate_percentile(ability)
        grade = self._calculate_grade_from_percentile(percentile)
        
        # 등급에 따른 기본 분위기
        if grade <= 2:
            # 상위 등급: 잘했다고 느낌
            base_mood = "well"
            thought_templates = [
                f"{subject} 잘본것 같다.",
                f"{subject} 괜찮은 것 같은데?",
                f"{subject}은 좀 자신있는 편이야."
            ]
        elif grade <= 4:
            # 중위 등급: 불확실
            base_mood = "uncertain"
            thought_templates = [
                f"{subject} 잘모르겠다.",
                f"{subject}은... 음... 잘모르겠어.",
                f"{subject} 좀 애매하다."
            ]
        else:
            # 하위 등급: 못봤다고 느낌
            base_mood = "bad"
            thought_templates = [
                f"{subject} 조졌다...",
                f"{subject} 너무 어려웠어.",
                f"{subject} 완전 망한 것 같아."
            ]
        
        # 체력이 낮으면 탐구 과목에서 실수 가능성 증가
        if subject in ["탐구1", "탐구2"]:
            if stamina <= 20:
                # 탐구 과목에서 체력 부족으로 실수
                stamina_penalty_templates = [
                    f"{subject} 피곤해서 실수했을 것 같아.",
                    f"{subject}은... 체력이 딸려서 시간이 부족했어.",
                    f"{subject} 마지막 쪽이 제대로 안 풀렸어."
                ]
                if random.random() < 0.6:  # 60% 확률로 체력 패널티 적용
                    thought_templates.extend(stamina_penalty_templates)
        
        # 멘탈이 낮으면 국어에서 실수 가능성 증가
        if subject == "국어":
            if mental <= 25:
                # 국어에서 멘탈 부족으로 실수
                mental_penalty_templates = [
                    f"{subject}... 긴장해서 실수했나?",
                    f"{subject} 작품 해석이... 혼란스러웠어.",
                    f"{subject} 비문학이 읽히지 않았어."
                ]
                if random.random() < 0.6:  # 60% 확률로 멘탈 패널티 적용
                    thought_templates.extend(mental_penalty_templates)
        
        # 전략 품질에 따른 추가 효과
        strategy_bonus = 0
        if strategy_quality == "VERY_GOOD":
            # 매우 좋은 전략: 확신 증가
            if base_mood == "well" or grade <= 2:
                thought_templates.extend([
                    f"{subject} 전략대로 잘 풀었어.",
                    f"{subject} 선생님이 알려주신 대로 했더니 쉬웠어."
                ])
        elif strategy_quality == "POOR":
            # 나쁜 전략: 확신 감소
            thought_templates.extend([
                f"{subject}... 어떻게 하는 게 맞았는지 모르겠어.",
                f"{subject} 시간 배분이 잘 안 됐어."
            ])
        
        # 랜덤 선택
        selected_thought = random.choice(thought_templates)
        
        print(f"[STUDENT_THOUGHT] {subject}: {selected_thought} (등급: {grade}, 체력: {stamina}, 멘탈: {mental}, 전략: {strategy_quality})")
        return selected_thought
    
    def _judge_advice_quality(self, username: str, advice: str, weak_subject: str, weakness_message: str) -> bool:
        """
        LLM을 사용하여 플레이어의 조언이 적절한지 판단
        chatbot_config.json에서 프롬프트 설정을 로드합니다.
        """
        try:
            # 먼저 부정적 키워드를 직접 체크하여 확실한 부정적 조언은 즉시 거부
            negative_direct_keywords = [
                "망해", "망하", "포기", "포기해", "그만둬", "그만", "안돼", "못해", 
                "별로", "좋지않", "좋지 않", "안좋", "안 좋", "나쁘", "싫", "미워",
                "에휴", "아이고", "제발", "짜증", "답답", "한심", "바보", "멍청",
                "쓸모없", "쓸모 없", "소용없", "소용 없", "시작도", "시작도 못해",
                "이딴", "저딴", "이런", "저런", "그냥", "망했", "망했어", "망해라",
                "좆같", "지랄", "죽어", "죽어라", "꺼져", "시발", "개같", "병신"
            ]
            
            advice_lower = advice.lower()
            for keyword in negative_direct_keywords:
                if keyword in advice_lower:
                    print(f"[ADVICE_JUDGE] 부정적 키워드 직접 감지: '{keyword}' in '{advice}' → NO")
                    return False
            
            if not self.client:
                # LLM이 없으면 기본적으로 적절하다고 판단 (절반 확률)
                import random
                return random.choice([True, False])
            
            # chatbot_config.json에서 판단 설정 로드
            judgment_config = self.config.get("mock_exam_advice_judgment", {})
            system_prompt = judgment_config.get(
                "system_prompt", 
                "당신은 교육 전문가입니다. 학생을 격려하고 도와주는 멘토의 조언이 적절한지 판단하세요. 부정적이고 해로운 조언은 절대 용납하지 마세요."
            )
            user_prompt_template = judgment_config.get(
                "user_prompt_template",
                "플레이어(멘토)가 재수생에게 다음과 같은 조언을 했습니다:\n{advice}\n\n이 조언이 학생에게 도움이 되고 격려가 되는 긍정적인 조언인지, 아니면 부정적이고 해로운 조언인지 판단해주세요.\n\n조언이 긍정적이고 격려적이면(예: '할 수 있어', '괜찮아', '응원해', '노력하면 돼' 등) \"YES\", 부정적이고 해로운 조언이면(예: '포기해', '망해', '안돼', '그만둬', 비꼬거나 비판적인 말 등) \"NO\"만 답변해주세요."
            )
            temperature = judgment_config.get("temperature", 0.3)
            max_tokens = judgment_config.get("max_tokens", 10)
            positive_keywords = judgment_config.get("positive_keywords", ["YES", "적절", "좋", "도움", "유용", "효과적", "격려", "긍정"])
            negative_keywords = judgment_config.get("negative_keywords", ["NO", "부적절", "나쁨", "무도움", "비효과적", "비판", "부정", "해롭", "해로운"])
            
            # 프롬프트 템플릿에 변수 치환 (advice만 사용)
            # 템플릿에 있는 변수만 format
            try:
                # advice 변수만 있는지 확인하고 format
                if "{advice}" in user_prompt_template:
                    judgment_prompt = user_prompt_template.format(advice=advice)
                elif "{weak_subject}" in user_prompt_template or "{weakness_message}" in user_prompt_template:
                    # 이전 형식 지원 (하위 호환성)
                    judgment_prompt = user_prompt_template.format(
                        weak_subject=weak_subject,
                        weakness_message=weakness_message,
                        advice=advice
                    )
                else:
                    # 변수가 없으면 그대로 사용하고 advice만 추가
                    judgment_prompt = user_prompt_template + f"\n\n조언: {advice}"
            except KeyError as e:
                # 변수 치환 실패 시 advice만 추가
                print(f"[WARN] Prompt template format error: {e}. Using advice directly.")
                judgment_prompt = user_prompt_template.replace("{advice}", advice) if "{advice}" in user_prompt_template else f"{user_prompt_template}\n\n조언: {advice}"

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": judgment_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            judgment = response.choices[0].message.content.strip().upper()
            
            print(f"[ADVICE_JUDGE] LLM 원본 응답: {response.choices[0].message.content.strip()}")
            
            # 키워드 기반 판단
            judgment_upper = judgment.upper()
            has_positive = any(keyword.upper() in judgment_upper for keyword in positive_keywords)
            has_negative = any(keyword.upper() in judgment_upper for keyword in negative_keywords)
            
            print(f"[ADVICE_JUDGE] Positive keywords found: {has_positive}, Negative keywords found: {has_negative}")
            print(f"[ADVICE_JUDGE] Judgment upper: {judgment_upper}")
            
            if has_positive:
                is_good = True
                print(f"[ADVICE_JUDGE] 긍정 키워드 발견 - YES로 판단")
            elif has_negative:
                is_good = False
                print(f"[ADVICE_JUDGE] 부정 키워드 발견 - NO로 판단")
            else:
                # 키워드가 없으면 응답 내용을 다시 확인
                # "YES" 또는 "NO"가 직접 포함되어 있는지 확인
                if "YES" in judgment_upper or "예" in judgment or "좋" in judgment or "긍정" in judgment:
                    is_good = True
                    print(f"[ADVICE_JUDGE] 직접 확인 - YES로 판단")
                elif "NO" in judgment_upper or "아니" in judgment or "부정" in judgment or "나쁨" in judgment:
                    is_good = False
                    print(f"[ADVICE_JUDGE] 직접 확인 - NO로 판단")
                else:
                    # 키워드가 없으면 LLM 응답을 다시 분석
                    # 응답이 명확하지 않으면 안전하게 부적절로 판단
                    if len(judgment_upper) > 0 and ("NO" in judgment_upper or "아니" in judgment or "부정" in judgment or "해롭" in judgment):
                        is_good = False
                        print(f"[ADVICE_JUDGE] 애매한 응답에서 부정 키워드 발견 - NO로 판단")
                    elif len(judgment_upper) > 0 and ("YES" in judgment_upper or "예" in judgment or "좋" in judgment):
                        is_good = True
                        print(f"[ADVICE_JUDGE] 애매한 응답에서 긍정 키워드 발견 - YES로 판단")
                    else:
                        # 응답이 완전히 불명확하면 안전을 위해 부적절로 판단 (보수적 접근)
                        is_good = False
                        print(f"[ADVICE_JUDGE] 응답 불명확 - 안전을 위해 NO로 판단")
            
            print(f"[ADVICE_JUDGE] 최종 판단 결과: {is_good} (judgment: '{judgment}', advice: '{advice[:50]}...')")
            return is_good
            
        except Exception as e:
            print(f"[ERROR] 조언 판단 실패: {e}")
            # 오류 시 안전을 위해 부적절로 판단 (보수적 접근)
            return False
    
    def _check_prompt_injection(self, user_message: str) -> bool:
        """
        프롬프트 공격(주입) 감지
        반환값: True면 공격으로 감지됨
        """
        injection_cfg = self.config.get("prompt_injection_detection", {})
        
        if not injection_cfg.get("enabled", True):
            return False
        
        warning_keywords = injection_cfg.get("warning_keywords", [])
        user_lower = user_message.lower()
        
        for keyword in warning_keywords:
            if keyword.lower() in user_lower:
                print(f"[SECURITY] 프롬프트 공격 감지: '{keyword}' 키워드 발견")
                return True
        
        return False
    
    def _get_narration(self, event_type: str, context: dict = None) -> str:
        """
        나레이션 메시지 생성
        event_type: "game_start", "state_transition"
        """
        try:
            if not self.config:
                return None
                
            narration_cfg = self.config.get("narration", {})
            
            if not narration_cfg.get("enabled", True):
                return None
            
            if event_type == "game_start":
                return narration_cfg.get("game_start", "")
            elif event_type == "state_transition":
                transitions = narration_cfg.get("state_transitions", {})
                if context:
                    transition_key = context.get("transition_key", "")
                    return transitions.get(transition_key, "")
                return None
            
            return None
        except Exception as e:
            print(f"[WARN] _get_narration 오류: {e}")
            return None
    
    def _get_affection_tone(self, affection: int) -> str:
        """호감도 구간에 따른 말투 지시사항 반환 (chatbot_config.json에서만 읽어옴)"""
        from services.utils.prompt_builder import get_affection_tone
        return get_affection_tone(self.config, affection)

    def _analyze_sentiment_with_llm(self, user_message: str) -> int:
        """
        LLM을 사용하여 사용자 메시지의 긍정/부정 정도를 분석하고 호감도 변화량 반환
        반환값: -3 ~ +3 (부정적일수록 음수, 긍정적일수록 양수)
        """
        if not self.client:
            return 0
        
        try:
            sentiment_prompt = f"""다음 사용자 메시지를 분석하여 선생님(멘토)에 대한 태도가 얼마나 긍정적인지 판단해주세요.

사용자 메시지: "{user_message}"

이 메시지는:
- 매우 긍정적 (격려, 감사, 응원, 신뢰 표현 등): 3
- 긍정적 (협조적, 수용적, 관심 표현 등): 2
- 약간 긍정적 (중립적이지만 긍정적 경향): 1
- 중립적 (단순 질문, 정보 요청 등): 0
- 약간 부정적 (불만, 반대, 거부감 등): -1
- 부정적 (비판, 불신, 거리두기 등): -2
- 매우 부정적 (적대적, 공격적, 완전 거부 등): -3

숫자 하나만 답변해주세요 (예: 2, -1, 0 등)."""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 감정 분석 전문가입니다. 사용자 메시지의 긍정/부정 정도를 정확하게 판단해주세요."},
                    {"role": "user", "content": sentiment_prompt}
                ],
                temperature=0.3,  # 일관성 있는 판단을 위해 낮은 temperature
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip()
            # 숫자만 추출
            try:
                change = int(result)
                return max(-3, min(3, change))  # -3 ~ +3 범위로 제한
            except ValueError:
                # 숫자 파싱 실패 시, 텍스트에서 숫자 찾기
                import re
                numbers = re.findall(r'-?\d+', result)
                if numbers:
                    change = int(numbers[0])
                    return max(-3, min(3, change))
                return 0
        except Exception as e:
            print(f"[WARN] 감정 분석 실패: {e}")
            return 0
    
    def _get_state_context(self, game_state: str) -> str:
        """
        게임 상태에 따른 컨텍스트 프롬프트 반환 (state JSON에서 로드)
        """
        state_info = self._get_state_info(game_state)
        context = state_info.get("context", "")

        # 하위 호환성: 기존 상태명도 지원
        if not context and game_state == "ice_break":
            return self._get_state_context("start")
        elif not context and game_state == "mentoring":
            return self._get_state_context("icebreak")

        return context
    
    def _build_system_prompt(self) -> str:
        """시스템 프롬프트 생성 (캐릭터 설정, 역할 지침, 대화 예시 포함)"""
        from services.utils.prompt_builder import build_system_prompt
        return build_system_prompt(self.config)

    def _build_prompt(self, user_message: str, context: str = None, username: str = "사용자", affection: int = 5, game_state: str = "ice_break", selected_subjects: list = None, subject_selected: bool = False, schedule_set: bool = False, official_mock_exam_grade_info: dict = None):
        """LLM 프롬프트 구성 (호감도 및 게임 상태 반영)"""
        from services.utils.prompt_builder import build_user_prompt, get_affection_tone
        from services.utils.career_manager import get_career_description
        
        if selected_subjects is None:
            selected_subjects = []

        # 호감도 말투 추가
        affection_tone = get_affection_tone(self.config, affection)
        
        # 진로 정보 추가
        career = self._get_career(username)
        career_info = ""
        if career:
            career_desc = get_career_description(career)
            career_info = f"[진로 목표]\n당신의 진로 목표는 '{career}'입니다. ({career_desc})\n플레이어(멘토)가 진로에 대해 물어보면 자연스럽게 자신의 진로 목표와 그 이유, 그리고 그 진로를 향한 열정을 표현하세요."

        # 게임 상태 컨텍스트
        state_context = self._get_state_context(game_state)
        
        # 프롬프트 빌드
        user_prompt = build_user_prompt(
            user_message=user_message,
            context=context,
            username=username,
            game_state=game_state,
            state_context=state_context,
            selected_subjects=selected_subjects,
            schedule_set=schedule_set,
            official_mock_exam_grade_info=official_mock_exam_grade_info,
            current_week=self._get_current_week(username),
            last_mock_exam_week=self.mock_exam_last_week.get(username, -1)
        )
        
        # 호감도 말투와 진로 정보를 앞에 추가
        prompt_parts = []
        if affection_tone.strip():
            prompt_parts.append(affection_tone.strip())
        if career_info:
            prompt_parts.append(career_info)
        
        # 기존 프롬프트와 결합
        if prompt_parts:
            return "\n\n".join(prompt_parts) + "\n\n" + user_prompt
        return user_prompt
    
    
    def generate_response(self, user_message: str, username: str = "사용자") -> dict:
        """
        사용자 메시지에 대한 챗봇 응답 생성 (통합 파이프라인)
        호감도 시스템 및 게임 상태 시스템 포함
        """
        try:
            # [0] 영구 저장소에서 사용자 데이터 로드
            self._load_user_data(username)

            # [0.1] 현재 상태 가져오기
            current_affection = self._get_affection(username)
            current_state = self._get_game_state(username)
            
            # [1] 초기 메시지(인사)
            if user_message.strip().lower() == 'init':
                try:
                    bot_name = self.config.get('name', '챗봇') if self.config else '챗봇'
                    # 게임 상태 초기화
                    self._set_game_state(username, "start")
                    # 대화 횟수 초기화
                    self._reset_conversation_count(username)
                    # 주 초기화
                    self.current_weeks[username] = 0
                    # 게임 날짜 초기화
                    self._set_game_date(username, "2023-11-17")
                    # 체력과 멘탈 초기화 (명시적으로 설정)
                    self._set_stamina(username, 30)
                    self._set_mental(username, 40)
                    # 진로 초기화 (없으면 랜덤 생성)
                    from services.utils.career_manager import initialize_career_for_user
                    existing_career = self._get_career(username)
                    career = initialize_career_for_user(username, existing_career)
                    self._set_career(username, career)
                    # 호감도 확인 (초기값 5)
                    current_affection = self._get_affection(username)
                    # 나레이션 생성 (start state의 narration 사용)
                    try:
                        start_state_info = self._get_state_info("start")
                        narration = start_state_info.get("narration")
                    except Exception as e:
                        print(f"[WARN] 나레이션 생성 실패: {e}")
                        narration = None
                    
                    # 안전하게 모든 값 가져오기
                    try:
                        abilities = self._get_abilities(username)
                    except Exception as e:
                        print(f"[WARN] 능력치 가져오기 실패: {e}")
                        abilities = {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0}
                    
                    try:
                        stamina = self._get_stamina(username)
                    except Exception as e:
                        print(f"[WARN] 체력 가져오기 실패: {e}")
                        stamina = 30
                    
                    try:
                        mental = self._get_mental(username)
                    except Exception as e:
                        print(f"[WARN] 멘탈 가져오기 실패: {e}")
                        mental = 40
                    
                    return {
                        'reply': f"게임이 시작되었습니다.",
                        'image': None,
                        'affection': current_affection,
                        'game_state': "start",
                        'selected_subjects': [],
                        'narration': narration,
                        'abilities': abilities,
                        'schedule': {},
                        'current_date': "2023-11-17",
                        'stamina': stamina,
                        'mental': mental
                    }
                except Exception as e:
                    print(f"[ERROR] init 메시지 처리 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    # 최소한의 응답 반환
                    return {
                        'reply': "게임이 시작되었습니다.",
                        'image': None,
                        'affection': 5,
                        'game_state': "start",
                        'selected_subjects': [],
                        'narration': None,
                        'abilities': {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0},
                        'schedule': {},
                        'current_date': "2023-11-17",
                        'stamina': 30,
                        'mental': 40
                    }
            
            # [1.1] 게임 상태 초기화 요청 처리
            if user_message.strip() == "__RESET_GAME_STATE__":
                # 모든 게임 상태 초기화
                self._set_game_state(username, "start")
                self._set_affection(username, 5)
                self._set_stamina(username, 30)
                self._set_mental(username, 40)
                self._set_abilities(username, {
                    "국어": 0,
                    "수학": 0,
                    "영어": 0,
                    "탐구1": 0,
                    "탐구2": 0
                })
                self._set_selected_subjects(username, [])
                self._set_schedule(username, {})
                self._reset_conversation_count(username)
                self.current_weeks[username] = 0
                self._set_game_date(username, "2023-11-17")
                # 진로 재초기화 (랜덤 생성)
                from services.utils.career_manager import initialize_career_for_user
                career = initialize_career_for_user(username, None)
                self._set_career(username, career)

                # 나레이션 생성 (start state의 narration 사용)
                try:
                    start_state_info = self._get_state_info("start")
                    narration = start_state_info.get("narration")
                except Exception as e:
                    print(f"[WARN] 나레이션 생성 실패: {e}")
                    narration = None
                return {
                    'reply': "게임이 완전히 초기화되었습니다. 다시 시작하세요!",
                    'image': None,
                    'affection': 5,
                    'game_state': "start",
                    'selected_subjects': [],
                    'narration': narration,
                    'abilities': {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0},
                    'schedule': {},
                    'current_date': "2023-11-17",
                    'stamina': 30,
                    'mental': 40
                }
            
            # [1.2] 디버깅 전용 히든 명령어 처리
            debug_response = self._handle_debug_command(user_message, username, current_state, current_affection)
            if debug_response:
                return debug_response
            
            # [1.3] 프롬프트 공격 감지
            if self._check_prompt_injection(user_message):
                injection_cfg = self.config.get("prompt_injection_detection", {})
                block_message = injection_cfg.get("block_message", "죄송해요, 그런 말은 할 수 없어요. 게임을 정상적으로 플레이해주세요.")
                return {
                    'reply': block_message,
                    'image': None,
                    'affection': current_affection,
                    'game_state': current_state,
                    'selected_subjects': self._get_selected_subjects(username),
                    'narration': None,
                    'abilities': self._get_abilities(username),
                    'schedule': self._get_schedule(username),
                    'current_date': self._get_game_date(username),
                    'stamina': self._get_stamina(username)
                }
            
            # [1.5] LLM으로 사용자 메시지의 긍정/부정 분석하여 호감도 변화 계산
            try:
                affection_change = self._analyze_sentiment_with_llm(user_message)
            except Exception as e:
                print(f"[WARN] 감정 분석 실패: {e}")
                affection_change = 0  # 기본값
            
            # 호감도가 낮을수록 변화가 작게 (신뢰 없음)
            if current_affection < 30:
                affection_change = int(affection_change * 0.7)
            # 호감도가 높을수록 변화가 크게 (신뢰 있음)
            elif current_affection > 70:
                affection_change = int(affection_change * 1.2)
            else:
                affection_change = int(affection_change)
            
            # 호감도 업데이트
            new_affection = max(0, min(100, current_affection + affection_change))
            self._set_affection(username, new_affection)

            # state_changed 변수 초기화 (자동 전이 체크 전)
            state_changed = False
            narration = None
            reply = None  # reply 변수 초기화
            mentoring_end_reply = None  # 멘토링 종료 메시지 초기화
            original_reply_on_game_end = None  # game_ended일 때 엔딩 메시지 백업용
            
            # june_exam_intro_reply 변수 선언 (6exam 처리에서 사용)
            june_exam_intro_reply = None
            
            # [1.5.9] exam_strategy 상태 처리 (Handler 사용)
            exam_strategy_reply = None
            exam_strategy_processed = False  # 전략 처리가 완료되었는지 플래그
            exam_strategy_user_input = None  # LLM 호출 시 사용할 전략 텍스트
            if current_state == "exam_strategy":
                # Handler로 처리
                handler_result = self.handler_registry.call_handle(
                    'exam_strategy', username, user_message,
                    {'current_state': current_state}
                )
                if handler_result:
                    if handler_result.get('skip_llm'):
                        exam_strategy_processed = True
                        exam_strategy_reply = handler_result.get('reply')
                    else:
                        exam_strategy_processed = True
                        exam_strategy_user_input = handler_result.get('user_input')

                    if handler_result.get('narration'):
                        if not narration:
                            narration = handler_result['narration']
                        else:
                            narration = f"{narration}\n\n{handler_result['narration']}"
            
            # [1.6] 상태 전환 체크 (state machine 기반)
            state_changed, transition_narration = self._check_state_transition(
                username,
                new_affection,
                affection_change,  # 이번 턴 호감도 증가량 전달
                user_message  # 유저 입력 메시지 전달 (user_input 트리거용)
            )
            new_state = self._get_game_state(username)

            # 상태 전환 시 나레이션 사용 (이미 설정된 나레이션이 있으면 덮어쓰지 않음)
            if state_changed and transition_narration:
                if narration:
                    narration = f"{narration}\n\n{transition_narration}"
                else:
                    narration = transition_narration
            
            # 학습시간표 관리 상태로 전이될 때 특별한 메시지 생성
            study_schedule_transition_reply = None
            if state_changed and new_state == "study_schedule":
                study_schedule_transition_reply = "14시간 안에 어떻게 분배를 해야할까요?"
            
            # [1.6.5] "멘토링 종료" 트리거 처리 (어떤 상태에서든 가능)
            week_advanced = False
            week_advance_narration = None
            week_result = None
            if "멘토링 종료" in user_message:
                week_result = self._advance_one_week(username, mentoring_end=True)
                week_advanced = True
                
                # 정규 모의고사인 경우 자동 전이
                if week_result.get('exam'):
                    exam_result = week_result['exam']
                    exam_name = exam_result.get('name', '')
                    
                    # 11월 수능 여부 확인
                    is_csat = exam_name == "수능"
                    
                    if is_csat:
                        exam_month = "2024-11"
                    else:
                        exam_month_str = exam_name.replace('월 모의고사', '').replace('월', '').zfill(2)
                        exam_month = f"2024-{exam_month_str}" if exam_month_str else None
                    
                    # 11월 수능인 경우 11exam 상태로 전이
                    if exam_month and exam_month.endswith("-11"):
                        # 11exam 성적 정보 초기화
                        self.csat_exam_scores[username] = {
                            "scores": None  # handler에서 계산
                        }
                        
                        # 상태를 11exam으로 전이
                        self._set_game_state(username, "11exam")
                        new_state = "11exam"
                        state_changed = True
                        
                        # 수능 완료 안내
                        week_advance_narration = f"{week_result['week']}주차가 완료되었습니다. 수능이 끝났습니다."
                        print(f"[11EXAM] 멘토링 종료로 인한 수능 - 11exam 상태로 전이")
                    # 6월 모의고사인 경우 6exam 상태로 전이
                    elif exam_month and exam_month.endswith("-06"):
                        # 6exam 진행 정보 초기화 (전략 관련 정보 제거)
                        self.exam_progress[username] = {
                            "current_subject": None,
                            "subject_order": ["국어", "수학", "영어", "탐구1", "탐구2"],
                            "subjects_completed": []
                        }
                        
                        # 상태를 6exam으로 전이
                        self._set_game_state(username, "6exam")
                        new_state = "6exam"
                        state_changed = True
                        
                        # 6월 모의고사 완료 안내
                        week_advance_narration = f"{week_result['week']}주차가 완료되었습니다. 6월 모의고사가 끝났습니다."
                        print(f"[6EXAM] 멘토링 종료로 인한 6월 모의고사 - 6exam 상태로 전이")
                    elif exam_month and exam_month.endswith("-09"):
                        # 9월 모의고사인 경우 9exam 상태로 전이
                        self.september_exam_problems[username] = {
                            "current_subject": None,
                            "subject_order": ["국어", "수학", "영어", "탐구1", "탐구2"],
                            "subjects_completed": []
                        }
                        
                        # 상태를 9exam으로 전이
                        self._set_game_state(username, "9exam")
                        new_state = "9exam"
                        state_changed = True
                        
                        # 9월 모의고사 완료 안내
                        week_advance_narration = f"{week_result['week']}주차가 완료되었습니다. 9월 모의고사가 끝났습니다."
                        print(f"[9EXAM] 멘토링 종료로 인한 9월 모의고사 - 9exam 상태로 전이")
                    elif exam_month and self._is_official_mock_exam_month(exam_month):
                        # 취약점 식별
                        exam_scores = exam_result.get('scores', {})
                        if exam_scores:
                            weak_subject = self._identify_weak_subject(exam_scores)
                            weakness_message = self._generate_weakness_message(weak_subject, exam_scores.get(weak_subject, {}))
                            
                            # 취약점 정보 저장
                            self.official_mock_exam_weakness[username] = {
                                "subject": weak_subject,
                                "message": weakness_message,
                                "scores": exam_scores,
                                "exam_name": exam_name
                            }
                            
                            # 취약점 메시지를 먼저 reply로 표시 (서가윤이 먼저 취약점을 얘기함)
                            state_info = self._get_state_info("official_mock_exam_feedback")
                            state_name = state_info.get("name", "정규모의고사 피드백") if state_info else "정규모의고사 피드백"
                            official_mock_exam_weakness_reply = weakness_message
                            if not reply:
                                reply = f"[{state_name}] {weakness_message}"
                            
                            # 상태를 official_mock_exam_feedback으로 전이
                            self._set_game_state(username, "official_mock_exam_feedback")
                            new_state = "official_mock_exam_feedback"
                            state_changed = True
                            
                            # 성적표 나레이션 (등급대별 반응은 포함하지 않음)
                            week_advance_narration = week_result['exam']['text']
                            
                            # 등급대별 반응을 reply로 저장 (서가윤이 말함)
                            if 'grade_reaction' in week_result['exam']:
                                official_mock_exam_grade_reaction_reply = week_result['exam']['grade_reaction']
                            
                            print(f"[OFFICIAL_MOCK_EXAM] 멘토링 종료로 인한 {exam_name} 자동 전이. 취약 과목: {weak_subject}")
                        else:
                            week_advance_narration = f"{week_result['week']}주차가 완료되었습니다."
                            if week_result['exam']:
                                week_advance_narration += f"\n\n{week_result['exam']['text']}"
                    else:
                        week_advance_narration = f"{week_result['week']}주차가 완료되었습니다."
                        if week_result['exam']:
                            week_advance_narration += f"\n\n{week_result['exam']['text']}"
                else:
                    week_advance_narration = f"{week_result['week']}주차가 완료되었습니다."
                
                print(f"[TIME] {username}이(가) '멘토링 종료'를 입력하여 시간을 1주 진행했습니다.")
            
            # "멘토링 종료" 처리 시 나레이션 추가 (상태 전이 나레이션보다 우선)
            if week_advanced and week_advance_narration:
                if narration:
                    narration = f"{narration}\n\n{week_advance_narration}"
                else:
                    narration = week_advance_narration
                
                # 멘토링 종료 시 특별 메시지 (자동 전이되지 않는 경우에만)
                # new_state가 daily_routine이거나 state_changed가 False인 경우에만 메시지 표시
                if not state_changed:
                    mentoring_end_reply = "선생님, 저 그럼 공부하러 갈게요."
                    print(f"[MENTORING_END] 멘토링 종료 메시지 설정: {mentoring_end_reply}")
            
            # [1.7] 선택과목 선택 처리 (icebreak 단계에서만)
            selected_subjects = self._get_selected_subjects(username)
            subject_selected_in_this_turn = False
            subjects_completed = False  # 선택과목 2개 모두 선택 완료 여부

            if new_state in ["icebreak", "mentoring"]:  # icebreak 또는 하위호환성을 위한 mentoring
                # 사용자 메시지에서 선택과목 추출 (여러 개 가능)
                parsed_subjects = self._parse_subject_from_message(user_message)

                if parsed_subjects:
                    # 새로 선택할 과목들만 필터링
                    new_subjects = []
                    for subject in parsed_subjects:
                        if subject not in selected_subjects:
                            new_subjects.append(subject)

                    if new_subjects:
                        # 남은 슬롯만큼만 추가 (최대 2개)
                        remaining_slots = 2 - len(selected_subjects)
                        if remaining_slots > 0:
                            # 최대 남은 슬롯 수만큼만 추가
                            subjects_to_add = new_subjects[:remaining_slots]
                            selected_subjects.extend(subjects_to_add)
                            self._set_selected_subjects(username, selected_subjects)
                            subject_selected_in_this_turn = True

                            added_subjects_str = ", ".join(subjects_to_add)
                            print(f"[SUBJECT] {username}이(가) '{added_subjects_str}' 과목을 선택했습니다.")

                            # 선택과목 2개 모두 완료되었는지 확인
                            if len(selected_subjects) >= 2:
                                # state machine을 통해 상태 전이 체크
                                subjects_state_changed, subjects_transition_narration = self._check_state_transition(
                                    username,
                                    new_affection,
                                    affection_change,  # 호감도 증가량 전달
                                    user_message  # 유저 입력 메시지 전달 (user_input 트리거용)
                                )

                                if subjects_state_changed:
                                    subjects_completed = True
                                    new_state = self._get_game_state(username)
                                    # 기존 narration이 없으면 새 narration 사용
                                    if not narration and subjects_transition_narration:
                                        narration = subjects_transition_narration
                                    print(f"[STATE_TRANSITION] 선택과목 선택 완료! 상태가 {new_state}로 전환되었습니다.")
                        else:
                            print(f"[SUBJECT] 이미 2개의 과목을 선택했습니다.")
                    else:
                        # 이미 선택된 과목들만 언급된 경우
                        mentioned_subjects = ", ".join([s for s in parsed_subjects if s in selected_subjects])
                        print(f"[SUBJECT] 이미 선택한 과목입니다: {mentioned_subjects}")
                
                # 선택과목 목록 요청 확인
                if "탐구과목" in user_message or "선택과목" in user_message or "과목 선택" in user_message or "과목 목록" in user_message:
                    subjects_list = self._get_subject_list_text()
                    # 프롬프트에 선택과목 목록 추가될 수 있도록 처리
            
            # [1.7.5] 사설모의고사 응시 처리
            mock_exam_processed = False  # 사설모의고사 피드백 처리 플래그
            official_mock_exam_processed = False  # 정규모의고사 피드백 처리 플래그
            june_exam_processed = False  # 6월 모의고사 피드백 처리 플래그
            september_exam_processed = False  # 9월 모의고사 피드백 처리 플래그
            mock_exam_scores = None
            weak_subject = None
            weakness_message = None
            mock_exam_weakness_reply = None  # 취약점 메시지를 reply에 포함시키기 위한 변수
            mock_exam_advice_reply = None  # 조언에 대한 서가윤의 반응 (LLM 생성)
            mock_exam_advice_user_input = None  # 사용자의 조언 내용
            mock_exam_advice_is_good = None  # 조언 적절성 플래그
            official_mock_exam_advice_reply = None  # 정규모의고사 조언에 대한 서가윤의 반응 (LLM 생성)
            official_mock_exam_advice_user_input = None  # 정규모의고사 사용자의 조언 내용
            official_mock_exam_advice_is_good = None  # 정규모의고사 조언 적절성 플래그
            
            if new_state == "mock_exam" and current_state != "mock_exam":
                # Handler로 처리 (on_enter 사용)
                handler_result = self.handler_registry.call_on_enter(
                    'mock_exam', username,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    if handler_result.get('skip_llm'):
                        mock_exam_processed = True
                        reply = handler_result.get('reply')

                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed

                    # data 저장
                    if handler_result.get('data'):
                        mock_exam_scores = handler_result['data'].get('mock_exam_scores')
                        weak_subject = handler_result['data'].get('weak_subject')
                        weakness_message = handler_result['data'].get('weakness_message')
                        mock_exam_weakness_reply = weakness_message
                        mock_exam_grade_reaction_reply = handler_result['data'].get('grade_reaction')
            
            # [1.7.5.5] 6exam 상태 처리 (전략 수집 → 시험 진행 → 피드백)
            # june_exam_intro_reply는 이미 위에서 선언됨
            june_subject_problem_reply = None  # 과목별 문제점 메시지
            june_exam_student_thoughts = []  # 시험 중 학생의 생각들
            june_exam_grade_reaction_reply = None  # 6월 모의고사 등급대별 반응 (서가윤이 reply로 말함)
            june_exam_advice_reply = None  # 조언에 대한 서가윤의 반응 (LLM 생성)
            mock_exam_grade_reaction_reply = None  # 사설모의고사 등급대별 반응 (서가윤이 reply로 말함)
            official_mock_exam_grade_reaction_reply = None  # 정규모의고사 등급대별 반응 (서가윤이 reply로 말함)
            
            # 6exam 상태 처리 (질문 시 바로 성적 발표 → 피드백)
            if new_state == "6exam":
                # Handler로 처리
                handler_result = self.handler_registry.call_handle(
                    '6exam', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    if handler_result.get('skip_llm'):
                        june_exam_processed = True
                        reply = handler_result.get('reply')

                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed
            
            # [1.7.5.7] 9exam 상태 처리 (6exam과 동일한 로직)
            september_subject_problem_reply = None  # 과목별 문제점 메시지
            september_exam_grade_reaction_reply = None  # 9월 모의고사 등급대별 반응 (서가윤이 reply로 말함)
            september_exam_advice_reply = None  # 조언에 대한 서가윤의 반응 (LLM 생성)
            september_exam_intro_reply = None
            
            # 9exam 상태 처리 (Handler 사용)
            if new_state == "9exam":
                handler_result = self.handler_registry.call_handle(
                    '9exam', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    if handler_result.get('skip_llm'):
                        september_exam_processed = True
                        reply = handler_result.get('reply')

                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed
            
            # [1.7.5.8] university_application 상태 처리 (대학 지원 및 엔딩)
            university_application_processed = False
            game_ended = False
            
            # university_application 상태 진입 시 (on_enter 호출)
            if new_state == "university_application" and current_state != "university_application":
                handler_result = self.handler_registry.call_on_enter(
                    'university_application', username,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    if handler_result.get('skip_llm'):
                        university_application_processed = True
                        reply = handler_result.get('reply')

                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed
            
            # university_application 상태에서 사용자 입력 처리
            if new_state == "university_application" or current_state == "university_application":
                handler_result = self.handler_registry.call_handle(
                    'university_application', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    # game_ended 플래그 확인
                    if handler_result.get('game_ended'):
                        game_ended = True
                    
                    # skip_llm이 True이고 reply가 None이 아닐 때만 처리 완료로 간주
                    # reply가 None이면 LLM 호출이 필요함
                    handler_reply = handler_result.get('reply')
                    if handler_result.get('skip_llm') and handler_reply is not None:
                        university_application_processed = True
                        reply = handler_reply
                        print(f"[UNIVERSITY_APPLICATION] handler에서 받은 reply 설정: '{reply[:100] if reply else 'None'}...'")
                    elif handler_reply is not None:
                        # skip_llm이 False이거나 없지만 reply가 있는 경우
                        reply = handler_reply
                        print(f"[UNIVERSITY_APPLICATION] handler에서 받은 reply 설정 (skip_llm=False): '{reply[:100] if reply else 'None'}...'")
                    
                    # game_ended이고 reply가 있으면 반드시 보존 (엔딩 메시지)
                    if handler_result.get('game_ended') and handler_reply:
                        reply = handler_reply  # 엔딩 메시지 강제 설정
                        # 엔딩 메시지 백업 (다른 로직에 의해 변경되는 것을 방지)
                        original_reply_on_game_end = handler_reply
                        print(f"[UNIVERSITY_APPLICATION] game_ended=True, 엔딩 reply 강제 설정: '{reply[:100] if reply else 'None'}...'")
                        print(f"[UNIVERSITY_APPLICATION] 엔딩 reply 백업 완료 (길이: {len(handler_reply) if handler_reply else 0})")

                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed
            
            # [1.7.5.8] 11exam 상태 처리 (수능)
            csat_exam_processed = False
            if new_state == "11exam":
                print(f"[DEBUG] 11exam 핸들러 호출: user_message={user_message}")
                handler_result = self.handler_registry.call_handle(
                    '11exam', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                print(f"[DEBUG] 11exam 핸들러 결과: handler_result={handler_result}")
                if handler_result:
                    if handler_result.get('skip_llm'):
                        csat_exam_processed = True
                        reply = handler_result.get('reply')
                    
                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed
            
            # [1.7.5.6] 6exam_feedback 상태 처리 (Handler 사용)
            if new_state == "6exam_feedback":
                handler_result = self.handler_registry.call_handle(
                    '6exam_feedback', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    if handler_result.get('skip_llm'):
                        june_exam_processed = True

                    if handler_result.get('reply'):
                        june_exam_advice_reply = handler_result.get('reply')

                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed

                    if handler_result.get('subject_problem_reply'):
                        june_subject_problem_reply = handler_result['subject_problem_reply']

            # [1.7.5.8] 9exam_feedback 상태 처리 (Handler 사용)
            if new_state == "9exam_feedback":
                handler_result = self.handler_registry.call_handle(
                    '9exam_feedback', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    if handler_result.get('skip_llm'):
                        september_exam_processed = True

                    if handler_result.get('reply'):
                        september_exam_advice_reply = handler_result.get('reply')

                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed

                    if handler_result.get('subject_problem_reply'):
                        september_subject_problem_reply = handler_result['subject_problem_reply']

            # [1.7.6] 사설모의고사 피드백 처리 (Handler 사용)
            if current_state == "mock_exam_feedback" and new_state != "mock_exam":
                handler_result = self.handler_registry.call_handle(
                    'mock_exam_feedback', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    # 재응시 요청 처리
                    if handler_result.get('retest'):
                        self._set_game_state(username, 'mock_exam')
                        new_state = 'mock_exam'
                        state_changed = True
                        mock_exam_processed = True

                    # LLM 호출 건너뛰기 설정
                    if handler_result.get('skip_llm') is not None:
                        if not handler_result.get('skip_llm'):
                            # LLM으로 반응 생성해야 함
                            mock_exam_advice_user_input = handler_result.get('advice_user_input')
                            mock_exam_advice_is_good = handler_result.get('advice_is_good')
                            mock_exam_advice_reply = None  # 나중에 LLM으로 생성
                        else:
                            mock_exam_processed = True

                    # 즉시 reply가 있으면 설정
                    if handler_result.get('reply'):
                        reply = handler_result.get('reply')
                        mock_exam_processed = True

                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed

            # [1.7.7] 정규모의고사 피드백 처리 (Handler 사용)
            official_mock_exam_grade_info = None  # 등급 정보 저장 (프롬프트에 추가용)
            if new_state == "official_mock_exam_feedback" or current_state == "official_mock_exam_feedback":
                # 등급 정보 계산 (프롬프트에 추가하기 위해)
                weakness_info = self.official_mock_exam_weakness.get(username, {})
                exam_scores = weakness_info.get("scores", {})
                if exam_scores:
                    average_grade = self._calculate_average_grade(exam_scores)
                    grade_reaction_text = self._generate_grade_reaction("official_mock_exam", average_grade)
                    official_mock_exam_grade_info = {
                        "average_grade": average_grade,
                        "grade_reaction": grade_reaction_text,
                        "scores": exam_scores
                    }

                # Handler 호출
                handler_result = self.handler_registry.call_handle(
                    'official_mock_exam_feedback', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    # LLM 호출 건너뛰기 설정
                    if handler_result.get('skip_llm') is not None:
                        if not handler_result.get('skip_llm'):
                            # LLM으로 반응 생성해야 함
                            official_mock_exam_advice_user_input = handler_result.get('advice_user_input')
                            official_mock_exam_advice_is_good = handler_result.get('advice_is_good')
                            official_mock_exam_advice_reply = None  # 나중에 LLM으로 생성
                        else:
                            official_mock_exam_processed = True

                    # 즉시 reply가 있으면 설정
                    if handler_result.get('reply'):
                        reply = handler_result.get('reply')
                        official_mock_exam_processed = True

                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed

            # [1.7.9] 일상루틴 단계에서 운동/휴식 조언 처리 (제거됨 - 운동은 시간표에서만 처리)
            stamina_recovered = False
            # 운동 조언에 따른 체력 증가 로직 제거

            # [1.7.10] 탐구과목 선택 처리 (selection 상태에서만, Handler 사용)
            subjects_selected = False
            selected_subjects = None
            if new_state == "selection" or current_state == "selection":
                # Handler로 처리
                handler_result = self.handler_registry.call_handle(
                    'selection', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    if handler_result.get('subjects_selected'):
                        subjects_selected = True
                        selected_subjects = handler_result.get('subjects')
                    
                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed
                        print(f"[STATE_TRANSITION] 탐구과목 선택 완료로 인해 {transition_to} 상태로 전이했습니다.")

            # [1.8] 시간표 처리 (학습 시간표 관리 상태에서만, Handler 사용)
            schedule_updated = False
            week_passed = False
            current_schedule = None  # 변수 선언
            if new_state == "study_schedule" or current_state == "study_schedule":
                # Handler로 처리
                handler_result = self.handler_registry.call_handle(
                    'study_schedule', username, user_message,
                    {'current_state': current_state, 'new_state': new_state}
                )
                if handler_result:
                    if handler_result.get('schedule_updated'):
                        schedule_updated = True
                        current_schedule = handler_result.get('schedule')
                    
                    # 헬퍼로 narration 및 전이 처리
                    narration, transition_to, handler_state_changed = self._process_handler_result(handler_result, narration)
                    if transition_to:
                        self._set_game_state(username, transition_to)
                        new_state = transition_to
                        state_changed = handler_state_changed
                        print(f"[STATE_TRANSITION] 시간표 설정 완료로 인해 {transition_to} 상태로 복귀했습니다.")
            
            # daily_routine 상태에서 대화 횟수 증가 등의 처리
            if new_state == "daily_routine":
                # 현재 시간표 가져오기
                if 'current_schedule' not in locals():
                    current_schedule = self._get_schedule(username)
                
                # 대화 횟수 증가 (init 메시지 제외)
                if user_message.strip().lower() != 'init':
                    self._increment_conversation_count(username)
                    conv_count = self._get_conversation_count(username)
                    print(f"[CONVERSATION] {username}의 대화 횟수: {conv_count}/5")
                    
                    # 대화 5번 후 자동으로 1주일 경과 처리
                    if conv_count >= 5:
                        # 주 증가 (먼저 증가해서 현재 주차 표시)
                        self._increment_week(username)
                        current_week = self._get_current_week(username)
                        
                        # 시간표에 따라 능력치 증가
                        if current_schedule:
                            self._apply_schedule_to_abilities(username)
                            print(f"[WEEK] {username}의 1주일이 경과했습니다. 능력치가 증가했습니다.")
                            print(f"[ABILITIES] 현재 능력치: {self._get_abilities(username)}")
                        
                        # 1주 경과 시 체력 -1
                        current_stamina = self._get_stamina(username)
                        new_stamina = max(0, current_stamina - 1)
                        self._set_stamina(username, new_stamina)
                        print(f"[STAMINA] {username}의 체력이 {current_stamina}에서 {new_stamina}로 변경되었습니다. (1주 경과로 -1)")
                        
                        # 대화 횟수 초기화
                        self._reset_conversation_count(username)
                        
                        # 날짜 7일 증가
                        current_date = self._get_game_date(username)
                        new_date = self._add_days_to_date(current_date, 7)
                        self._set_game_date(username, new_date)
                        
                        week_passed = True
                        
                        # 1주 기간 동안 시험이 있었는지 확인 (현재 날짜부터 7일 후까지)
                        exam_month = self._check_exam_in_period(current_date, new_date)
                        exam_scores = None
                        exam_scores_text = ""
                        
                        if exam_month:
                            # 시험 성적 계산
                            exam_scores = self._calculate_exam_scores(username, exam_month)
                            exam_name = "수능" if exam_month.endswith("-11") else f"{exam_month[-2:]}월 모의고사"
                            
                            # 11월 수능인 경우 11exam 상태로 전이
                            if exam_month and exam_month.endswith("-11"):
                                # 11exam 성적 정보 초기화
                                self.csat_exam_scores[username] = {
                                    "scores": None  # handler에서 계산
                                }
                                
                                # 상태를 11exam으로 전이
                                self._set_game_state(username, "11exam")
                                new_state = "11exam"
                                state_changed = True
                                
                                # 수능 완료 안내
                                print(f"[11EXAM] {username}의 수능 자동 진입 - 11exam 상태로 전이")
                            # 6월 모의고사인 경우 6exam 상태로 전이
                            elif exam_month and exam_month.endswith("-06"):
                                # 6exam 진행 정보 초기화 (전략 관련 정보 제거)
                                self.exam_progress[username] = {
                                    "current_subject": None,
                                    "subject_order": ["국어", "수학", "영어", "탐구1", "탐구2"],
                                    "subjects_completed": []
                                }
                                
                                # 상태를 6exam으로 전이
                                self._set_game_state(username, "6exam")
                                new_state = "6exam"
                                state_changed = True
                                
                                # 6월 모의고사 완료 안내
                                print(f"[6EXAM] {username}의 6월 모의고사 자동 진입 - 6exam 상태로 전이")
                            elif exam_month and exam_month.endswith("-09"):
                                # 9월 모의고사인 경우 9exam 상태로 전이
                                self.september_exam_problems[username] = {
                                    "current_subject": None,
                                    "subject_order": ["국어", "수학", "영어", "탐구1", "탐구2"],
                                    "subjects_completed": []
                                }
                                
                                # 상태를 9exam으로 전이
                                self._set_game_state(username, "9exam")
                                new_state = "9exam"
                                state_changed = True
                                
                                # 9월 모의고사 완료 안내
                                print(f"[9EXAM] {username}의 9월 모의고사 자동 진입 - 9exam 상태로 전이")
                            # 정규 모의고사인 경우 자동으로 official_mock_exam_feedback으로 전이
                            elif self._is_official_mock_exam_month(exam_month):
                                # 취약점 식별
                                weak_subject = self._identify_weak_subject(exam_scores)
                                weakness_message = self._generate_weakness_message(weak_subject, exam_scores.get(weak_subject, {}))
                                
                                # 취약점 정보 저장
                                self.official_mock_exam_weakness[username] = {
                                    "subject": weak_subject,
                                    "message": weakness_message,
                                    "scores": exam_scores,
                                    "exam_name": exam_name
                                }
                                
                                # 취약점 메시지를 먼저 reply로 표시 (서가윤이 먼저 취약점을 얘기함)
                                state_info = self._get_state_info("official_mock_exam_feedback")
                                state_name = state_info.get("name", "정규모의고사 피드백") if state_info else "정규모의고사 피드백"
                                official_mock_exam_weakness_reply = weakness_message
                                if not reply:
                                    reply = f"[{state_name}] {weakness_message}"
                                
                                # 성적표 나레이션 생성
                                subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
                                score_lines = []
                                for subject in subjects:
                                    if subject in exam_scores:
                                        score_data = exam_scores[subject]
                                        score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                                
                                # 평균 등급 계산 및 반응 생성 (서가윤이 reply로 말함)
                                average_grade = self._calculate_average_grade(exam_scores)
                                grade_reaction = self._generate_grade_reaction("official_mock_exam", average_grade)
                                
                                # 나레이션에는 성적표만 포함
                                exam_scores_text = f"\n\n{exam_name} 성적이 발표되었습니다:\n" + "\n".join(score_lines)
                                
                                # 등급대별 반응은 서가윤이 reply로 말함 (나중에 reply에 추가)
                                official_mock_exam_grade_reaction_reply = grade_reaction
                                
                                # 나레이션에 성적표 추가
                                if not narration:
                                    narration = exam_scores_text.strip()
                                else:
                                    narration = f"{narration}\n\n{exam_scores_text.strip()}"
                                
                                # 상태를 official_mock_exam_feedback으로 전이
                                self._set_game_state(username, "official_mock_exam_feedback")
                                new_state = "official_mock_exam_feedback"
                                state_changed = True
                                print(f"[OFFICIAL_MOCK_EXAM] {username}의 {exam_name} 자동 전이. 취약 과목: {weak_subject}")
                            else:
                                # 정규 모의고사가 아닌 경우 기존 로직
                                exam_scores_text = f"\n\n{exam_name} 성적이 발표되었습니다:\n"
                                
                                subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
                                score_lines = []
                                for subject in subjects:
                                    if subject in exam_scores:
                                        score_data = exam_scores[subject]
                                        score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                                
                                exam_scores_text += "\n".join(score_lines)
                        
                        # 나레이션 메시지 (6월, 9월, 11월, 정규 모의고사가 아닌 경우에만)
                        if exam_month:
                            if exam_month.endswith("-06") or exam_month.endswith("-09") or exam_month.endswith("-11"):
                                # 6월, 9월, 11월 모의고사인 경우 성적표 나레이션만 (이미 위에서 설정됨)
                                pass  # narration은 이미 위에서 설정됨
                            elif self._is_official_mock_exam_month(exam_month):
                                # 정규 모의고사인 경우 성적표 나레이션만 (이미 위에서 설정됨)
                                pass  # narration은 이미 위에서 설정됨
                            else:
                                # 일반 시험인 경우
                                narration = f"{current_week}주차가 완료되었습니다. 다시 일상 루틴 단계입니다. 다음 중 하나를 입력하여 다음 행동을 선택하세요. '학습시간표 관리','사설모의고사 응시','멘토링 종료'"
                        if exam_scores_text:
                            narration += exam_scores_text
                        else:
                            # 시험이 없는 경우
                            narration = f"{current_week}주차가 완료되었습니다. 다시 일상 루틴 단계입니다. 다음 중 하나를 입력하여 다음 행동을 선택하세요. '학습시간표 관리','사설모의고사 응시','멘토링 종료'"
            
            # [2] RAG 검색
            try:
                context, similarity, metadata = self._search_similar(
                    query=user_message,
                    threshold=0.45,
                    top_k=5
                )
                has_context = (context is not None)
            except Exception as e:
                print(f"[WARN] RAG 검색 실패: {e}")
                context, similarity, metadata = None, None, None
                has_context = False
            
            # [3] 프롬프트 구성 (업데이트된 호감도 및 게임 상태 반영)
            current_schedule_for_prompt = self._get_schedule(username)
            schedule_set = bool(current_schedule_for_prompt)
            
            prompt = self._build_prompt(
                user_message=user_message,
                context=context,
                username=username,
                affection=new_affection,
                game_state=new_state,
                selected_subjects=selected_subjects if new_state == "mentoring" else [],
                subject_selected=subject_selected_in_this_turn,
                schedule_set=schedule_set,
                official_mock_exam_grade_info=official_mock_exam_grade_info
            )
            
            # 선택과목 목록 요청 시 프롬프트에 추가
            if new_state in ["icebreak", "mentoring"] and ("탐구과목" in user_message or "선택과목" in user_message or "과목 선택" in user_message or "과목 목록" in user_message):
                subjects_list = self._get_subject_list_text()
                prompt += f"\n\n[선택과목 목록]\n{subjects_list}\n\n사용자가 위 목록 중에서 선택과목을 고를 수 있도록 안내하세요. (최대 2개)"
            
            # reply 변수 초기화
            reply = None
            
            # [3.5] 대화 5번 후 자동 처리 (LLM 호출 전)
            if week_passed:
                # 호감도에 따른 공부하러 가는 메시지 생성
                auto_study_message = self._get_study_message_by_affection(new_affection)
                reply = auto_study_message
                
                # 주차 완료 메시지에도 상태 접두사 추가
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {reply}"
                # 나레이션도 추가
                if narration is None:
                    current_week = self._get_current_week(username)
                    narration = f"{current_week}주차가 완료되었습니다. 설정한 공부 시간만큼 실력이 향상되었어요!"
                # 주차 완료 시 시험 점수도 확인
                exam_month = self._get_current_exam_month(username)
                if exam_month:
                    exam_scores = self._calculate_exam_scores(username, exam_month)
                    if exam_scores:
                        exam_name = "수능" if exam_month.endswith("-11") else f"{exam_month[-2:]}월 모의고사"
                        exam_scores_text = f"\n\n{exam_name} 성적이 발표되었습니다:\n"
                        subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
                        score_lines = []
                        for subject in subjects:
                            if subject in exam_scores:
                                score_data = exam_scores[subject]
                                score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                        if score_lines:
                            exam_scores_text += "\n".join(score_lines)
                            narration += exam_scores_text
            if not week_passed:
                # [4] LLM 응답 생성
                print(f"\n{'='*50}")
                print(f"[USER] {username}: {user_message}")
                print(f"[GAME_STATE] {current_state}" + (f" → {new_state}" if state_changed else ""))
                print(f"[AFFECTION] {current_affection} → {new_affection} (변화: {affection_change:+.1f})")
                print(f"[RAG] Context found: {has_context}")
                if has_context:
                    print(f"[RAG] Similarity: {similarity:.4f}")
                    print(f"[RAG] Context: {str(context)[:100]}...")
                print(f"[LLM] Calling API...")
                
                # mock_exam_feedback 또는 official_mock_exam_feedback에서 이미 처리된 경우 LLM 호출 건너뛰기
                # 또는 6exam/9exam 상태에서 질문이 아닌 경우, 6exam_feedback/9exam_feedback에서 조언 처리 중인 경우
                # 또는 게임이 종료된 경우 (university_application에서 합격 처리 완료)
                processed = False
                if game_ended:
                    processed = True
                    # game_ended일 때는 handler의 reply를 사용 (이미 설정됨)
                    if not reply:
                        print(f"[WARN] [GAME_ENDED] 게임 종료되었지만 reply가 없습니다. 기본 메시지 사용.")
                        reply = "감사합니다. 멘토 덕분에 여기까지 올 수 있었어요."
                    print(f"[GAME_ENDED] 게임 종료 - LLM 호출 건너뛰기 (reply: '{reply[:50] if reply else 'None'}...')")
                elif mock_exam_processed or official_mock_exam_processed or june_exam_processed or september_exam_processed:
                    processed = True
                    if june_exam_processed:
                        print("[6EXAM] 6월 모의고사 처리 중 - LLM 호출 건너뛰기")
                    elif september_exam_processed:
                        print("[9EXAM] 9월 모의고사 처리 중 - LLM 호출 건너뛰기")
                    else:
                        print("[MOCK_EXAM_FEEDBACK] 피드백 처리 완료 - LLM 호출 건너뛰기")
                elif university_application_processed:
                    processed = True
                    print("[UNIVERSITY_APPLICATION] 대학 지원 처리 완료 - LLM 호출 건너뛰기")
                
                # exam_strategy 상태에서 전략 입력 시 전용 LLM 호출
                if exam_strategy_processed and exam_strategy_user_input:
                    if not self.client:
                        exam_strategy_reply = "알겠습니다."
                    else:
                        try:
                            # 전략에 대한 자연스러운 응답 생성
                            system_prompt = "당신은 서가윤입니다. 선생님(멘토)이 시험 전략을 알려주면 그 전략을 이해하고 확신하거나 확인하는 것처럼 자연스럽게 응답하세요."
                            strategy_prompt = f"선생님이 '{exam_strategy_user_input}'라고 시험 전략을 알려주셨습니다. 이 전략을 이해했다는 의미의 자연스러운 응답을 해주세요. (30자 이내로 간단히)"
                            
                            response = self.client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": strategy_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=100
                            )
                            
                            if response and response.choices and len(response.choices) > 0:
                                exam_strategy_reply = response.choices[0].message.content.strip()
                                if not exam_strategy_reply:
                                    exam_strategy_reply = "알겠습니다."
                            else:
                                exam_strategy_reply = "알겠습니다."
                        except Exception as e:
                            print(f"[ERROR] 시험 전략 응답 생성 실패: {e}")
                            exam_strategy_reply = "알겠습니다."
                    
                    processed = True
                    print("[EXAM_STRATEGY] 전략 처리 완료 - 전용 LLM 호출로 응답 생성")
                
                # mock_exam_feedback에서 조언 처리 완료 시 서가윤의 반응 생성
                if mock_exam_advice_user_input is not None and mock_exam_advice_reply is None:
                    if not self.client:
                        if mock_exam_advice_is_good:
                            mock_exam_advice_reply = "감사해요! 좋은 조언이었어요."
                        else:
                            mock_exam_advice_reply = "음... 알겠습니다."
                    else:
                        try:
                            # 조언에 대한 자연스러운 응답 생성
                            if mock_exam_advice_is_good:
                                system_prompt = "당신은 서가윤입니다. 선생님(멘토)의 좋은 조언을 듣고 감사하고 기뻐하는 반응을 자연스럽게 표현하세요."
                                advice_prompt = f"선생님이 '{mock_exam_advice_user_input}'라고 조언을 해주셨고, 이 조언이 도움이 되었습니다. 감사하고 기뻐하는 반응을 해주세요. (30자 이내로 간단히)"
                            else:
                                system_prompt = "당신은 서가윤입니다. 선생님(멘토)의 부적절한 조언을 듣고 당황하거나 어색해하는 반응을 자연스럽게 표현하세요."
                                advice_prompt = f"선생님이 '{mock_exam_advice_user_input}'라고 조언을 해주셨지만, 이 조언이 도움이 되지 않았습니다. 당황하거나 어색해하는 반응을 해주세요. (30자 이내로 간단히)"
                            
                            response = self.client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": advice_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=100
                            )
                            
                            if response and response.choices and len(response.choices) > 0:
                                mock_exam_advice_reply = response.choices[0].message.content.strip()
                                if not mock_exam_advice_reply:
                                    mock_exam_advice_reply = "감사해요!" if mock_exam_advice_is_good else "알겠습니다..."
                            else:
                                mock_exam_advice_reply = "감사해요!" if mock_exam_advice_is_good else "알겠습니다..."
                        except Exception as e:
                            print(f"[ERROR] 조언 반응 생성 실패: {e}")
                            mock_exam_advice_reply = "감사해요!" if mock_exam_advice_is_good else "알겠습니다..."
                    
                    processed = True
                    print(f"[MOCK_EXAM_ADVICE] 조언 반응 생성 완료: {mock_exam_advice_reply}")
                
                # official_mock_exam_feedback에서 조언 처리 완료 시 서가윤의 반응 생성
                if official_mock_exam_advice_user_input is not None and official_mock_exam_advice_reply is None:
                    if not self.client:
                        if official_mock_exam_advice_is_good:
                            official_mock_exam_advice_reply = "감사해요! 좋은 조언이었어요."
                        else:
                            official_mock_exam_advice_reply = "음... 알겠습니다."
                    else:
                        try:
                            # 조언에 대한 자연스러운 응답 생성
                            if official_mock_exam_advice_is_good:
                                system_prompt = "당신은 서가윤입니다. 선생님(멘토)의 좋은 조언을 듣고 감사하고 기뻐하는 반응을 자연스럽게 표현하세요."
                                advice_prompt = f"선생님이 '{official_mock_exam_advice_user_input}'라고 조언을 해주셨고, 이 조언이 도움이 되었습니다. 감사하고 기뻐하는 반응을 해주세요. (30자 이내로 간단히)"
                            else:
                                system_prompt = "당신은 서가윤입니다. 선생님(멘토)의 부적절한 조언을 듣고 당황하거나 어색해하는 반응을 자연스럽게 표현하세요."
                                advice_prompt = f"선생님이 '{official_mock_exam_advice_user_input}'라고 조언을 해주셨지만, 이 조언이 도움이 되지 않았습니다. 당황하거나 어색해하는 반응을 해주세요. (30자 이내로 간단히)"
                            
                            response = self.client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": advice_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=100
                            )
                            
                            if response and response.choices and len(response.choices) > 0:
                                official_mock_exam_advice_reply = response.choices[0].message.content.strip()
                                if not official_mock_exam_advice_reply:
                                    official_mock_exam_advice_reply = "감사해요!" if official_mock_exam_advice_is_good else "알겠습니다..."
                            else:
                                official_mock_exam_advice_reply = "감사해요!" if official_mock_exam_advice_is_good else "알겠습니다..."
                        except Exception as e:
                            print(f"[ERROR] 정규모의고사 조언 반응 생성 실패: {e}")
                            official_mock_exam_advice_reply = "감사해요!" if official_mock_exam_advice_is_good else "알겠습니다..."
                    
                    processed = True
                    print(f"[OFFICIAL_MOCK_EXAM_ADVICE] 조언 반응 생성 완료: {official_mock_exam_advice_reply}")
                
                if not self.client and not processed:
                    # OpenAI Client 확인
                    print("[WARN] OpenAI Client가 초기화되지 않았습니다. 기본 응답을 반환합니다.")
                    reply = "죄송해요, 현재 AI 서비스에 연결할 수 없어요. 잠시 후 다시 시도해주세요."
                    # 기본 메시지에도 상태 접두사 추가
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    reply = f"[{state_name}] {reply}"
                elif not processed:
                    try:
                        # 시스템 프롬프트 생성 (캐릭터 설정, 역할 지침, 대화 예시 포함)
                        system_prompt = self._build_system_prompt()

                        response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        if not response or not response.choices or len(response.choices) == 0:
                            print("[WARN] LLM 응답이 비어있습니다.")
                            reply = "죄송해요, 응답을 생성할 수 없어요. 다시 시도해주세요."
                            # 에러 메시지에도 상태 접두사 추가
                            state_info = self._get_state_info(new_state)
                            state_name = state_info.get("name", new_state)
                            reply = f"[{state_name}] {reply}"
                        else:
                            reply = response.choices[0].message.content
                            if not reply or not reply.strip():
                                reply = "죄송해요, 응답을 생성할 수 없어요. 다시 시도해주세요."
                                # 빈 응답 에러 메시지에도 상태 접두사 추가
                                state_info = self._get_state_info(new_state)
                                state_name = state_info.get("name", new_state)
                                reply = f"[{state_name}] {reply}"
                            else:
                                # 응답 앞에 [state명] 추가
                                state_info = self._get_state_info(new_state)
                                state_name = state_info.get("name", new_state)
                                reply = f"[{state_name}] {reply}"
                                
                                # 일상루틴 단계에서 시간표가 이미 설정된 경우, 시간표 관련 내용 필터링
                                if new_state == "daily_routine" and schedule_set:
                                    schedule_keywords = ["시간표", "시간 분배", "학습 시간", "시간 관리", "시간표 관리", 
                                                         "시간을", "시간이", "시간으로", "시간으로는", "시간을 설정",
                                                         "공부 시간", "공부시간", "시간표를", "시간표가", "시간표에"]
                                    reply_lower = reply.lower()
                                    # 시간표 관련 키워드가 포함되어 있으면 기본 응답으로 대체
                                    if any(keyword in reply_lower for keyword in schedule_keywords):
                                        # LLM 응답에서 시간표 관련 부분을 제거하거나 기본 응답으로 대체
                                        print(f"[SCHEDULE_FILTER] 시간표 관련 키워드 감지, 응답 필터링: {reply[:100]}")
                                        # 기본 응답 유지 (시간표 관련 내용을 자연스럽게 피하도록 이미 프롬프트에 지시했으므로)
                                        # 만약 LLM이 계속 시간표를 언급하면 여기서 추가 필터링 가능
                    except Exception as e:
                        print(f"[ERROR] LLM 호출 실패: {e}")
                        import traceback
                        traceback.print_exc()
                        reply = "죄송해요, 일시적인 오류가 발생했어요. 다시 시도해주세요."
                        # 에러 메시지에도 상태 접두사 추가
                        if reply:
                            state_info = self._get_state_info(new_state)
                            state_name = state_info.get("name", new_state)
                            reply = f"[{state_name}] {reply}"
            
            # 학습시간표 관리 상태로 전이될 때 특별한 메시지 처리
            # 단, game_ended일 때는 handler의 reply를 보존하기 위해 건너뛰기
            if study_schedule_transition_reply and not game_ended:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {study_schedule_transition_reply}"
            
            # 멘토링 종료 시 특별 메시지 처리 (정규 모의고사나 6exam_feedback으로 전이되지 않는 경우에만)
            # 단, game_ended일 때는 handler의 reply를 보존하기 위해 건너뛰기
            if week_advanced and mentoring_end_reply and new_state != "6exam_feedback" and not game_ended:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {mentoring_end_reply}"
                print(f"[MENTORING_END] 멘토링 종료 메시지 적용: {reply}")
            
            # reply가 없으면 기본 메시지 추가 (상태 접두사 포함)
            # 단, game_ended일 때는 handler의 reply가 반드시 있어야 함
            if not reply:
                if game_ended:
                    print(f"[WARN] [GAME_ENDED] 게임 종료되었지만 reply가 없습니다. 기본 메시지 사용.")
                    reply = "감사합니다. 멘토 덕분에 여기까지 올 수 있었어요."
                else:
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    reply = f"[{state_name}]"
            
            # game_ended일 때 reply가 있는지 최종 확인 및 로그
            # 만약 다른 로직에 의해 reply가 변경되었다면 원래 reply로 복원
            if game_ended:
                if original_reply_on_game_end and reply != original_reply_on_game_end:
                    print(f"[WARN] [GAME_ENDED] reply가 변경되었습니다. 원래 reply로 복원합니다.")
                    print(f"[WARN] 원래 reply: '{original_reply_on_game_end[:100]}...'")
                    print(f"[WARN] 변경된 reply: '{reply[:100] if reply else 'None'}...'")
                    reply = original_reply_on_game_end
                print(f"[GAME_ENDED] 최종 reply 확인: '{reply[:150] if reply else 'None'}...'")
                print(f"[GAME_ENDED] reply 길이: {len(reply) if reply else 0}")
            
            # 상태 전환 시 나레이션은 별도로 반환 (프론트엔드에서 처리)
            # reply에는 추가 메시지 없음 (나레이션으로 처리)
            
            # 선택과목 선택 시 확인 메시지
            if subject_selected_in_this_turn:
                current_selected = self._get_selected_subjects(username)
                if len(current_selected) == 2:
                    subjects_text = ", ".join(current_selected)
                    reply += f"\n\n(선택과목이 모두 선택되었습니다: {subjects_text})"
                elif len(current_selected) == 1:
                    reply += f"\n\n(선택과목 '{current_selected[0]}'이(가) 선택되었습니다. {2 - len(current_selected)}개 더 선택할 수 있어요.)"
                else:
                    # 여러 개 한번에 선택된 경우 (이론적으로는 발생하지 않지만 안전장치)
                    subjects_text = ", ".join(current_selected)
                    if len(current_selected) < 2:
                        reply += f"\n\n(선택과목 {subjects_text}이(가) 선택되었습니다. {2 - len(current_selected)}개 더 선택할 수 있어요.)"
                    else:
                        reply += f"\n\n(선택과목이 모두 선택되었습니다: {subjects_text})"
            
            # 선택과목 완료 시 나레이션은 이미 state machine에서 설정됨
            # (subjects_completed는 더 이상 필요하지 않음 - state machine이 처리)
            
            # game_ended일 때는 handler의 reply를 보존하기 위해 모든 추가 메시지 처리 건너뛰기
            if not game_ended:
                # 등급대별 반응을 reply에 추가 (서가윤이 성적에 대해 말함)
                # 6월 모의고사 등급대별 반응 (문제점 메시지가 없을 때만 추가)
                if june_exam_grade_reaction_reply and not june_subject_problem_reply and not june_exam_advice_reply:
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    if reply:
                        reply = f"[{state_name}] {june_exam_grade_reaction_reply}\n\n{reply}"
                    else:
                        reply = f"[{state_name}] {june_exam_grade_reaction_reply}"
                    print(f"[6EXAM_GRADE_REACTION] 등급대별 반응 설정: {reply}")
            
            # 사설모의고사 등급대별 반응을 취약점 메시지 다음에 추가
            if mock_exam_grade_reaction_reply and new_state == "mock_exam_feedback":
                # mock_exam_feedback 상태에서 취약점 메시지 다음에 등급대별 반응 추가
                if reply and mock_exam_weakness_reply and mock_exam_weakness_reply in reply:
                    # 취약점 메시지가 있으면 그 다음에 등급대별 반응 추가
                    reply = reply.replace(mock_exam_weakness_reply, f"{mock_exam_weakness_reply}\n\n{mock_exam_grade_reaction_reply}")
                elif reply:
                    # 취약점 메시지가 없으면 등급대별 반응만 추가
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    reply = f"[{state_name}] {mock_exam_grade_reaction_reply}\n\n{reply}"
                else:
                    # reply가 없으면 등급대별 반응만 표시
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    reply = f"[{state_name}] {mock_exam_grade_reaction_reply}"
                print(f"[MOCK_EXAM_GRADE_REACTION] 등급대별 반응 추가: {reply}")
            
            # 정규모의고사 등급대별 반응
            if official_mock_exam_grade_reaction_reply:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                if reply:
                    reply = f"[{state_name}] {official_mock_exam_grade_reaction_reply}\n\n{reply}"
                else:
                    reply = f"[{state_name}] {official_mock_exam_grade_reaction_reply}"
                print(f"[OFFICIAL_MOCK_EXAM_GRADE_REACTION] 등급대별 반응 설정: {reply}")
            
            # 시험 전략 수립 상태 메시지 처리
            if exam_strategy_reply and new_state in ["exam_strategy", "daily_routine"]:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {exam_strategy_reply}"
                print(f"[EXAM_STRATEGY] 전략 메시지 설정: {reply}")
            
            # 사설모의고사 취약점 메시지를 가장 먼저 표시 (우선순위 최우선)
            # mock_exam 상태에서 이미 설정되었지만, 혹시 누락되면 여기서 보장
            if mock_exam_weakness_reply and (new_state == "mock_exam_feedback" or new_state == "mock_exam"):
                # reply가 비어있거나 취약점 메시지가 없으면 추가
                if not reply or (mock_exam_weakness_reply not in reply):
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state) if state_info else "사설모의고사 응시"
                    if not reply:
                        reply = f"[{state_name}] {mock_exam_weakness_reply}"
                    else:
                        # 기존 reply 앞에 취약점 메시지 추가 (최우선)
                        if reply.startswith("[") and "]" in reply:
                            prefix_end = reply.find("]") + 1
                            prefix = reply[:prefix_end]
                            body = reply[prefix_end:].strip()
                            reply = f"{prefix} {mock_exam_weakness_reply}\n\n{body}"
                        else:
                            reply = f"{mock_exam_weakness_reply}\n\n{reply}"
                    print(f"[MOCK_EXAM_WEAKNESS] 취약점 메시지 우선 표시 (보장): {reply}")
            
            # 사설모의고사 조언에 대한 서가윤의 반응 처리 (일상루틴단계에서 표시)
            if mock_exam_advice_reply and new_state == "daily_routine":
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {mock_exam_advice_reply}"
                print(f"[MOCK_EXAM_ADVICE] 조언 반응 메시지 설정: {reply}")
            
            # 정규모의고사 조언에 대한 서가윤의 반응 처리 (일상루틴단계에서 표시)
            if official_mock_exam_advice_reply and new_state == "daily_routine":
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {official_mock_exam_advice_reply}"
                print(f"[OFFICIAL_MOCK_EXAM_ADVICE] 조언 반응 메시지 설정: {reply}")
            
            # 6월 모의고사 조언에 대한 서가윤의 반응 처리 (최우선)
            # 조언 반응이 있으면 이것을 먼저 표시하고, 다음 과목 문제점도 함께 표시
            if june_exam_advice_reply:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                # 조언 반응과 다음 과목 문제점을 함께 표시
                if june_subject_problem_reply:
                    reply = f"[{state_name}] {june_exam_advice_reply}\n\n{june_subject_problem_reply}"
                    print(f"[6EXAM_ADVICE] 조언 반응 + 다음 과목 문제점 함께 표시: {reply}")
                else:
                    reply = f"[{state_name}] {june_exam_advice_reply}"
                    print(f"[6EXAM_ADVICE] 조언 반응 메시지 설정: {reply}")
            # 6월 모의고사 과목별 문제점 메시지를 reply에 추가 (조언 반응이 없을 때만)
            elif june_subject_problem_reply:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {june_subject_problem_reply}"
                print(f"[6EXAM_SUBJECT_PROBLEM] 문제점 메시지 설정: {reply}")
            # 6월 모의고사 초기 메시지 (조언 반응과 문제점 메시지가 모두 없을 때만)
            elif june_exam_intro_reply and (new_state == "6exam_feedback" or new_state == "6exam"):
                # 6exam 또는 6exam_feedback 상태로 전이될 때 초기 메시지를 reply로 설정
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {june_exam_intro_reply}"
                print(f"[6EXAM_INTRO] 초기 메시지 설정: {reply}")
            
            # 9월 모의고사 등급대별 반응을 reply에 추가 (6exam과 동일)
            if september_exam_grade_reaction_reply and not september_subject_problem_reply and not september_exam_advice_reply:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                if reply:
                    reply = f"[{state_name}] {september_exam_grade_reaction_reply}\n\n{reply}"
                else:
                    reply = f"[{state_name}] {september_exam_grade_reaction_reply}"
                print(f"[9EXAM_GRADE_REACTION] 등급대별 반응 설정: {reply}")
            
            # 9월 모의고사 조언 반응을 reply에 추가 (6exam과 동일)
            if september_exam_advice_reply:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                # 조언 반응과 다음 과목 문제점을 함께 표시
                if september_subject_problem_reply:
                    reply = f"[{state_name}] {september_exam_advice_reply}\n\n{september_subject_problem_reply}"
                    print(f"[9EXAM_ADVICE] 조언 반응 + 다음 과목 문제점 함께 표시: {reply}")
                else:
                    reply = f"[{state_name}] {september_exam_advice_reply}"
                    print(f"[9EXAM_ADVICE] 조언 반응 메시지 설정: {reply}")
            # 9월 모의고사 과목별 문제점 메시지를 reply에 추가 (조언 반응이 없을 때만)
            elif september_subject_problem_reply:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {september_subject_problem_reply}"
                print(f"[9EXAM_SUBJECT_PROBLEM] 문제점 메시지 설정: {reply}")
            # 9월 모의고사 초기 메시지 (조언 반응과 문제점 메시지가 모두 없을 때만)
            elif september_exam_intro_reply and (new_state == "9exam_feedback" or new_state == "9exam"):
                # 9exam 또는 9exam_feedback 상태로 전이될 때 초기 메시지를 reply로 설정
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {september_exam_intro_reply}"
                print(f"[9EXAM_INTRO] 초기 메시지 설정: {reply}")
            
            # 정규모의고사 취약점 메시지를 reply에 추가 (접두사 유지)
            # official_mock_exam_feedback 상태로 전환된 직후에도 취약점 메시지가 없으면 추가
            if new_state == "official_mock_exam_feedback":
                # 취약점 정보 가져오기
                weakness_info = self.official_mock_exam_weakness.get(username, {})
                official_mock_exam_weakness_reply = weakness_info.get("message")
                
                # reply가 비어있거나 취약점 메시지가 없으면 추가
                if official_mock_exam_weakness_reply:
                    if not reply or (official_mock_exam_weakness_reply not in reply):
                        state_info = self._get_state_info(new_state)
                        state_name = state_info.get("name", new_state) if state_info else "정규모의고사 피드백"
                        if not reply:
                            reply = f"[{state_name}] {official_mock_exam_weakness_reply}"
                        else:
                            # 기존 reply 앞에 취약점 메시지 추가 (최우선)
                            if reply.startswith("[") and "]" in reply:
                                prefix_end = reply.find("]") + 1
                                prefix = reply[:prefix_end]
                                body = reply[prefix_end:].strip()
                                reply = f"{prefix} {official_mock_exam_weakness_reply}\n\n{body}"
                            else:
                                reply = f"{official_mock_exam_weakness_reply}\n\n{reply}"
                        print(f"[OFFICIAL_MOCK_EXAM_WEAKNESS] 취약점 메시지 우선 표시 (보장): {reply}")
            
            # 사설모의고사 취약점 메시지를 reply에 추가 (접두사 유지)
            # mock_exam_feedback 상태로 전환된 직후에도 취약점 메시지가 없으면 추가
            if mock_exam_weakness_reply:
                # mock_exam_feedback 상태에서 reply가 비어있거나 취약점 메시지가 없으면 추가
                if new_state == "mock_exam_feedback":
                    if not reply or (mock_exam_weakness_reply not in reply):
                        state_info = self._get_state_info(new_state)
                        state_name = state_info.get("name", new_state)
                        if not reply:
                            reply = f"[{state_name}] {mock_exam_weakness_reply}"
                        else:
                            # 기존 reply 앞에 취약점 메시지 추가
                            if reply.startswith("[") and "]" in reply:
                                prefix_end = reply.find("]") + 1
                                prefix = reply[:prefix_end]
                                body = reply[prefix_end:].strip()
                                reply = f"{prefix} {mock_exam_weakness_reply}\n\n{body}"
                            else:
                                reply = f"{mock_exam_weakness_reply}\n\n{reply}"
                        print(f"[MOCK_EXAM_WEAKNESS] mock_exam_feedback 상태에서 취약점 메시지 추가: {reply}")
                elif new_state != "mock_exam_feedback":
                    # reply가 이미 있으면 추가, 없으면 취약점 메시지로 시작
                    if reply:
                        # reply에 이미 접두사가 있을 수 있으므로, 접두사가 있으면 유지
                        if reply.startswith("[") and "]" in reply:
                            # 접두사와 본문 분리
                            prefix_end = reply.find("]") + 1
                            prefix = reply[:prefix_end]
                            body = reply[prefix_end:].strip()
                            reply = f"{prefix} {mock_exam_weakness_reply}\n\n{body}"
                        else:
                            reply = f"{mock_exam_weakness_reply}\n\n{reply}"
                    else:
                        # reply가 없으면 취약점 메시지에 접두사 추가
                        state_info = self._get_state_info(new_state)
                        state_name = state_info.get("name", new_state)
                        reply = f"[{state_name}] {mock_exam_weakness_reply}"
            
            # 시간표 업데이트 시 확인 메시지
            # game_ended일 때는 건너뛰기
            if schedule_updated and not week_passed and not game_ended:
                schedule = self._get_schedule(username)
                schedule_text = ", ".join([f"{k} {v}시간" for k, v in schedule.items()])
                total = sum(schedule.values())
                reply += f"\n\n(시간표가 설정되었습니다: {schedule_text} (총 {total}시간))"
            
            # 대화 횟수 안내 (daily_routine 상태이고 시간표가 설정된 경우)
            # game_ended일 때는 건너뛰기
            if new_state == "daily_routine" and not week_passed and not game_ended:
                conv_count = self._get_conversation_count(username)
                schedule = self._get_schedule(username)
                if schedule:
                    remaining = 5 - conv_count
                    if remaining > 0:
                        reply += f"\n\n(대화 {remaining}번 후 1주일이 지나며 능력치가 증가합니다.)"
            
            # 최종 안전장치: reply에 접두사가 없으면 추가 (study_schedule 등 모든 상태에서)
            # 단, university_application 상태이거나 game_ended인 경우에는 접두사 추가하지 않음 (서가윤의 직접적인 반응이므로)
            if reply and not (reply.startswith("[") and reply.find("]") > 0 and reply.find("]") < 50):
                if new_state != "university_application" and not game_ended:
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    reply = f"[{state_name}] {reply}"
            
            print(f"[BOT] {reply}")
            print(f"{'='*50}\n")
            
            # [5] 메모리 저장(선택)
            if self.memory:
                self.memory.save_context(
                    {"input": user_message},
                    {"output": reply}
                )

            # [5.5] 엔딩 상태의 이미지 설정
            # 엔딩 상태(to_states가 빈 리스트)인 경우 state JSON에 정의된 이미지를 사용
            response_image = None
            try:
                state_info = self._get_state_info(new_state)
                if state_info:
                    # 엔딩 state 체크: to_states가 비어있거나 state 이름에 ending이 포함된 경우
                    to_states = state_info.get('to_states', [])
                    state_name = state_info.get('name', new_state)
                    if not to_states or 'ending' in new_state.lower():
                        # 엔딩 상태인 경우 state JSON에 정의된 이미지 사용
                        state_image = state_info.get('image')
                        if state_image:
                            # 이미지 경로 앞에 /가 없으면 추가
                            if not state_image.startswith('/'):
                                response_image = '/' + state_image
                            else:
                                response_image = state_image
                            print(f"[ENDING_IMAGE] {new_state} 엔딩 이미지 설정: {response_image}")
            except Exception as e:
                print(f"[WARN] 엔딩 이미지 설정 중 오류: {e}")
                response_image = None

            # [6] 응답 반환 (호감도, 게임 상태, 선택과목, 나레이션, 능력치, 시간표, 날짜, 체력 포함)
            return {
                'reply': reply,
                'image': response_image,
                'affection': new_affection,
                'game_state': new_state,
                'selected_subjects': self._get_selected_subjects(username),
                'narration': narration,
                'abilities': self._get_abilities(username),
                'schedule': self._get_schedule(username),
                'current_date': self._get_game_date(username),
                'stamina': self._get_stamina(username),
                'mental': self._get_mental(username),
                'game_ended': game_ended  # 엔딩 플래그 (university_application에서 설정)
            }
        except Exception as e:
            import traceback
            print(f"[ERROR] 응답 생성 실패: {e}")
            print(f"[ERROR] Traceback:")
            traceback.print_exc()
            try:
                current_affection = self._get_affection(username)
                current_state = self._get_game_state(username)
                selected_subjects = self._get_selected_subjects(username)
                abilities = self._get_abilities(username)
                schedule = self._get_schedule(username)
                current_date = self._get_game_date(username)
                stamina = self._get_stamina(username)
            except Exception as inner_e:
                print(f"[ERROR] 오류 복구 중 추가 오류: {inner_e}")
                # 기본값 사용
                current_affection = 5
                current_state = "ice_break"
                selected_subjects = []
                abilities = {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0}
                schedule = {}
                current_date = "2023-11-17"
                stamina = 30
            
            return {
                'reply': f"죄송해요, 일시적인 오류가 발생했어요. 다시 시도해주세요.",
                'image': None,
                'affection': current_affection,
                'game_state': current_state,
                'selected_subjects': selected_subjects,
                'narration': None,
                'abilities': abilities,
                'schedule': schedule,
                'current_date': current_date,
                'stamina': stamina
            }


# ============================================================================
# 싱글톤 패턴
# ============================================================================
# ChatbotService 인스턴스를 앱 전체에서 재사용
# (매번 새로 초기화하면 비효율적)

_chatbot_service = None

def get_chatbot_service():
    """
    챗봇 서비스 인스턴스 반환 (싱글톤)
    
    첫 호출 시 인스턴스 생성, 이후 재사용
    """
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service


# ============================================================================
# 테스트용 메인 함수
# ============================================================================

if __name__ == "__main__":
    """
    로컬 테스트용
    
    실행 방법:
    python services/chatbot_service.py
    """
    print("챗봇 서비스 테스트")
    print("=" * 50)
    
    service = get_chatbot_service()
    
    # 초기화 테스트
    response = service.generate_response("init", "테스터")
    print(f"초기 응답: {response}")
    
    # 일반 대화 테스트
    response = service.generate_response("안녕하세요!", "테스터")
    print(f"응답: {response}")
