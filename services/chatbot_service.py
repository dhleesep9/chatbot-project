"""
🎯 챗봇 서비스 - 구현 파일

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

        # 1.7. Trigger Registry 초기화 (자동으로 모든 트리거 로드)
        from services.triggers.trigger_registry import TriggerRegistry
        self.trigger_registry = TriggerRegistry()
        print(f"[ChatbotService] trigger registry loaded: {self.trigger_registry.list_triggers()}")

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

        # 9.6. 사설모의고사 취약점 정보 저장 (피드백용)
        self.mock_exam_weakness = {}  # {username: {"subject": str, "message": str}}
        print("[ChatbotService] 사설모의고사 취약점 저장 시스템 초기화 완료")
        
        # 9.7. 정규모의고사 취약점 정보 저장 (피드백용)
        self.official_mock_exam_weakness = {}  # {username: {"subject": str, "message": str}}
        print("[ChatbotService] 정규모의고사 취약점 저장 시스템 초기화 완료")
        
        # 9.8. 6월 모의고사 문제점 추적 시스템
        # {username: {"scores": {...}, "subjects": {"국어": {"problem": str, "solved": bool}, ...}, "current_subject": str, "completed_count": int}}
        self.june_exam_problems = {}
        print("[ChatbotService] 6월 모의고사 문제점 추적 시스템 초기화 완료")
        
        # 9. 대화 횟수 추적 (daily_routine 상태에서만)
        self.conversation_counts = {}  # {username: count}
        print("[ChatbotService] 대화 횟수 시스템 초기화 완료")

        # 10. 현재 주(week) 추적
        self.current_weeks = {}  # {username: week_number}
        print("[ChatbotService] 주(week) 추적 시스템 초기화 완료")

        # 11. 게임 날짜 저장
        self.game_dates = {}  # {username: "2023-11-17"}
        print("[ChatbotService] 게임 날짜 시스템 초기화 완료")

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
        """
        사용자 게임 데이터를 JSON 파일로 저장
        """
        try:
            user_data = {
                "affection": self._get_affection(username),
                "game_state": self._get_game_state(username),
                "abilities": self._get_abilities(username),
                "selected_subjects": self._get_selected_subjects(username),
                "schedule": self._get_schedule(username),
                "conversation_count": self._get_conversation_count(username),
                "current_week": self._get_current_week(username),
                "game_date": self._get_game_date(username),
                "stamina": self._get_stamina(username),
                "mental": self._get_mental(username)
            }

            user_file = BASE_DIR / f"data/users/{username}.json"
            user_file.parent.mkdir(parents=True, exist_ok=True)

            with open(user_file, "w", encoding="utf-8") as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)

            print(f"[STORAGE] {username} 데이터 저장 완료")
        except Exception as e:
            print(f"[ERROR] {username} 데이터 저장 실패: {e}")

    def _load_user_data(self, username: str):
        """
        사용자 게임 데이터를 JSON 파일에서 로드
        """
        try:
            user_file = BASE_DIR / f"data/users/{username}.json"

            if not user_file.exists():
                print(f"[STORAGE] {username} 저장 파일 없음 (새 유저)")
                return

            with open(user_file, "r", encoding="utf-8") as f:
                user_data = json.load(f)

            # 데이터 로드
            self.affections[username] = user_data.get("affection", 5)
            self.game_states[username] = user_data.get("game_state", "start")
            self.abilities[username] = user_data.get("abilities", {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0})
            self.selected_subjects[username] = user_data.get("selected_subjects", [])
            self.schedules[username] = user_data.get("schedule", {})
            self.conversation_counts[username] = user_data.get("conversation_count", 0)
            self.current_weeks[username] = user_data.get("current_week", 0)
            self.game_dates[username] = user_data.get("game_date", "2023-11-17")
            self.staminas[username] = user_data.get("stamina", 30)
            self.mentals[username] = user_data.get("mental", 40)

            print(f"[STORAGE] {username} 데이터 로드 완료")
        except Exception as e:
            print(f"[ERROR] {username} 데이터 로드 실패: {e}")

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
        사용자의 체력 설정
        """
        self.staminas[username] = max(0, stamina)  # 체력은 0 이상
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
        """
        체력에 따른 능력치 증가 효율 계산
        공식: 효율(%) = 100 + (체력 - 30)
        예시:
        - 체력 30: 100%
        - 체력 31: 101%
        - 체력 29: 99%
        - 체력 20: 90%
        - 체력 100: 170%
        """
        return 100 + (stamina - 30)
    
    def _calculate_mental_efficiency(self, mental: int) -> float:
        """
        멘탈에 따른 능력치 증가 효율 계산
        공식: 효율(%) = 100 + (멘탈 - 40)
        예시:
        - 멘탈 40: 100%
        - 멘탈 50: 110%
        - 멘탈 30: 90%
        - 멘탈 100: 160%
        """
        return 100 + (mental - 40)
    
    def _calculate_combined_efficiency(self, stamina: int, mental: int) -> float:
        """
        체력과 멘탈의 곱연산으로 최종 효율 계산
        공식: (체력 효율 * 멘탈 효율) / 100
        예시:
        - 체력 31(101%), 멘탈 50(110%): 101 * 110 / 100 = 111.1%
        - 체력 30(100%), 멘탈 40(100%): 100 * 100 / 100 = 100%
        """
        stamina_eff = self._calculate_stamina_efficiency(stamina)
        mental_eff = self._calculate_mental_efficiency(mental)
        return (stamina_eff * mental_eff) / 100.0
    
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
        
        # 트리거가 등록되어 있는지 확인
        if not self.trigger_registry.has_trigger(trigger_type):
            print(f"[WARN] Trigger type '{trigger_type}' not found in registry. Available triggers: {self.trigger_registry.list_triggers()}")
            return False

        # 트리거 실행 컨텍스트 구성
        context = {
            'username': username,
            'user_message': user_message,
            'affection_increased': affection_increased,
            'current_state': self._get_game_state(username),
            'june_exam_problems': getattr(self, 'june_exam_problems', {}),
            'service': self  # 트리거가 서비스 메서드에 접근할 수 있도록
        }
        
        print(f"[TRIGGER_EVAL] Evaluating trigger '{trigger_type}' with user_message: '{user_message}'")

        # 트리거 레지스트리를 통해 동적으로 트리거 실행
        result = self.trigger_registry.evaluate_trigger(trigger_type, transition, context)
        print(f"[TRIGGER_EVAL] Trigger '{trigger_type}' result: {result}")
        
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

        # 현재 상태 정보 가져오기 (별도 JSON에서 로드)
        state_info = self._get_state_info(current_state)
        transitions = state_info.get("transitions", [])
        print(f"[STATE_CHECK] Found {len(transitions)} transitions for {current_state}")

        # 각 전이 조건 확인
        for transition in transitions:
            trigger_type = transition.get('trigger_type')
            next_state = transition.get('next_state')
            print(f"[STATE_CHECK] Checking transition: {trigger_type} -> {next_state}")
            
            result = self._evaluate_transition_condition(username, transition, affection_increased, user_message)
            print(f"[STATE_CHECK] Transition evaluation result: {result} for trigger_type '{trigger_type}'")
            
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
    
    def _apply_schedule_to_abilities(self, username: str):
        """
        시간표에 따라 능력치 증가
        시간당 +1 증가 (체력에 따른 효율 적용)
        """
        schedule = self._get_schedule(username)
        if not schedule:
            return
        
        abilities = self._get_abilities(username)
        stamina = self._get_stamina(username)
        mental = self._get_mental(username)
        efficiency = self._calculate_combined_efficiency(stamina, mental) / 100.0  # 효율을 배율로 변환 (1.0 = 100%)
        
        for subject, hours in schedule.items():
            if subject in abilities:
                # 체력과 멘탈의 곱연산 효율 적용: 시간 * 효율
                increased = hours * efficiency
                abilities[subject] = min(2500, abilities[subject] + increased)  # 최대 2500
        
        self._set_abilities(username, abilities)
    
    def _advance_one_week(self, username: str) -> dict:
        """
        1주일을 진행시키는 통합 메서드
        시간표에 따라 능력치를 증가시키고, 날짜와 주차를 업데이트합니다.
        
        Returns:
            dict: 시험 결과 정보 (시험이 있었으면 포함)
        """
        current_schedule = self._get_schedule(username)
        current_date = self._get_game_date(username)
        
        # 시간표에 따라 능력치 증가
        if current_schedule:
            self._apply_schedule_to_abilities(username)
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
        """
        능력치를 백분위로 변환
        공식: 2 * sqrt(능력치)
        """
        import math
        if ability <= 0:
            return 0.0
        percentile = 2 * math.sqrt(ability)
        return min(100.0, percentile)  # 최대 100%
    
    def _calculate_grade_from_percentile(self, percentile: float) -> int:
        """
        백분위를 등급으로 변환 (수능 등급 체계)
        1등급: 96~100
        2등급: 89~95
        3등급: 77~88
        4등급: 60~76
        5등급: 40~59
        6등급: 23~39
        7등급: 11~22
        8등급: 4~10
        9등급: 1~3
        """
        if percentile >= 96:
            return 1
        elif percentile >= 89:
            return 2
        elif percentile >= 77:
            return 3
        elif percentile >= 60:
            return 4
        elif percentile >= 40:
            return 5
        elif percentile >= 23:
            return 6
        elif percentile >= 11:
            return 7
        elif percentile >= 4:
            return 8
        else:
            return 9
    
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
    
    def _calculate_exam_scores(self, username: str, exam_month: str) -> dict:
        """
        능력치를 기반으로 시험 성적 계산
        반환값: {"국어": {"grade": 1, "percentile": 85.5}, "수학": {"grade": 2, "percentile": 90.2}, ...}
        """
        abilities = self._get_abilities(username)
        scores = {}
        
        for subject, ability in abilities.items():
            percentile = self._calculate_percentile(ability)
            grade = self._calculate_grade_from_percentile(percentile)
            scores[subject] = {
                "grade": grade,
                "percentile": round(percentile, 1)
            }
        
        print(f"[EXAM] {username}의 {exam_month} 시험 성적 계산: {scores}")
        return scores
    
    def _calculate_mock_exam_scores(self, username: str) -> dict:
        """
        사설모의고사 성적 계산 (능력치 기반)
        """
        abilities = self._get_abilities(username)
        scores = {}
        
        for subject, ability in abilities.items():
            percentile = self._calculate_percentile(ability)
            grade = self._calculate_grade_from_percentile(percentile)
            scores[subject] = {
                "grade": grade,
                "percentile": round(percentile, 1)
            }
        
        print(f"[MOCK_EXAM] {username}의 사설모의고사 성적 계산: {scores}")
        return scores
    
    def _is_official_mock_exam_month(self, exam_month: str) -> bool:
        """
        정규모의고사 월인지 확인 (3, 4, 5, 7, 10월)
        6월, 9월, 수능(11월)은 False 반환
        """
        if not exam_month:
            return False
        try:
            month_num = int(exam_month.split("-")[1])
            return month_num in [3, 4, 5, 7, 10]
        except:
            return False
    
    def _identify_weak_subject(self, exam_scores: dict) -> str:
        """
        시험 점수에서 가장 취약한 과목 식별 (등급이 가장 낮은 과목)
        """
        if not exam_scores:
            return "수학"  # 기본값
        
        # 등급이 가장 높은(숫자가 큰) 과목을 취약 과목으로 선택
        weak_subject = max(exam_scores.items(), key=lambda x: x[1]['grade'])
        return weak_subject[0]
    
    def _generate_weakness_message(self, subject: str, score_data: dict) -> str:
        """
        취약 과목에 대한 취약점 메시지 생성 (과목별 다양한 예시)
        """
        weakness_examples = {
            "국어": [
                "국어에서 선택과목 시간에 시간을 다 써버려서 비문학 지문을 제대로 읽지 못했어요...",
                "국어에서 문학 작품 해석이 너무 어려웠어요. 작가의 의도를 파악하지 못했어요.",
                "국어 비문학 지문이 너무 길어서 읽는 속도가 느렸어요. 시간이 부족했어요.",
                "국어에서 고전 문학 부분을 제대로 이해하지 못했어요. 한자어가 많아서 어려웠어요.",
                "국어 화법 작문에서는 시간이 부족해서 대충 썼어요. 구조화된 글쓰기가 어려웠어요."
            ],
            "수학": [
                "수학에서 미적분 문제를 풀다가 시간이 너무 많이 걸렸어요...",
                "수학 기하 문제에서 도형을 그려도 풀이 방법이 생각이 안 났어요.",
                "수학에서 확률과 통계 부분을 완전히 틀렸어요. 경우의 수를 세는 게 헷갈렸어요.",
                "수학에서 삼각함수 문제가 너무 어려웠어요. 공식을 외웠는데 적용이 안 됐어요.",
                "수학 계산 실수가 너무 많았어요. 과정은 맞는데 답이 틀렸어요."
            ],
            "영어": [
                "영어에서 독해 지문을 읽고 문제를 풀 때 시간이 부족했어요...",
                "영어 어휘 문제에서 모르는 단어가 너무 많아서 문맥으로 유추했는데 틀렸어요.",
                "영어 문법 문제를 풀 때 시제를 헷갈려서 틀렸어요.",
                "영어에서 빈칸 채우기 문제가 어려웠어요. 문맥을 파악하지 못했어요.",
                "영어 작문 문제에서 표현이 자연스럽지 않아서 점수를 많이 깎였어요."
            ],
            "탐구1": [
                "탐구1에서 개념 문제는 알겠는데, 응용 문제가 너무 어려웠어요...",
                "탐구1에서 실험 문제를 풀 때 실험 과정을 제대로 이해하지 못했어요.",
                "탐구1에서 그래프 분석 문제가 헷갈렸어요. 데이터를 읽는 게 어려웠어요.",
                "탐구1에서 서술형 문제에서 답은 맞는데 표현이 부족해서 점수를 못 받았어요.",
                "탐구1에서 선택지가 비슷비슷해서 구분하기가 어려웠어요."
            ],
            "탐구2": [
                "탐구2에서 시간 분배가 안 되어서 마지막 문제들을 대충 풀었어요...",
                "탐구2에서 개념 연결 문제가 너무 어려웠어요. 서로 다른 개념을 연결하는 게 힘들었어요.",
                "탐구2에서 계산 문제에서 단위 변환을 실수했어요.",
                "탐구2에서 문제 해석이 어려웠어요. 문제가 뭘 요구하는지 모르겠었어요.",
                "탐구2에서 기출 문제는 풀었는데, 새로 나온 유형은 전혀 몰랐어요."
            ]
        }
        
        import random
        examples = weakness_examples.get(subject, weakness_examples["수학"])
        return random.choice(examples)
    
    def _generate_june_subject_problem(self, subject: str, score_data: dict) -> str:
        """
        6월 모의고사 과목별 문제점 메시지 생성
        """
        problem_examples = {
            "국어": [
                "작품 해석이 너무 어려웠어요. 작가의 의도를 파악하지 못했어요.",
                "선택과목 시간에 시간을 다 써버려서 비문학 지문을 제대로 읽지 못했어요.",
                "비문학 지문이 너무 길어서 읽는 속도가 느렸어요. 시간이 부족했어요.",
                "고전 문학 부분을 제대로 이해하지 못했어요. 한자어가 많아서 어려웠어요.",
                "화법 작문에서는 시간이 부족해서 대충 썼어요. 구조화된 글쓰기가 어려웠어요."
            ],
            "수학": [
                "미적분 문제를 풀다가 시간이 너무 많이 걸렸어요.",
                "기하 문제에서 도형을 그려도 풀이 방법이 생각이 안 났어요.",
                "확률과 통계 부분을 완전히 틀렸어요. 경우의 수를 세는 게 헷갈렸어요.",
                "삼각함수 문제가 너무 어려웠어요. 공식을 외웠는데 적용이 안 됐어요.",
                "계산 실수가 너무 많았어요. 과정은 맞는데 답이 틀렸어요."
            ],
            "영어": [
                "독해 지문을 읽고 문제를 풀 때 시간이 부족했어요.",
                "어휘 문제에서 모르는 단어가 너무 많아서 문맥으로 유추했는데 틀렸어요.",
                "문법 문제를 풀 때 시제를 헷갈려서 틀렸어요.",
                "빈칸 채우기 문제가 어려웠어요. 문맥을 파악하지 못했어요.",
                "작문 문제에서 표현이 자연스럽지 않아서 점수를 많이 깎였어요."
            ],
            "탐구1": [
                "개념 문제는 알겠는데, 응용 문제가 너무 어려웠어요.",
                "실험 문제를 풀 때 실험 과정을 제대로 이해하지 못했어요.",
                "그래프 분석 문제가 헷갈렸어요. 데이터를 읽는 게 어려웠어요.",
                "서술형 문제에서 답은 맞는데 표현이 부족해서 점수를 못 받았어요.",
                "선택지가 비슷비슷해서 구분하기가 어려웠어요."
            ],
            "탐구2": [
                "시간 분배가 안 되어서 마지막 문제들을 대충 풀었어요.",
                "개념 연결 문제가 너무 어려웠어요. 서로 다른 개념을 연결하는 게 힘들었어요.",
                "계산 문제에서 단위 변환을 실수했어요.",
                "문제 해석이 어려웠어요. 문제가 뭘 요구하는지 모르겠었어요.",
                "기출 문제는 풀었는데, 새로 나온 유형은 전혀 몰랐어요."
            ]
        }
        
        import random
        examples = problem_examples.get(subject, problem_examples["수학"])
        return random.choice(examples)
    
    def _check_if_advice_given(self, user_message: str) -> bool:
        """
        사용자 메시지에 조언이 포함되어 있는지 확인
        """
        advice_keywords = ["이렇게", "이런", "조언", "팁", "방법", "해보", "시도", "추천", "제안", "도움", "알려", "가르쳐"]
        user_lower = user_message.lower()
        
        for keyword in advice_keywords:
            if keyword in user_lower:
                return True
        
        # 메시지가 충분히 길면 조언으로 간주
        if len(user_message.strip()) > 10:
            return True
        
        return False
    
    def _judge_advice_quality(self, username: str, advice: str, weak_subject: str, weakness_message: str) -> bool:
        """
        LLM을 사용하여 플레이어의 조언이 적절한지 판단
        chatbot_config.json에서 프롬프트 설정을 로드합니다.
        """
        try:
            if not self.client:
                # LLM이 없으면 기본적으로 적절하다고 판단 (절반 확률)
                import random
                return random.choice([True, False])
            
            # chatbot_config.json에서 판단 설정 로드
            judgment_config = self.config.get("mock_exam_advice_judgment", {})
            system_prompt = judgment_config.get(
                "system_prompt", 
                "당신은 교육 전문가입니다. 조언의 적절성을 판단하세요."
            )
            user_prompt_template = judgment_config.get(
                "user_prompt_template",
                "플레이어(멘토)가 다음과 같은 조언을 했습니다:\n{advice}\n\n이 조언이 긍정적인 말투를 사용했는지만 판단해주세요. 취약점 해결 여부나 조언의 적절성은 전혀 고려하지 마세요.\n\n조언이 긍정적인 말투(격려, 칭찬, 위로, 다정한 표현 등)를 사용했다면 무조건 \"YES\", 부정적이거나 비판적인 말투를 사용했다면 \"NO\"만 답변해주세요."
            )
            temperature = judgment_config.get("temperature", 0.3)
            max_tokens = judgment_config.get("max_tokens", 10)
            positive_keywords = judgment_config.get("positive_keywords", ["YES", "적절", "좋", "도움", "유용", "효과적", "격려", "긍정"])
            negative_keywords = judgment_config.get("negative_keywords", ["NO", "부적절", "나쁨", "무도움", "비효과적", "비판", "부정"])
            
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
                    # 키워드가 없으면 기본적으로 적절하다고 판단 (긍정적 말투면 보상)
                    is_good = True
                    print(f"[ADVICE_JUDGE] 키워드 없음 - 기본값 YES로 판단")
            
            print(f"[ADVICE_JUDGE] 최종 판단 결과: {is_good} (judgment: '{judgment}')")
            return is_good
            
        except Exception as e:
            print(f"[ERROR] 조언 판단 실패: {e}")
            # 오류 시 기본적으로 적절하다고 판단
            return True
    
    def _judge_confession_advice(self, username: str, advice: str) -> bool:
        """
        LLM을 사용하여 플레이어의 조언이 받아들이라는 의미인지 거절하라는 의미인지 판단
        
        Returns:
            bool: 받아들이라는 조언이면 True, 거절하라는 조언이면 False
        """
        try:
            if not self.client:
                # LLM이 없으면 기본적으로 거절로 판단 (절반 확률)
                import random
                return random.choice([True, False])
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 의미 분석 전문가입니다. 플레이어의 조언이 고백을 받아들이라는 의미인지 거절하라는 의미인지 판단하세요."},
                    {"role": "user", "content": f"서가윤은 재수생이고 목표는 대학 합격입니다. 누군가가 서가윤에게 고백했습니다.\n\n플레이어(멘토)가 다음과 같이 조언했습니다:\n{advice}\n\n이 조언이 '고백을 받아들이라는 의미'라면 \"ACCEPT\", '고백을 거절하라는 의미'라면 \"REJECT\"만 답변해주세요."}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            judgment = response.choices[0].message.content.strip().upper()
            print(f"[CONFESSION_JUDGE] LLM 원본 응답: {response.choices[0].message.content.strip()}")
            
            # 판단
            if "ACCEPT" in judgment or "받아들" in judgment:
                should_accept = True
                print(f"[CONFESSION_JUDGE] 받아들이라는 의미로 판단")
            elif "REJECT" in judgment or "거절" in judgment:
                should_accept = False
                print(f"[CONFESSION_JUDGE] 거절하라는 의미로 판단")
            else:
                # 명확하지 않으면 거절로 판단
                should_accept = False
                print(f"[CONFESSION_JUDGE] 명확하지 않음 - 기본값 거절로 판단")
            
            print(f"[CONFESSION_JUDGE] 최종 판단 결과: {should_accept} (judgment: '{judgment}')")
            return should_accept
            
        except Exception as e:
            print(f"[ERROR] 고백 조언 판단 실패: {e}")
            # 오류 시 기본적으로 거절로 판단
            return False
    
    def _judge_confession_explanation(self, username: str, explanation: str) -> bool:
        """
        LLM을 사용하여 플레이어의 고백 거절 설명이 논리적이고 설득력 있는지 판단
        chatbot_config.json에서 프롬프트 설정을 로드합니다.
        
        Returns:
            bool: 논리적이고 설득력 있으면 True (고백 거절), 아니면 False (고백 수락)
        """
        try:
            if not self.client:
                # LLM이 없으면 기본적으로 논리적이라고 판단 (절반 확률)
                import random
                return random.choice([True, False])
            
            # chatbot_config.json에서 판단 설정 로드
            judgment_config = self.config.get("confession_judgment", {})
            system_prompt = judgment_config.get(
                "system_prompt", 
                "당신은 교육 전문가입니다. 재수생이 목표 대학에 합격하기 위해 집중해야 하는 상황에서, 고백을 거절하도록 설득하는 논리적인 설명인지 판단하세요."
            )
            user_prompt_template = judgment_config.get(
                "user_prompt_template",
                "서가윤은 재수생이고, 목표는 대학 합격입니다. 누군가가 서가윤에게 고백했습니다.\n\n플레이어(멘토)가 다음과 같이 설명했습니다:\n{explanation}\n\n이 설명이 논리적이고 설득력이 있는지 판단해주세요.\n\n설명이 논리적이고 설득력이 있어서 고백을 거절하도록 설득할 수 있다면 \"YES\", 논리적이지 않거나 설득력이 없어서 고백을 받아들이게 된다면 \"NO\"만 답변해주세요."
            )
            temperature = judgment_config.get("temperature", 0.3)
            max_tokens = judgment_config.get("max_tokens", 10)
            positive_keywords = judgment_config.get("positive_keywords", ["YES", "논리적", "설득력", "좋", "적절", "타당"])
            negative_keywords = judgment_config.get("negative_keywords", ["NO", "논리적이지", "설득력 없", "부적절", "타당하지"])
            
            # 프롬프트 템플릿에 변수 치환
            try:
                judgment_prompt = user_prompt_template.format(explanation=explanation)
            except KeyError as e:
                print(f"[WARN] Prompt template format error: {e}. Using explanation directly.")
                judgment_prompt = user_prompt_template.replace("{explanation}", explanation) if "{explanation}" in user_prompt_template else f"{user_prompt_template}\n\n설명: {explanation}"

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
            
            print(f"[CONFESSION_JUDGE] LLM 원본 응답: {response.choices[0].message.content.strip()}")
            
            # 키워드 기반 판단
            judgment_upper = judgment.upper()
            has_positive = any(keyword.upper() in judgment_upper for keyword in positive_keywords)
            has_negative = any(keyword.upper() in judgment_upper for keyword in negative_keywords)
            
            print(f"[CONFESSION_JUDGE] Positive keywords found: {has_positive}, Negative keywords found: {has_negative}")
            
            if has_positive:
                is_logical = True
                print(f"[CONFESSION_JUDGE] 긍정 키워드 발견 - 논리적(거절)로 판단")
            elif has_negative:
                is_logical = False
                print(f"[CONFESSION_JUDGE] 부정 키워드 발견 - 논리적이지 않음(수락)로 판단")
            else:
                # 키워드가 없으면 응답 내용을 다시 확인
                if "YES" in judgment_upper or "예" in judgment or "논리" in judgment:
                    is_logical = True
                    print(f"[CONFESSION_JUDGE] 직접 확인 - 논리적(거절)로 판단")
                elif "NO" in judgment_upper or "아니" in judgment:
                    is_logical = False
                    print(f"[CONFESSION_JUDGE] 직접 확인 - 논리적이지 않음(수락)로 판단")
                else:
                    # 키워드가 없으면 기본적으로 논리적이라고 판단
                    is_logical = True
                    print(f"[CONFESSION_JUDGE] 키워드 없음 - 기본값 논리적(거절)로 판단")
            
            print(f"[CONFESSION_JUDGE] 최종 판단 결과: {is_logical} (judgment: '{judgment}')")
            return is_logical
            
        except Exception as e:
            print(f"[ERROR] 고백 설명 판단 실패: {e}")
            # 오류 시 기본적으로 논리적이라고 판단
            return True
    
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
        """
        호감도 구간에 따른 말투 지시사항 반환 (chatbot_config.json에서만 읽어옴)
        """
        affection_config = self.config.get("affection_tone", {})

        # config가 없으면 경고하고 빈 문자열 반환
        if not affection_config:
            print("[WARN] chatbot_config.json에 affection_tone 설정이 없습니다.")
            return ""

        # 호감도 구간에 따라 config에서 읽어오기
        tone_config = None
        if affection <= 10:
            tone_config = affection_config.get("very_low", {})
        elif affection <= 30:
            tone_config = affection_config.get("low", {})
        elif affection <= 50:
            tone_config = affection_config.get("medium", {})
        elif affection <= 70:
            tone_config = affection_config.get("high", {})
        else:  # 70~100
            tone_config = affection_config.get("very_high", {})

        # tone 필드가 배열이면 조인, 문자열이면 그대로 반환
        tone = tone_config.get("tone", None)
        if tone is None:
            print(f"[WARN] 호감도 구간 설정이 없습니다. (affection: {affection})")
            return ""

        # 배열인 경우 \n으로 조인
        if isinstance(tone, list):
            return "\n".join(tone)
        # 문자열인 경우 그대로 반환 (하위 호환성)
        elif isinstance(tone, str):
            return tone
        else:
            print(f"[WARN] tone 필드 형식이 올바르지 않습니다. (affection: {affection})")
            return ""
        self._save_user_data(username)  # 변경사항 저장

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
        """
        시스템 프롬프트 생성 (캐릭터 설정, 역할 지침, 대화 예시 포함)
        """
        if not self.config:
            return "당신은 재수생입니다."

        system_parts = []

        # 1. 기본 캐릭터 정보
        character = self.config.get("character", {})
        if character:
            bot_name = self.config.get("name", "챗봇")
            system_parts.append(f"## 캐릭터 정보")
            system_parts.append(f"당신은 '{bot_name}'입니다.")

            # 나이, 대학, 전공
            if character.get("age"):
                system_parts.append(f"- 나이: {character.get('age')}세")
            if character.get("university"):
                system_parts.append(f"- 대학/상태: {character.get('university')}")
            if character.get("major"):
                system_parts.append(f"- 전공/목표: {character.get('major')}")

            # 성격
            if character.get("personality"):
                system_parts.append(f"\n### 성격")
                system_parts.append(character.get("personality"))

            # 배경
            if character.get("background"):
                system_parts.append(f"\n### 배경")
                system_parts.append(character.get("background"))

            # 주요 고민사항
            concerns = character.get("major_concerns", [])
            if concerns:
                system_parts.append(f"\n### 주요 고민사항")
                for concern in concerns:
                    system_parts.append(f"- {concern}")

            # 도움이 필요한 부분
            needs_help = character.get("needs_help_with", [])
            if needs_help:
                system_parts.append(f"\n### 도움이 필요한 부분")
                for need in needs_help:
                    system_parts.append(f"- {need}")

            # 역할 지침
            role_directives = character.get("role_directives", {})
            if role_directives:
                system_parts.append(f"\n## 역할 지침")

                # 반드시 따라야 할 규칙
                must_follow = role_directives.get("must_follow_rules", [])
                if must_follow:
                    system_parts.append(f"\n### ✅ 반드시 따라야 할 규칙:")
                    for i, rule in enumerate(must_follow, 1):
                        system_parts.append(f"{i}. {rule}")

                # 절대 하지 말아야 할 것
                must_not = role_directives.get("must_not_do", [])
                if must_not:
                    system_parts.append(f"\n### 🚫 절대 하지 말아야 할 것:")
                    for i, rule in enumerate(must_not, 1):
                        system_parts.append(f"{i}. {rule}")

        # 2. 대화 예시
        dialogue_examples = self.config.get("dialogue_examples", {})
        if dialogue_examples:
            system_parts.append(f"\n## 대화 예시")

            # 도움 요청 시
            asking = dialogue_examples.get("asking_for_help", [])
            if asking:
                system_parts.append(f"\n### 도움을 요청할 때:")
                for example in asking:
                    system_parts.append(f"- \"{example}\"")

            # 불안감 표현 시
            anxiety = dialogue_examples.get("expressing_anxiety", [])
            if anxiety:
                system_parts.append(f"\n### 불안감을 표현할 때:")
                for example in anxiety:
                    system_parts.append(f"- \"{example}\"")

            # 멘토 조언에 반응할 때
            reacting = dialogue_examples.get("reacting_to_mentor_advice", [])
            if reacting:
                system_parts.append(f"\n### 멘토의 조언에 반응할 때:")
                for example in reacting:
                    system_parts.append(f"- \"{example}\"")

        return "\n".join(system_parts)

    def _build_prompt(self, user_message: str, context: str = None, username: str = "사용자", affection: int = 5, game_state: str = "ice_break", selected_subjects: list = None, subject_selected: bool = False, schedule_set: bool = False):
        """
        LLM 프롬프트 구성 (호감도 및 게임 상태 반영)
        호감도 프롬프트만 사용
        """
        if selected_subjects is None:
            selected_subjects = []

        # 프롬프트 시작 (호감도 말투가 메인)
        prompt_parts = []

        # 호감도에 따른 말투 추가 (가장 먼저)
        affection_tone = self._get_affection_tone(affection)
        prompt_parts.append(affection_tone.strip())

        # 게임 상태 컨텍스트 추가
        state_context = self._get_state_context(game_state)
        if state_context.strip():
            prompt_parts.append(state_context.strip())

        # 선택과목 정보 추가 (icebreak 또는 mentoring 단계)
        if game_state in ["icebreak", "mentoring"]:
            if selected_subjects:
                subjects_text = ", ".join(selected_subjects)
                prompt_parts.append(f"[현재 선택된 탐구과목: {subjects_text}]")
                if len(selected_subjects) < 2:
                    prompt_parts.append(f"(아직 {2 - len(selected_subjects)}개 더 선택할 수 있습니다.)")
            else:
                prompt_parts.append("[선택된 탐구과목: 없음]")
                prompt_parts.append("(아직 탐구과목을 선택하지 않았습니다. 자연스럽게 선택과목을 선택하도록 유도하세요.)")

        # 시간표 설정 안내 (daily_routine 단계에서는 14시간 제한 정보를 주지 않음)
        if game_state == "daily_routine":
            if not schedule_set:
                prompt_parts.append("[중요] 아직 주간 학습 시간표가 설정되지 않았습니다. 플레이어에게 '학습 시간표 관리'를 통해 시간표를 설정하도록 자연스럽게 안내하세요. 14시간 제한이나 구체적인 시간표 형식은 언급하지 마세요.")
            else:
                # 시간표가 이미 설정된 경우, 시간표에 대해 언급하지 말 것
                prompt_parts.append("[중요] 시간표는 이미 설정되어 있습니다. 시간표, 학습 시간, 시간표 관리, 시간 분배 등 시간표와 관련된 내용은 절대 언급하지 마세요. 시간표가 언급되면 자연스럽게 다른 주제로 대화를 이어가세요.")
        
        # 6exam_feedback 상태에서는 절대로 여러 과목을 한 번에 말하지 않도록 지시
        if game_state == "6exam_feedback":
            prompt_parts.append("[중요] 절대로 여러 과목(국어, 수학, 영어, 탐구1, 탐구2)을 한 번에 말하지 마세요. 현재 대화하고 있는 과목 하나만 얘기하세요. 예를 들어, 국어에 대해 얘기하고 있다면 국어만 언급하고 수학, 영어, 탐구 등을 함께 말하지 마세요.")

        # 프롬프트 조립
        sys_prompt = "\n\n".join(prompt_parts)

        prompt = sys_prompt.strip() + "\n\n"
        if context:
            prompt += "[참고 정보]\n" + context.strip() + "\n\n"
        prompt += f"{username}: {user_message.strip()}"
        return prompt
    
    
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
                        'stamina': stamina
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
                        'stamina': 30
                    }
            
            # [1.1] 게임 상태 초기화 요청 처리
            if user_message.strip() == "__RESET_GAME_STATE__":
                # 모든 게임 상태 초기화
                self._set_game_state(username, "start")
                self._set_affection(username, 5)
                self._set_stamina(username, 30)
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
                    'stamina': 30
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
            mentoring_end_reply = None  # 멘토링 종료 메시지 초기화
            confession_reply_set = False  # 고백 이벤트 reply 설정 여부
            
            # [1.5.8] 고백 이벤트 처리 (트리거 없이 직접 구현)
            confession_triggered = False
            if "고백" in user_message and "이벤트" in user_message and current_state == "daily_routine":
                # 고백 이벤트 시작: confession 상태로 전환
                self._set_game_state(username, "confession")
                new_state = "confession"
                state_changed = True
                confession_triggered = True
                
                # 서가윤의 고백 상황 안내 나레이션
                if not narration:
                    narration = "어느 날, 서가윤이 당신에게 말했다. '선생님... 오늘 누군가 저한테 고백했어요. 어떻게 해야 할지 모르겠어요.'"
                else:
                    narration = f"{narration}\n\n어느 날, 서가윤이 당신에게 말했다. '선생님... 오늘 누군가 저한테 고백했어요. 어떻게 해야 할지 모르겠어요.'"
                
                print(f"[CONFESSION] 고백 이벤트 시작 - {username}")
            
            # [1.5.9] 전역 체크: 어떤 상태에서든 "6월 모의고사 응시" 입력 시 6exam으로 전이
            if "6월 모의고사" in user_message or "6월모의고사" in user_message.replace(" ", ""):
                # 6월 모의고사 성적 계산
                june_exam_scores = self._calculate_mock_exam_scores(username)
                
                # 성적표 나레이션 생성 (요청 형식: "6월 모의고사 성적이 발표 되었습니다: 국어 -등급 (백분위 Y%) 수학 -등급 (백분위 Y%) ...")
                score_parts = []
                for subject in ["국어", "수학", "영어", "탐구1", "탐구2"]:
                    if subject in june_exam_scores:
                        score_data = june_exam_scores[subject]
                        score_parts.append(f"{subject} {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                
                june_exam_narration = "6월 모의고사 성적이 발표 되었습니다: " + " ".join(score_parts)
                
                if not narration:
                    narration = june_exam_narration
                else:
                    narration = f"{narration}\n\n{june_exam_narration}"
                
                # 문제점 추적 시스템 초기화 (6exam 상태 처리 전에 미리 초기화)
                self.june_exam_problems[username] = {
                    "scores": june_exam_scores,
                    "subjects": {
                        "국어": {"problem": None, "solved": False},
                        "수학": {"problem": None, "solved": False},
                        "영어": {"problem": None, "solved": False},
                        "탐구1": {"problem": None, "solved": False},
                        "탐구2": {"problem": None, "solved": False}
                    },
                    "current_subject": None,
                    "completed_count": 0,
                    "subject_order": ["국어", "수학", "영어", "탐구1", "탐구2"]
                }
                
                # 상태를 6exam으로 전이 (나중에 6exam 상태 처리에서 6exam_feedback으로 자동 전이됨)
                self._set_game_state(username, "6exam")
                new_state = "6exam"
                state_changed = True
                print(f"[6EXAM] {username}의 6월 모의고사 응시로 6exam 상태로 전이")

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
                week_result = self._advance_one_week(username)
                week_advanced = True
                
                # 정규 모의고사인 경우 자동 전이
                if week_result.get('exam'):
                    exam_result = week_result['exam']
                    exam_name = exam_result.get('name', '')
                    exam_month_str = exam_name.replace('월 모의고사', '').replace('월', '').zfill(2)
                    exam_month = f"2024-{exam_month_str}" if exam_month_str and exam_month_str != '11' else None
                    
                    # 6월 모의고사인 경우 6exam 상태로 전이
                    if exam_month and exam_month.endswith("-06"):
                        # 6월 모의고사인 경우 6exam 상태로 전이
                        exam_scores = exam_result.get('scores', {})
                        if exam_scores:
                            # 상태를 6exam으로 전이
                            self._set_game_state(username, "6exam")
                            new_state = "6exam"
                            state_changed = True
                            
                            # 성적표 나레이션
                            week_advance_narration = week_result['exam']['text']
                            print(f"[6EXAM] 멘토링 종료로 인한 6월 모의고사 - 6exam 상태로 전이")
                        else:
                            week_advance_narration = f"{week_result['week']}주차가 완료되었습니다."
                            if week_result['exam']:
                                week_advance_narration += f"\n\n{week_result['exam']['text']}"
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
                            
                            # 상태를 official_mock_exam_feedback으로 전이
                            self._set_game_state(username, "official_mock_exam_feedback")
                            new_state = "official_mock_exam_feedback"
                            state_changed = True
                            
                            # 성적표 나레이션
                            week_advance_narration = week_result['exam']['text']
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
                
                # 멘토링 종료 시 특별 메시지 (정규 모의고사로 자동 전이되지 않는 경우에만)
                if not (week_result.get('exam') and week_result['exam'].get('name') and 
                        self._is_official_mock_exam_month(f"2024-{week_result['exam']['name'].replace('월 모의고사', '').replace('월', '').zfill(2) if week_result['exam']['name'] != '수능' else '00'}")):
                    mentoring_end_reply = "선생님, 저 그럼 공부하러 갈게요."
                    print(f"[MENTORING_END] 멘토링 종료 메시지 설정: {mentoring_end_reply}")
            
            # "멘토링 종료" 처리 시 나레이션 추가 (상태 전이 나레이션보다 우선)
            if week_advanced and week_advance_narration:
                if narration:
                    narration = f"{narration}\n\n{week_advance_narration}"
                else:
                    narration = week_advance_narration
            
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
            mock_exam_processed = False
            mock_exam_scores = None
            weak_subject = None
            weakness_message = None
            mock_exam_weakness_reply = None  # 취약점 메시지를 reply에 포함시키기 위한 변수
            
            if new_state == "mock_exam" and current_state != "mock_exam":
                # 사설모의고사 응시 - 성적표 생성
                mock_exam_scores = self._calculate_mock_exam_scores(username)
                weak_subject = self._identify_weak_subject(mock_exam_scores)
                weakness_message = self._generate_weakness_message(weak_subject, mock_exam_scores.get(weak_subject, {}))
                
                # 성적표 나레이션 생성 (취약점 메시지는 나레이션에 포함하지 않음)
                score_lines = []
                for subject, score_data in mock_exam_scores.items():
                    score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                
                mock_exam_narration = "사설모의고사 성적표가 발표되었습니다:\n" + "\n".join(score_lines)
                
                if not narration:
                    narration = mock_exam_narration
                else:
                    narration = f"{narration}\n\n{mock_exam_narration}"
                
                # 취약점 메시지는 reply에 포함시킬 플래그 설정 (나중에 reply에 추가)
                mock_exam_weakness_reply = weakness_message
                
                # 취약점 정보 저장 (피드백에서 사용)
                self.mock_exam_weakness[username] = {
                    "subject": weak_subject,
                    "message": weakness_message
                }
                
                # 상태를 mock_exam_feedback으로 전이
                self._set_game_state(username, "mock_exam_feedback")
                new_state = "mock_exam_feedback"
                mock_exam_processed = True
                print(f"[MOCK_EXAM] {username}의 사설모의고사 성적표 생성 완료. 취약 과목: {weak_subject}")
            
            # [1.7.5.5] 6exam 상태 처리 (6exam_feedback으로 자동 전이)
            june_exam_intro_reply = None  # 6exam에서 6exam_feedback으로 전이 시 초기 메시지
            june_subject_problem_reply = None  # 과목별 문제점 메시지
            
            # 6exam 상태 처리 (6exam_feedback으로 자동 전이)
            if new_state == "6exam":
                # 6exam으로 전이될 때 성적표가 없으면 계산
                problem_info = self.june_exam_problems.get(username, {})
                if not problem_info or not problem_info.get("scores"):
                    june_exam_scores = self._calculate_mock_exam_scores(username)
                    
                    # 성적표 나레이션 생성 (한 번만, 요청 형식)
                    if not narration or "6월 모의고사 성적" not in narration:
                        score_parts = []
                        for subject in ["국어", "수학", "영어", "탐구1", "탐구2"]:
                            if subject in june_exam_scores:
                                score_data = june_exam_scores[subject]
                                score_parts.append(f"{subject} {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                        
                        june_exam_narration = "6월 모의고사 성적이 발표 되었습니다: " + " ".join(score_parts)
                        
                        if not narration:
                            narration = june_exam_narration
                        else:
                            narration = f"{narration}\n\n{june_exam_narration}"
                    
                    # 문제점 추적 시스템 초기화
                    self.june_exam_problems[username] = {
                        "scores": june_exam_scores,
                        "subjects": {
                            "국어": {"problem": None, "solved": False},
                            "수학": {"problem": None, "solved": False},
                            "영어": {"problem": None, "solved": False},
                            "탐구1": {"problem": None, "solved": False},
                            "탐구2": {"problem": None, "solved": False}
                        },
                        "current_subject": None,  # 현재 대화 중인 과목
                        "completed_count": 0,  # 완료한 과목 수
                        "subject_order": ["국어", "수학", "영어", "탐구1", "탐구2"]  # 과목 순서
                    }
                else:
                    # 이미 문제점 정보가 있는 경우
                    june_exam_scores = problem_info.get("scores", {})
                
                # "많은 생각이 든 시험이었어요" 메시지를 reply에 추가 (별도 메시지로)
                june_exam_intro_reply = "많은 생각이 든 시험이었어요. 과목별로 저에게 피드백을 해주세요."
                
                # 상태를 6exam_feedback으로 자동 전이
                self._set_game_state(username, "6exam_feedback")
                new_state = "6exam_feedback"
                state_changed = True
                print(f"[6EXAM] {username}의 6exam 상태 - 6exam_feedback으로 자동 전이")
            
            # [1.7.5.6] 6exam_feedback 상태 처리 (과목별 문제점 파악)
            if new_state == "6exam_feedback":
                problem_info = self.june_exam_problems.get(username, {})
                if not problem_info:
                    # 문제점 정보가 없으면 초기화
                    june_exam_scores = self._calculate_mock_exam_scores(username)
                    problem_info = {
                        "scores": june_exam_scores,
                        "subjects": {
                            "국어": {"problem": None, "solved": False},
                            "수학": {"problem": None, "solved": False},
                            "영어": {"problem": None, "solved": False},
                            "탐구1": {"problem": None, "solved": False},
                            "탐구2": {"problem": None, "solved": False}
                        },
                        "current_subject": None,
                        "completed_count": 0,
                        "subject_order": ["국어", "수학", "영어", "탐구1", "탐구2"]
                    }
                    self.june_exam_problems[username] = problem_info
                
                subjects = problem_info.get("subjects", {})
                current_subject = problem_info.get("current_subject")
                completed_count = problem_info.get("completed_count", 0)
                subject_order = problem_info.get("subject_order", ["국어", "수학", "영어", "탐구1", "탐구2"])
                
                # 다음 대화할 과목 찾기
                next_subject = None
                for subject in subject_order:
                    if not subjects.get(subject, {}).get("solved", False):
                        next_subject = subject
                        break
                
                # 트리거를 사용하여 첫 번째 과목 문제점 제시 확인
                trigger_context = {
                    'username': username,
                    'user_message': user_message,
                    'current_state': new_state,
                    'june_exam_problems': self.june_exam_problems,
                    'service': self
                }
                
                # june_exam_subject_problem 트리거 확인
                subject_problem_trigger = {
                    "trigger_type": "june_exam_subject_problem",
                    "conditions": {}
                }
                
                if self.trigger_registry.evaluate_trigger("june_exam_subject_problem", subject_problem_trigger, trigger_context) and next_subject:
                    # 첫 번째 과목의 문제점 생성
                    subject_scores = problem_info.get("scores", {}).get(next_subject, {})
                    subject_problem = self._generate_june_subject_problem(next_subject, subject_scores)
                    
                    # 현재 과목 설정 및 문제점 저장
                    subjects[next_subject]["problem"] = subject_problem
                    problem_info["current_subject"] = next_subject
                    problem_info["subjects"] = subjects
                    self.june_exam_problems[username] = problem_info
                    
                    # reply에 문제점 메시지 추가 (과목별로 하나씩만: "국어에서는 ~~ 이랬어요" 형식)
                    june_subject_problem_reply = f"{next_subject}에서는 {subject_problem}"
                    print(f"[6EXAM_FEEDBACK] {next_subject} 과목 문제점: {subject_problem}")
                
                # 트리거를 사용하여 조언 제시 확인
                advice_given_trigger = {
                    "trigger_type": "june_exam_advice_given",
                    "conditions": {}
                }
                
                # june_exam_advice_given 트리거 확인
                if self.trigger_registry.evaluate_trigger("june_exam_advice_given", advice_given_trigger, trigger_context):
                    # 현재 과목이 있고 아직 해결되지 않은 경우에만 처리
                    if current_subject and not subjects.get(current_subject, {}).get("solved", False):
                        # LLM으로 해결방안 적절성 판단
                        current_problem = subjects.get(current_subject, {}).get("problem", "")
                        is_solution_good = self._judge_advice_quality(username, user_message, current_subject, current_problem)
                        
                        if is_solution_good:
                            # 해결방안이 적절함: 해당과목 +100, 멘탈 +5, 호감도 +2
                            abilities = self._get_abilities(username)
                            if current_subject in abilities:
                                abilities[current_subject] = min(2500, abilities[current_subject] + 100)
                                self._set_abilities(username, abilities)
                            
                            current_mental = self._get_mental(username)
                            new_mental = min(100, current_mental + 5)
                            self._set_mental(username, new_mental)
                            
                            new_affection = min(100, new_affection + 2)
                            self._set_affection(username, new_affection)
                            
                            # 현재 과목 완료 처리
                            subjects[current_subject]["solved"] = True
                            completed_count += 1
                            problem_info["completed_count"] = completed_count
                            problem_info["subjects"] = subjects
                            problem_info["current_subject"] = None
                            self.june_exam_problems[username] = problem_info
                            
                            if not narration:
                                narration = f"적절한 조언이였습니다 {current_subject}과목 능력치 +100 멘탈 +5 호감도 +2"
                            else:
                                narration = f"{narration}\n\n적절한 조언이였습니다 {current_subject}과목 능력치 +100 멘탈 +5 호감도 +2"
                            
                            print(f"[6EXAM_FEEDBACK] {current_subject} 해결방안 적절함 - 능력치 +100, 멘탈 +5, 완료: {completed_count}/5")
                            
                            # 모든 과목 완료 확인
                            if completed_count >= 5:
                                self._set_game_state(username, "daily_routine")
                                new_state = "daily_routine"
                                if narration:
                                    narration = f"{narration}\n\n모든 과목의 문제점을 해결했습니다. 일상 루틴으로 돌아갑니다."
                                else:
                                    narration = "모든 과목의 문제점을 해결했습니다. 일상 루틴으로 돌아갑니다."
                                
                                # 문제점 정보 초기화
                                if username in self.june_exam_problems:
                                    del self.june_exam_problems[username]
                                
                                print(f"[6EXAM_FEEDBACK] 모든 과목 완료 - daily_routine으로 전이")
                            else:
                                # 다음 과목 찾기
                                next_subject_after = None
                                for subject in subject_order:
                                    if not subjects.get(subject, {}).get("solved", False):
                                        next_subject_after = subject
                                        break
                                
                                # 다음 과목이 있으면 자동으로 다음 과목 문제점 제시
                                if next_subject_after:
                                    subject_scores = problem_info.get("scores", {}).get(next_subject_after, {})
                                    subject_problem = self._generate_june_subject_problem(next_subject_after, subject_scores)
                                    
                                    subjects[next_subject_after]["problem"] = subject_problem
                                    problem_info["current_subject"] = next_subject_after
                                    problem_info["subjects"] = subjects
                                    self.june_exam_problems[username] = problem_info
                                    
                                    june_subject_problem_reply = f"{next_subject_after}는 {subject_problem}"
                                    print(f"[6EXAM_FEEDBACK] 다음 과목({next_subject_after}) 문제점: {subject_problem}")
                        else:
                            # 해결방안이 부적절함: 호감도 -2
                            new_affection = max(0, new_affection - 2)
                            self._set_affection(username, new_affection)
                            
                            # 현재 과목 완료 처리 (부적절한 조언이어도 다음 과목으로 진행)
                            subjects[current_subject]["solved"] = True
                            completed_count += 1
                            problem_info["completed_count"] = completed_count
                            problem_info["subjects"] = subjects
                            problem_info["current_subject"] = None
                            self.june_exam_problems[username] = problem_info
                            
                            if not narration:
                                narration = f"적절하지 않은 조언이였습니다. 호감도 -2"
                            else:
                                narration = f"{narration}\n\n적절하지 않은 조언이였습니다. 호감도 -2"
                            
                            print(f"[6EXAM_FEEDBACK] {current_subject} 해결방안 부적절함 - 호감도 -2, 완료: {completed_count}/5")
                            
                            # 모든 과목 완료 확인
                            if completed_count >= 5:
                                self._set_game_state(username, "daily_routine")
                                new_state = "daily_routine"
                                if narration:
                                    narration = f"{narration}\n\n모든 과목의 문제점을 해결했습니다. 일상 루틴으로 돌아갑니다."
                                else:
                                    narration = "모든 과목의 문제점을 해결했습니다. 일상 루틴으로 돌아갑니다."
                                
                                # 문제점 정보 초기화
                                if username in self.june_exam_problems:
                                    del self.june_exam_problems[username]
                                
                                print(f"[6EXAM_FEEDBACK] 모든 과목 완료 - daily_routine으로 전이")
                            else:
                                # 다음 과목 찾기
                                next_subject_after = None
                                for subject in subject_order:
                                    if not subjects.get(subject, {}).get("solved", False):
                                        next_subject_after = subject
                                        break
                                
                                # 다음 과목이 있으면 자동으로 다음 과목 문제점 제시
                                if next_subject_after:
                                    subject_scores = problem_info.get("scores", {}).get(next_subject_after, {})
                                    subject_problem = self._generate_june_subject_problem(next_subject_after, subject_scores)
                                    
                                    subjects[next_subject_after]["problem"] = subject_problem
                                    problem_info["current_subject"] = next_subject_after
                                    problem_info["subjects"] = subjects
                                    self.june_exam_problems[username] = problem_info
                                    
                                    june_subject_problem_reply = f"{next_subject_after}는 {subject_problem}"
                                    print(f"[6EXAM_FEEDBACK] 다음 과목({next_subject_after}) 문제점: {subject_problem}")
            
            # [1.7.6] 사설모의고사 피드백 처리 (조언 판단)
            if new_state == "mock_exam_feedback":
                # 저장된 취약점 정보 가져오기
                weakness_info = self.mock_exam_weakness.get(username, {})
                current_weak_subject = weakness_info.get("subject")
                current_weakness_message = weakness_info.get("message")
                
                if current_weak_subject and current_weakness_message:
                    # 취약점이 언급되었고, 플레이어가 조언을 주었는지 확인
                    advice_given = self._check_if_advice_given(user_message)
                    
                    if advice_given:
                        # LLM으로 조언 적절성 판단
                        is_advice_good = self._judge_advice_quality(username, user_message, current_weak_subject, current_weakness_message)
                        
                        if is_advice_good:
                            # 조언이 적절함: 호감도 +2, 멘탈 +5, 해당과목 +10
                            new_affection = min(100, new_affection + 2)
                            self._set_affection(username, new_affection)
                            
                            current_mental = self._get_mental(username)
                            new_mental = min(100, current_mental + 5)
                            self._set_mental(username, new_mental)
                            
                            abilities = self._get_abilities(username)
                            if current_weak_subject in abilities:
                                abilities[current_weak_subject] = min(2500, abilities[current_weak_subject] + 10)
                                self._set_abilities(username, abilities)
                            
                            if not narration:
                                narration = f"좋은 조언이었어요! {current_weak_subject} 능력치가 10, 호감도 +2, 멘탈 +5 증가했습니다."
                            else:
                                narration = f"{narration}\n\n좋은 조언이었어요! 호감도 +2, 멘탈 +5, {current_weak_subject} 능력치 +10"
                            
                            # 취약점 정보 초기화 (한 번만 보상)
                            if username in self.mock_exam_weakness:
                                del self.mock_exam_weakness[username]
                            
                            # 피드백 완료 후 일상루틴으로 전이
                            self._set_game_state(username, "daily_routine")
                            new_state = "daily_routine"
                            if narration:
                                narration = f"{narration}\n\n일상 루틴으로 돌아갑니다."
                            else:
                                narration = "일상 루틴으로 돌아갑니다."
                            
                            print(f"[MOCK_EXAM] 조언 적절함 - 호감도 +2, 멘탈 +5, {current_weak_subject} +10, daily_routine으로 전이")
                        else:
                            # 조언이 부적절함: 호감도 -2, 멘탈 -2
                            new_affection = max(0, new_affection - 2)
                            self._set_affection(username, new_affection)
                            
                            current_mental = self._get_mental(username)
                            new_mental = max(0, current_mental - 2)
                            self._set_mental(username, new_mental)
                            
                            if not narration:
                                narration = "조언이 잘못되었어요. 호감도와 멘탈이 감소했습니다."
                            else:
                                narration = f"{narration}\n\n조언이 잘못되었어요. 호감도 -2, 멘탈 -2"
                            
                            # 취약점 정보 초기화 (한 번만 페널티)
                            if username in self.mock_exam_weakness:
                                del self.mock_exam_weakness[username]
                            
                            # 피드백 완료 후 일상루틴으로 전이
                            self._set_game_state(username, "daily_routine")
                            new_state = "daily_routine"
                            if narration:
                                narration = f"{narration}\n\n일상 루틴으로 돌아갑니다."
                            else:
                                narration = "일상 루틴으로 돌아갑니다."
                            
                            print(f"[MOCK_EXAM] 조언 부적절함 - 호감도 -2, 멘탈 -2, daily_routine으로 전이")
            
            # [1.7.7] 정규모의고사 피드백 처리 (조언 판단)
            if new_state == "official_mock_exam_feedback":
                # 저장된 취약점 정보 가져오기
                weakness_info = self.official_mock_exam_weakness.get(username, {})
                current_weak_subject = weakness_info.get("subject")
                current_weakness_message = weakness_info.get("message")
                
                if current_weak_subject and current_weakness_message:
                    # 취약점이 언급되었고, 플레이어가 조언을 주었는지 확인
                    advice_given = self._check_if_advice_given(user_message)
                    
                    if advice_given:
                        # LLM으로 조언 적절성 판단
                        is_advice_good = self._judge_advice_quality(username, user_message, current_weak_subject, current_weakness_message)
                        
                        if is_advice_good:
                            # 조언이 적절함: 호감도 +2, 멘탈 +5, 해당과목 +10
                            new_affection = min(100, new_affection + 2)
                            self._set_affection(username, new_affection)
                            
                            current_mental = self._get_mental(username)
                            new_mental = min(100, current_mental + 5)
                            self._set_mental(username, new_mental)
                            
                            abilities = self._get_abilities(username)
                            if current_weak_subject in abilities:
                                abilities[current_weak_subject] = min(2500, abilities[current_weak_subject] + 10)
                                self._set_abilities(username, abilities)
                            
                            if not narration:
                                narration = f"좋은 조언이었어요! {current_weak_subject} 능력치가 10, 호감도 +2, 멘탈 +5 증가했습니다."
                            else:
                                narration = f"{narration}\n\n좋은 조언이었어요! 호감도 +2, 멘탈 +5, {current_weak_subject} 능력치 +10"
                            
                            # 취약점 정보 초기화
                            if username in self.official_mock_exam_weakness:
                                del self.official_mock_exam_weakness[username]
                            
                            # 피드백 완료 후 일상루틴으로 전이
                            self._set_game_state(username, "daily_routine")
                            new_state = "daily_routine"
                            if narration:
                                narration = f"{narration}\n\n일상 루틴으로 돌아갑니다."
                            else:
                                narration = "일상 루틴으로 돌아갑니다."
                            
                            print(f"[OFFICIAL_MOCK_EXAM] 조언 적절함 - 호감도 +2, 멘탈 +5, {current_weak_subject} +10, daily_routine으로 전이")
                        else:
                            # 조언이 부적절함: 호감도 -2, 멘탈 -2
                            new_affection = max(0, new_affection - 2)
                            self._set_affection(username, new_affection)
                            
                            current_mental = self._get_mental(username)
                            new_mental = max(0, current_mental - 2)
                            self._set_mental(username, new_mental)
                            
                            if not narration:
                                narration = "조언이 잘못되었어요. 호감도와 멘탈이 감소했습니다."
                            else:
                                narration = f"{narration}\n\n조언이 잘못되었어요. 호감도 -2, 멘탈 -2"
                            
                            # 취약점 정보 초기화
                            if username in self.official_mock_exam_weakness:
                                del self.official_mock_exam_weakness[username]
                            
                            # 피드백 완료 후 일상루틴으로 전이
                            self._set_game_state(username, "daily_routine")
                            new_state = "daily_routine"
                            if narration:
                                narration = f"{narration}\n\n일상 루틴으로 돌아갑니다."
                            else:
                                narration = "일상 루틴으로 돌아갑니다."
                            
                            print(f"[OFFICIAL_MOCK_EXAM] 조언 부적절함 - 호감도 -2, 멘탈 -2, daily_routine으로 전이")
            
            # [1.7.8] 고백 이벤트 처리 (confession 상태에서 조언 입력 처리)
            if new_state == "confession" and not confession_triggered:
                # "고백 이벤트" 입력이 아닌 경우에만 조언 처리
                # 플레이어의 조언이 있는지 확인 (충분히 긴 메시지)
                advice_given = len(user_message.strip()) > 5
                
                if advice_given:
                    # LLM으로 조언의 의도 판단 (받아들이라는 의미인지 거절하라는 의미인지)
                    should_accept = self._judge_confession_advice(username, user_message)
                    
                    # 서가윤의 조언에 대한 반응을 reply로 설정
                    if should_accept:
                        # 받아들이라는 조언: 고백 수락 (멘탈 상승)
                        current_mental = self._get_mental(username)
                        self._set_mental(username, min(100, current_mental + 20))
                        confession_reply = "어... 선생님 말씀이 맞네요. 제가 너무 크게 생각한 것 같아요. 그럼 받아보는 게 어떨까요?"
                        
                        print(f"[CONFESSION] 받아들이라는 조언 - 고백 수락, 멘탈 상승")
                    else:
                        # 거절하라는 조언: 고백 거절 (능력치 유지)
                        confession_reply = "선생님 말씀이 맞아요. 지금은 공부에 집중해야 할 때니까요. 제가 거절하는 게 맞을 것 같아요."
                        
                        print(f"[CONFESSION] 거절하라는 조언 - 고백 거절, 능력치 유지")
                    
                    # reply 설정
                    reply = confession_reply
                    confession_reply_set = True  # 플래그 설정
                    
                    # 결과 나레이션 설정
                    if should_accept:
                        narration = "서가윤이 고백을 받아들이기로 결정했습니다."
                    else:
                        narration = "서가윤이 고백을 거절하기로 결정했습니다."
                    
                    # 결과 후 daily_routine으로 전이
                    self._set_game_state(username, "daily_routine")
                    new_state = "daily_routine"
                    narration = f"{narration}\n\n일상 루틴으로 돌아갑니다."
            
            # [1.7.9] 일상루틴 단계에서 운동/휴식 조언 처리
            stamina_recovered = False
            if new_state == "daily_routine":
                # 사용자 메시지에서 운동/휴식 관련 조언 확인
                exercise_keywords = ["운동", "운동하", "체력 회복", "활동", "스트레칭"]
                rest_keywords = ["휴식", "쉬", "휴식하", "쉬어", "편히", "안정"]
                
                user_message_lower = user_message.lower()
                has_exercise_advice = any(keyword in user_message_lower for keyword in exercise_keywords)
                has_rest_advice = any(keyword in user_message_lower for keyword in rest_keywords)
                
                if has_exercise_advice or has_rest_advice:
                    current_stamina = self._get_stamina(username)
                    new_stamina = min(100, current_stamina + 3)  # 최대 100
                    self._set_stamina(username, new_stamina)
                    stamina_recovered = True
                    
                    advice_type = "운동" if has_exercise_advice else "휴식"
                    if not narration:
                        narration = f"{advice_type} 조언을 따라 체력이 3 회복되었습니다. (현재 체력: {new_stamina})"
                    else:
                        narration = f"{narration}\n\n{advice_type} 조언을 따라 체력이 3 회복되었습니다. (현재 체력: {new_stamina})"
                    
                    print(f"[STAMINA_RECOVER] {username}의 체력이 {current_stamina}에서 {new_stamina}로 회복되었습니다. ({advice_type} 조언)")
            
            # [1.8] 시간표 처리 (학습 시간표 관리 상태에서만)
            schedule_updated = False
            week_passed = False
            # 학습 시간표 관리 상태에서만 시간표 파싱 및 설정 허용
            if new_state == "study_schedule" or current_state == "study_schedule":
                # 현재 시간표 가져오기 (처리 전)
                current_schedule = self._get_schedule(username)
                
                parsed_schedule = self._parse_schedule_from_message(user_message, username)
                if parsed_schedule:
                    total_hours = sum(parsed_schedule.values())
                    if total_hours <= 14:
                        self._set_schedule(username, parsed_schedule)
                        schedule_updated = True
                        current_schedule = parsed_schedule  # 업데이트된 스케줄 사용
                        print(f"[SCHEDULE] {username}의 시간표가 설정되었습니다: {parsed_schedule}")
                        
                        # 학습 시간표 관리 상태에서 시간표를 설정하면 일상 루틴으로 복귀
                        self._set_game_state(username, "daily_routine")
                        new_state = "daily_routine"
                        state_changed = True
                        if not narration:
                            narration = "시간표 설정을 완료했습니다. 일상 루틴으로 돌아갑니다."
                        print(f"[STATE_TRANSITION] 시간표 설정 완료로 인해 daily_routine 상태로 복귀했습니다.")
                    else:
                        print(f"[SCHEDULE] 총 시간이 14시간을 초과합니다: {total_hours}시간")
                        if not narration:
                            narration = f"총 시간이 14시간을 초과합니다. ({total_hours}시간) 14시간 이하로 다시 설정해주세요."
                        else:
                            narration = f"{narration}\n\n총 시간이 14시간을 초과합니다. ({total_hours}시간) 14시간 이하로 다시 설정해주세요."
            
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
                            
                            # 6월 모의고사인 경우 6exam 상태로 전이
                            if exam_month and exam_month.endswith("-06"):
                                # 상태를 6exam으로 전이
                                self._set_game_state(username, "6exam")
                                new_state = "6exam"
                                state_changed = True
                                
                                # 성적표 나레이션 생성
                                subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
                                score_lines = []
                                for subject in subjects:
                                    if subject in exam_scores:
                                        score_data = exam_scores[subject]
                                        score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                                
                                exam_scores_text = f"\n\n{exam_name} 성적이 발표되었습니다:\n" + "\n".join(score_lines)
                                if not confession_reply_set:
                                    narration = exam_scores_text if exam_scores_text else f"{current_week}주차가 완료되었습니다."
                                print(f"[6EXAM] {username}의 6월 모의고사로 6exam 상태로 전이")
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
                                
                                # 성적표 나레이션 생성
                                subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
                                score_lines = []
                                for subject in subjects:
                                    if subject in exam_scores:
                                        score_data = exam_scores[subject]
                                        score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                                
                                exam_scores_text = f"\n\n{exam_name} 성적이 발표되었습니다:\n" + "\n".join(score_lines)
                                
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
                        
                        # 나레이션 메시지 (6월 모의고사나 정규 모의고사가 아닌 경우에만)
                        if exam_month:
                            if exam_month.endswith("-06"):
                                # 6월 모의고사인 경우 성적표 나레이션만 (이미 위에서 설정됨)
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
                schedule_set=schedule_set
            )
            
            # 선택과목 목록 요청 시 프롬프트에 추가
            if new_state in ["icebreak", "mentoring"] and ("탐구과목" in user_message or "선택과목" in user_message or "과목 선택" in user_message or "과목 목록" in user_message):
                subjects_list = self._get_subject_list_text()
                prompt += f"\n\n[선택과목 목록]\n{subjects_list}\n\n사용자가 위 목록 중에서 선택과목을 고를 수 있도록 안내하세요. (최대 2개)"
            
            # reply 변수 초기화 (confession 처리에서 이미 설정했으면 건너뛰기)
            if not confession_reply_set:
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
                
                # confession 상태에서 이미 조언 처리된 경우 LLM 호출 건너뛰기
                confession_processed = False
                if new_state == "confession" and not confession_triggered:
                    # 이미 2881라인에서 advice_given이 설정됨
                    if len(user_message.strip()) > 5:
                        confession_processed = True
                        print("[CONFESSION] 조언 처리 완료 - LLM 호출 건너뛰기")
                
                if not self.client and not confession_processed:
                    # OpenAI Client 확인
                    print("[WARN] OpenAI Client가 초기화되지 않았습니다. 기본 응답을 반환합니다.")
                    reply = "죄송해요, 현재 AI 서비스에 연결할 수 없어요. 잠시 후 다시 시도해주세요."
                    # 기본 메시지에도 상태 접두사 추가
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    reply = f"[{state_name}] {reply}"
                elif not confession_processed:
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
            if study_schedule_transition_reply:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {study_schedule_transition_reply}"
            
            # 멘토링 종료 시 특별 메시지 처리 (정규 모의고사나 6exam_feedback으로 전이되지 않는 경우에만)
            if week_advanced and mentoring_end_reply and new_state != "6exam_feedback":
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {mentoring_end_reply}"
                print(f"[MENTORING_END] 멘토링 종료 메시지 적용: {reply}")
            
            # reply가 없으면 기본 메시지 추가 (상태 접두사 포함)
            if not reply:
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] 안녕하세요."
            
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
            
            # 6월 모의고사 초기 메시지 및 과목별 문제점 메시지를 reply에 추가
            # 초기 메시지는 별도로 처리 (겹치지 않도록)
            if june_exam_intro_reply and new_state == "6exam_feedback":
                # 6exam_feedback 상태로 전이될 때만 초기 메시지를 reply로 설정
                # 다른 메시지와 겹치지 않도록 독립적으로 처리
                state_info = self._get_state_info(new_state)
                state_name = state_info.get("name", new_state)
                reply = f"[{state_name}] {june_exam_intro_reply}"
                print(f"[6EXAM_INTRO] 초기 메시지 설정: {reply}")
            
            if june_subject_problem_reply:
                # 과목별 문제점 메시지는 독립적으로 설정 (초기 메시지와 겹치지 않도록)
                # 초기 메시지가 이미 설정되어 있으면 문제점 메시지로 교체
                if june_exam_intro_reply and reply and june_exam_intro_reply in reply:
                    # 초기 메시지가 있는 경우 문제점 메시지로 교체
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    reply = f"[{state_name}] {june_subject_problem_reply}"
                elif reply and not (june_exam_intro_reply and june_exam_intro_reply in reply):
                    # 초기 메시지가 없는 경우 기존 reply와 함께
                    if reply.startswith("[") and "]" in reply:
                        prefix_end = reply.find("]") + 1
                        prefix = reply[:prefix_end]
                        body = reply[prefix_end:].strip()
                        reply = f"{prefix} {june_subject_problem_reply}\n\n{body}"
                    else:
                        reply = f"{june_subject_problem_reply}\n\n{reply}"
                else:
                    # reply가 없거나 초기 메시지가 있는 경우 문제점 메시지만 표시
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    reply = f"[{state_name}] {june_subject_problem_reply}"
            
            # 정규모의고사 취약점 메시지를 reply에 추가 (접두사 유지)
            official_mock_exam_weakness_reply = None
            if new_state == "official_mock_exam_feedback":
                weakness_info = self.official_mock_exam_weakness.get(username, {})
                official_mock_exam_weakness_reply = weakness_info.get("message")
            
            if official_mock_exam_weakness_reply:
                # reply가 이미 있으면 추가, 없으면 취약점 메시지로 시작
                if reply:
                    # reply에 이미 접두사가 있을 수 있으므로, 접두사가 있으면 유지
                    if reply.startswith("[") and "]" in reply:
                        # 접두사와 본문 분리
                        prefix_end = reply.find("]") + 1
                        prefix = reply[:prefix_end]
                        body = reply[prefix_end:].strip()
                        reply = f"{prefix} {official_mock_exam_weakness_reply}\n\n{body}"
                    else:
                        reply = f"{official_mock_exam_weakness_reply}\n\n{reply}"
                else:
                    # reply가 없으면 취약점 메시지에 접두사 추가
                    state_info = self._get_state_info(new_state)
                    state_name = state_info.get("name", new_state)
                    reply = f"[{state_name}] {official_mock_exam_weakness_reply}"
            
            # 사설모의고사 취약점 메시지를 reply에 추가 (접두사 유지)
            if mock_exam_weakness_reply:
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
            if schedule_updated and not week_passed:
                schedule = self._get_schedule(username)
                schedule_text = ", ".join([f"{k} {v}시간" for k, v in schedule.items()])
                total = sum(schedule.values())
                reply += f"\n\n(시간표가 설정되었습니다: {schedule_text} (총 {total}시간))"
            
            # 대화 횟수 안내 (daily_routine 상태이고 시간표가 설정된 경우)
            if new_state == "daily_routine" and not week_passed:
                conv_count = self._get_conversation_count(username)
                schedule = self._get_schedule(username)
                if schedule:
                    remaining = 5 - conv_count
                    if remaining > 0:
                        reply += f"\n\n(대화 {remaining}번 후 1주일이 지나며 능력치가 증가합니다.)"
            
            # 최종 안전장치: reply에 접두사가 없으면 추가 (study_schedule 등 모든 상태에서)
            if reply and not (reply.startswith("[") and reply.find("]") > 0 and reply.find("]") < 50):
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
            
            # [6] 응답 반환 (호감도, 게임 상태, 선택과목, 나레이션, 능력치, 시간표, 날짜, 체력 포함)
            return {
                'reply': reply,
                'image': None,
                'affection': new_affection,
                'game_state': new_state,
                'selected_subjects': self._get_selected_subjects(username),
                'narration': narration,
                'abilities': self._get_abilities(username),
                'schedule': self._get_schedule(username),
                'current_date': self._get_game_date(username),
                'stamina': self._get_stamina(username),
                'mental': self._get_mental(username)
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
