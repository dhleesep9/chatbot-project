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
                        'stamina': self._get_stamina(username)
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
            'stamina': self._get_stamina(username)
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
            'stamina': self._get_stamina(username)
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
            'stamina': self._get_stamina(username)
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
                "stamina": self._get_stamina(username)
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
        else:
            print(f"[WARN] 잘못된 게임 상태: {state}. 유효한 상태: {valid_states}")
    
    def _evaluate_transition_condition(self, username: str, transition: dict, affection_increased: int, user_message: str = "") -> bool:
        """
        전이 조건 평가 (state machine 기반)

        Args:
            username: 사용자 이름
            transition: 전이 정보 딕셔너리
            affection_increased: 이번 턴 호감도 증가량
            user_message: 사용자 입력 메시지

        Returns:
            조건 만족 여부
        """
        trigger_type = transition.get("trigger_type")
        conditions = transition.get("conditions", {})

        if trigger_type == "affection_increase":
            # 호감도 증가량 트리거 (start → icebreak)
            min_increase = conditions.get("affection_increase_min", 1)
            return affection_increased >= min_increase

        elif trigger_type == "affection_threshold":
            # 호감도 절대값 트리거 (icebreak → daily_routine)
            min_affection = conditions.get("affection_min", 10)
            current_affection = self._get_affection(username)
            return current_affection >= min_affection

        elif trigger_type == "affection_and_subjects":
            # 호감도 달성 + 탐구과목 선택 트리거 (복합 조건)
            min_affection = conditions.get("affection_min", 10)
            subjects_count = conditions.get("subjects_count", 2)

            current_affection = self._get_affection(username)
            selected_subjects = self._get_selected_subjects(username)

            affection_met = current_affection >= min_affection
            subjects_met = len(selected_subjects) >= subjects_count

            return affection_met and subjects_met

        elif trigger_type == "user_input":
            # 유저 입력 포함 트리거
            input_equals = conditions.get("input_equals", "")
            if not input_equals:
                return False

            # 대소문자 구분 없이 포함 여부 체크
            user_message_lower = user_message.lower()
            input_equals_lower = input_equals.lower()

            is_contained = input_equals_lower in user_message_lower

            if is_contained:
                print(f"[TRIGGER] user_input 트리거 발동: '{input_equals}' in '{user_message}'")

            return is_contained

        elif trigger_type == "subject_selection":
            # 탐구과목 선택 트리거
            from services.subject_selection import parse_subjects_from_message, validate_subject_count

            required_count = conditions.get("required_count", 2)

            # 메시지에서 과목 추출
            found_subjects = parse_subjects_from_message(user_message)

            # 필요한 개수만큼 선택되었는지 확인
            if validate_subject_count(found_subjects, required_count):
                # 선택된 과목을 저장
                self._set_selected_subjects(username, found_subjects)
                print(f"[TRIGGER] subject_selection 트리거 발동: {found_subjects}")
                return True

            return False

        # 알 수 없는 트리거 타입
        print(f"[WARN] Unknown trigger_type: {trigger_type}")
        return False

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
            print(f"[STATE_CHECK] Checking transition: {transition.get('trigger_type')} -> {transition.get('next_state')}")
            if self._evaluate_transition_condition(username, transition, affection_increased, user_message):
                next_state = transition.get("next_state")
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
    
    def _reset_conversation_count(self, username: str):
        """
        사용자의 대화 횟수 초기화
        """
        self.conversation_counts[username] = 0
    
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
        efficiency = self._calculate_stamina_efficiency(stamina) / 100.0  # 효율을 배율로 변환 (1.0 = 100%)
        
        for subject, hours in schedule.items():
            if subject in abilities:
                # 체력에 따른 효율 적용: 시간 * 효율
                increased = hours * efficiency
                abilities[subject] = min(2500, abilities[subject] + increased)  # 최대 2500
        
        self._set_abilities(username, abilities)
    
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

        # 시간표 설정 안내 (daily_routine 단계)
        if game_state == "daily_routine":
            if not schedule_set:
                prompt_parts.append("[중요] 아직 주간 학습 시간표가 설정되지 않았습니다. 플레이어에게 14시간을 자유롭게 분배하여 시간표를 설정하도록 안내하세요. 예: '수학4시간 국어4시간 영어4시간 탐구1 1시간 탐구2 1시간'")

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

            # [1.6] 상태 전환 체크 (state machine 기반)
            state_changed, transition_narration = self._check_state_transition(
                username,
                new_affection,
                affection_change,  # 이번 턴 호감도 증가량 전달
                user_message  # 유저 입력 메시지 전달 (user_input 트리거용)
            )
            new_state = self._get_game_state(username)

            # 상태 전환 시 나레이션 사용
            narration = None
            if state_changed and transition_narration:
                narration = transition_narration
            
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
            
            # [1.8] 시간표 처리 (일상 루프 단계에서만)
            schedule_updated = False
            week_passed = False
            if new_state == "daily_routine":
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
                    else:
                        print(f"[SCHEDULE] 총 시간이 14시간을 초과합니다: {total_hours}시간")
                
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
                        
                        # 체력 변동 (30에서 ±1씩 랜덤 변동)
                        import random
                        current_stamina = self._get_stamina(username)
                        stamina_change = random.choice([-1, 1])  # -1 또는 +1
                        new_stamina = max(0, current_stamina + stamina_change)
                        self._set_stamina(username, new_stamina)
                        print(f"[STAMINA] {username}의 체력이 {current_stamina}에서 {new_stamina}로 변경되었습니다.")
                        
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
                            
                            # 나레이션에 성적 정보 추가
                            exam_name = "수능" if exam_month.endswith("-11") else f"{exam_month[-2:]}월 모의고사"
                            exam_scores_text = f"\n\n{exam_name} 성적이 발표되었습니다:\n"
                            
                            subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
                            score_lines = []
                            for subject in subjects:
                                if subject in exam_scores:
                                    score_data = exam_scores[subject]
                                    score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                            
                            exam_scores_text += "\n".join(score_lines)
                        
                        # 나레이션 메시지
                        narration = f"{current_week}주차가 완료되었습니다. 설정한 공부 시간만큼 실력이 향상되었어요!"
                        if exam_scores_text:
                            narration += exam_scores_text
            
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
            
            # [3.5] 대화 5번 후 자동 처리 (LLM 호출 전)
            if week_passed:
                # 호감도에 따른 공부하러 가는 메시지 생성
                auto_study_message = self._get_study_message_by_affection(new_affection)
                reply = auto_study_message
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
                
                # OpenAI Client 확인
                if not self.client:
                    print("[WARN] OpenAI Client가 초기화되지 않았습니다. 기본 응답을 반환합니다.")
                    reply = "죄송해요, 현재 AI 서비스에 연결할 수 없어요. 잠시 후 다시 시도해주세요."
                else:
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
                        else:
                            reply = response.choices[0].message.content
                            if not reply or not reply.strip():
                                reply = "죄송해요, 응답을 생성할 수 없어요. 다시 시도해주세요."
                            else:
                                # 응답 앞에 [state명] 추가
                                state_info = self._get_state_info(new_state)
                                state_name = state_info.get("name", new_state)
                                reply = f"[{state_name}] {reply}"
                    except Exception as e:
                        print(f"[ERROR] LLM 호출 실패: {e}")
                        import traceback
                        traceback.print_exc()
                        reply = "죄송해요, 일시적인 오류가 발생했어요. 다시 시도해주세요."
            
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
                'stamina': self._get_stamina(username)
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
