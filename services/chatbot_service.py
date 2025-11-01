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
        
        # 10. 멘탈 저장 (기본값 50)
        self.mentals = {}  # {username: mental_value}
        print("[ChatbotService] 멘탈 시스템 초기화 완료")

        # 9. 대화 횟수 추적 (daily_routine 상태에서만)
        self.conversation_counts = {}  # {username: count}
        print("[ChatbotService] 대화 횟수 시스템 초기화 완료")

        # 10. 현재 주(week) 추적
        self.current_weeks = {}  # {username: week_number}
        print("[ChatbotService] 주(week) 추적 시스템 초기화 완료")

        # 11. 게임 날짜 저장
        self.game_dates = {}  # {username: "2023-11-17"}
        print("[ChatbotService] 게임 날짜 시스템 초기화 완료")
        
        # 12. 시험 직후 자책 상태 저장 (최근 시험 성적이 나쁘면 True)
        self.exam_disappointment = {}  # {username: True/False}
        print("[ChatbotService] 시험 직후 자책 상태 초기화 완료")
        
        # 13. 시험 직후 문제점 저장 (시험 후 랜덤으로 선택된 문제점)
        self.exam_issues = {}  # {username: {"question": "국어 푸는데 시간이 부족해서...", "expected_advice": "모르는 문제는 넘어가"}}
        print("[ChatbotService] 시험 직후 문제점 초기화 완료")
        
        # 14. 재수생 고민 상태 저장
        self.student_concerns = {}  # {username: {"concern": "고민 내용", "keywords": ["키워드1", "키워드2"], "category": "카테고리"}}
        print("[ChatbotService] 재수생 고민 상태 초기화 완료")
        
        # 15. 사설모의고사 이후 "어땠냐" 물어볼 차례인지 저장
        self.awaiting_exam_feedback = {}  # {username: True/False}
        print("[ChatbotService] 사설모의고사 피드백 대기 상태 초기화 완료")
        
        # 16. 사설모의고사 진행 주차 저장 (1주에 1번만 볼 수 있도록)
        self.private_exam_weeks = {}  # {username: week_num (주차 번호)}
        print("[ChatbotService] 사설모의고사 진행 주차 초기화 완료")
        
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
    
    def _get_mental(self, username: str) -> int:
        """
        사용자의 현재 멘탈 반환 (없으면 기본값 50)
        """
        return self.mentals.get(username, 50)
    
    def _set_mental(self, username: str, mental: int):
        """
        사용자의 멘탈 설정
        """
        self.mentals[username] = max(0, min(100, mental))  # 멘탈은 0~100 범위
    
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
        멘탈에 따른 상승효율 계산
        공식: 효율(%) = 100 + (멘탈 - 50)
        예시:
        - 멘탈 50: 100%
        - 멘탈 51: 101%
        - 멘탈 49: 99%
        - 멘탈 30: 80%
        - 멘탈 100: 150%
        """
        return 100 + (mental - 50)
    
    def _is_low_grade(self, ability: float) -> bool:
        """
        능력치가 7~9등급인지 확인
        능력치 -> 백분위 -> 등급 변환하여 7, 8, 9등급인지 확인
        """
        percentile = self._calculate_percentile(ability)
        grade = self._calculate_grade_from_percentile(percentile)
        return grade in [7, 8, 9]
    
    def _get_game_state(self, username: str) -> str:
        """
        사용자의 현재 게임 상태 반환 (없으면 "ice_break")
        """
        return self.game_states.get(username, "ice_break")
    
    def _set_game_state(self, username: str, state: str):
        """
        사용자의 게임 상태 설정
        """
        valid_states = ["ice_break", "mentoring", "daily_routine"]
        if state in valid_states:
            self.game_states[username] = state
            print(f"[GAME_STATE] {username}의 상태가 {state}로 변경되었습니다.")
        else:
            print(f"[WARN] 잘못된 게임 상태: {state}")
    
    def _check_state_transition(self, username: str, new_affection: int) -> bool:
        """
        상태 전환 조건 체크 및 전환
        반환값: 전환이 일어났는지 여부
        
        참고: 선택과목 완료로 인한 상태 전환은 [1.7] 단계에서 직접 처리됩니다.
        """
        current_state = self._get_game_state(username)
        
        # daily_routine 상태에서는 호감도가 낮아져도 이전 상태로 전이되지 않음
        if current_state == "daily_routine":
            return False
        
        # 아이스 브레이크 → 멘토링: 호감도 10 이상 달성 시
        if current_state == "ice_break" and new_affection >= 10:
            self._set_game_state(username, "mentoring")
            return True
        
        # 멘토링 → ice_break: 호감도 10 미만으로 떨어질 때 (daily_routine이 아닌 경우만)
        if current_state == "mentoring" and new_affection < 10:
            # daily_routine 상태가 아닐 때만 ice_break로 전이
            self._set_game_state(username, "ice_break")
            return True
        
        # 멘토링 → 일상 루프는 [1.7]에서 선택과목 완료 시 직접 처리되므로 여기서는 제거
        # (선택과목 완료 메시지를 나레이션으로 표시하기 위해)
        
        return False
    
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
    
    def _set_exam_disappointment(self, username: str, is_disappointed: bool):
        """
        시험 직후 자책 상태 설정
        """
        self.exam_disappointment[username] = is_disappointed
    
    def _get_exam_disappointment(self, username: str) -> bool:
        """
        시험 직후 자책 상태 반환
        """
        return self.exam_disappointment.get(username, False)
    
    def _get_exam_issue_combinations(self) -> list:
        """
        시험 직후 문제점-조언 조합 반환
        """
        return [
            # 국어 관련 (7개)
            {
                "question": "국어 푸는데 시간이 부족해서 뒤에 부분은 보지도 못했어요",
                "subject": "국어",
                "expected_advice": ["모르는 문제는", "넘어가", "그냥 넘어가", "건너뛰", "포기", "스킵", "바로 넘어가"]
            },
            {
                "question": "국어 비문학 지문을 너무 자세히 읽다가 뒤 문제들을 못 봤어요",
                "subject": "국어",
                "expected_advice": ["빨리 읽", "요지 파악", "핵심만", "꼼꼼히 읽지 말", "스킵", "대략적으로 읽"]
            },
            {
                "question": "국어 문학 작품을 매번 처음부터 다 읽으려고 해서 시간을 많이 썼어요",
                "subject": "국어",
                "expected_advice": ["문제 중심", "지문 일부", "핵심만", "빠르게", "스킵", "문제 먼저 보"]
            },
            {
                "question": "국어 시간 분배를 잘못해서 마지막 문제들을 못 봤어요",
                "subject": "국어",
                "expected_advice": ["시간 관리", "시간 분배", "체크", "시간 확인", "분배", "계획"]
            },
            {
                "question": "국어 어휘 문제를 한 글자 한 글자 너무 꼼꼼하게 풀었어요",
                "subject": "국어",
                "expected_advice": ["빨리 풀", "빠르게", "신속하게", "스피드", "체감으로"]
            },
            {
                "question": "국어 비문학에서 찾기 문제를 너무 꼼꼼히 찾다가 시간을 다 썼어요",
                "subject": "국어",
                "expected_advice": ["대략적", "요지만", "핵심만", "빠르게", "대충 찾"]
            },
            {
                "question": "국어 문법 문제를 반복해서 확인하다가 시간이 부족했어요",
                "subject": "국어",
                "expected_advice": ["넘어가", "확인하지 말", "그냥 넘어가", "시간 아껴", "스킵"]
            },
            # 수학 관련 (8개)
            {
                "question": "수학 문제 이해는 되는데 시간이 부족해서 못 풀었어요",
                "subject": "수학",
                "expected_advice": ["빨리 풀", "시간 단축", "속도를 올려", "시간 관리", "시간 분배", "빠르게"]
            },
            {
                "question": "수학 계산을 계속 반복해서 틀렸는데 시간을 너무 많이 써버렸어요",
                "subject": "수학",
                "expected_advice": ["넘어가", "포기", "확인하지 말", "그냥 넘어가", "시간 관리", "체크 안 해"]
            },
            {
                "question": "수학 고난도 문제에만 집중하다가 쉬운 문제들을 못 봤어요",
                "subject": "수학",
                "expected_advice": ["쉬운 문제", "순서", "전략", "선택과 집중", "포기", "기본 문제 먼저"]
            },
            {
                "question": "수학 풀이 과정을 너무 자세히 쓰다가 시간이 부족했어요",
                "subject": "수학",
                "expected_advice": ["간략히", "핵심만", "줄여서", "짧게", "요약", "축약"]
            },
            {
                "question": "수학 풀이법을 고민하다가 결국 못 풀고 시간만 많이 썼어요",
                "subject": "수학",
                "expected_advice": ["포기", "그냥 넘어가", "스킵", "시간 낭비", "빠르게 결정"]
            },
            {
                "question": "수학 계산 실수가 걱정되어 계속 재계산하다가 시간이 부족했어요",
                "subject": "수학",
                "expected_advice": ["확인 안 해", "재계산 안 해", "넘어가", "그냥 넘어가", "시간 아껴"]
            },
            {
                "question": "수학 문제 읽는 속도가 너무 느려서 모든 문제를 못 봤어요",
                "subject": "수학",
                "expected_advice": ["빨리 읽", "스피드", "빠르게 읽", "핵심만", "대략적"]
            },
            {
                "question": "수학 문제를 앞에서부터 하나씩 풀다가 뒤 문제들을 못 봤어요",
                "subject": "수학",
                "expected_advice": ["순서", "전략", "난이도 순", "쉬운 거 먼저", "시간 분배"]
            },
            # 영어 관련 (8개)
            {
                "question": "영어 단어를 너무 오래 생각하다가 시간이 부족했어요",
                "subject": "영어",
                "expected_advice": ["빨리 풀", "쉽게 생각", "고민하지 말", "즉각 판단", "빠르게 풀", "체감으로"]
            },
            {
                "question": "영어 지문을 처음부터 끝까지 다 읽어서 문제 푸는데 시간이 부족했어요",
                "subject": "영어",
                "expected_advice": ["문제 먼저", "지문 안 읽", "핵심만", "스키밍", "빠르게 읽", "문제 중심"]
            },
            {
                "question": "영어 정답을 확신하지 못해 고민하다가 시간을 너무 써버렸어요",
                "subject": "영어",
                "expected_advice": ["빨리 결정", "즉각 판단", "고민하지 말", "첫 느낌", "확신하지 말", "빠르게 선택"]
            },
            {
                "question": "영어 단어 뜻을 계속 추론하려다가 시간이 부족했어요",
                "subject": "영어",
                "expected_advice": ["추론 안 해", "넘어가", "그냥 넘어가", "스킵", "시간 아껴"]
            },
            {
                "question": "영어 빈칸 추론 문제를 너무 오래 생각하다가 시간이 부족했어요",
                "subject": "영어",
                "expected_advice": ["빨리 결정", "빠르게", "체감으로", "즉각 판단", "포기"]
            },
            {
                "question": "영어 어법 문제를 계속 다시 읽어보다가 시간이 부족했어요",
                "subject": "영어",
                "expected_advice": ["다시 안 읽", "한 번에", "빠르게 결정", "확인 안 해", "넘어가"]
            },
            {
                "question": "영어 문장 순서 문제를 여러 번 재배치하다가 시간을 다 썼어요",
                "subject": "영어",
                "expected_advice": ["빠르게 결정", "재배치 안 해", "한 번에", "시간 아껴", "첫 느낌"]
            },
            {
                "question": "영어 시간이 부족해서 마지막 지문을 거의 못 봤어요",
                "subject": "영어",
                "expected_advice": ["시간 분배", "시간 관리", "계획", "순서", "전략", "속도"]
            },
            # 탐구 관련 (8개)
            {
                "question": "탐구 문제가 너무 어려워서 시간 분배를 잘 못했어요",
                "subject": "탐구1",
                "expected_advice": ["시간 관리", "시간 분배", "전략", "순서", "계획", "체크"]
            },
            {
                "question": "탐구 개념을 완전히 기억하려고 하다가 시간이 부족했어요",
                "subject": "탐구1",
                "expected_advice": ["대략적", "추론", "빠르게", "완벽하지 말", "가능성", "체감으로"]
            },
            {
                "question": "탐구 계산 문제에서 실수를 찾기 위해 계속 다시 풀어서 시간이 부족했어요",
                "subject": "탐구1",
                "expected_advice": ["넘어가", "다시 하지 말", "시간 관리", "그냥 넘어가", "확인하지 말", "체크 안 해"]
            },
            {
                "question": "탐구 지문을 너무 자세히 읽어서 문제를 못 봤어요",
                "subject": "탐구2",
                "expected_advice": ["빨리 읽", "핵심만", "스킵", "대략적", "요지만", "빠르게 읽"]
            },
            {
                "question": "탐구 개념 정확히 떠올리려다가 시간을 너무 많이 썼어요",
                "subject": "탐구2",
                "expected_advice": ["추론", "대략적", "가능성", "완벽하지 말", "빠르게", "체감"]
            },
            {
                "question": "탐구 그래프 분석을 너무 꼼꼼히 하다가 시간이 부족했어요",
                "subject": "탐구1",
                "expected_advice": ["대략적", "패턴만", "핵심만", "빠르게", "세밀하게 안 해"]
            },
            {
                "question": "탐구 낯선 개념 문제에 계속 시간을 써버렸어요",
                "subject": "탐구2",
                "expected_advice": ["넘어가", "포기", "그냥 넘어가", "스킵", "시간 낭비"]
            },
            {
                "question": "탐구 시간이 너무 부족해서 뒷부분 문제들을 다 찍었어요",
                "subject": "탐구1",
                "expected_advice": ["시간 분배", "시간 관리", "계획", "순서", "전략", "앞 문제 빨리"]
            }
        ]
    
    def _set_exam_issue(self, username: str, issue: dict):
        """
        시험 직후 문제점 설정
        """
        self.exam_issues[username] = issue
    
    def _get_exam_issue(self, username: str) -> dict:
        """
        시험 직후 문제점 반환
        """
        return self.exam_issues.get(username, None)
    
    def _get_student_concern_combinations(self) -> list:
        """
        재수생 고민 조합 반환
        """
        return [
            # 학업 및 성적에 대한 불안감
            {
                "concern": "공부를 하는데도 성적이 오르지 않으면 어떡하지? 작년과 똑같은 실수를 반복할까 봐 두렵다.",
                "keywords": ["성적", "오르지", "반복", "실수", "두려", "노력", "안정", "단계", "과정", "꾸준", "신뢰", "노하우", "방법"],
                "category": "성적 향상 불안"
            },
            {
                "concern": "매달 치르는 모의고사 성적에 일희일비하게 되며, 긴 수험 생활로 인한 번아웃이나 슬럼프를 겪기 쉽다.",
                "keywords": ["번아웃", "슬럼프", "모의고사", "일희일비", "휴식", "여유", "쉬어", "리프레시", "휴가", "조금 쉬어"],
                "category": "슬럼프와 번아웃"
            },
            {
                "concern": "지금 내가 하는 공부 방법이 맞는지, 이 학원이나 인강이 나에게 정말 도움이 되는지 끊임없이 의심하고 불안해한다.",
                "keywords": ["공부 방법", "학원", "인강", "의심", "불안", "자신", "현재", "지금", "신뢰", "믿고", "맞는", "확신"],
                "category": "공부 방법 불신"
            },
            {
                "concern": "실전에서 제 실력을 발휘하지 못할까 봐 느끼는 극심한 긴장감과 압박감이 있다.",
                "keywords": ["실전", "긴장", "압박", "발휘", "연습", "모의고사", "시험장", "체득", "익숙", "준비", "연습량"],
                "category": "수능 당일 공포"
            },
            # 심리적 압박과 자존감
            {
                "concern": "이번이 마지막 기회라는 생각에 스스로를 몰아붙이며 심리적 압박이 극대화된다.",
                "keywords": ["마지막", "압박", "몰아붙", "여유", "마음", "긴장", "편안", "자연스럽", "부담", "무리", "자신"],
                "category": "극심한 압박감"
            },
            {
                "concern": "이미 대학에 합격해 캠퍼스 생활을 즐기는 친구들을 보며 소외감, 박탈감, 조바심을 느낀다.",
                "keywords": ["친구", "소외감", "박탈감", "조바심", "비교", "나 자신", "속도", "개인", "타인", "시선", "집중"],
                "category": "비교와 박탈감"
            },
            {
                "concern": "스스로를 실패자나 뒤처진 사람으로 여기며 자존감이 크게 낮아지고 우울감을 느끼기 쉽다.",
                "keywords": ["실패자", "뒤처진", "자존감", "우울", "가치", "존재", "긍정", "새로운", "기회", "각자의", "타이밍"],
                "category": "자존감 하락"
            },
            {
                "concern": "혼자 공부하는 시간이 길어지면서 사회와 단절된 듯한 고립감이나 외로움을 많이 느낀다.",
                "keywords": ["고립감", "외로움", "단절", "사람", "관계", "친구", "가족", "대화", "소통", "만나", "연락", "시간"],
                "category": "고립감"
            },
            # 대인관계 및 사회적 시선
            {
                "concern": "부모님의 경제적 지원과 기대를 받으면서 '이번에도 실패하면 안 된다'는 부담감과 죄송한 마음을 동시에 가진다.",
                "keywords": ["부모님", "경제", "지원", "부담", "죄송", "기대", "진심", "노력", "미래", "기쁘", "지원받", "감사"],
                "category": "부모님에 대한 죄송함"
            },
            {
                "concern": "대학생이 된 친구들과 공감대 형성이 어려워지고, 연락을 피하게 되면서 자연스럽게 관계가 멀어진다.",
                "keywords": ["친구", "공감대", "연락", "관계", "멀어", "이해", "지지", "긴 우정", "오래", "바쁜", "각자의"],
                "category": "친구 관계 소원함"
            },
            {
                "concern": "'왜 재수해?', '어느 대학 목표야?' 등 주변 사람들의 관심이나 질문이 큰 스트레스로 다가온다.",
                "keywords": ["재수", "대학", "목표", "질문", "시선", "남의", "지금", "집중", "과정", "무시", "신경"],
                "category": "외부 시선"
            },
            # 미래와 진로에 대한 불확실성
            {
                "concern": "만약 이번에도 실패하면 어떡하지? 또 실패할까 봐 두렵다.",
                "keywords": ["실패", "두렵", "미래", "다음", "대안", "계획", "힘", "노력", "기회", "포기하지", "좋은", "결과"],
                "category": "또 실패하면 공포"
            },
            {
                "concern": "재수에 실패할 경우 삼수, 사수까지 이어질 수 있다는 두려움이 있다.",
                "keywords": ["삼수", "사수", "두려", "기간", "시간", "지금", "열심히", "이번", "신중", "계획", "노력"],
                "category": "N수로의 연장"
            },
            {
                "concern": "또래보다 1~2년 늦어지는 것이 인생 전체에서 뒤처지는 것은 아닌지 걱정한다.",
                "keywords": ["늦어", "뒤처", "인생", "또래", "개인", "속도", "나만의", "타이밍", "기회", "다른", "시간"],
                "category": "인생 뒤처짐 걱정"
            },
            # 경제적·신체적 부담
            {
                "concern": "학원비, 교재비, 생활비 등 만만치 않은 경제적 비용이 부모님과 본인 모두에게 큰 부담이 된다.",
                "keywords": ["경제", "비용", "학원비", "교재비", "생활비", "부담", "알뜰히", "효율", "가치", "투자"],
                "category": "재수 비용"
            },
            {
                "concern": "장시간 앉아서 공부해야 하므로 체력이 급격히 떨어지고 건강 관리에 어려움을 겪는다.",
                "keywords": ["체력", "건강", "운동", "몸", "스트레칭", "활동", "관리", "숙면", "규칙", "생활", "밥"],
                "category": "체력 저하"
            }
        ]
    
    def _set_student_concern(self, username: str, concern: dict):
        """
        재수생 고민 설정
        """
        self.student_concerns[username] = concern
    
    def _get_student_concern(self, username: str) -> dict:
        """
        재수생 고민 반환
        """
        return self.student_concerns.get(username, None)
    
    def _set_awaiting_exam_feedback(self, username: str, awaiting: bool):
        """
        사설모의고사 피드백 대기 상태 설정
        """
        self.awaiting_exam_feedback[username] = awaiting
    
    def _get_awaiting_exam_feedback(self, username: str) -> bool:
        """
        사설모의고사 피드백 대기 상태 반환
        """
        return self.awaiting_exam_feedback.get(username, False)
    
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
        시간당 +1 증가 (체력과 멘탈에 따른 효율 적용)
        """
        schedule = self._get_schedule(username)
        if not schedule:
            return
        
        abilities = self._get_abilities(username)
        stamina = self._get_stamina(username)
        mental = self._get_mental(username)
        
        # 체력 효율
        stamina_efficiency = self._calculate_stamina_efficiency(stamina) / 100.0
        # 멘탈 효율
        mental_efficiency = self._calculate_mental_efficiency(mental) / 100.0
        
        # 총 효율 = 체력 효율 * 멘탈 효율 (두 효율을 곱하여 적용)
        total_efficiency = stamina_efficiency * mental_efficiency
        
        for subject, hours in schedule.items():
            if subject in abilities:
                # 체력과 멘탈에 따른 효율 적용: 시간 * 총효율
                increased = hours * total_efficiency
                # 소수점 첫째자리로 반올림
                abilities[subject] = round(min(2500, abilities[subject] + increased), 1)  # 최대 2500
        
        self._set_abilities(username, abilities)
        print(f"[SCHEDULE] 체력 효율: {stamina_efficiency:.2f}x, 멘탈 효율: {mental_efficiency:.2f}x, 총 효율: {total_efficiency:.2f}x")
    
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
        체력과 멘탈 상태에 따른 패널티 적용
        
        반환값: {"국어": {"grade": 1, "percentile": 85.5}, "수학": {"grade": 2, "percentile": 90.2}, ...}
        """
        import random
        abilities = self._get_abilities(username)
        stamina = self._get_stamina(username)
        mental = self._get_mental(username)
        scores = {}
        
        for subject, ability in abilities.items():
            percentile = self._calculate_percentile(ability)
            
            # 1. 체력 50 이하일 때 탐구과목 백분위 -10 (음수 불가)
            if stamina <= 50 and subject in ["탐구1", "탐구2"]:
                percentile = max(0, percentile - 10)
            
            # 2. 멘탈 40 미만일 때 전과목 백분위 -10~0 랜덤
            if mental < 40:
                penalty = random.uniform(-10, 0)
                percentile = max(0, percentile + penalty)
            
            grade = self._calculate_grade_from_percentile(percentile)
            scores[subject] = {
                "grade": grade,
                "percentile": round(percentile, 1)
            }
        
        print(f"[EXAM] {username}의 {exam_month} 시험 성적 계산 (체력: {stamina}, 멘탈: {mental}): {scores}")
        return scores
    
    def _check_prompt_injection(self, user_message: str) -> bool:
        """
        프롬프트 공격(주입) 감지
        반환값: True면 공격으로 감지됨
        """
        narration_cfg = self.config.get("narration", {})
        injection_cfg = narration_cfg.get("prompt_injection_detection", {})
        
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
        호감도 구간에 따른 말투 지시사항 반환 (config에서 읽어옴)
        """
        affection_config = self.config.get("affection_tone", {})
        
        # config가 없으면 기본값 사용
        if not affection_config:
            return self._get_default_affection_tone(affection)
        
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
            return self._get_default_affection_tone(affection)
        
        # 배열인 경우 \n으로 조인
        if isinstance(tone, list):
            return "\n".join(tone)
        # 문자열인 경우 그대로 반환 (하위 호환성)
        elif isinstance(tone, str):
            return tone
        else:
            return self._get_default_affection_tone(affection)
    
    def _get_default_affection_tone(self, affection: int) -> str:
        """
        기본 호감도 말투 (config가 없을 때 사용)
        """
        if affection <= 10:
            return """
[호감도: 매우 낮음 (0~10)]
- 매우 조심스럽고 방어적인 말투를 사용하세요.
- '그건 저도 알아요...', '그런데 거기선...' 같은 방어적인 표현을 자주 사용하세요.
- 선생님을 완전히 낯선 사람처럼 대하세요.
- 짧고 신중하게 대답하며, 자세한 설명을 하지 마세요.
- 거리를 최대한 두며 불신감을 보이세요.
- 절대로 먼저 대화를 시작하거나 주제를 제안하지 마세요. 대답만 하세요.
"""
        elif affection <= 30:
            return """
[호감도: 낮음 (10~30)]
- 여전히 조심스럽고 방어적인 말투를 사용하세요.
- 하지만 '그럼... 좀 해볼게요'처럼 약간의 개방 신호를 보이세요.
- 선생님에게 여전히 거리를 두지만, 가끔 의지하고 싶어하는 모습을 보이세요.
- 짧게 대답하되, 약간 길게 설명할 수도 있어요.
- 불신과 신뢰 사이에서 갈등하는 모습을 보이세요.
- 절대로 먼저 대화를 시작하거나 주제를 제안하지 마세요. 대답만 하세요.
"""
        elif affection <= 50:
            return """
[호감도: 보통 (30~50)]
- 조금씩 신뢰를 보이는 말투로 변화하세요.
- '그럼 한번 해볼게요...', '선생님 말씀 듣고 해봤는데...' 같은 표현을 사용하세요.
- 여전히 조심스럽지만, 조금 더 편하게 대화하세요.
- 감정 기복이 있지만 좋을 때는 웃는 모습을 보이세요.
- 선생님에게 의지하고 싶어하는 마음을 표현하세요.
- 절대로 먼저 대화를 시작하거나 주제를 제안하지 마세요. 대답만 하세요.
"""
        elif affection <= 70:
            return """
[호감도: 높음 (50~70)]
- 신뢰가 쌓인 말투로 변화하세요.
- '선생님 덕분에...', '이제 좀 자신감이 생겼어요' 같은 표현을 사용하세요.
- 더 편하게 대화하며, 자신의 감정을 솔직하게 표현하세요.
- 선생님을 믿고 의지하는 모습을 보이세요.
- 밈이나 웃는 표현을 자주 사용하세요.
- 절대로 먼저 대화를 시작하거나 주제를 제안하지 마세요. 대답만 하세요.
"""
        else:  # 70~100
            return """
[호감도: 매우 높음 (70~100)]
- 완전히 신뢰하는 말투로 변화하세요.
- '선생님 정말 고마워요!', '선생님 덕분에 이렇게까지 올 수 있었어요!' 같은 표현을 사용하세요.
- 매우 편하고 친근하게 대화하세요.
- 선생님에게 감사하고 의지하는 마음을 자주 표현하세요.
- 자신감이 생긴 모습을 보이되, 여전히 겸손하게 대하세요.
- 웃는 표현과 긍정적인 말투를 자주 사용하세요.
- 절대로 먼저 대화를 시작하거나 주제를 제안하지 마세요. 대답만 하세요.
"""
    
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
        게임 상태에 따른 컨텍스트 프롬프트 반환
        """
        if game_state == "ice_break":
            return """
[게임 상태: 아이스 브레이크 단계]
- 현재는 캐릭터와 서로를 알아가는 단계입니다.
- 사용자는 멘토이고, 당신은 재수생 서가윤입니다.
- 목표: 호감도를 10까지 올려서 신뢰를 쌓는 것입니다.
- 이 단계에서는 대화를 통해 서가윤의 성격, 상황, 불안감 등을 파악하세요.
- 아직 완전한 신뢰는 없으니 방어적이고 조심스러운 말투를 유지하세요.
- 호감도가 10이 되면 다음 단계(멘토링)로 넘어갑니다.
"""
        elif game_state == "mentoring":
            return """
[게임 상태: 멘토링 단계]
- 이제 본격적인 멘토링 단계입니다.
- 호감도 10을 달성하여 서가윤이 선생님(멘토)에게 어느 정도 신뢰를 보이기 시작했습니다.
- 이 단계에서는 구체적인 학습 방법, 과목 선택, 진로 상담 등 멘토링 활동을 진행할 수 있습니다.
- 서가윤은 여전히 불안하고 방어적이지만, 선생님의 조언을 듣고 시도해볼 의지가 생겼습니다.
- 사용자가 선택과목을 아직 선택하지 않았다면, 자연스럽게 선택과목을 선택하도록 유도하세요.
- 선택과목은 최대 2개까지 선택할 수 있습니다.
"""
        elif game_state == "daily_routine":
            return """
[게임 상태: 일상 루프 단계]
- 이제 재수생의 하루 루틴을 관리하는 단계입니다.
- 플레이어가 14시간을 자유롭게 분배하여 학습 계획을 세울 수 있습니다.
- 시간표를 정하면 그에 따라 각 과목의 실력이 증가합니다.
- 시간표 예시: "수학4시간 국어4시간 영어4시간 탐구1 1시간 탐구2 1시간"
- 총 14시간을 초과할 수 없습니다.
- 대화를 5번 하면 자동으로 1주일이 지나며 설정된 시간표에 따라 능력치가 증가합니다.
"""
        else:
            return ""
    
    def _build_prompt(self, user_message: str, context: str = None, username: str = "사용자", affection: int = 5, game_state: str = "ice_break", selected_subjects: list = None, subject_selected: bool = False, schedule_set: bool = False, exam_disappointment: bool = False, exam_issue: dict = None):
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
        
        # 시험 직후 자책 상태 추가
        if exam_disappointment:
            if exam_issue and exam_issue.get("question"):
                # 특정 문제점이 있으면 그것만 언급
                prompt_parts.append(f"""
[시험 직후 자책 상태]
- 최근 모의고사(또는 사설모의고사) 성적이 2등급 미만이었습니다.
- 자신은 열심히 했는데 성적이 오르지 않았다고 자책하고 있습니다.
- 멘토(플레이어)의 조언이 절실히 필요합니다.
- 자연스럽게 다음과 같이 문제점을 토로하세요: "{exam_issue['question']}"
""")
            else:
                # 특정 문제점이 없으면 예시 제공
                prompt_parts.append("""
[시험 직후 자책 상태]
- 최근 모의고사(또는 사설모의고사) 성적이 2등급 미만이었습니다.
- 자신은 열심히 했는데 성적이 오르지 않았다고 자책하고 있습니다.
- 멘토(플레이어)의 조언이 절실히 필요합니다.
- 자연스럽게 자신의 문제점을 토로하세요. 예시:
  - "국어 푸는데 시간이 부족해서 뒤에 부분은 보지도 못했어요"
  - "수학 문제 이해는 되는데 시간이 부족해서 못 풀었어요"
  - "영어 단어를 너무 오래 생각하다가 시간이 부족했어요"
  - "탐구 문제가 너무 어려워서 시간 분배를 잘 못했어요"
""")
        
        # 선택과목 정보 추가 (멘토링 단계)
        if game_state == "mentoring":
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
                prompt_parts.append("[제한] 시간표가 설정되기 전까지는 사설모의고사나 기타 행동을 할 수 없습니다. 오직 시간표 설정에만 집중해야 합니다.")
        
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
            # [0] 현재 상태 가져오기
            current_affection = self._get_affection(username)
            current_state = self._get_game_state(username)
            
            # 기본 변수 초기화
            reply = None
            narration = None
            
            # [1] 초기 메시지(인사)
            if user_message.strip().lower() == 'init':
                try:
                    bot_name = self.config.get('name', '챗봇') if self.config else '챗봇'
                    # 게임 상태 초기화
                    self._set_game_state(username, "ice_break")
                    # 대화 횟수 초기화
                    self._reset_conversation_count(username)
                    # 주 초기화
                    self.current_weeks[username] = 0
                    # 게임 날짜 초기화
                    self._set_game_date(username, "2023-11-17")
                    # 시험 자책 상태 초기화
                    self._set_exam_disappointment(username, False)
                    # 시험 문제점 초기화
                    self.exam_issues[username] = None
                    # 재수생 고민 초기화
                    self.student_concerns[username] = None
                    # 사설모의고사 피드백 대기 상태 초기화
                    self.awaiting_exam_feedback[username] = False
                    # 호감도 확인 (초기값 5)
                    current_affection = self._get_affection(username)
                    # 나레이션 생성
                    try:
                        narration = self._get_narration("game_start")
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
                        'game_state': "ice_break",
                        'selected_subjects': [],
                        'narration': narration,
                        'abilities': abilities,
                        'schedule': {},
                        'current_date': "2023-11-17",
                        'stamina': stamina,
                        'mental': 50
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
                        'game_state': "ice_break",
                        'selected_subjects': [],
                        'narration': None,
                        'abilities': {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0},
                        'schedule': {},
                        'current_date': "2023-11-17",
                        'stamina': 30,
                        'mental': 50
                    }
            
            # [1.1] 게임 상태 초기화 요청 처리
            if user_message.strip() == "__RESET_GAME_STATE__":
                # 모든 게임 상태 초기화
                self._set_game_state(username, "ice_break")
                self._set_affection(username, 5)
                self._set_stamina(username, 30)
                self._set_mental(username, 50)
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
                self._set_exam_disappointment(username, False)
                self.exam_issues[username] = None
                self.student_concerns[username] = None
                self.awaiting_exam_feedback[username] = False
                
                try:
                    narration = self._get_narration("game_start")
                except Exception as e:
                    print(f"[WARN] 나레이션 생성 실패: {e}")
                    narration = None
                return {
                    'reply': "게임이 완전히 초기화되었습니다. 다시 시작하세요!",
                    'image': None,
                    'affection': 5,
                    'game_state': "ice_break",
                    'selected_subjects': [],
                    'narration': narration,
                    'abilities': {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0},
                    'schedule': {},
                    'current_date': "2023-11-17",
                    'stamina': 30,
                    'mental': 50
                }
            
            # [1.2] 디버깅 전용 히든 명령어 처리
            user_message_clean = user_message.strip()
            
            # "1주스킵" 명령어: 1주일 자동 스킵
            if user_message_clean == "1주스킵":
                if current_state == "daily_routine":
                    current_schedule = self._get_schedule(username)
                    if current_schedule:
                        # 시간표에 따라 능력치 증가
                        self._apply_schedule_to_abilities(username)
                        print(f"[DEBUG] 1주 스킵: 능력치 증가 완료")
                    
                    # 주 증가
                    self._increment_week(username)
                    current_week = self._get_current_week(username)
                    
                    # 체력 변동 (매주마다 -1씩 감소)
                    current_stamina = self._get_stamina(username)
                    stamina_change = -1  # 매주 -1씩 감소
                    new_stamina = max(0, current_stamina + stamina_change)
                    self._set_stamina(username, new_stamina)
                    print(f"[STAMINA] {username}의 체력이 {current_stamina}에서 {new_stamina}로 변경되었습니다. (1주 스킵으로 -1)")
                    
                    # 대화 횟수 초기화
                    self._reset_conversation_count(username)
                    
                    # 날짜 7일 증가
                    current_date = self._get_game_date(username)
                    new_date = self._add_days_to_date(current_date, 7)
                    self._set_game_date(username, new_date)
                    
                    # 1주 기간 동안 시험이 있었는지 확인
                    exam_month = self._check_exam_in_period(current_date, new_date)
                    exam_scores = None
                    exam_scores_text = ""
                    
                    if exam_month:
                        exam_scores = self._calculate_exam_scores(username, exam_month)
                        
                        # 2등급 미만 감지 및 자책 상태 설정
                        has_bad_grade = False
                        for subject, score_data in exam_scores.items():
                            if score_data.get('percentile', 100) < 89:  # 2등급 미만
                                has_bad_grade = True
                                break
                        
                        if has_bad_grade:
                            self._set_exam_disappointment(username, True)
                            
                            # 랜덤으로 문제점 선택
                            import random
                            issue_combinations = self._get_exam_issue_combinations()
                            selected_issue = random.choice(issue_combinations)
                            self._set_exam_issue(username, selected_issue)
                            print(f"[EXAM_DISAPPOINTMENT] {username}의 성적이 나빠서 자책 상태로 설정됨 (문제점: {selected_issue['question']})")
                        
                        exam_name = "수능" if exam_month.endswith("-11") else f"{exam_month[-2:]}월 모의고사"
                        exam_scores_text = f"\n\n{exam_name} 성적이 발표되었습니다:\n"
                        subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
                        score_lines = []
                        for subject in subjects:
                            if subject in exam_scores:
                                score_data = exam_scores[subject]
                                score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                        exam_scores_text += "\n".join(score_lines)
                    
                    # 호감도에 따른 공부하러 가는 메시지 생성
                    study_message = self._get_study_message_by_affection(current_affection)
                    
                    narration = f"{current_week}주차가 완료되었습니다. 설정한 공부 시간만큼 실력이 향상되었어요!"
                    if exam_scores_text:
                        narration += exam_scores_text
                    
                    return {
                        'reply': "",  # 빈 메시지로 나레이션이 먼저 표시되도록
                        'image': None,
                        'affection': current_affection,
                        'game_state': current_state,
                        'selected_subjects': self._get_selected_subjects(username),
                        'narration': narration,
                        'abilities': self._get_abilities(username),
                        'schedule': self._get_schedule(username),
                        'current_date': new_date,
                        'stamina': self._get_stamina(username),
                        'mental': self._get_mental(username)
                    }
                else:
                    return {
                        'reply': "daily_routine 상태에서만 사용할 수 있습니다.",
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
            
            # "4주스킵" 명령어: 4주일 자동 스킵
            if user_message_clean == "4주스킵":
                if current_state == "daily_routine":
                    current_schedule = self._get_schedule(username)
                    
                    # 4주 동안 반복
                    narration_parts = []
                    for week_num in range(4):
                        if current_schedule:
                            self._apply_schedule_to_abilities(username)
                        
                        self._increment_week(username)
                        current_week = self._get_current_week(username)
                        
                        # 체력 변동 (매주마다 -1씩 감소, 4주이므로 -4)
                        if week_num == 0:  # 첫 주차에만 체력 감소 (총 4주이므로 -4)
                            current_stamina = self._get_stamina(username)
                            stamina_change = -4  # 4주이므로 -4
                            new_stamina = max(0, current_stamina + stamina_change)
                            self._set_stamina(username, new_stamina)
                            print(f"[STAMINA] {username}의 체력이 {current_stamina}에서 {new_stamina}로 변경되었습니다. (4주 스킵으로 -4)")
                        
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
                    
                    narration = f"4주가 완료되었습니다. 설정한 공부 시간만큼 실력이 향상되었어요!"
                    if narration_parts:
                        narration += "\n\n" + "\n".join(narration_parts)
                    
                    # 호감도에 따른 공부하러 가는 메시지 생성
                    study_message = self._get_study_message_by_affection(current_affection)
                    
                    return {
                        'reply': "",  # 빈 메시지로 나레이션이 먼저 표시되도록
                        'image': None,
                        'affection': current_affection,
                        'game_state': current_state,
                        'selected_subjects': self._get_selected_subjects(username),
                        'narration': narration,
                        'abilities': self._get_abilities(username),
                        'schedule': self._get_schedule(username),
                        'current_date': self._get_game_date(username),
                        'stamina': self._get_stamina(username),
                        'mental': self._get_mental(username)
                    }
                else:
                    return {
                        'reply': "daily_routine 상태에서만 사용할 수 있습니다.",
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
            
            # "호감도5올리기" 명령어: 호감도 5 증가
            if user_message_clean == "호감도5올리기":
                try:
                    new_affection = min(100, current_affection + 5)
                    self._set_affection(username, new_affection)
                    print(f"[DEBUG] 호감도 증가: {current_affection} -> {new_affection}")
                    
                    # 안전하게 모든 값 가져오기
                    try:
                        selected_subjects = self._get_selected_subjects(username)
                    except:
                        selected_subjects = []
                    
                    try:
                        abilities = self._get_abilities(username)
                    except:
                        abilities = {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0}
                    
                    try:
                        schedule = self._get_schedule(username)
                    except:
                        schedule = {}
                    
                    try:
                        current_date = self._get_game_date(username)
                    except:
                        current_date = "2023-11-17"
                    
                    try:
                        stamina = self._get_stamina(username)
                    except:
                        stamina = 30
                    
                    return {
                        'reply': f"호감도가 {current_affection}에서 {new_affection}으로 증가했습니다! (디버그 모드)",
                        'image': None,
                        'affection': new_affection,
                        'game_state': current_state,
                        'selected_subjects': selected_subjects,
                        'narration': None,
                        'abilities': abilities,
                        'schedule': schedule,
                        'current_date': current_date,
                        'stamina': stamina
                    }
                except Exception as e:
                    print(f"[ERROR] 호감도5올리기 명령어 처리 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    # 기본 응답 반환
                    return {
                        'reply': f"호감도가 증가했습니다! (디버그 모드)",
                        'image': None,
                        'affection': min(100, current_affection + 5),
                        'game_state': current_state,
                        'selected_subjects': [],
                        'narration': None,
                        'abilities': {"국어": 0, "수학": 0, "영어": 0, "탐구1": 0, "탐구2": 0},
                        'schedule': {},
                        'current_date': "2023-11-17",
                        'stamina': 30,
                        'mental': 50
                    }
            
            # "만점" 명령어: 모든 능력치를 2500으로 설정
            if user_message_clean == "만점":
                try:
                    max_abilities = {
                        "국어": 2500,
                        "수학": 2500,
                        "영어": 2500,
                        "탐구1": 2500,
                        "탐구2": 2500
                    }
                    self._set_abilities(username, max_abilities)
                    print(f"[DEBUG] 모든 능력치를 2500으로 설정했습니다.")
                    
                    # 안전하게 모든 값 가져오기
                    try:
                        selected_subjects = self._get_selected_subjects(username)
                    except:
                        selected_subjects = []
                    
                    try:
                        schedule = self._get_schedule(username)
                    except:
                        schedule = {}
                    
                    try:
                        current_date = self._get_game_date(username)
                    except:
                        current_date = "2023-11-17"
                    
                    try:
                        stamina = self._get_stamina(username)
                    except:
                        stamina = 30
                    
                    return {
                        'reply': "모든 능력치가 2500으로 설정되었습니다! (디버그 모드)",
                        'image': None,
                        'affection': current_affection,
                        'game_state': current_state,
                        'selected_subjects': selected_subjects,
                        'narration': None,
                        'abilities': max_abilities,
                        'schedule': schedule,
                        'current_date': current_date,
                        'stamina': stamina
                    }
                except Exception as e:
                    print(f"[ERROR] 만점 명령어 처리 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    # 기본 응답 반환
                    return {
                        'reply': "능력치가 설정되었습니다! (디버그 모드)",
                        'image': None,
                        'affection': current_affection,
                        'game_state': current_state,
                        'selected_subjects': [],
                        'narration': None,
                        'abilities': {"국어": 2500, "수학": 2500, "영어": 2500, "탐구1": 2500, "탐구2": 2500},
                        'schedule': {},
                        'current_date': "2023-11-17",
                        'stamina': 30,
                        'mental': 50
                    }
            
            # [1.3] 프롬프트 공격 감지
            if self._check_prompt_injection(user_message):
                injection_cfg = self.config.get("narration", {}).get("prompt_injection_detection", {})
                block_message = injection_cfg.get("block_message", "죄송해요, 그런 말은 할 수 없어요. 게임을 정상적으로 플레이해주세요.")
                return {
                    'reply': "",  # 빈 메시지로 나레이션만 표시
                    'image': None,
                    'affection': current_affection,
                    'game_state': current_state,
                    'selected_subjects': self._get_selected_subjects(username),
                    'narration': "프롬프트 수정 감지! 정상적으로 이용해주세요",
                    'abilities': self._get_abilities(username),
                    'schedule': self._get_schedule(username),
                    'current_date': self._get_game_date(username),
                    'stamina': self._get_stamina(username),
                    'mental': self._get_mental(username)
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
            
            # [1.6] 상태 전환 체크
            state_changed = self._check_state_transition(username, new_affection)
            new_state = self._get_game_state(username)
            
            # 상태 전환 시 나레이션 생성
            narration = None
            if state_changed:
                narration = self._get_narration("state_transition", {
                    "transition_key": f"{current_state}_to_{new_state}"
                })
            
            # [1.7] 선택과목 선택 처리 (멘토링 단계에서만)
            selected_subjects = self._get_selected_subjects(username)
            subject_selected_in_this_turn = False
            subjects_completed = False  # 선택과목 2개 모두 선택 완료 여부
            
            if new_state == "mentoring":
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
                                subjects_completed = True
                                # 상태를 daily_routine으로 전환
                                self._set_game_state(username, "daily_routine")
                                new_state = "daily_routine"
                                print(f"[STATE_TRANSITION] 선택과목 선택 완료! 상태가 daily_routine으로 전환되었습니다.")
                        else:
                            print(f"[SUBJECT] 이미 2개의 과목을 선택했습니다.")
                    else:
                        # 이미 선택된 과목들만 언급된 경우
                        mentioned_subjects = ", ".join([s for s in parsed_subjects if s in selected_subjects])
                        print(f"[SUBJECT] 이미 선택한 과목입니다: {mentioned_subjects}")
                else:
                    # 선택과목이 이미 2개 모두 선택되어 있고, 아직 상태 전환이 안 된 경우 체크
                    if len(selected_subjects) >= 2 and new_state == "mentoring":
                        subjects_completed = True
                        self._set_game_state(username, "daily_routine")
                        new_state = "daily_routine"
                        print(f"[STATE_TRANSITION] 선택과목이 이미 완료되어 상태가 daily_routine으로 전환되었습니다.")
                
                # 선택과목 목록 요청 확인
                if "탐구과목" in user_message or "선택과목" in user_message or "과목 선택" in user_message or "과목 목록" in user_message:
                    subjects_list = self._get_subject_list_text()
                    # 프롬프트에 선택과목 목록 추가될 수 있도록 처리
            
            # [1.8] 시간표 처리 (일상 루프 단계에서만)
            schedule_updated = False
            week_passed = False
            private_exam_taken = False  # 사설모의고사 시험 여부
            private_exam_scores = None  # 사설모의고사 성적표
            if new_state == "daily_routine":
                # 현재 시간표 가져오기 (처리 전)
                current_schedule = self._get_schedule(username)
                
                # 사설모의고사 선택지 감지
                private_exam_keywords = ["사설모의고사", "사설 모의고사", "사설", "사설고사", "사설 모의"]
                user_msg_lower = user_message.lower()
                is_private_exam_request = any(keyword in user_msg_lower for keyword in private_exam_keywords)
                
                if is_private_exam_request:
                    # 시간표가 설정되지 않았으면 사설모의고사 불가
                    if not current_schedule:
                        # 사설모의고사 처리를 중단하고, 대신 안내 메시지 표시
                        is_private_exam_request = False
                        # 아래에서 일반 대화로 처리됨
                        print(f"[PRIVATE_EXAM] {username}이(가) 사설모의고사를 요청했으나 시간표가 설정되지 않아 불가능합니다.")
                    
                    # 1주에 1번만 볼 수 있도록 체크
                    if is_private_exam_request:
                        current_week = self.current_weeks.get(username, 0)
                        last_private_exam_week = self.private_exam_weeks.get(username, -1)
                        
                        if last_private_exam_week >= current_week:
                            # 이미 이번 주에 사설모의고사를 봤음
                            is_private_exam_request = False
                            print(f"[PRIVATE_EXAM] {username}이(가) 사설모의고사를 요청했으나 이번 주에 이미 봤습니다 (주차: {current_week})")
                
                if is_private_exam_request:
                    import random
                    print(f"[PRIVATE_EXAM] {username}이(가) 사설모의고사를 선택했습니다.")
                    
                    # 1. 랜덤으로 임의의 과목 능력치 1~10 증가
                    abilities = self._get_abilities(username)
                    subjects = list(abilities.keys())
                    selected_subject = random.choice(subjects)
                    ability_increase = random.randint(1, 10)
                    abilities[selected_subject] = min(2500, abilities[selected_subject] + ability_increase)
                    self._set_abilities(username, abilities)
                    print(f"[PRIVATE_EXAM] {selected_subject} 능력치가 {ability_increase}만큼 증가했습니다. (현재: {abilities[selected_subject]})")
                    
                    # 2. 멘탈 -10, 체력 -10 감소
                    current_mental = self._get_mental(username)
                    new_mental = max(0, current_mental - 10)
                    self._set_mental(username, new_mental)
                    print(f"[PRIVATE_EXAM] 멘탈이 {current_mental}에서 {new_mental}로 변경되었습니다. (-10)")
                    
                    current_stamina = self._get_stamina(username)
                    new_stamina = max(0, current_stamina - 10)
                    self._set_stamina(username, new_stamina)
                    print(f"[PRIVATE_EXAM] 체력이 {current_stamina}에서 {new_stamina}로 변경되었습니다. (-10)")
                    
                    # 3. 성적표 계산
                    private_exam_scores = self._calculate_exam_scores(username, "사설모의고사")
                    private_exam_taken = True
                    print(f"[PRIVATE_EXAM] 사설모의고사 성적표: {private_exam_scores}")
                    
                    # 2등급 미만 감지 및 자책 상태 설정
                    has_bad_grade = False
                    for subject, score_data in private_exam_scores.items():
                        if score_data.get('percentile', 100) < 89:  # 2등급 미만
                            has_bad_grade = True
                            break
                    
                    # 사설모의고사는 바로 자책 상태 설정하지 않고, 다음 대화에서 "어땠냐" 물어보면 그때 문제점 저장
                    # exam_disappointment는 다음 대화에서 설정
                    if has_bad_grade:
                        # 다음 대화에서 피드백을 받을 차례임을 표시
                        self._set_awaiting_exam_feedback(username, True)
                        print(f"[PRIVATE_EXAM] {username}의 사설모의고사 성적이 나빠서 다음 대화에서 피드백을 받을 준비됨")
                    else:
                        self._set_awaiting_exam_feedback(username, False)
                    
                    # 사설모의고사 진행 주차 저장
                    current_week = self.current_weeks.get(username, 0)
                    self.private_exam_weeks[username] = current_week
                    print(f"[PRIVATE_EXAM] 사설모의고사 진행 주차 기록: {current_week}")
                
                parsed_schedule = self._parse_schedule_from_message(user_message, username)
                if parsed_schedule:
                    total_hours = sum(parsed_schedule.values())
                    if total_hours <= 14:
                        self._set_schedule(username, parsed_schedule)
                        schedule_updated = True
                        current_schedule = parsed_schedule  # 업데이트된 스케줄 사용
                        print(f"[SCHEDULE] {username}의 시간표가 설정되었습니다: {parsed_schedule}")
                        
                        # 정기 모의고사 후 시간표 설정 시 성적 발표 + 문제점 표시
                        awaiting_exam_feedback = self._get_awaiting_exam_feedback(username)
                        if awaiting_exam_feedback and not reply:
                            # 정기 모의고사 문제점이 설정되어 있는지 확인
                            current_exam_issue = self._get_exam_issue(username)
                            if current_exam_issue:
                                print(f"[REGULAR_EXAM] 시간표 설정 후 정기 모의고사 문제점 표시 중...")
                                
                                # 성적 발표 나레이션 추가
                                regular_exam_scores = self._get_regular_exam_scores(username) if hasattr(self, '_get_regular_exam_scores') else None
                                if regular_exam_scores:
                                    exam_month = self._get_regular_exam_month(username) if hasattr(self, '_get_regular_exam_month') else None
                                    if exam_month:
                                        exam_name = "수능" if exam_month.endswith("-11") else f"{exam_month[-2:]}월 모의고사"
                                        exam_scores_text = f"\n\n{exam_name} 성적이 발표되었습니다:\n"
                                        
                                        subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
                                        score_lines = []
                                        for subject in subjects:
                                            if subject in regular_exam_scores:
                                                score_data = regular_exam_scores[subject]
                                                score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                                        
                                        exam_scores_text += "\n".join(score_lines)
                                        
                                        # narration이 None일 수 있으므로 빈 문자열로 초기화
                                        if narration is None:
                                            narration = ""
                                        narration += exam_scores_text
                                
                                # 문제점 표시
                                reply = current_exam_issue.get("question", "")
                                print(f"[REGULAR_EXAM] 재수생이 문제점을 얘기함: {reply}")
                    else:
                        print(f"[SCHEDULE] 총 시간이 14시간을 초과합니다: {total_hours}시간")
                
                # 대화 횟수 증가 (init 메시지 제외, 시간표가 설정된 경우에만, 단 이번 턴에 시간표를 설정한 경우는 제외)
                current_schedule_after_update = self._get_schedule(username)  # schedule_updated 후 다시 가져오기
                if user_message.strip().lower() != 'init' and current_schedule_after_update and not schedule_updated:
                    self._increment_conversation_count(username)
                    conv_count = self._get_conversation_count(username)
                    print(f"[CONVERSATION] {username}의 대화 횟수: {conv_count}/5")
                elif current_schedule_after_update:
                    conv_count = self._get_conversation_count(username)
                else:
                    conv_count = 0
                
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
                    
                    # 체력 변동 (매주마다 -1씩 감소)
                    current_stamina = self._get_stamina(username)
                    stamina_change = -1  # 매주 -1씩 감소
                    new_stamina = max(0, current_stamina + stamina_change)
                    self._set_stamina(username, new_stamina)
                    print(f"[STAMINA] {username}의 체력이 {current_stamina}에서 {new_stamina}로 변경되었습니다. (주차 경과로 -1)")
                    
                    # 대화 횟수 초기화
                    self._reset_conversation_count(username)
                    
                    # 시간표 초기화 (다음 주에 다시 시간표를 짜야 함)
                    self._set_schedule(username, {})
                    print(f"[SCHEDULE] {username}의 시간표가 초기화되었습니다. 다음 주에 시간표를 다시 설정해야 합니다.")
                    
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
                        
                        # 2등급 미만 감지 및 자책 상태 설정
                        has_bad_grade = False
                        for subject, score_data in exam_scores.items():
                            if score_data.get('percentile', 100) < 89:  # 2등급 미만
                                has_bad_grade = True
                                break
                        
                        # 정기 모의고사도 사설모의고사와 동일하게 바로 문제점 선택
                        if has_bad_grade:
                            # 다음 대화에서 피드백을 받을 차례임을 표시
                            self._set_awaiting_exam_feedback(username, True)
                            print(f"[EXAM_DISAPPOINTMENT] {username}의 정기 모의고사 성적이 나빠서 문제점 선택 중...")
                            
                            import random
                            
                            # 각 과목별로 등급이 낮은 과목(7등급 이상)을 찾아서 문제점 선택
                            worst_subject = None
                            worst_grade = 0
                            worst_subject_name = None
                            
                            for subject_name, score_data in exam_scores.items():
                                grade = score_data.get("grade", 9)
                                if grade > worst_grade:
                                    worst_grade = grade
                                    worst_subject = subject_name
                                    worst_subject_name = subject_name
                            
                            # 등급별 문제점 목록 가져오기
                            issue_combinations = self._get_exam_issue_combinations()
                            
                            # 해당 과목과 관련된 문제점만 필터링
                            filtered_issues = []
                            for issue in issue_combinations:
                                issue_subject = issue.get("subject")
                                if issue_subject is None or issue_subject == worst_subject:
                                    filtered_issues.append(issue)
                            
                            if not filtered_issues:
                                filtered_issues = issue_combinations
                            
                            selected_issue = random.choice(filtered_issues)
                            
                            # {subject} 플레이스홀더를 실제 과목명으로 치환
                            question_template = selected_issue.get("question", "")
                            if worst_subject_name and "{subject}" in question_template:
                                question_text = question_template.replace("{subject}", worst_subject_name)
                            else:
                                question_text = question_template
                            
                            # 조언 요청 메시지 추가
                            advice_request = " 조언해주세요." if "조언해주세요" not in question_text else ""
                            final_question = question_text + advice_request
                            
                            # 문제점 정보에 실제 질문 텍스트와 과목명 추가
                            selected_issue_copy = selected_issue.copy()
                            selected_issue_copy["question"] = final_question
                            if worst_subject_name:
                                selected_issue_copy["subject"] = worst_subject_name
                            
                            # 자책 상태 및 문제점 설정
                            self._set_exam_disappointment(username, True)
                            self._set_exam_issue(username, selected_issue_copy)
                            
                            # 재수생 고민으로 설정 (is_exam_feedback 플래그 추가)
                            exam_concern = {
                                "concern": final_question,
                                "keywords": [],
                                "category": "exam_feedback",
                                "is_exam_feedback": True,
                                "subject": worst_subject_name
                            }
                            self._set_student_concern(username, exam_concern)
                            
                            # 정기 모의고사 성적과 문제점 정보를 저장하여 다음 대화에서 사용
                            if hasattr(self, 'regular_exam_scores'):
                                self._set_regular_exam_scores(username, exam_scores)
                                self._set_regular_exam_month(username, exam_month)
                            
                            print(f"[REGULAR_EXAM] 정기 모의고사 문제점 선택 완료 ({worst_subject_name}, {worst_grade}등급): {final_question}")
                        else:
                            self._set_awaiting_exam_feedback(username, False)
                    
                    # 나레이션 메시지 (시간표 설정 요청 먼저, 성적 발표는 시간표 설정 후)
                    narration = f"{current_week}주차가 완료되었습니다. 설정한 공부 시간만큼 실력이 향상되었어요!"
                    narration += "\n\n다음 주를 위해 새로운 학습 시간표를 설정해주세요. (총 14시간)"
            
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
            
            # 프롬프트 생성 (exam_disappointment는 더 이상 사용하지 않으므로 False 전달)
            prompt = self._build_prompt(
                user_message=user_message,
                context=context,
                username=username,
                affection=new_affection,
                game_state=new_state,
                selected_subjects=selected_subjects if new_state == "mentoring" else [],
                subject_selected=subject_selected_in_this_turn,
                schedule_set=schedule_set,
                exam_disappointment=False,  # 더 이상 사용하지 않음
                exam_issue=None  # 더 이상 사용하지 않음
            )
            
            # 선택과목 목록 요청 시 프롬프트에 추가
            if new_state == "mentoring" and ("탐구과목" in user_message or "선택과목" in user_message or "과목 선택" in user_message or "과목 목록" in user_message):
                subjects_list = self._get_subject_list_text()
                prompt += f"\n\n[선택과목 목록]\n{subjects_list}\n\n사용자가 위 목록 중에서 선택과목을 고를 수 있도록 안내하세요. (최대 2개)"
            
            # [3.5] 대화 5번 후 자동 처리 (LLM 호출 전)
            if week_passed:
                # 호감도에 따른 공부하러 가는 메시지 생성 (이건 다음 대화 턴에서)
                auto_study_message = self._get_study_message_by_affection(new_affection)
                reply = ""  # 빈 메시지로 나레이션이 먼저 표시되도록
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
            
            # 사설모의고사/정기 모의고사 이후 조언 피드백 처리 (LLM 응답 생성보다 먼저 실행)
            # 주의: 현재 턴에서 사설모의고사를 치르고 문제점을 말한 경우는 제외해야 함 (다음 턴에서 조언 판단)
            advice_processed = False
            awaiting_feedback = self._get_awaiting_exam_feedback(username)
            
            # 정기 모의고사인지 확인
            regular_exam_scores = self._get_regular_exam_scores(username) if hasattr(self, '_get_regular_exam_scores') else None
            is_regular_exam = regular_exam_scores is not None
            
            # 현재 시간표 상태 확인
            current_schedule_for_check = self._get_schedule(username)
            schedule_set_for_advice = bool(current_schedule_for_check)
            
            print(f"[DEBUG] 조언 피드백 체크: awaiting_feedback={awaiting_feedback}, private_exam_taken={private_exam_taken}, is_regular_exam={is_regular_exam}, schedule_set={schedule_set_for_advice}")
            
            # 정기 모의고사인 경우 시간표가 설정되어 있지 않으면 조언 피드백 처리하지 않음
            if awaiting_feedback and not private_exam_taken:
                if is_regular_exam and not schedule_set_for_advice:
                    # 정기 모의고사인데 시간표가 없으면 조언 피드백 처리하지 않음
                    print(f"[DEBUG] 정기 모의고사인데 시간표가 없어서 조언 피드백 처리 안 함. 시간표 설정을 먼저 요청.")
                    advice_processed = False
                else:
                    # 재수생이 이미 고민을 말한 경우는 조언 피드백 처리 (현재 턴에서 사설모의고사를 치르지 않은 경우만)
                    current_concern = self._get_student_concern(username)
                    print(f"[DEBUG] 조언 피드백 체크: current_concern={current_concern}, is_exam_feedback={current_concern.get('is_exam_feedback') if current_concern else None}")
                    if current_concern and current_concern.get("is_exam_feedback"):
                        # LLM으로 조언 적절성 판단 (플레이어가 처음 한 말만 조언으로 처리)
                        # 상태를 먼저 초기화하여 이후 메시지는 조언으로 처리되지 않도록 함
                        self._set_awaiting_exam_feedback(username, False)
                        
                        exam_type = "정기 모의고사" if is_regular_exam else "사설모의고사"
                        advice_judgment_prompt = f"""다음은 재수생 서가윤이 {exam_type}에서 겪은 문제와 플레이어의 조언입니다.

재수생의 문제: {current_concern.get('concern', '')}

플레이어의 조언: {user_message}

위 조언이 재수생의 문제를 해결하는 데 도움이 되는지 판단하세요.
- 도움이 되면: "적절"
- 도움이 안 되면: "부적절"

답변은 "적절" 또는 "부적절"만 하세요."""
                        
                        judgment_narration = None
                        if self.client:
                            try:
                                judgment_response = self.client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[
                                        {"role": "system", "content": "당신은 학습 조언 평가 전문가입니다. 재수생의 문제점에 대한 조언이 적절한지 판단해주세요."},
                                        {"role": "user", "content": advice_judgment_prompt}
                                    ],
                                    temperature=0.3,
                                    max_tokens=10
                                )
                                
                                if judgment_response and judgment_response.choices:
                                    judgment = judgment_response.choices[0].message.content.strip()
                                    is_advice_appropriate = "적절" in judgment
                                    
                                    if is_advice_appropriate:
                                        # 조언 적절: 호감도 +2, 멘탈 +5, 해당 과목 +10
                                        current_affection_for_advice = self._get_affection(username)
                                        new_affection_for_advice = min(100, current_affection_for_advice + 2)
                                        self._set_affection(username, new_affection_for_advice)
                                        new_affection = new_affection_for_advice
                                        
                                        current_mental_for_advice = self._get_mental(username)
                                        new_mental_for_advice = min(100, current_mental_for_advice + 5)
                                        self._set_mental(username, new_mental_for_advice)
                                        
                                        # 해당 과목 능력치 +10
                                        current_issue = self._get_exam_issue(username)
                                        subject_name = None
                                        if current_issue and current_issue.get("subject"):
                                            subject_name = current_issue["subject"]
                                        elif current_concern and current_concern.get("subject"):
                                            subject_name = current_concern["subject"]
                                        
                                        if subject_name:
                                            abilities = self._get_abilities(username)
                                            if subject_name in abilities:
                                                old_ability = abilities[subject_name]
                                                abilities[subject_name] = min(2500, abilities[subject_name] + 10)
                                                self._set_abilities(username, abilities)
                                                print(f"[ADVICE][SKILL] {subject_name} 능력치 {old_ability:.1f}→{abilities[subject_name]:.1f} (+10)")
                                        
                                        # 나레이션 생성 (능력치 변화 정보 포함)
                                        subject_info = f", {subject_name} +10" if subject_name else ""
                                        judgment_narration = f"적절한 조언입니다! 호감도 +2, 멘탈 +5{subject_info}"
                                        
                                        # 상태 초기화
                                        self._set_student_concern(username, None)
                                        self._set_exam_disappointment(username, False)
                                        self._set_exam_issue(username, None)
                                        
                                        print(f"[EXAM_FEEDBACK][SUCCESS] LLM 판단: 조언 적절! 호감도 {current_affection_for_advice}→{new_affection_for_advice} (+2), 멘탈 {current_mental_for_advice}→{new_mental_for_advice} (+5)")
                                    else:
                                        # 조언 부적절: 호감도 -2, 멘탈 -5
                                        current_affection_for_advice = self._get_affection(username)
                                        new_affection_for_advice = max(0, current_affection_for_advice - 2)
                                        self._set_affection(username, new_affection_for_advice)
                                        new_affection = new_affection_for_advice
                                        
                                        current_mental_for_advice = self._get_mental(username)
                                        new_mental_for_advice = max(0, current_mental_for_advice - 5)
                                        self._set_mental(username, new_mental_for_advice)
                                        
                                        # 나레이션 생성 (능력치 변화 정보 포함)
                                        judgment_narration = "적절하지 못한 조언입니다. 호감도 -2, 멘탈 -5"
                                        
                                        # 상태 초기화
                                        self._set_student_concern(username, None)
                                        self._set_exam_disappointment(username, False)
                                        self._set_exam_issue(username, None)
                                        
                                        print(f"[EXAM_FEEDBACK][FAILURE] LLM 판단: 조언 부적절... 호감도 {current_affection_for_advice}→{new_affection_for_advice} (-2), 멘탈 {current_mental_for_advice}→{new_mental_for_advice} (-5)")
                            except Exception as e:
                                print(f"[ERROR] 조언 판단 생성 실패: {e}")
                                # 실패 시 기본값: 부적절로 처리
                                current_affection_for_advice = self._get_affection(username)
                                new_affection_for_advice = max(0, current_affection_for_advice - 2)
                                self._set_affection(username, new_affection_for_advice)
                                new_affection = new_affection_for_advice
                                
                                current_mental_for_advice = self._get_mental(username)
                                new_mental_for_advice = max(0, current_mental_for_advice - 5)
                                self._set_mental(username, new_mental_for_advice)
                                
                                judgment_narration = "적절하지 못한 조언입니다. 호감도 -2, 멘탈 -5"
                                
                                self._set_student_concern(username, None)
                                self._set_exam_disappointment(username, False)
                                self._set_exam_issue(username, None)
                    
                    # 나레이션 추가 (서가윤의 말보다 앞에 표시되도록)
                    if judgment_narration:
                        if narration:
                            narration = judgment_narration + "\n\n" + narration
                        else:
                            narration = judgment_narration
                    
                    # 조언 판단 후 서가윤의 반응 생성
                    advice_processed = True
                    if self.client:
                        try:
                            is_appropriate = "적절한 조언입니다" in judgment_narration if judgment_narration else False
                            reaction_prompt = f"""재수생 서가윤이 플레이어의 조언을 듣고 반응합니다.

플레이어의 조언: {user_message}
재수생이 겪은 문제: {current_concern.get('concern', '')}

조언이 적절했는지: {"적절함" if is_appropriate else "부적절함"}
현재 호감도: {new_affection}

서가윤의 반응을 생성하세요:
- 조언이 적절했을 때: "감사합니다", "알겠습니다", "다음엔 그렇게 해볼게요" 등의 긍정적인 반응
- 조언이 부적절했을 때: "음...", "그렇군요", "알겠습니다" 등의 중립적이거나 약간 아쉬운 반응

짧고 자연스럽게 한 문장으로 반응하세요. 존댓말을 사용하세요."""
                            
                            reaction_response = self.client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {"role": "system", "content": "당신은 재수생 서가윤입니다. 플레이어의 조언에 대해 자연스럽게 반응하세요."},
                                    {"role": "user", "content": reaction_prompt}
                                ],
                                temperature=0.7,
                                max_tokens=100
                            )
                            
                            if reaction_response and reaction_response.choices and len(reaction_response.choices) > 0:
                                reply = reaction_response.choices[0].message.content
                                print(f"[ADVICE_REACTION] 서가윤의 조언 반응 생성 완료: {reply}")
                            else:
                                reply = "감사합니다..." if is_appropriate else "음... 알겠습니다."
                        except Exception as e:
                            print(f"[ERROR] 조언 반응 생성 실패: {e}")
                            reply = "감사합니다..." if "적절한 조언입니다" in judgment_narration else "음... 알겠습니다."
                    else:
                        reply = "감사합니다..." if "적절한 조언입니다" in judgment_narration else "음... 알겠습니다."
            
            if not advice_processed and not week_passed and not private_exam_taken and reply is None:
                # [4] LLM 응답 생성 (조언이 처리되지 않은 경우만)
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
                        response = self.client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": ""},
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
                    except Exception as e:
                        print(f"[ERROR] LLM 호출 실패: {e}")
                        import traceback
                        traceback.print_exc()
                        reply = "죄송해요, 일시적인 오류가 발생했어요. 다시 시도해주세요."
            
            # 상태 전환 시 나레이션은 별도로 반환 (프론트엔드에서 처리)
            # reply에는 추가 메시지 없음 (나레이션으로 처리)
            
            # 운동 조언 감지 및 체력 증가 (봇의 응답이나 사용자 메시지에서 운동 관련 키워드 확인)
            exercise_keywords = ["운동", "체력", "활동", "몸", "건강", "달리기", "조깅", "헬스", "운동하", "몸을 움직"]
            exercise_advice_keywords = ["운동하", "운동해", "운동을", "운동해야", "운동해야해", "운동해봐", "운동하세요", "운동하라", "조깅하", "달리기"]
            exercise_mentioned = False
            
            # 사용자 메시지나 봇 응답에서 운동 키워드 확인
            user_msg_lower = user_message.lower()
            
            # 사용자가 운동을 하라고 조언한 경우
            user_advice_exercise = any(keyword in user_msg_lower for keyword in exercise_advice_keywords)
            
            if reply:
                reply_lower = reply.lower()
                
                # 사용자가 운동을 하라고 조언한 경우 (봇이 운동을 하게 됨)
                if user_advice_exercise:
                    current_stamina = self._get_stamina(username)
                    new_stamina = max(0, current_stamina + 2)  # 체력 +2 증가
                    self._set_stamina(username, new_stamina)
                    print(f"[STAMINA] 사용자의 운동 조언으로 인해 {username}의 체력이 {current_stamina}에서 {new_stamina}로 증가했습니다. (+2)")
                
                # 봇이 운동을 조언하는 경우 (봇 응답에 운동 관련 키워드 포함)
                elif any(keyword in reply_lower for keyword in exercise_keywords):
                    exercise_mentioned = True
                    # 사용자 메시지에 "운동"이 직접 언급되지 않은 경우만 (봇의 조언으로 인식)
                    if not any(keyword in user_msg_lower for keyword in exercise_keywords):
                        current_stamina = self._get_stamina(username)
                        new_stamina = max(0, current_stamina + 2)  # 체력 +2 증가
                        self._set_stamina(username, new_stamina)
                        print(f"[STAMINA] 봇의 운동 조언으로 인해 {username}의 체력이 {current_stamina}에서 {new_stamina}로 증가했습니다. (+2)")
            
            # daily_routine 상태에서 재수생 고민 처리 (사설모의고사 직후는 제외)
            if current_state == "daily_routine" and not private_exam_taken:
                # 현재 고민 상태 확인
                current_concern = self._get_student_concern(username)
                
                # 고민이 이미 있는 경우, 조언 감지 및 처리
                if current_concern:
                    user_msg_lower = user_message.lower()
                    # 조언 키워드 감지
                    concern_keywords = current_concern.get("keywords", [])
                    is_concern_advice_given = any(keyword in user_msg_lower for keyword in concern_keywords)
                    
                    if is_concern_advice_given:
                        # 조언 성공: 호감도 +1, 멘탈 +3
                        current_affection_for_concern = self._get_affection(username)
                        new_affection_for_concern = min(100, current_affection_for_concern + 1)
                        self._set_affection(username, new_affection_for_concern)
                        
                        current_mental_for_concern = self._get_mental(username)
                        new_mental_for_concern = min(100, current_mental_for_concern + 3)
                        self._set_mental(username, new_mental_for_concern)
                        
                        # 고민 해소
                        self._set_student_concern(username, None)
                        
                        print(f"[CONCERN][SUCCESS] 재수생 고민 조언 성공! 호감도 {current_affection_for_concern}→{new_affection_for_concern} (+1), 멘탈 {current_mental_for_concern}→{new_mental_for_concern} (+3)")
                        # new_affection 반영
                        if new_affection != new_affection_for_concern:
                            new_affection = self._get_affection(username)
                    else:
                        # 조언 실패: 호감도 -1
                        current_affection_for_concern = self._get_affection(username)
                        new_affection_for_concern = max(0, current_affection_for_concern - 1)
                        self._set_affection(username, new_affection_for_concern)
                        
                        # 고민 해소
                        self._set_student_concern(username, None)
                        
                        print(f"[CONCERN][FAILURE] 재수생 고민 조언 실패... 호감도 {current_affection_for_concern}→{new_affection_for_concern} (-1)")
                        # new_affection 반영
                        if new_affection != new_affection_for_concern:
                            new_affection = self._get_affection(username)
                # 고민이 없는 경우, "고민 있냐" 키워드로 고민 요청
                else:
                    concern_request_keywords = ["고민", "걱정", "불안", "어려워", "힘들어", "두려워"]
                    user_msg_lower = user_message.lower()
                    is_concern_request = any(keyword in user_msg_lower for keyword in concern_request_keywords)
                    
                    if is_concern_request:
                        # 랜덤으로 고민 선택
                        import random
                        concern_combinations = self._get_student_concern_combinations()
                        selected_concern = random.choice(concern_combinations)
                        self._set_student_concern(username, selected_concern)
                        
                        # 재수생이 고민을 얘기하는 응답으로 변경
                        reply = selected_concern.get("concern", "")
                        print(f"[CONCERN] 재수생 고민 생성: {selected_concern.get('category', 'Unknown')}")
            
            
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
            
            # 선택과목 완료 시 특별 메시지 및 상태 전환
            if subjects_completed:
                narration = "선택과목이 모두 선택되었습니다! 이제 14시간으로 스케줄을 짜보세요."
                print(f"[NARRATION] 선택과목 완료 메시지: {narration}")
            
            # 시간표 업데이트 시 확인 메시지 (사설모의고사 직후는 제외)
            if schedule_updated and not week_passed and not private_exam_taken:
                schedule = self._get_schedule(username)
                schedule_text = ", ".join([f"{k} {v}시간" for k, v in schedule.items()])
                total = sum(schedule.values())
                reply += f"\n\n(시간표가 설정되었습니다: {schedule_text} (총 {total}시간))"
            
            # 대화 횟수 안내 (daily_routine 상태이고 시간표가 설정된 경우, 단 사설모의고사 직후는 제외)
            if new_state == "daily_routine" and not week_passed and not private_exam_taken:
                conv_count = self._get_conversation_count(username)
                schedule = self._get_schedule(username)
                if schedule:
                    remaining = 5 - conv_count
                    if remaining > 0:
                        reply += f"\n\n(대화 {remaining}번 후 1주일이 지나며 능력치가 증가합니다.)"
            
            if reply is not None:
                print(f"[BOT] {reply}")
            else:
                print(f"[BOT] (응답 없음)")
            print(f"{'='*50}\n")
            
            # [5] 메모리 저장(선택)
            if self.memory and reply is not None:
                self.memory.save_context(
                    {"input": user_message},
                    {"output": reply}
                )
            
            # [6] 사설모의고사 성적표 나레이션 추가 (직후 문제점도 표시)
            if private_exam_taken and private_exam_scores:
                exam_scores_text = "\n\n사설모의고사 성적이 발표되었습니다:\n"
                subjects = ["국어", "수학", "영어", "탐구1", "탐구2"]
                score_lines = []
                for subject in subjects:
                    if subject in private_exam_scores:
                        score_data = private_exam_scores[subject]
                        score_lines.append(f"- {subject}: {score_data['grade']}등급 (백분위 {score_data['percentile']}%)")
                exam_scores_text += "\n".join(score_lines)
                if narration:
                    narration = exam_scores_text + "\n\n" + narration  # 나레이션이 먼저 오도록
                else:
                    narration = exam_scores_text
                
                # 사설모의고사 직후 성적이 나쁘면 바로 문제점 표시
                awaiting_feedback = self._get_awaiting_exam_feedback(username)
                if awaiting_feedback and not reply:  # reply가 설정되지 않았을 때만
                    print(f"[PRIVATE_EXAM] 사설모의고사 직후 문제점 표시 중...")
                    import random
                    
                    # 각 과목별로 등급이 낮은 과목 찾기
                    worst_subject = None
                    worst_grade = 0
                    worst_subject_name = None
                    
                    for subject_name, score_data in private_exam_scores.items():
                        grade = score_data.get("grade", 9)
                        if grade > worst_grade:
                            worst_grade = grade
                            worst_subject = subject_name
                            worst_subject_name = subject_name
                    
                    # 등급별 문제점 목록 가져오기
                    issue_combinations = self._get_exam_issue_combinations()
                    
                    # 해당 과목과 관련된 문제점만 필터링
                    filtered_issues = []
                    for issue in issue_combinations:
                        issue_subject = issue.get("subject")
                        if issue_subject is None or issue_subject == worst_subject:
                            filtered_issues.append(issue)
                    
                    if not filtered_issues:
                        filtered_issues = issue_combinations
                    
                    selected_issue = random.choice(filtered_issues)
                    
                    # {subject} 플레이스홀더를 실제 과목명으로 치환
                    question_template = selected_issue.get("question", "")
                    if worst_subject_name and "{subject}" in question_template:
                        question_text = question_template.replace("{subject}", worst_subject_name)
                    else:
                        question_text = question_template
                    
                    # 조언 요청 메시지 추가
                    advice_request = " 조언해주세요." if "조언해주세요" not in question_text else ""
                    final_question = question_text + advice_request
                    
                    # 문제점 정보에 실제 질문 텍스트와 과목명 추가
                    selected_issue_copy = selected_issue.copy()
                    selected_issue_copy["question"] = final_question
                    if worst_subject_name:
                        selected_issue_copy["subject"] = worst_subject_name
                    
                    # 자책 상태 및 문제점 설정
                    self._set_exam_disappointment(username, True)
                    self._set_exam_issue(username, selected_issue_copy)
                    
                    # 재수생 고민으로 설정
                    exam_concern = {
                        "concern": final_question,
                        "keywords": [],
                        "category": "exam_feedback",
                        "is_exam_feedback": True,
                        "subject": worst_subject_name
                    }
                    self._set_student_concern(username, exam_concern)
                    
                    # 재수생이 문제점을 얘기하는 응답으로 설정
                    reply = final_question
                    print(f"[PRIVATE_EXAM] 재수생이 문제점을 얘기함 ({worst_subject_name}, {worst_grade}등급): {final_question}")
            
            # [7] 응답 반환 (호감도, 게임 상태, 선택과목, 나레이션, 능력치, 시간표, 날짜, 체력, 멘탈, 사설모의고사 성적표 포함)
            # reply가 None이면 빈 문자열로 처리
            if reply is None:
                reply = ""
            
            result = {
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
            
            # 사설모의고사 성적표 추가
            if private_exam_taken and private_exam_scores:
                result['exam_scores'] = private_exam_scores
                result['exam_month'] = "사설모의고사"
            
            return result
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
