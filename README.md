# HateSlop 3기 엔지니어x프로듀서 합동 캐릭터 챗봇 프로젝트

GOAL)

- AI를 이용해 빠르게 개념에 대해 학습하고 실습을 진행합니다.
- AI를 적극적으로 활용하여 코드를 작성하세요.
- AI가 코드를 짜는 것을 보며 AI가 할 수 있는 것과 내가 할 수 있는 것에 대한 성찰을 얻으세요.
- 앞으로 코드는 사람이 짜지 않을 것입니다. 그 시간에 AI가 할 수 없는 것과 본인 내실에 집중하여 몸값을 기르세요.
- 바이브코딩 등 현재 유행하는 모든 AI 기법을 체화하는 것까지가 프로젝트의 목적입니다.

> 운영진이 최신 Claude 4.5 모델과 함께 구성한 모범답안은 `answer-sheet` 브랜치에 있습니다.  
> 답안을 공개하고 AI 활용을 장려하는 이유는 다음과 같습니다.
>
> 첫째, Hateslop 학회원은 스스로 배우고자 하는 의지가 검증된 사람들로, 자기주도적 학습이 전제되어 있습니다.  
> 우리는 여러분이 단순히 제출을 위한 과제를 작성하지 않을 것이라는 믿음을 가지고 있습니다.
>
> 둘째, 오늘날 AI로 정답을 찾는 것은 어렵지 않습니다.  
> 중요한 것은 그 정답에 이르기까지의 사고 과정과 추론 능력, 그리고 더 나은 답을 도출하려는 문제 해결력을 기르는 일입니다.
>
> 그렇기 때문에 단순히 결과를 복제하는 데 그치지 말고, AI를 도구로 삼아 스스로 사고하고 탐구하며 성장하길 바랍니다.

> \*바이브코딩 교육은 학회 커리큘럼에 맞춰 추후 진행될 예정입니다.

> \*해당 프로젝트 답안지가 어떻게 작성되었는지 궁금하시다면, .cursor/rules 의 내용을 살펴보세요. 해당 내용을 LLM에게 지침으로 주고 Task 를 기반으로 바이브코딩한 것입니다. 당연히 해당 문서들도 모두 AI와 함께 작성하였습니다.

TIPS)

- 유료) 바이브코딩 툴을 이용한다면 그를 활용하세요.
- 무료) repomix 를 이용해 코드베이스 전체를 google ai studio 에 넣어서 정확한 내용 기반으로 LLM 과 분석하세요. (Google AI Studio 를 쓰는 이유는 처리할 수 있는 Token 수가 1M으로 타 서비스 대비 압도적으로 많고 무료이기 때문)

  [repomix 활용방법](https://hateslop.notion.site/AgentOps-285219be4e0b8068974cc572a53bf20a)

  [gitingest : GitHub 저장소를 LLM 친화적인 텍스트로 변환하는 도구](https://discuss.pytorch.kr/t/gitingest-github-llm/6896)

  [deepwiki : Github 기반 프로젝트 분석방법](https://deepwiki.org/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## ⚡ 빠른 시작

- hateslop organization에서 fork 한 것이라 가정
- docker desktop 설치 및 실행한 상태라 가정

```bash
# 1. Fork & Clone
git clone https://github.com/YOUR_USERNAME/chatbot-project.git
cd chatbot-project

# 2. .env 파일 생성 및 API 키 입력
cp .env.example .env
nano .env  # OPENAI_API_KEY 입력

# 3. Docker 실행
docker compose up --build

# 4. 브라우저에서 http://localhost:5001 접속
```

정상 작동 화면

메인 페이지

![메인 페이지](static/images/hateslop/example1.png)

상세 페이지

![상세 페이지](static/images/hateslop/example2.png)

채팅 페이지

![채팅 페이지](static/images/hateslop/example3.png)

## 📚 문서 가이드

| 문서                                        | 내용                         | 비고     |
| ------------------------------------------- | ---------------------------- | -------- |
| [README.md](README.md) ⭐⭐                 | 프로젝트 개요                | 현재문서 |
| [ARCHITECTURE.md](ARCHITECTURE.md) ⭐⭐     | 시스템 아키텍처              | 필독     |
| [DOCKER-GUIDE.md](DOCKER-GUIDE.md) ⭐⭐     | 개발 환경 구성               | 필독     |
| [RENDER-GUIDE.md](RENDER-GUIDE.md) ⭐⭐     | 배포 (Render - 무료, 권장)   | 필독     |
| [ADVANCED_TOPICS.md](ADVANCED_TOPICS.md) 🚀 | 성능 개선 & 최신 기술 트렌드 | (심화)   |

---

## 🎯 프로젝트 개요

- 📖 학습 목표: RAG, Embedding, LLM, Vector Database
- 👥 협업 방식: 프로듀서가 기획한 내용을 바탕으로 캐릭터 챗봇을 완성
- 🚀 배포: Render.com (무료) 또는 Railway를 통한 프로덕션 배포
- 🐳 환경: Docker로 일관된 개발 환경 보장

### 핵심 기능

- 🤖 OpenAI GPT 기반 대화 생성
- 📚 RAG (Retrieval-Augmented Generation)를 통한 지식 기반 답변
- 💾 ChromaDB를 활용한 임베딩 벡터 저장
- 🧠 LangChain 기반 대화 메모리 관리
- 🎨 Vanilla JavaScript 기반 웹 인터페이스
- 🐳 Docker를 통한 환경 일관성 보장

### 기술 스택

- Backend: Flask (Python 3.11)
- AI: OpenAI API, LangChain, ChromaDB
- Frontend: Vanilla JavaScript, HTML, CSS
- Deployment: Docker, Render.com (권장) / Railway
- Version Control: Git, GitHub

## 🏗️ 프로젝트 구조

```
chatbot-project/
├── app.py                     # 🚫 템플릿 (수정 원한다면 의존성 있는 파일함께 수정)
├── services/
│   ├── __init__.py
│   └── chatbot_service.py     # ✏️ 학회원 구현 파일 (AI 로직)
├── config/
│   └── chatbot_config.json    # ✏️ 챗봇 설정 (예시)
├── static/
│   ├── data/
│   │     └── chardb_text/   # ✏️ 텍스트 데이터 (예시)
│   ├── images/
│   │   └── something/           # ✏️ 이미지 파일
│   ├── videos/
│   │   └── something/           # ✏️ 비디오 파일 (선택)
│   ├── css/
│   │   └── style.css          # # ✏️ 학회원 구현 파일 (스타일)
│   └── js/
│       └── chatbot.js         # # ✏️ 학회원 구현 파일 (Front 로직)
├── templates/
│   ├── index.html             # ✏️ 학회원 구현 파일
│   ├── detail.html            # ✏️ 학회원 구현 파일
│   └── chat.html              # ✏️ 학회원 구현 파일
├── Dockerfile                 # 🚫 템플릿
├── docker-compose.yml         # 🚫 템플릿
├── requirements.txt           # 🚫 템플릿
├── .env.example               # 참고용
└── README.md                  # 현재 파일
```

### static/js/chatbot.js

JS-파이썬 매핑:

- 이 JS 파일은 `chat.html`에서 동적으로 로드되어, 사용자 메시지를 `/api/chat`으로 보내고, 서버(파이썬) 응답을 화면에 표시하는 역할을 합니다.

- `chatbot.js` 참고:
  - 기본 메시지 전송 로직(이벤트 리스너, fetch API, DOM 업데이트)은 `chatbot.js`를 예시로 삼으면 됩니다.
  - 단, 현재 프론트엔드는 백엔드에서 이미지 경로를 전달할 경우에만 이미지를 표시하도록 되어 있습니다. 이미지 검색 기능을 구현하기 전까지는 이미지가 표시되지 않습니다.
  - 추가적으로, 응답 형태나 포맷이 달라질 경우(예: JSON 구조 변경), 그에 맞게 프런트 처리 로직도 수정해야 합니다.

### static/data/chatbot/ 폴더

임베딩 벡터 / 필요한 데이터 저장:

- 각 팀은 static/data/chatbot/ 폴더 아래에, 임베딩 결과나 기타 필요한 텍스트, 이미지, 스크립트 파일 등을 저장합니다.
- `chatbot_service.py`에서 임베딩 데이터를 불러올 때도 이 경로를 기준으로 맞춰주세요.

### 추가 패키지 requirements.txt

임베딩 패키지, 기타 라이브러리:

- 예: `numpy`, `pandas`, `openai`, `scikit-learn` 등등.
- 새로운 라이브러리를 사용하면, 반드시 `requirements.txt`에 추가하여 다른 팀원/환경에서도 동일한 버전으로 설치 가능하도록 해주세요.
- 해당 내용을 추가하게 되면 Docker 이미지를 새롭게 `build` 해야 합니다. 자세한 가이드는 [DOCK-GUIDE.md](DOCKER-GUIDE.md)에서 "상황 2: 새로운 Python 라이브러리를 추가하는 경우" 를 참고하세요.

### 📁 파일별 역할

#### 🚫 템플릿 파일

> _커스텀 원하시면 수정하셔도 되지만, 의존성을 가진 파일을 같이 수정하셔야 합니다._

- `app.py`: Flask 애플리케이션 핵심 로직
- `templates/*.html`: 웹 UI 템플릿
- `static/css/`, `static/js/`: 프론트엔드 리소스
- `Dockerfile`, `docker-compose.yml`: Docker 설정
- `requirements.txt`: Python 의존성

#### ✏️ 작성/수정할 파일

- `services/chatbot_service.py`: AI 로직 구현 (RAG, Embedding, LLM)
- `config/chatbot_config.json`: 챗봇 설정 (이름, 성격, 시스템 프롬프트)
- `static/data//*`: 텍스트 데이터 (json, markdown, txt 자유롭게 사용하시면 됩니다.)
- `static/images//*`: 챗봇 관련 이미지

## 📚 학습 자료

### 공식 문서

1. OpenAI API Documentation
   - https://platform.openai.com/docs
2. LangChain Documentation
   - https://python.langchain.com/docs
3. ChromaDB Documentation
   - https://docs.trychroma.com/

### 추천 논문

1. RAG 기초: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)

   - https://arxiv.org/abs/2005.11401

2. Self-RAG: "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2024)
   - https://arxiv.org/abs/2310.11511

더 많은 자료: [ADVANCED_TOPICS.md](ADVANCED_TOPICS.md#-관련-논문-및-연구)

## 👥 협업 워크플로우

### Git 협업 방식 (Fork & Collaborator)

워크플로우 단계별 설명

#### 1️⃣ 초기 셋업 (조원A)

```bash
# HateSlop Organization에서 Fork
# hateslop 올가니케이션 GitHub 웹에서 Fork 버튼 클릭

# Clone & 초기 설정
git clone https://github.com/조원A/chatbot-project.git
cd chatbot-project

# 개발 환경 구축
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
docker compose up --build
```

#### 2️⃣ Collaborator 초대 (조원A)

1. GitHub Repository 페이지 → Settings 탭
2. 왼쪽 메뉴 → Collaborators
3. Add people → 조원B의 GitHub 아이디 입력
4. 조원B 이메일로 초대 링크 발송

#### 3️⃣ 협업 시작 (조원B)

```bash
# 초대 수락 후 Clone
git clone https://github.com/조원A/chatbot-project.git
cd chatbot-project

# 개발 브랜치 생성
git checkout -b feature/chatbot-service

# 개발 환경 구축
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력
docker compose up --build

# 작업 후 커밋 & 푸시
git add .
git commit -m "feat: implement RAG search logic"
git push origin feature/chatbot-service
```

#### 4️⃣ Pull Request & 코드 리뷰

1. 조원B: GitHub에서 New Pull Request 생성
   - Base: `조원A/chatbot-project` (main)
   - Compare: `feature/chatbot-service`
2. 조원A: PR 리뷰 및 피드백
3. 조원B: 피드백 반영 후 추가 커밋
4. 조원A: 리뷰 완료 후 Merge

#### 5️⃣ 포트폴리오 저장 (조원B)

```bash
# 조원A의 레포지토리를 조원B 계정으로 Fork
# GitHub 웹에서 조원A/chatbot-project → Fork 버튼 클릭

# 본인 레포지토리에 최종 작업물 저장 완료
# URL: https://github.com/조원B/chatbot-project
```

### 📋 협업 규칙 (권장사항)

- 브랜치 전략

  - `main`: 안정적인 배포 버전
  - `feature/*`: 기능 개발 브랜치
  - `fix/*`: 버그 수정 브랜치

- 커밋 컨벤션

  ```
  feat: 새로운 기능 추가
  fix: 버그 수정
  docs: 문서 수정
  style: 코드 포맷팅, 세미콜론 누락 등
  refactor: 코드 리팩토링
  test: 테스트 코드
  chore: 빌드 작업, 패키지 매니저 설정 등
  ```

- PR 템플릿 (권장)

  ```markdown
  ## 작업 내용

  - [ ] RAG 검색 로직 구현
  - [ ] ChromaDB 연동
  - [ ] 테스트 완료

  ## 테스트 방법

  1. Docker 환경 실행
  2. http://localhost:5001/chat 접속
  3. 대화 테스트

  ## 스크린샷

  (선택사항)
  ```

---

## 📦 최종 제출물 안내

### 🎯 제출 요구사항

과제 완료 후 아래 2가지를 제출해주세요:

#### 1️⃣ 배포된 애플리케이션 URL

```
🌐 배포 URL: https://your-app-name.onrender.com
```

> 📝 배포 방법: [RENDER-GUIDE.md](RENDER-GUIDE.md) 참고

#### 2️⃣ 프로젝트 README.md 작성

팀별로 Fork한 Repository의 README.md에 다음 내용을 상세히 작성해주세요:

---

### 📋 README.md 필수 작성 항목

#### 1. 📐 시스템 아키텍처

프로젝트의 전체 구조를 설명해주세요.

예시:

```markdown
## 🏗️ 시스템 아키텍처

### 전체 구조도

[다이어그램 또는 이미지]

### 데이터 흐름

사용자 입력 → Flask API → ChatbotService → RAG 검색 (ChromaDB) → OpenAI API → 응답 생성
```

#### 2. 🛠️ 사용한 기술 스택

예시:

```markdown
## 🛠️ 기술 스택

### Backend

- Flask 3.0: RESTful API 서버
- OpenAI API (gpt-4o-mini): 대화 생성 엔진
- ChromaDB: 벡터 데이터베이스 (임베딩 저장/검색)
- LangChain: LLM 통합 및 메모리 관리

### Frontend

- Vanilla JavaScript: 프레임워크 없는 순수 JS
- HTML5/CSS3: 반응형 UI

### Infrastructure

- Docker: 컨테이너화
- Render.com: 클라우드 배포
```

#### 3. 💡 기술 선택 이유

각 기술을 선택한 이유를 구체적으로 설명해주세요.

예시:

```markdown
## 💡 기술 선택 이유

### ChromaDB를 선택한 이유

- 이유 1: Python 네이티브 지원으로 Flask와 통합이 쉬움
- 이유 2: 별도 서버 설치 없이 임베디드 모드로 사용 가능
- 이유 3: 벡터 유사도 검색이 빠르고 정확함

### RAG 패턴을 적용한 이유

- 문제 인식: LLM은 학습 데이터에 없는 최신 정보나 특정 도메인 지식에 약함
- 해결 방법: ChromaDB에 서강대 관련 지식을 저장하고, 관련 정보를 검색하여 프롬프트에 포함
- 효과: 환각(Hallucination) 감소 및 정확한 답변 생성
```

#### 4. ⚠️ 개발 시 겪은 문제점

예시:

```markdown
## ⚠️ 개발 중 문제점

### 문제 1: RAG 검색 결과의 품질 문제

- 현상: 사용자 질문과 무관한 문서가 검색됨
- 원인: 임베딩 모델이 한국어 유사도를 제대로 판단하지 못함
- 증상: "학식 추천해줘" 질문에 "도서관 위치" 답변 반환

### 문제 2: Docker 환경에서 ChromaDB 데이터 손실

- 현상: 컨테이너 재시작 시 임베딩 데이터가 사라짐
- 원인: Volume 마운트 설정 누락
```

#### 5. ✅ 문제 해결 방법

예시:

````markdown
## ✅ 해결 방법

### 문제 1 해결: 유사도 임계값 조정

시도한 방법들:

1. ❌ 임베딩 모델 변경 → 큰 효과 없음
2. ✅ 유사도 점수 임계값 0.7로 상향 조정 → 정확도 85% 달성
3. ✅ 메타데이터 필터링 추가 (카테고리별 검색)

최종 구현 코드:
\```python
def \_search_similar(self, query: str, threshold=0.7):
results = self.collection.query(
query_embeddings=embedding,
n_results=5
) # 유사도 필터링
filtered = [r for r in results if r['distance'] < threshold]
return filtered
\```

### 문제 2 해결: Docker Volume 설정

docker-compose.yml 수정:
\```yaml
volumes:

- ./static/data/chatbot/chardb_embedding:/app/static/data/chatbot/chardb_embedding
  \```
````

#### 6. 🚀 성능 개선 노력

예시:

```markdown
## 🚀 성능 개선

### 개선 1: 응답 속도 최적화

- Before: 평균 5초 소요
- After: 평균 2초로 단축 (60% 개선)
- 방법:
  - ChromaDB 쿼리 결과 캐싱
  - OpenAI API 호출 시 max_tokens 제한

### 개선 2: 메모리 사용량 감소

- Before: Docker 컨테이너 메모리 800MB 사용
- After: 400MB로 절반 감소
- 방법: 불필요한 라이브러리 제거, 임베딩 벡터 차원 축소
```

#### 7. 😔 아쉬웠던 점

예시:

```markdown
## 😔 아쉬웠던 점

### 1. 멀티모달 기능 미구현

- 계획: 이미지 임베딩을 통한 이미지 검색 기능
- 현실: 시간 부족으로 텍스트 검색만 구현
- 향후 계획: CLIP 모델을 활용한 이미지-텍스트 통합 검색 도입

### 2. 테스트 코드 부족

- 현황: 핵심 로직에 대한 단위 테스트 없음
- 문제: 리팩토링 시 기존 기능 동작 보장 어려움
- 교훈: TDD(Test-Driven Development) 방식 도입 필요성 느낌
```

#### 8. 🤔 회고 및 성찰

예시:

```markdown
## 🤔 회고 및 성찰

### 기술적 성장

- RAG 이해도 향상: 이론으로만 알던 RAG를 실제 구현하며 내부 동작 원리 이해
- 프롬프트 엔지니어링: 시스템 프롬프트 최적화를 통해 답변 품질 30% 개선
- Vector Database 경험: ChromaDB를 통해 벡터 검색의 강력함을 체감

### 협업 경험

- Git 협업: PR 리뷰를 통해 코드 품질 향상
- 역할 분담: 프로듀서-엔지니어 간 명확한 업무 분담으로 효율성 증가

### 아쉬운 점 및 개선 방향

- 시간 관리: 초반 설계에 시간을 더 투자했다면 리팩토링 시간 단축 가능
- 문서화: 개발 중 문서화를 소홀히 하여 나중에 일괄 작성 → 부담 증가
- 다음 프로젝트에서는: 애자일 방식으로 1주 단위 스프린트 도입 계획
```

---

### 🎤 최종 발표 PPT 가이드

위의 README.md 내용을 기반으로 팀별 최종 발표 PPT를 작성해주세요.
