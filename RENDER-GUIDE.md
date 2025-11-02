## 🚀 Render.com 배포 가이드

> Railway 무료 플랜의 크레딧 제한 때문에 배포가 어렵나요?
>
> Render.com은 완전 무료이며 Docker를 지원하고, ChromaDB + LangChain을 제한 없이 사용할 수 있습니다!

---

### 🤔 Render.com, 왜 추천할까요?

- 완전 무료: 크레딧 제한 없음, 학생들이 부담 없이 사용 가능
- Docker 네이티브 지원: Railway와 동일하게 Dockerfile 그대로 사용
- ChromaDB + LangChain 완벽 지원: Vector Database와 메모리 관리 학습 가능
- 자동 재배포: GitHub 연동 시 git push만으로 자동 배포
- 간단한 설정: 웹 대시보드에서 클릭 몇 번으로 완료

무료 플랜 제한:

- ⚠️ 15분 동안 요청이 없으면 서비스가 sleep 모드로 전환 (다음 요청 시 재시작, 약 30초 소요)
- 💚 교육용 데모/포트폴리오로는 충분
  원래 vercel 을 이용하려고 했는데, 학습용으로는 충분치 않아서 다음에 파이널 프로젝트때 활용해보는 걸로 합시다!

---

### 🔧 배포 환경 구축을 위한 준비물

배포를 시작하기 전에 다음을 확인해주세요:

- [ ] 로컬에서 완벽히 동작하는 프로젝트: `DOCKER-GUIDE.md`를 따라 로컬(`http://localhost:5001`)에서 챗봇이 완벽하게 동작하는 것을 확인
- [ ] Render 계정: [Render 홈페이지](https://render.com/)에서 GitHub 계정으로 가입 (필수!)
- [ ] GitHub 저장소: 프로젝트가 GitHub에 push되어 있어야 함

---

### 🚀 Render 웹 대시보드를 통한 배포

#### 1단계: Render 계정 생성 및 GitHub 연동

1. https://render.com/ 접속
2. `Get Started for Free` 버튼 클릭
3. GitHub 계정으로 가입
4. GitHub 연동 승인

#### 2단계: 새 Web Service 생성

1. Render 대시보드에서 `New +` 버튼 클릭
2. `Web Service` 선택
3. `Build and deploy from a Git repository` 선택
4. `Next` 클릭

#### 3단계: GitHub 저장소 연결

1. 본인의 `chatbot-project` 저장소 repository 찾기
2. `Connect` 버튼 클릭

> 저장소가 안 보인다면?
>
> - `Configure account` 링크 클릭
> - GitHub에서 Render에 저장소 접근 권한 부여
> - 다시 돌아와서 새로고침

#### 4단계: 서비스 설정

기본 정보:

- Name: `sogang-chatbot` (또는 원하는 이름, URL의 일부가 됩니다)
- Region: `Singapore` (한국과 가장 가까움)
- Branch: `main` (또는 배포할 브랜치)

빌드 설정:

- Runtime: `Docker` 선택
- Dockerfile Path: `Dockerfile` (기본값 그대로)

Plan:

- Instance Type: `Free` 선택

#### 5단계: 환경 변수 설정

페이지 하단의 Environment Variables 섹션에서:

1. `Add Environment Variable` 버튼 클릭
2. 다음 변수들을 추가:

| Key              | Value                         |
| ---------------- | ----------------------------- |
| `OPENAI_API_KEY` | `sk-your-actual-api-key-here` |
| `FLASK_ENV`      | `production`                  |
| `PORT`           | `5000`                        |

> ⚠️ `OPENAI_API_KEY`에 본인의 실제 API 키를 입력하세요!

#### 6단계: 배포 시작!

1. 모든 설정을 확인
2. 페이지 하단의 `Create Web Service` 버튼 클릭
3. 자동으로 빌드 및 배포가 시작됩니다! 🎉

#### 7단계: 배포 진행 상황 확인

대시보드에서 실시간 로그를 볼 수 있습니다:

배포 단계:

1. 📦 Building - Docker 이미지 빌드 중 (약 3-5분)
2. 🚀 Deploying - 컨테이너 실행 중
3. ✅ Live - 배포 완료!

로그에서 다음과 같은 메시지를 확인:

```
==> Building...
==> Deploying...
==> Your service is live 🎉
```

#### 8단계: 웹사이트 접속

배포가 완료되면:

1. 대시보드 상단에 URL이 표시됩니다
   - 예: `https://sogang-chatbot.onrender.com`
2. URL을 클릭하여 챗봇 접속!

> ⚠️ 첫 접속 시 30초 정도 걸릴 수 있습니다 (서비스 시작 중)

---

### 🔄 코드 변경 후 재배포

코드를 수정한 후:

```bash
git add .
git commit -m "Update chatbot"
git push origin main
```

자동으로 재배포됩니다! GitHub에 push하면 Render가 자동 감지하여 빌드 시작 🎉

### 💡 로컬 Docker vs Render Docker: 무엇이 같고 다를까?

| 항목        | 로컬 환경 (Docker Desktop) | Render 환경      |
| ----------- | -------------------------- | ---------------- |
| 기반 기술   | Docker                     | Docker           |
| Dockerfile  | 동일한 파일 사용           | 동일한 파일 사용 |
| Python 버전 | 3.11                       | 3.11             |
| 의존성      | requirements.txt           | requirements.txt |
| ChromaDB    | ✅ 작동                    | ✅ 작동          |
| LangChain   | ✅ 작동                    | ✅ 작동          |
| 차이점      | 항상 실행                  | 15분 후 sleep    |

핵심: 로컬에서 완벽히 동작한다면, Render에서도 동일하게 동작합니다!

---

### 🛠️ 추가 설정

#### 커스텀 도메인 설정

무료 플랜에서도 커스텀 도메인을 연결할 수 있습니다:

1. `Settings` 탭 → `Custom Domains` 섹션
2. 본인의 도메인 추가 (예: `chatbot.yourdomain.com`)
3. DNS 설정 안내에 따라 CNAME 레코드 추가

#### 환경 변수 수정

환경 변수를 나중에 변경하려면:

1. `Environment` 탭 클릭
2. 변수 수정 또는 추가
3. `Save Changes` 클릭
4. 자동으로 재배포됩니다

#### 수동 재배포

코드 변경 없이 서비스를 다시 시작하려면:

1. 대시보드 우측 상단 `Manual Deploy` 메뉴
2. `Deploy latest commit` 클릭

---

### 🐛 Troubleshooting

#### 문제 1: 빌드가 실패해요

증상: Build failed 메시지

해결 방법:

1. 로그 확인:
   - `Logs` 탭에서 에러 메시지 확인
2. 흔한 원인:
   - `requirements.txt`에 오타
   - Dockerfile 경로 잘못 지정
   - 로컬에서 Docker 빌드 테스트:
     ```bash
     docker compose build
     ```

#### 문제 2: 배포는 성공했는데 500 에러가 발생해요

증상: 사이트에 접속하면 `Internal Server Error`

해결 방법:

1. 로그 확인:

   - `Logs` 탭 → `Deploy Logs` 또는 `Runtime Logs`
   - Python 에러 메시지 확인

2. 환경 변수 확인:

   - `Environment` 탭에서 `OPENAI_API_KEY` 확인
   - 올바른 API 키인지 검증

3. ChromaDB 데이터 확인:
   - `static/data/chatbot/chardb_embedding/` 폴더가 Git에 포함되었는지 확인
   - `.gitignore`에서 제외되지 않았는지 확인

#### 문제 3: 사이트가 너무 느려요 (첫 접속 시)

증상: 첫 접속 시 30초 이상 걸림

원인: 무료 플랜의 sleep 모드 (정상 동작)

해결 방법:

방법 1: 그대로 사용 (권장)

- 교육용/포트폴리오로는 충분
- 데모 전에 미리 한 번 접속하여 깨워두기

방법 2: 외부 Ping 서비스 사용

- [UptimeRobot](https://uptimerobot.com/) 같은 서비스로 5분마다 ping
- 서비스가 sleep 모드로 가는 것을 방지
- ⚠️ 하지만 Render 정책상 권장하지 않음

방법 3: 유료 플랜 업그레이드

- Starter Plan: $7/월
- Sleep 모드 없음, 더 많은 메모리

#### 문제 4: ChromaDB 데이터가 사라져요

증상: 재배포 후 Vector DB 데이터가 초기화됨

원인: Render의 컨테이너는 재시작 시 파일 시스템이 초기화됩니다.

해결 방법:

방법 1: 임베딩 데이터를 Git에 포함 (권장)

```bash
# .gitignore에서 ChromaDB 데이터 제외하지 않기
git add static/data/chatbot/chardb_embedding/
git commit -m "Add ChromaDB embeddings"
git push origin main
```

방법 2: Render Persistent Disk 사용 (고급)

1. `Settings` → `Disks`
2. `Add Disk` 클릭
3. Mount Path: `/app/static/data/chatbot/chardb_embedding`
4. Size: 1GB (무료)

> ⚠️ 무료 플랜에서도 1GB 디스크는 무료입니다!

#### 문제 5: 이미지나 CSS가 안 보여요

증상: 웹사이트는 뜨는데 스타일이 깨짐

해결 방법:

1. `static/` 폴더가 Git에 포함되었는지 확인

   ```bash
   git ls-files static/
   ```

2. Flask 라우팅 확인 (`app.py`의 static 경로)

3. 브라우저 개발자 도구 (F12) → Network 탭에서 404 에러 확인

#### 문제 6: "Failed to fetch repository" 에러

증상: Render가 GitHub 저장소에 접근 못함

해결 방법:

1. Render 대시보드에서 `Account Settings` 클릭
2. `Git Providers` → GitHub `Reconnect` 클릭
3. GitHub에서 Render 앱 권한 재승인
4. 저장소 접근 권한 확인
