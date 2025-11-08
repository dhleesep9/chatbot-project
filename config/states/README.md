# State Machine Configuration Guide

이 디렉토리는 게임 상태(state) 정보를 관리하는 JSON 파일들을 포함합니다.

## 📁 파일 구조

```
config/states/
├── README.md           # 이 파일
├── start.json          # 시작단계
├── icebreak.json       # 아이스브레이크단계
└── daily_routine.json  # 일상루틴단계
```

---

## 📋 State JSON 파일 구조

각 state JSON 파일은 다음과 같은 구조를 가집니다:

```json
{
  "name": "상태 이름 (화면에 표시됨)",
  "description": "상태에 대한 설명",
  "from_states": ["이전 상태들"],
  "to_states": ["다음 상태들"],
  "narration": "상태 진입 시 표시할 나레이션 (선택사항)",
  "transitions": [
    {
      "name": "전이 이름",
      "trigger_type": "트리거 타입",
      "conditions": { "조건들" },
      "next_state": "다음 상태",
      "transition_narration": "전이 시 표시할 나레이션"
    }
  ]
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `name` | string | 화면에 표시될 상태 이름 (예: "시작단계") |
| `description` | string | 상태에 대한 설명 |
| `from_states` | array | 이 상태로 전이 가능한 이전 상태들 |
| `to_states` | array | 이 상태에서 전이 가능한 다음 상태들 |
| `narration` | string\|null | 상태 진입 시 표시할 나레이션 |
| `transitions` | array | 상태 전이 규칙들 |

---

## 🎯 트리거 타입 (Trigger Types)

### 1. `affection_increase`
**호감도 증가량 체크**

호감도가 일정량 이상 증가했을 때 트리거됩니다.

```json
{
  "trigger_type": "affection_increase",
  "conditions": {
    "affection_increase_min": 1
  }
}
```

**조건 필드:**
- `affection_increase_min`: 최소 호감도 증가량

**사용 예시:**
- start → icebreak: 첫 대화로 호감도가 1 이상 증가

---

### 2. `affection_threshold`
**호감도 절대값 체크**

현재 호감도가 특정 값 이상일 때 트리거됩니다.

```json
{
  "trigger_type": "affection_threshold",
  "conditions": {
    "affection_min": 10
  }
}
```

**조건 필드:**
- `affection_min`: 최소 호감도 (절대값)

**사용 예시:**
- icebreak → daily_routine: 호감도 10 달성

---

### 3. `affection_and_subjects`
**호감도 + 탐구과목 복합 조건**

호감도와 선택과목 개수를 모두 체크합니다.

```json
{
  "trigger_type": "affection_and_subjects",
  "conditions": {
    "affection_min": 10,
    "subjects_count": 2
  }
}
```

**조건 필드:**
- `affection_min`: 최소 호감도
- `subjects_count`: 최소 선택과목 개수

**사용 예시:**
- 호감도 10 + 탐구과목 2개 선택 완료 시

---

## 🔄 상태 흐름 예시

```
start (시작단계)
  │
  │ [trigger: affection_increase >= 1]
  ↓
icebreak (아이스브레이크단계)
  │
  │ [trigger: affection_threshold >= 10]
  ↓
daily_routine (일상루틴단계)
```

---

## 📝 새로운 State 추가하기

1. **JSON 파일 생성**
   ```bash
   config/states/new_state.json
   ```

2. **State 정보 작성**
   ```json
   {
     "name": "새로운단계",
     "description": "새로운 단계 설명",
     "from_states": ["이전_상태"],
     "to_states": ["다음_상태"],
     "narration": null,
     "transitions": [...]
   }
   ```

3. **config/chatbot_config.json 수정**
   ```json
   "state_machine": {
     "available_states": ["start", "icebreak", "daily_routine", "new_state"]
   }
   ```

4. **서버 재시작**

---

## ⚠️ 주의사항

1. **JSON 문법**: 유효한 JSON 형식을 유지해야 합니다
2. **순환 참조 방지**: from_states와 to_states가 순환하지 않도록 주의
3. **파일명 = 상태명**: JSON 파일명과 상태 ID가 일치해야 합니다
4. **트리거 우선순위**: transitions 배열의 순서대로 평가됩니다

---

## 🛠️ 트러블슈팅

### State 로드 실패
```
[WARN] State 파일 없음: config/states/xxx.json
```
→ JSON 파일이 존재하는지 확인하세요

### 트리거 동작 안함
```
[WARN] Unknown trigger_type: xxx
```
→ trigger_type이 올바른지 확인하세요

### 상태 전이 안됨
- conditions 값이 올바른지 확인
- 로그에서 `[STATE_TRANSITION]` 메시지 확인
