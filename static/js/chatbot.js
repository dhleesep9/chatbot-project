console.log("챗봇 JS 로드 완료");

// DOM 요소
const chatArea = document.querySelector(".chat-area");
const username = chatArea ? chatArea.dataset.username : "사용자";
const chatLog = document.getElementById("chat-log");
const userMessageInput = document.getElementById("user-message");
const sendBtn = document.getElementById("send-btn");
const videoBtn = document.getElementById("videoBtn");
const imageBtn = document.getElementById("imageBtn");

// localStorage 키
const STORAGE_KEY_PREFIX = `chatbot_game_state_${username}_`;
const CHAT_LOG_KEY = `${STORAGE_KEY_PREFIX}chat_log`;
const GAME_STATE_KEY = `${STORAGE_KEY_PREFIX}game_state`;

// 메시지 전송 함수
async function sendMessage(isInitial = false) {
  let message;

  if (isInitial) {
    message = "init";
  } else {
    message = userMessageInput.value.trim();
    if (!message) return;

    appendMessage("user", message);
    userMessageInput.value = "";
  }

  // 로딩 표시
  const loadingId = appendMessage("bot", "생각 중...");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: message,
        username: username,
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("[API] 전체 응답 데이터:", data);

    // 로딩 메시지 제거
    removeMessage(loadingId);

    // 응답 데이터 검증
    if (!data || typeof data !== "object") {
      throw new Error("서버 응답이 올바르지 않습니다.");
    }

    // 능력치 업데이트
    if (data.abilities !== undefined) {
      updateAbilitiesDisplay(data.abilities);
    }

    // 호감도 업데이트
    if (data.affection !== undefined) {
      updateAffectionDisplay(data.affection);
    }

    // 체력 업데이트
    if (data.stamina !== undefined) {
      updateStaminaDisplay(data.stamina);
    }

    // 멘탈 업데이트 (mental 값이 있으면 상태 계산 및 표시)
    if (data.mental !== undefined) {
      updateMentalDisplay(data.mental, data.stamina);
    }

    // 시간표 업데이트
    if (data.schedule !== undefined) {
      updateScheduleDisplay(data.schedule, data.game_state || "daily_routine");
    }

    // 게임 날짜 업데이트
    if (data.current_date) {
      updateGameDate(data.current_date);
    }

    // 응답 파싱 및 표시 (나레이션보다 먼저 표시)
    let replyText, imagePath;
    if (data.reply) {
      if (typeof data.reply === "object" && data.reply !== null) {
        replyText = data.reply.reply || data.reply;
        imagePath = data.reply.image || null;
      } else {
        replyText = data.reply;
        imagePath = null;
      }
      if (replyText) {
        appendMessage("bot", replyText, imagePath);
      }
    } else {
      // reply가 없는 경우 기본 메시지 표시
      console.warn("[API] 응답에 reply가 없습니다:", data);
      appendMessage("bot", "안녕하세요! 게임을 시작하겠습니다.");
    }

    // 나레이션 표시 (reply 이후에 표시)
    if (data.narration) {
      appendNarration(data.narration);
    }

    // 사설모의고사 성적표 표시 (나레이션에 포함되어 있으므로 추가 UI는 선택사항)
    // 나레이션에 이미 성적표가 포함되어 있음

    // 현재 호감도 업데이트
    if (data.affection !== undefined) {
      currentAffection = data.affection;
    }

    // 현재 체력 업데이트 (stamina가 정의되지 않았으면 기본값 유지)
    if (data.stamina !== undefined) {
      currentStamina = data.stamina;
    }

    // 서가윤 뻐끔뻐끔 애니메이션 시작 (호감도 전달)
    startSpeakingAnimation(currentAffection);

    // 게임 상태 저장 (F5 새로고침 시 복원하기 위해)
    saveGameState(data);
  } catch (err) {
    console.error("메시지 전송 에러:", err);
    removeMessage(loadingId);

    // 에러 메시지 표시 (초기화 직후가 아닌 경우에만)
    const errorMessage = err.message || "알 수 없는 오류";
    console.error("[ERROR] 상세 오류:", errorMessage);

    // 네트워크 오류나 초기화 직후가 아닌 경우에만 에러 메시지 표시
    if (!message || message !== "init") {
      appendMessage(
        "bot",
        "죄송합니다. 오류가 발생했습니다. 다시 시도해주세요."
      );
    } else {
      // init 메시지 실패 시 조용히 처리 (페이지가 막 리로드되었을 수 있음)
      console.warn("[INIT] 초기 메시지 전송 실패 (무시됨):", errorMessage);
    }
  }
}

// 능력치 표시 업데이트
function updateAbilitiesDisplay(abilities) {
  if (!abilities) {
    return;
  }

  const abilitiesDisplay = document.getElementById("abilities-display");
  if (abilitiesDisplay) {
    abilitiesDisplay.style.display = "block";
  }

  const abilityNames = ["국어", "수학", "영어", "탐구1", "탐구2"];
  abilityNames.forEach((name) => {
    const abilityElem = document.getElementById(`ability-${name}`);
    if (abilityElem && abilities[name] !== undefined) {
      // 소수점 첫째자리까지만 표시
      const value = parseFloat(abilities[name]);
      if (!isNaN(value)) {
        abilityElem.textContent = value.toFixed(1);
      } else {
        abilityElem.textContent = abilities[name];
      }
    }
  });
}

// 호감도 표시 업데이트
function updateAffectionDisplay(affection) {
  const affectionValue = document.getElementById("affection-value");
  if (affectionValue) {
    affectionValue.textContent = affection;
  }

  // 현재 호감도 저장
  currentAffection = affection;

  // 호감도 변경 시 기본 이미지 업데이트 (애니메이션 중이 아닐 때만)
  if (!speakingAnimationInterval) {
    const sideImage = document.querySelector(".side-image");
    if (sideImage) {
      const defaultImage = getDefaultImageByAffectionAndStamina(
        affection,
        currentStamina
      );
      sideImage.src = defaultImage;
    }
  }
}

// 체력 표시 업데이트
function updateStaminaDisplay(stamina) {
  const staminaValue = document.getElementById("stamina-value");
  const staminaEfficiency = document.getElementById("stamina-efficiency");

  if (staminaValue) {
    staminaValue.textContent = stamina;
  }

  // 체력에 따른 효율 계산: 효율(%) = 100 + (체력 - 30)
  const efficiency = 100 + (stamina - 30);
  if (staminaEfficiency) {
    staminaEfficiency.textContent = `(효율: ${efficiency}%)`;
  }

  // 현재 체력 저장
  currentStamina = stamina;

  // 체력 변경 시 이미지 업데이트 (애니메이션 중이 아닐 때만)
  if (!speakingAnimationInterval) {
    const sideImage = document.querySelector(".side-image");
    if (sideImage) {
      const defaultImage = getDefaultImageByAffectionAndStamina(
        currentAffection,
        currentStamina
      );
      sideImage.src = defaultImage;
    }
  }
}

// 멘탈 표시 및 상태 업데이트
function updateMentalDisplay(mental, stamina) {
  // 멘탈 값 표시
  const mentalValue = document.getElementById("mental-value");
  if (mentalValue) {
    mentalValue.textContent = mental;
  }

  // 상태 계산
  const statuses = [];

  // 병듦 상태: 체력 10 이하
  if (stamina !== undefined && stamina <= 10) {
    statuses.push({ type: "sick", text: "병듦" });
  }

  // 번아웃 상태: 멘탈 40 미만
  if (mental < 40) {
    statuses.push({ type: "burnout", text: "번아웃" });
  }

  // 평범한 상태: 위 상태가 없을 때만 표시
  if (statuses.length === 0) {
    statuses.push({ type: "normal", text: "평범" });
  }

  // 상태 표시 업데이트
  const statusContainer = document.getElementById("character-status");
  if (statusContainer) {
    statusContainer.innerHTML = "";
    statuses.forEach((status) => {
      const badge = document.createElement("div");
      badge.className = `status-badge status-${status.type}`;
      badge.textContent = status.text;
      statusContainer.appendChild(badge);
    });
  }
}

// 시간표 표시 업데이트
function updateScheduleDisplay(schedule, gameState) {
  const scheduleDisplay = document.getElementById("schedule-display");
  if (!scheduleDisplay) {
    return;
  }

  // daily_routine 상태일 때만 표시
  if (gameState === "daily_routine") {
    scheduleDisplay.style.display = "block";
  } else {
    scheduleDisplay.style.display = "none";
    return;
  }

  if (!schedule || typeof schedule !== "object") {
    return;
  }

  const subjects = ["국어", "수학", "영어", "탐구1", "탐구2"];
  let total = 0;

  subjects.forEach((subject) => {
    const hours = schedule[subject] || 0;
    total += hours;
    const elem = document.getElementById(`schedule-${subject}`);
    if (elem) {
      elem.textContent = `${hours}시간`;
    }
  });

  // 총 시간 표시
  const totalElem = document.getElementById("schedule-total");
  if (totalElem) {
    totalElem.textContent = `${total}시간`;
  }
}

// 나레이션 메시지 추가
function appendNarration(text) {
  const messageId = `narration-${messageIdCounter++}`;
  const narrationElem = document.createElement("div");
  narrationElem.classList.add("message", "narration");
  narrationElem.id = messageId;
  narrationElem.textContent = text;

  if (chatLog) {
    chatLog.appendChild(narrationElem);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  return messageId;
}

// 게임 상태 저장 함수
function saveGameState(data) {
  try {
    // 채팅 로그 저장
    if (chatLog) {
      const chatLogHTML = chatLog.innerHTML;
      localStorage.setItem(CHAT_LOG_KEY, chatLogHTML);
    }

    // 게임 상태 저장
    const gameState = {
      abilities: data.abilities,
      mental: data.mental,
      affection: data.affection,
      stamina: data.stamina,
      schedule: data.schedule,
      current_date: data.current_date,
      game_state: data.game_state,
      selected_subjects: data.selected_subjects,
      timestamp: Date.now(),
    };
    localStorage.setItem(GAME_STATE_KEY, JSON.stringify(gameState));

    console.log("[SAVE] 게임 상태 저장 완료");
  } catch (err) {
    console.error("[SAVE] 게임 상태 저장 실패:", err);
  }
}

// 게임 상태 복원 함수
function loadGameState() {
  try {
    // 채팅 로그 복원
    const savedChatLog = localStorage.getItem(CHAT_LOG_KEY);
    if (savedChatLog && chatLog) {
      chatLog.innerHTML = savedChatLog;
      chatLog.scrollTop = chatLog.scrollHeight;

      // messageIdCounter 복원 (복원된 메시지 수만큼 증가)
      const messageElements = chatLog.querySelectorAll(
        '[id^="msg-"], [id^="narration-"]'
      );
      messageIdCounter = messageElements.length;

      console.log(
        "[LOAD] 채팅 로그 복원 완료 (메시지 수:",
        messageIdCounter,
        ")"
      );
    }

    // 게임 상태 복원
    const savedGameState = localStorage.getItem(GAME_STATE_KEY);
    if (savedGameState) {
      const gameState = JSON.parse(savedGameState);

      // 능력치 복원
      if (gameState.abilities) {
        updateAbilitiesDisplay(gameState.abilities);
      }

      // 호감도 복원
      if (gameState.affection !== undefined) {
        updateAffectionDisplay(gameState.affection);
      }

      // 체력 복원
      if (gameState.stamina !== undefined) {
        updateStaminaDisplay(gameState.stamina);
      }

      // 멘탈 복원 및 상태 표시
      if (gameState.mental !== undefined) {
        updateMentalDisplay(gameState.mental, gameState.stamina);
      }

      // 시간표 복원
      if (gameState.schedule) {
        updateScheduleDisplay(
          gameState.schedule,
          gameState.game_state || "daily_routine"
        );
      }

      // 게임 날짜 복원
      if (gameState.current_date) {
        updateGameDate(gameState.current_date);
      }

      console.log("[LOAD] 게임 상태 복원 완료");
      return true; // 복원 성공
    }

    return false; // 저장된 상태 없음
  } catch (err) {
    console.error("[LOAD] 게임 상태 복원 실패:", err);
    return false;
  }
}

// 메시지 DOM에 추가
let messageIdCounter = 0;
function appendMessage(sender, text, imageSrc = null) {
  const messageId = `msg-${messageIdCounter++}`;
  const messageElem = document.createElement("div");
  messageElem.classList.add("message", sender);
  messageElem.id = messageId;

  if (sender === "user") {
    messageElem.textContent = text;
  } else {
    // 이미지가 있으면 먼저 표시
    if (imageSrc) {
      const botImg = document.createElement("img");
      botImg.classList.add("bot-big-img");
      botImg.src = imageSrc;
      botImg.alt = "챗봇 이미지";
      messageElem.appendChild(botImg);
    }

    // 텍스트 추가
    const textContainer = document.createElement("div");
    textContainer.classList.add("bot-text-container");
    textContainer.textContent = text;
    messageElem.appendChild(textContainer);
  }

  if (chatLog) {
    chatLog.appendChild(messageElem);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  return messageId;
}

// 메시지 제거
function removeMessage(messageId) {
  const elem = document.getElementById(messageId);
  if (elem) {
    elem.remove();
  }
}

// 서가윤 뻐끔뻐끔 애니메이션
let speakingAnimationInterval = null;
let speakingAnimationTimeout = null;
let currentAffection = 5; // 현재 호감도 저장
let currentStamina = 30; // 현재 체력 저장

// 호감도에 따른 이미지 프리픽스 반환
function getImagePrefixByAffection(affection) {
  if (affection < 10) {
    return "하";
  } else if (affection < 30) {
    return "중하";
  } else if (affection < 50) {
    return "중";
  } else if (affection < 70) {
    return "중상";
  } else {
    return "상";
  }
}

// 호감도와 체력에 따른 기본 이미지 경로 반환
function getDefaultImageByAffectionAndStamina(affection, stamina) {
  const basePath = "static/images/chatbot/";

  // 체력이 10 미만이면 체력 10미만 이미지 사용
  if (stamina < 10) {
    return basePath + "서가윤_체력10미만.png";
  }

  // 체력이 10 이상이면 호감도에 따라 이미지 선택
  const prefix = getImagePrefixByAffection(affection);
  return basePath + prefix + "-0.png";
}

// 호감도에 따른 기본 이미지 경로 반환 (하위 호환성을 위해 유지)
function getDefaultImageByAffection(affection) {
  return getDefaultImageByAffectionAndStamina(affection, currentStamina);
}

function startSpeakingAnimation(affection = null) {
  const sideImage = document.querySelector(".side-image");
  if (!sideImage) return;

  // 호감도가 전달되지 않으면 현재 저장된 호감도 사용
  const targetAffection = affection !== null ? affection : currentAffection;

  // 체력이 10 미만이면 애니메이션 사용하지 않음 (체력 10미만 이미지만 사용)
  if (currentStamina < 10) {
    const defaultImage = getDefaultImageByAffectionAndStamina(
      targetAffection,
      currentStamina
    );
    sideImage.src = defaultImage;
    return;
  }

  // 기존 애니메이션 중지
  stopSpeakingAnimation(targetAffection);

  // 호감도에 따른 이미지 프리픽스 결정
  const prefix = getImagePrefixByAffection(targetAffection);
  const basePath = "static/images/chatbot/";
  const image0 = basePath + prefix + "-0.png";
  const image1 = basePath + prefix + "-1.png";

  // 현재 이미지가 기본 이미지면 -0으로 시작
  let currentImage = 0;
  sideImage.src = image0;

  // 메시지 길이에 따라 애니메이션 시간 계산 (최소 1초, 최대 5초)
  const lastBotMessage = document.querySelector(
    ".message.bot:last-child .bot-text-container"
  );
  const messageLength = lastBotMessage ? lastBotMessage.textContent.length : 50;
  const duration = Math.min(Math.max(messageLength * 30, 1000), 5000); // 글자당 30ms, 최소 1초, 최대 5초

  // 뻐끔뻐끔 애니메이션 (약 150ms마다 이미지 교체)
  let startTime = Date.now();
  speakingAnimationInterval = setInterval(() => {
    const elapsed = Date.now() - startTime;
    if (elapsed >= duration) {
      stopSpeakingAnimation(targetAffection);
      return;
    }

    // 0과 1을 번갈아가며 표시
    currentImage = currentImage === 0 ? 1 : 0;
    sideImage.src = currentImage === 0 ? image0 : image1;
  }, 150); // 150ms마다 이미지 교체

  // 애니메이션 자동 종료 타이머
  speakingAnimationTimeout = setTimeout(() => {
    stopSpeakingAnimation(targetAffection);
  }, duration);
}

function stopSpeakingAnimation(affection = null) {
  if (speakingAnimationInterval) {
    clearInterval(speakingAnimationInterval);
    speakingAnimationInterval = null;
  }
  if (speakingAnimationTimeout) {
    clearTimeout(speakingAnimationTimeout);
    speakingAnimationTimeout = null;
  }

  // 기본 이미지로 복원 (호감도와 체력에 따라)
  const sideImage = document.querySelector(".side-image");
  if (sideImage) {
    const targetAffection = affection !== null ? affection : currentAffection;
    const defaultImage = getDefaultImageByAffectionAndStamina(
      targetAffection,
      currentStamina
    );
    sideImage.src = defaultImage;
  }
}

// 엔터키로 전송
if (userMessageInput) {
  userMessageInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      sendMessage();
    }
  });
}

// 전송 버튼
if (sendBtn) {
  sendBtn.addEventListener("click", () => sendMessage());
}

// 모달 열기/닫기
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = "block";
  }
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.style.display = "none";
  }
}

// 미디어 버튼 이벤트
if (videoBtn) {
  videoBtn.addEventListener("click", () => openModal("videoModal"));
}

if (imageBtn) {
  imageBtn.addEventListener("click", () => openModal("imageModal"));
}

// 모달 닫기 버튼
document.querySelectorAll(".modal-close").forEach((btn) => {
  btn.addEventListener("click", () => {
    const modalId = btn.dataset.closeModal;
    closeModal(modalId);
  });
});

// 모달 배경 클릭 시 닫기
document.querySelectorAll(".modal").forEach((modal) => {
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      modal.style.display = "none";
    }
  });
});

// F5 키 입력 감지 및 게임 상태 초기화
document.addEventListener("keydown", (event) => {
  // F5 키 (keyCode 116 또는 key === "F5")
  if (event.key === "F5" || event.keyCode === 116) {
    event.preventDefault(); // 기본 새로고침 동작 방지

    // 사용자 확인 없이 바로 초기화
    resetGame(true); // skipConfirm = true
  }
});

// UI 레이아웃 컨테이너 드래그 기능
function initDraggableLayout() {
  const container = document.getElementById("ui-layout-container");
  if (!container) return;

  const header = container.querySelector(".ui-layout-header");
  if (!header) return;

  let isDragging = false;
  let currentX = 0;
  let currentY = 0;
  let initialX = 0;
  let initialY = 0;

  // 드래그 시작 (헤더를 클릭했을 때)
  header.addEventListener("mousedown", (e) => {
    e.preventDefault();
    e.stopPropagation();

    isDragging = true;
    container.style.zIndex = "1000"; // 드래그 중일 때 최상위로
    container.style.opacity = "0.95"; // 드래그 중 투명도 변경

    // 현재 위치 가져오기
    const rect = container.getBoundingClientRect();
    initialX = e.clientX - rect.left;
    initialY = e.clientY - rect.top;

    header.style.cursor = "grabbing";
  });

  // 드래그 중
  document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;

    e.preventDefault();

    currentX = e.clientX - initialX;
    currentY = e.clientY - initialY;

    // 화면 경계 체크
    const maxX = window.innerWidth - container.offsetWidth;
    const maxY = window.innerHeight - container.offsetHeight;

    currentX = Math.max(0, Math.min(currentX, maxX));
    currentY = Math.max(0, Math.min(currentY, maxY));

    // 컨테이너 이동
    container.style.left = currentX + "px";
    container.style.top = currentY + "px";
    container.style.right = "auto";
    container.style.bottom = "auto";
  });

  // 드래그 종료
  document.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      container.style.zIndex = "20";
      container.style.opacity = "1";
      header.style.cursor = "grab";

      // 위치 저장
      const rect = container.getBoundingClientRect();
      localStorage.setItem(
        "ui-layout-position",
        JSON.stringify({
          left: rect.left + "px",
          top: rect.top + "px",
        })
      );
    }
  });

  // 저장된 위치 복원
  const saved = localStorage.getItem("ui-layout-position");
  if (saved) {
    try {
      const position = JSON.parse(saved);
      if (position.left !== undefined && position.top !== undefined) {
        container.style.left = position.left;
        container.style.top = position.top;
        container.style.right = "auto";
        container.style.bottom = "auto";
      }
    } catch (e) {
      console.error("위치 복원 실패:", e);
    }
  }
}

// (구버전) UI 요소 드래그 기능 (사용 안 함 - 레이아웃 컨테이너로 대체됨)
function initDraggableElements() {
  // 드래그 가능한 요소들
  const draggableElements = [
    { id: "affection-display", titleSelector: ".affection-label" },
    { id: "calendar-display", titleSelector: ".calendar-title" },
  ];

  draggableElements.forEach(({ id, titleSelector }) => {
    const element = document.getElementById(id);
    if (!element) return;

    const titleElement = element.querySelector(titleSelector);
    if (!titleElement) return;

    let isDragging = false;
    let currentX = 0;
    let currentY = 0;
    let initialX = 0;
    let initialY = 0;

    // 드래그 시작
    titleElement.addEventListener("mousedown", (e) => {
      // 텍스트 선택 방지
      e.preventDefault();

      isDragging = true;
      element.style.zIndex = "1000"; // 드래그 중일 때 최상위로
      element.style.opacity = "0.9"; // 드래그 중 투명도 변경

      // 현재 위치 가져오기 (getBoundingClientRect 사용)
      const rect = element.getBoundingClientRect();
      initialX = e.clientX - rect.left;
      initialY = e.clientY - rect.top;

      titleElement.style.cursor = "grabbing";
    });

    // 드래그 중
    document.addEventListener("mousemove", (e) => {
      if (!isDragging) return;

      e.preventDefault();

      currentX = e.clientX - initialX;
      currentY = e.clientY - initialY;

      // 화면 경계 체크
      const maxX = window.innerWidth - element.offsetWidth;
      const maxY = window.innerHeight - element.offsetHeight;

      currentX = Math.max(0, Math.min(currentX, maxX));
      currentY = Math.max(0, Math.min(currentY, maxY));

      // 자유롭게 드래그 (충돌 체크 없이)
      element.style.left = currentX + "px";
      element.style.top = currentY + "px";
      element.style.right = "auto"; // right 값 제거
      element.style.bottom = "auto"; // bottom 값도 제거
    });

    // 드래그 종료
    document.addEventListener("mouseup", () => {
      if (isDragging) {
        isDragging = false;
        element.style.zIndex = "10";
        element.style.opacity = "1";
        titleElement.style.cursor = "grab";

        // 위치 저장 (선택사항)
        const rect = element.getBoundingClientRect();
        localStorage.setItem(
          `${id}-position`,
          JSON.stringify({
            left: rect.left + "px",
            top: rect.top + "px",
          })
        );
      }
    });
  });

  // 저장된 위치 복원 (선택사항)
  draggableElements.forEach(({ id }) => {
    const saved = localStorage.getItem(`${id}-position`);
    if (saved) {
      try {
        const position = JSON.parse(saved);
        const element = document.getElementById(id);
        if (
          element &&
          position.left !== undefined &&
          position.top !== undefined
        ) {
          element.style.left = position.left;
          element.style.top = position.top;
          element.style.right = "auto";
          element.style.bottom = "auto";
        }
      } catch (e) {
        console.error("위치 복원 실패:", e);
      }
    }
  });
}

// 드래그 스크롤 기능 추가
function initDragScroll() {
  const chatLog = document.getElementById("chat-log");
  if (!chatLog) return;

  let isDragging = false;
  let startY = 0;
  let scrollTop = 0;

  chatLog.addEventListener("mousedown", (e) => {
    // 텍스트 선택 중이면 드래그 스크롤 비활성화
    const selection = window.getSelection();
    if (selection.toString().length > 0) {
      return;
    }

    isDragging = true;
    chatLog.style.cursor = "grabbing";
    startY = e.pageY - chatLog.offsetTop;
    scrollTop = chatLog.scrollTop;
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    e.preventDefault();
    const y = e.pageY - chatLog.offsetTop;
    const walk = (y - startY) * 2; // 스크롤 속도 조절
    chatLog.scrollTop = scrollTop - walk;
  });

  document.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      chatLog.style.cursor = "grab";
    }
  });

  // 마우스가 영역을 벗어나면 드래그 종료
  chatLog.addEventListener("mouseleave", () => {
    if (isDragging) {
      isDragging = false;
      chatLog.style.cursor = "grab";
    }
  });
}

// 게임 초기화 함수 (skipConfirm: 확인 대화상자 건너뛰기)
async function resetGame(skipConfirm = false) {
  // 사용자 확인 (skipConfirm이 false일 때만)
  if (!skipConfirm) {
    if (
      !confirm(
        "게임을 완전히 초기화하시겠습니까?\n\n모든 진행 상황이 삭제됩니다:\n- 호감도\n- 능력치\n- 선택과목\n- 시간표\n- 채팅 기록\n- 게임 날짜\n\n이 작업은 되돌릴 수 없습니다."
      )
    ) {
      return;
    }
  }

  try {
    const username =
      document.querySelector(".chat-area")?.dataset.username || "사용자";

    // localStorage 완전 삭제 (먼저 실행)
    localStorage.removeItem(CHAT_LOG_KEY);
    localStorage.removeItem(GAME_STATE_KEY);
    // 다른 관련 localStorage도 삭제 (UI 위치 등)
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(STORAGE_KEY_PREFIX)) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach((key) => localStorage.removeItem(key));

    console.log("[RESET] localStorage 삭제 완료");

    // 서버에 초기화 요청 (에러가 발생해도 계속 진행)
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: "__RESET_GAME_STATE__",
          username: username,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("[RESET] 서버 초기화 완료:", data);
      } else {
        console.warn("[RESET] 서버 응답 오류:", response.status);
      }
    } catch (serverErr) {
      console.warn("[RESET] 서버 초기화 요청 실패 (계속 진행):", serverErr);
    }

    console.log("[RESET] 게임 상태 초기화 완료, 페이지 리로드 중...");

    // 초기화 플래그 설정 (리로드 후 자동 init 방지)
    localStorage.setItem(`${STORAGE_KEY_PREFIX}just_reset`, "true");

    // 페이지 전체 리로드 (즉시)
    setTimeout(() => {
      location.reload();
    }, 100); // 짧은 지연으로 localStorage 저장 확실히 보장
  } catch (err) {
    console.error("[RESET] 게임 상태 초기화 중 오류:", err);
    // 에러가 발생해도 localStorage는 이미 삭제되었으므로 리로드
    alert(
      "초기화 중 오류가 발생했지만 계속 진행합니다. 페이지를 새로고침합니다."
    );
    location.reload();
  }
}

// 초기화 버튼 이벤트 리스너
const resetBtn = document.getElementById("reset-btn");
if (resetBtn) {
  resetBtn.addEventListener("click", resetGame);
}

// 페이지 로드 시 게임 상태 복원 또는 초기화
window.addEventListener("load", () => {
  console.log("페이지 로드 완료");

  // 드래그 스크롤 초기화
  initDragScroll();

  // UI 레이아웃 컨테이너 드래그 기능 초기화
  initDraggableLayout();

  // 달력 초기화
  if (typeof initCalendar === "function") {
    initCalendar();
  }

  // 게임 상태 복원 시도
  const stateRestored = loadGameState();

  setTimeout(() => {
    // 초기화 직후 플래그 확인
    const justReset = localStorage.getItem(`${STORAGE_KEY_PREFIX}just_reset`);

    if (justReset === "true") {
      // 초기화 직후: 플래그 제거
      localStorage.removeItem(`${STORAGE_KEY_PREFIX}just_reset`);
      console.log("[INIT] 초기화 직후 - 기본값 설정 및 초기 메시지 전송");

      // 초기 UI 설정
      updateAbilitiesDisplay({ 국어: 0, 수학: 0, 영어: 0, 탐구1: 0, 탐구2: 0 });
      updateAffectionDisplay(5);
      currentAffection = 5;
      updateScheduleDisplay({}, "ice_break");
      updateGameDate("2023-11-17");

      // 초기 메시지 전송 (즉시)
      if (chatLog && chatLog.childElementCount === 0) {
        console.log("[INIT] 초기 메시지 요청");
        sendMessage(true);
      }
    } else if (!stateRestored && chatLog && chatLog.childElementCount === 0) {
      // 저장된 상태가 없거나 채팅 로그가 비어있을 때만 초기 메시지 전송
      console.log("저장된 게임 상태가 없어 초기 메시지 요청");
      sendMessage(true);
      // 초기 능력치 표시
      updateAbilitiesDisplay({ 국어: 0, 수학: 0, 영어: 0, 탐구1: 0, 탐구2: 0 });
      // 초기 호감도 표시 (이미지도 함께 업데이트됨)
      updateAffectionDisplay(5);
      currentAffection = 5;
      // 초기 게임 날짜 설정
      updateGameDate("2023-11-17");
    } else if (stateRestored) {
      console.log(
        "게임 상태가 복원되었습니다. 초기 메시지를 전송하지 않습니다."
      );
      // 게임 날짜가 없으면 초기값 설정
      const savedGameState = localStorage.getItem(GAME_STATE_KEY);
      if (savedGameState) {
        const gameState = JSON.parse(savedGameState);
        if (!gameState.current_date) {
          updateGameDate("2023-11-17");
        }

        // 복원된 호감도에 따라 이미지도 업데이트
        if (gameState.affection !== undefined) {
          currentAffection = gameState.affection;
          // 애니메이션 중이 아닐 때만 이미지 업데이트
          if (!speakingAnimationInterval) {
            const sideImage = document.querySelector(".side-image");
            if (sideImage) {
              const savedStamina =
                gameState.stamina !== undefined ? gameState.stamina : 30;
              const defaultImage = getDefaultImageByAffectionAndStamina(
                gameState.affection,
                savedStamina
              );
              sideImage.src = defaultImage;
            }
          }
        }
      } else {
        updateGameDate("2023-11-17");
      }
    }
  }, 500);
});
