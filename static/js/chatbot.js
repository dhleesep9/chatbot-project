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

    // 자신감 업데이트
    if (data.confidence !== undefined) {
      updateConfidenceDisplay(data.confidence);
    }

    // 체력 업데이트
    if (data.stamina !== undefined) {
      currentStamina = data.stamina;
      updateStaminaDisplay(data.stamina);
    }

    // 멘탈 업데이트
    if (data.mental !== undefined) {
      currentMental = data.mental;
      updateMentalDisplay(data.mental);
    }

    // 서가윤 상태 업데이트
    if (data.stamina !== undefined || data.mental !== undefined || data.confidence !== undefined) {
      updateCharacterStatus(data.stamina, data.mental, data.confidence);
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
      if (Array.isArray(data.reply)) {
        // 배열인 경우 각 요소를 별도의 메시지로 표시
        console.log("[REPLY] 배열 형식 reply 감지, 개수:", data.reply.length);
        data.reply.forEach((msg, index) => {
          if (msg) {
            // 첫 번째 메시지에만 이미지 포함
            appendMessage("bot", msg, index === 0 ? data.image : null);
          }
        });
      } else if (typeof data.reply === "object" && data.reply !== null) {
        replyText = data.reply.reply || data.reply;
        imagePath = data.reply.image || null;
        if (replyText) {
          appendMessage("bot", replyText, imagePath);
        }
      } else {
        replyText = data.reply;
        imagePath = null;
        if (replyText) {
          appendMessage("bot", replyText, imagePath);
        }
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

    // 현재 호감도 업데이트
    if (data.affection !== undefined) {
      currentAffection = data.affection;
    }

    // 상태별 이미지 설정
    // 서버에서 data.image를 보내면 그것을 최우선으로 사용
    const sideImage = document.querySelector(".side-image");
    if (data.image && sideImage) {
      // state에 지정된 이미지가 있으면 저장 (다른 함수가 덮어쓰지 못하도록)
      currentStateImage = data.image;

      console.log(`[IMAGE] ${data.game_state} state 이미지 설정 (고정):`, data.image);
      sideImage.src = data.image;

      // state 이미지가 있을 때는 절대 애니메이션 실행하지 않음
      // (애니메이션이 이미지를 변경할 수 있음)
    } else {
      // state 이미지가 없으면 일반 애니메이션 실행 (호감도 기반)
      currentStateImage = null; // state 이미지 없음
      startSpeakingAnimation(currentAffection);
    }

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
      // 능력치를 정수로 표시
      const abilityValue = Math.floor(abilities[name]);
      abilityElem.textContent = abilityValue;
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

  // state 이미지가 지정되어 있으면 이미지 변경 금지
  if (currentStateImage) {
    return;
  }

  // 호감도 변경 시 기본 이미지 업데이트 (애니메이션 중이 아닐 때만, 비정상 상태가 아닐 때만)
  if (!speakingAnimationInterval) {
    const sideImage = document.querySelector(".side-image");
    if (sideImage) {
      // 실제 상태 값을 확인 (체력이 10 이하이면 질병 상태, 멘탈이 10 이하이면 번아웃 상태 유지)
      const isDisease = currentStamina !== undefined && currentStamina <= 10;
      const isBurnout = currentMental !== undefined && currentMental <= 10;

      // 질병이나 번아웃 상태가 아니면 호감도 이미지로 변경
      if (!isDisease && !isBurnout) {
        const defaultImage = getDefaultImageByAffection(affection);
        sideImage.src = defaultImage;
      }
      // 질병 상태면 질병 이미지 유지
      else if (isDisease) {
        sideImage.src = "/static/images/chatbot/질병-0.png";
      }
      // 번아웃 상태면 번아웃 이미지 유지
      else if (isBurnout) {
        sideImage.src = "/static/images/chatbot/번아웃-0.png";
      }
    }
  }
}

// 자신감 표시 업데이트
function updateConfidenceDisplay(confidence) {
  const confidenceValue = document.getElementById("confidence-value");
  if (confidenceValue) {
    confidenceValue.textContent = confidence;
  }

  // 현재 자신감 저장
  currentConfidence = confidence;

  // state 이미지가 지정되어 있으면 이미지 변경 금지
  if (currentStateImage) {
    return;
  }

  // 모든 상태이상 체크 (우선순위: 질병 > 번아웃 > 자신감)
  const sideImage = document.querySelector(".side-image");
  if (sideImage && !speakingAnimationInterval) {
    const isDisease = currentStamina !== undefined && currentStamina <= 10;
    const isBurnout = currentMental !== undefined && currentMental <= 10;
    const isDiscouraged = confidence <= 10;
    const isArrogant = confidence >= 90;

    // 질병 최우선
    if (isDisease) {
      sideImage.src = "/static/images/chatbot/질병-0.png";
    }
    // 번아웃 우선
    else if (isBurnout) {
      sideImage.src = "/static/images/chatbot/번아웃-0.png";
    }
    // 의기소침
    else if (isDiscouraged) {
      sideImage.src = "/static/images/chatbot/기죽음-0.png";
    }
    // 오만
    else if (isArrogant) {
      sideImage.src = "/static/images/chatbot/오만-0.png";
    }
    // 정상: 호감도 이미지
    else {
      const defaultImage = getDefaultImageByAffection(currentAffection);
      sideImage.src = defaultImage;
    }

    updateCharacterStatus(currentStamina, currentMental, confidence);
  }
}

// 체력 표시 업데이트
function updateStaminaDisplay(stamina) {
  const staminaValue = document.getElementById("stamina-value");
  const staminaEfficiency = document.getElementById("stamina-efficiency");

  if (staminaValue) {
    staminaValue.textContent = stamina;
  }

  // 현재 체력 저장 (번아웃 상태 확인용)
  currentStamina = stamina;

  // 체력에 따른 효율 계산: 효율(%) = 100 + (체력 - 30)
  const efficiency = 100 + (stamina - 30);
  if (staminaEfficiency) {
    staminaEfficiency.textContent = `(효율: ${efficiency}%)`;
  }
}

// 멘탈 표시 업데이트
function updateMentalDisplay(mental) {
  const mentalValue = document.getElementById("mental-value");
  const mentalEfficiency = document.getElementById("mental-efficiency");

  if (mentalValue) {
    mentalValue.textContent = mental;
  }

  // 현재 멘탈 저장 (혼란 상태 확인용)
  currentMental = mental;

  // 멘탈에 따른 효율 계산: 효율(%) = 100 + (멘탈 - 40)
  const efficiency = 100 + (mental - 40);
  if (mentalEfficiency) {
    mentalEfficiency.textContent = `(효율: ${efficiency}%)`;
  }
}

// 서가윤 상태 표시 업데이트
function updateCharacterStatus(stamina, mental, confidence) {
  const statusDisplay = document.getElementById("character-status-display");
  const statusValue = document.getElementById("status-value");
  const statusDescription = document.getElementById("status-description");
  const sideImage = document.querySelector(".side-image");

  if (!statusDisplay || !statusValue || !statusDescription) {
    return;
  }

  // 상태 정보 배열 (우선순위 순: 질병 > 번아웃 > 자신감)
  const statuses = [];

  // 체력이 10 이하일 때 질병 (최우선)
  if (stamina !== undefined && stamina <= 10) {
    statuses.push({
      name: "질병",
      description: "체력이 너무 낮아 아픈 상태입니다. 치료가 필요해요.",
      class: "status-disease",
      image: "/static/images/chatbot/질병-0.png",
    });
  }

  // 멘탈이 10 이하일 때 번아웃
  if (mental !== undefined && mental <= 10) {
    statuses.push({
      name: "번아웃",
      description: "멘탈이 너무 낮아 지쳤습니다. 휴식이 필요해요.",
      class: "status-burnout",
      image: "/static/images/chatbot/번아웃-0.png",
    });
  }

  // 자신감이 10 이하일 때 의기소침
  if (confidence !== undefined && confidence <= 10) {
    statuses.push({
      name: "의기소침",
      description: "자신감이 너무 낮아 기가 죽은 상태입니다.",
      class: "status-discouraged",
      image: "/static/images/chatbot/기죽음-0.png",
    });
  }

  // 자신감이 90 이상일 때 오만
  if (confidence !== undefined && confidence >= 90) {
    statuses.push({
      name: "오만",
      description: "자신감이 너무 높아 오만한 상태입니다.",
      class: "status-arrogant",
      image: "/static/images/chatbot/오만-0.png",
    });
  }

  // 상태가 없는 경우 정상 상태
  if (statuses.length === 0) {
    statuses.push({
      name: "정상",
      description: "건강한 상태입니다.",
      class: "status-normal",
      image: null, // 정상일 때는 호감도에 따른 이미지 사용
    });
  }

  // 첫 번째 상태를 메인 상태로 사용 (우선순위)
  const mainStatus = statuses[0];

  // 상태 표시 업데이트
  statusDisplay.className = `character-status-display ${mainStatus.class}`;
  statusValue.textContent = mainStatus.name;

  // 여러 상태가 있을 때 설명 결합
  if (statuses.length > 1) {
    statusDescription.textContent = statuses
      .map((s) => s.description)
      .join(" ");
  } else {
    statusDescription.textContent = mainStatus.description;
  }

  // state 이미지가 지정되어 있으면 이미지 변경 금지
  if (currentStateImage) {
    return;
  }

  // 이미지 업데이트
  if (sideImage && !speakingAnimationInterval) {
    if (mainStatus.image) {
      // 비정상 상태 이미지로 변경
      sideImage.src = mainStatus.image;
    } else {
      // 정상 상태로 복귀 시 호감도에 따른 이미지로 복원
      // 모든 상태이상을 체크
      const isActuallyDisease =
        currentStamina !== undefined && currentStamina <= 10;
      const isActuallyBurnout =
        currentMental !== undefined && currentMental <= 10;
      const isActuallyDiscouraged =
        currentConfidence !== undefined && currentConfidence <= 10;
      const isActuallyArrogant =
        currentConfidence !== undefined && currentConfidence >= 90;

      if (!isActuallyDisease && !isActuallyBurnout && !isActuallyDiscouraged && !isActuallyArrogant) {
        const defaultImage = getDefaultImageByAffection(currentAffection);
        sideImage.src = defaultImage;
      }
      // 질병 최우선
      else if (isActuallyDisease) {
        sideImage.src = "/static/images/chatbot/질병-0.png";
      }
      // 번아웃 우선
      else if (isActuallyBurnout) {
        sideImage.src = "/static/images/chatbot/번아웃-0.png";
      }
      // 의기소침
      else if (isActuallyDiscouraged) {
        sideImage.src = "/static/images/chatbot/기죽음-0.png";
      }
      // 오만
      else if (isActuallyArrogant) {
        sideImage.src = "/static/images/chatbot/오만-0.png";
      }
    }
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
  narrationElem.style.whiteSpace = "pre-wrap";

  if (chatLog) {
    chatLog.appendChild(narrationElem);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  return messageId;
}

// 게임 상태 저장 함수 (localStorage에 모든 데이터 저장)
function saveGameState(data) {
  try {
    // 채팅 로그 저장
    if (chatLog) {
      const chatLogHTML = chatLog.innerHTML;
      localStorage.setItem(CHAT_LOG_KEY, chatLogHTML);
    }

    // 기존 게임 상태 불러오기 (없으면 빈 객체)
    const existingState = localStorage.getItem(GAME_STATE_KEY);
    let gameState = existingState ? JSON.parse(existingState) : {};

    // 서버 응답 데이터로 업데이트 (모든 필드 포함)
    if (data.abilities !== undefined) gameState.abilities = data.abilities;
    if (data.affection !== undefined) gameState.affection = data.affection;
    if (data.stamina !== undefined) gameState.stamina = data.stamina;
    if (data.mental !== undefined) gameState.mental = data.mental;
    if (data.schedule !== undefined) gameState.schedule = data.schedule;
    if (data.current_date !== undefined)
      gameState.current_date = data.current_date;
    if (data.game_state !== undefined) gameState.game_state = data.game_state;
    if (data.selected_subjects !== undefined)
      gameState.selected_subjects = data.selected_subjects;
    if (data.image !== undefined) gameState.state_image = data.image;

    // 추가 필드들도 저장 (서버가 응답에 포함하는 경우)
    if (data.conversation_count !== undefined)
      gameState.conversation_count = data.conversation_count;
    if (data.current_week !== undefined)
      gameState.current_week = data.current_week;
    if (data.mock_exam_last_week !== undefined)
      gameState.mock_exam_last_week = data.mock_exam_last_week;
    if (data.career !== undefined) gameState.career = data.career;
    if (data.narration !== undefined) gameState.last_narration = data.narration;

    // 타임스탬프 업데이트
    gameState.timestamp = Date.now();
    gameState.last_saved = new Date().toISOString();

    // localStorage에 저장
    localStorage.setItem(GAME_STATE_KEY, JSON.stringify(gameState));

    console.log("[SAVE] 게임 상태 localStorage 저장 완료 (모든 데이터 포함)");
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
        currentStamina = gameState.stamina;
        updateStaminaDisplay(gameState.stamina);
      }

      // 멘탈 복원
      if (gameState.mental !== undefined) {
        currentMental = gameState.mental;
        updateMentalDisplay(gameState.mental);
      }

      // 서가윤 상태 복원
      if (gameState.stamina !== undefined || gameState.mental !== undefined || gameState.confidence !== undefined) {
        updateCharacterStatus(gameState.stamina, gameState.mental, gameState.confidence);
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

      // state 이미지 복원
      if (gameState.state_image) {
        currentStateImage = gameState.state_image;
        const sideImage = document.querySelector(".side-image");
        if (sideImage) {
          sideImage.src = currentStateImage;
          console.log("[LOAD] state 이미지 복원:", currentStateImage);
        }
      } else {
        currentStateImage = null;
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
let currentMental = 40; // 현재 멘탈 저장
let currentConfidence = 50; // 현재 자신감 저장
let currentStateImage = null; // 현재 state에 지정된 이미지 (이미지 변경 금지용)

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

// 호감도에 따른 기본 이미지 경로 반환
function getDefaultImageByAffection(affection) {
  const prefix = getImagePrefixByAffection(affection);
  const basePath = "/static/images/chatbot/";
  return basePath + prefix + "-0.png";
}

function startSpeakingAnimation(affection = null) {
  const sideImage = document.querySelector(".side-image");
  if (!sideImage) return;

  // state 이미지가 지정되어 있으면 애니메이션 실행하지 않음
  if (currentStateImage) {
    return;
  }

  // 모든 상태이상을 우선적으로 확인 (우선순위: 질병 > 번아웃 > 자신감)
  const isDisease = currentStamina !== undefined && currentStamina <= 10;
  const isBurnout = currentMental !== undefined && currentMental <= 10;
  const isDiscouraged = currentConfidence !== undefined && currentConfidence <= 10;
  const isArrogant = currentConfidence !== undefined && currentConfidence >= 90;

  // 호감도가 전달되지 않으면 현재 저장된 호감도 사용
  const targetAffection = affection !== null ? affection : currentAffection;

  // 기존 애니메이션 중지
  stopSpeakingAnimation(targetAffection);

  // 비정상 상태에 따라 이미지 프리픽스 결정 (질병이 최우선)
  let prefix;
  let basePath;
  if (isDisease) {
    // 질병 상태: 무조건 질병 이미지 사용 (최우선)
    prefix = "질병";
    basePath = "/static/images/chatbot/";
  } else if (isBurnout) {
    // 번아웃 상태: 무조건 번아웃 이미지 사용
    prefix = "번아웃";
    basePath = "/static/images/chatbot/";
  } else if (isDiscouraged) {
    // 의기소침 상태: 무조건 기죽음 이미지 사용
    prefix = "기죽음";
    basePath = "/static/images/chatbot/";
  } else if (isArrogant) {
    // 오만 상태: 무조건 오만 이미지 사용
    prefix = "오만";
    basePath = "/static/images/chatbot/";
  } else {
    // 정상 상태: 호감도에 따른 이미지 프리픽스 결정
    prefix = getImagePrefixByAffection(targetAffection);
    basePath = "/static/images/chatbot/";
  }

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

  // state 이미지가 지정되어 있으면 이미지 변경하지 않음
  if (currentStateImage) {
    return;
  }

  // 애니메이션 종료 후 모든 상태이상 확인하여 이미지 업데이트 (우선순위: 질병 > 번아웃 > 자신감)
  const sideImage = document.querySelector(".side-image");
  if (sideImage) {
    // 질병 최우선
    if (currentStamina !== undefined && currentStamina <= 10) {
      // 질병 이미지 표시
      sideImage.src = "/static/images/chatbot/질병-0.png";
    } else if (currentMental !== undefined && currentMental <= 10) {
      // 번아웃 이미지 표시
      sideImage.src = "/static/images/chatbot/번아웃-0.png";
    } else if (currentConfidence !== undefined && currentConfidence <= 10) {
      // 의기소침 이미지 표시
      sideImage.src = "/static/images/chatbot/기죽음-0.png";
    } else if (currentConfidence !== undefined && currentConfidence >= 90) {
      // 오만 이미지 표시
      sideImage.src = "/static/images/chatbot/오만-0.png";
    } else {
      // 정상 상태: 호감도에 따른 이미지로 복원
      const targetAffection = affection !== null ? affection : currentAffection;
      const defaultImage = getDefaultImageByAffection(targetAffection);
      sideImage.src = defaultImage;
    }
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
      updateCharacterStatus(30, 40, 50); // 기본 체력 30, 멘탈 40, 자신감 50으로 상태 업데이트

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
      updateCharacterStatus(30, 40, 50); // 기본 체력 30, 멘탈 40, 자신감 50으로 상태 업데이트
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
              const defaultImage = getDefaultImageByAffection(
                gameState.affection
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
