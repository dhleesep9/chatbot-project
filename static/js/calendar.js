// 달력 시스템

// 모의고사 일정 정의 (날짜는 정확한 시험일로 수정 필요)
const examSchedule = {
  "2024-03-07": "3월 모의고사",
  "2024-04-04": "4월 모의고사",
  "2024-05-09": "5월 모의고사",
  "2024-06-06": "6월 모의고사",
  "2024-07-11": "7월 모의고사",
  "2024-09-05": "9월 모의고사",
  "2024-10-17": "10월 모의고사",
  "2024-11-14": "수능",
};

// 현재 보고 있는 달력 월 (년, 월)
let currentCalendarYear = 2023;
let currentCalendarMonth = 11; // 0-based (11 = 12월)
let gameCurrentDate = "2023-11-17"; // 게임 내 현재 날짜

// 달력 초기화
function initCalendar() {
  updateCalendar(currentCalendarYear, currentCalendarMonth);

  // 이전/다음 버튼 이벤트
  const prevBtn = document.getElementById("calendar-prev");
  const nextBtn = document.getElementById("calendar-next");

  if (prevBtn) {
    prevBtn.addEventListener("click", () => {
      if (currentCalendarMonth === 0) {
        currentCalendarMonth = 11;
        currentCalendarYear--;
      } else {
        currentCalendarMonth--;
      }
      // 2023년 11월 17일 이전으로는 가지 못함
      const minDate = new Date(2023, 10, 17); // 2023년 11월
      const checkDate = new Date(currentCalendarYear, currentCalendarMonth, 1);
      if (checkDate < minDate) {
        currentCalendarYear = 2023;
        currentCalendarMonth = 10; // 11월 (0-based)
      }
      updateCalendar(currentCalendarYear, currentCalendarMonth);
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", () => {
      if (currentCalendarMonth === 11) {
        currentCalendarMonth = 0;
        currentCalendarYear++;
      } else {
        currentCalendarMonth++;
      }
      // 2025년 2월까지만
      if (
        currentCalendarYear > 2025 ||
        (currentCalendarYear === 2025 && currentCalendarMonth > 1)
      ) {
        currentCalendarYear = 2025;
        currentCalendarMonth = 1; // 2월 (0-based)
      }
      updateCalendar(currentCalendarYear, currentCalendarMonth);
    });
  }
}

// 달력 업데이트
function updateCalendar(year, month) {
  const monthYearElem = document.getElementById("calendar-month-year");
  monthYearElem.textContent = `${year}년 ${month + 1}월`;

  const grid = document.getElementById("calendar-grid");
  grid.innerHTML = "";

  // 요일 헤더
  const dayNames = ["일", "월", "화", "수", "목", "금", "토"];
  dayNames.forEach((day) => {
    const header = document.createElement("div");
    header.className = "calendar-day-header";
    header.textContent = day;
    grid.appendChild(header);
  });

  // 첫 날짜와 마지막 날짜
  const firstDay = new Date(year, month, 1);
  const lastDay = new Date(year, month + 1, 0);
  const startDate = new Date(firstDay);
  startDate.setDate(startDate.getDate() - firstDay.getDay()); // 이전 달의 일요일부터 시작

  // 달력 범위: 2023년 11월 17일 ~ 2025년 2월
  const minDate = new Date(2023, 10, 17); // 2023년 11월 17일
  const maxDate = new Date(2025, 1, 28); // 2025년 2월 28일

  // 현재 날짜를 게임 날짜로 설정
  const today = new Date(gameCurrentDate);

  // 42개 날짜 셀 생성 (6주)
  for (let i = 0; i < 42; i++) {
    const date = new Date(startDate);
    date.setDate(startDate.getDate() + i);

    // 범위 체크
    if (date < minDate || date > maxDate) {
      const emptyDay = document.createElement("div");
      emptyDay.className = "calendar-day other-month";
      grid.appendChild(emptyDay);
      continue;
    }

    const day = document.createElement("div");
    day.className = "calendar-day";

    // 다른 달인지 체크
    if (date.getMonth() !== month) {
      day.classList.add("other-month");
    }

    // 일요일/토요일 스타일
    const dayOfWeek = date.getDay();
    if (dayOfWeek === 0) {
      day.classList.add("sunday");
    } else if (dayOfWeek === 6) {
      day.classList.add("saturday");
    }

    // 오늘(게임 현재 날짜) 체크
    if (date.toDateString() === today.toDateString()) {
      day.classList.add("today");
    }

    // 모의고사 일정 체크
    const dateStr = formatDate(date);
    const dayNumber = date.getDate();

    if (examSchedule[dateStr]) {
      day.classList.add("event");
      const tooltip = document.createElement("div");
      tooltip.className = "event-tooltip";
      tooltip.textContent = examSchedule[dateStr];
      day.appendChild(tooltip);
      day.innerHTML = `<span>${dayNumber}</span>${tooltip.outerHTML}`;
    } else {
      day.textContent = dayNumber;
    }

    grid.appendChild(day);
  }
}

// 날짜 포맷팅 (YYYY-MM-DD)
function formatDate(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

// 게임 날짜 업데이트
function updateGameDate(dateStr) {
  if (!dateStr) return;

  gameCurrentDate = dateStr;
  const date = new Date(dateStr + "T00:00:00"); // 시간대 문제 해결
  const year = date.getFullYear();
  const month = date.getMonth();

  // 현재 보고 있는 달력이 게임 날짜와 다르면 업데이트
  if (currentCalendarYear !== year || currentCalendarMonth !== month) {
    currentCalendarYear = year;
    currentCalendarMonth = month;
    updateCalendar(year, month);
  } else {
    updateCalendar(currentCalendarYear, currentCalendarMonth);
  }

  // 현재 날짜 표시 업데이트
  const currentDateElem = document.getElementById("current-date-display");
  if (currentDateElem) {
    const formattedDate = `${year}년 ${month + 1}월 ${date.getDate()}일`;
    currentDateElem.textContent = `현재 날짜: ${formattedDate}`;
  }
}
