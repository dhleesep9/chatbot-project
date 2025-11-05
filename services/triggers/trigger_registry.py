"""트리거 레지스트리 시스템

services/triggers/ 디렉토리의 모든 트리거 파일을 자동으로 로드하고 관리합니다.

사용법:
    registry = TriggerRegistry()
    result = registry.evaluate_trigger("affection_increase", transition, context)
"""

import importlib
import os
from pathlib import Path
from typing import Dict, Callable


class TriggerRegistry:
    """트리거 자동 로딩 및 실행 관리 클래스"""

    def __init__(self):
        """트리거 레지스트리 초기화 및 자동 로딩"""
        self.triggers: Dict[str, Callable] = {}
        self._load_all_triggers()

    def _load_all_triggers(self):
        """services/triggers/ 디렉토리의 모든 트리거 파일 자동 로딩"""
        triggers_dir = Path(__file__).parent

        # base_trigger.py와 trigger_registry.py는 제외
        exclude_files = {'base_trigger.py', 'trigger_registry.py', '__init__.py'}

        for file_path in triggers_dir.glob("*.py"):
            filename = file_path.name

            if filename in exclude_files:
                continue

            # 파일명에서 .py 제거하여 트리거 이름 추출
            trigger_name = filename[:-3]

            try:
                # 동적으로 모듈 임포트
                module = importlib.import_module(f"services.triggers.{trigger_name}")

                # evaluate 함수가 있는지 확인
                if hasattr(module, 'evaluate'):
                    self.triggers[trigger_name] = module.evaluate
                    print(f"[TRIGGER REGISTRY] Loaded trigger: {trigger_name}")
                else:
                    print(f"[WARN] {filename} has no evaluate() function")

            except Exception as e:
                print(f"[ERROR] Failed to load trigger {filename}: {e}")

    def evaluate_trigger(self, trigger_type: str, transition: dict, context: dict) -> bool:
        """
        지정된 트리거 타입의 evaluate 함수 실행

        Args:
            trigger_type: 트리거 타입 (파일명과 동일)
            transition: state JSON의 transition 객체
            context: 트리거 실행 컨텍스트

        Returns:
            bool: 트리거 조건 만족 여부
        """
        if trigger_type not in self.triggers:
            print(f"[WARN] Unknown trigger_type: {trigger_type}")
            return False

        try:
            trigger_func = self.triggers[trigger_type]
            return trigger_func(transition, context)
        except Exception as e:
            print(f"[ERROR] Trigger '{trigger_type}' evaluation failed: {e}")
            return False

    def list_triggers(self) -> list:
        """등록된 모든 트리거 타입 목록 반환"""
        return list(self.triggers.keys())

    def has_trigger(self, trigger_type: str) -> bool:
        """지정된 트리거 타입이 등록되어 있는지 확인"""
        return trigger_type in self.triggers
