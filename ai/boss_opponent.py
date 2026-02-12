"""학습된 보스 모델을 상대로 사용하는 래퍼"""
from __future__ import annotations
import random

from config import ACTION_REPEAT, NUM_BOSS_ACTIONS
from ai.observation import build_observation
from ai.action_mapping import get_action_mask


class BossOpponent:
    """학습된 보스 모델 래퍼"""

    def __init__(self, model_path: str | None = None):
        self.model = None
        self._action_counter = 0
        self._current_action = 0

        if model_path:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)

    def get_action(self, world) -> int:
        """보스 액션 반환 (action_repeat 주기로 새 액션 선택)"""
        if self._action_counter <= 0:
            if self.model is not None:
                obs = build_observation(world)
                action, _ = self.model.predict(obs, deterministic=False)
                self._current_action = int(action)
            else:
                mask = get_action_mask(world)
                available = [i for i in range(NUM_BOSS_ACTIONS) if mask[i] > 0]
                self._current_action = random.choice(available) if available else 0
            self._action_counter = ACTION_REPEAT

        self._action_counter -= 1
        return self._current_action

    def reset(self):
        self._action_counter = 0
        self._current_action = 0
