"""보스 환경 (학습된 플레이어 모델 상대) - co-training용"""
from __future__ import annotations
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import (
    OBSERVATION_DIM, NUM_BOSS_ACTIONS, MAX_EPISODE_STEPS, ACTION_REPEAT,
    PLAYER_ACTION_REPEAT,
)
from core.game_world import GameWorld
from core.physics import angle_between
from ai.observation import build_observation
from ai.player_observation import build_player_observation
from ai.reward import RewardAccumulator, compute_reward
from ai.action_mapping import get_action_mask

# PlayerEnv의 이동 방향 매핑
MOVE_DIRS = [
    (0.0, 0.0), (0.0, -1.0), (0.707, -0.707), (1.0, 0.0),
    (0.707, 0.707), (0.0, 1.0), (-0.707, 0.707), (-1.0, 0.0),
    (-0.707, -0.707),
]


class BossEnvVsPlayer(gym.Env):
    """보스가 에이전트, 학습된 플레이어 모델이 상대"""
    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None, player_model_path: str | None = None,
                 kill_reward_coef: float = 1.0, action_repeat: int = ACTION_REPEAT):
        super().__init__()
        self.render_mode = render_mode
        self.action_repeat = action_repeat
        self.kill_reward_coef = kill_reward_coef

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_BOSS_ACTIONS)

        self.world = GameWorld()

        # 학습된 플레이어 모델 로드
        self._player_model = None
        self._player_action_counter = 0
        self._current_player_action = None
        if player_model_path:
            from stable_baselines3 import PPO
            self._player_model = PPO.load(player_model_path)

    def _get_player_action(self):
        """학습된 플레이어 모델로 행동 결정"""
        if self._player_action_counter <= 0:
            if self._player_model is not None:
                obs = build_player_observation(self.world)
                action, _ = self._player_model.predict(obs, deterministic=False)
                self._current_player_action = action
            else:
                # 폴백: 랜덤
                self._current_player_action = np.array([
                    np.random.randint(9),
                    np.random.randint(2),
                    np.random.randint(2),
                ])
            self._player_action_counter = PLAYER_ACTION_REPEAT

        self._player_action_counter -= 1

        # 디코딩
        act = self._current_player_action
        move_dir = MOVE_DIRS[int(act[0])]
        attack = bool(act[1])
        dodge = bool(act[2])
        aim_angle = angle_between(
            self.world.player.transform.x, self.world.player.transform.y,
            self.world.boss.transform.x, self.world.boss.transform.y,
        )
        return move_dir, aim_angle, attack, dodge

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self._player_action_counter = 0
        obs = build_observation(self.world)
        info = {'action_mask': get_action_mask(self.world)}
        return obs, info

    def step(self, action: int):
        prev_player_hp = self.world.player.health.current
        prev_boss_hp = self.world.boss.health.current
        prev_boss_phase = self.world.boss.phase

        executed = self.world.apply_boss_action(int(action))

        for _ in range(self.action_repeat):
            move_dir, aim_angle, attack, dodge = self._get_player_action()
            self.world.apply_player_action(move_dir, aim_angle, attack, dodge)
            self.world.tick()

            if self.world.done:
                break

        # 보상 (기존 보스 보상 + λ)
        acc = RewardAccumulator()
        acc.damage_dealt = max(0, prev_player_hp - self.world.player.health.current)
        acc.damage_taken = max(0, prev_boss_hp - self.world.boss.health.current)
        acc.phase_transition = self.world.boss.phase > prev_boss_phase
        acc.player_killed = self.world.player.health.current <= 0
        acc.boss_died = self.world.boss.health.current <= 0
        acc.distance_to_player = self.world.get_distance()
        acc.player_hp_ratio = self.world.player.health.ratio
        acc.boss_hp_ratio = self.world.boss.health.ratio
        acc.action_id = int(action)
        acc.action_executed = executed

        reward = compute_reward(acc, self.world.boss.recent_actions,
                                kill_reward_coef=self.kill_reward_coef)

        terminated = (self.world.player.health.current <= 0
                      or self.world.boss.health.current <= 0)
        truncated = self.world.step_count >= MAX_EPISODE_STEPS

        obs = build_observation(self.world)
        info = {
            'action_mask': get_action_mask(self.world),
            'boss_hp': self.world.boss.health.ratio,
            'player_hp': self.world.player.health.ratio,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        pass
