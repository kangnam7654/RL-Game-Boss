"""Player Gymnasium 환경 - 플레이어 RL 에이전트 학습용"""
from __future__ import annotations
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import (
    PLAYER_OBS_DIM, MAX_EPISODE_STEPS, PLAYER_ACTION_REPEAT,
    PLAYER_DODGE_SPEED,
)
from core.game_world import GameWorld
from core.physics import distance, angle_between
from ai.player_observation import build_player_observation
from ai.player_reward import PlayerRewardAccumulator, compute_player_reward
from ai.boss_opponent import BossOpponent

# 이동 방향 매핑: 0=정지, 1=↑, 2=↗, 3=→, 4=↘, 5=↓, 6=↙, 7=←, 8=↖
MOVE_DIRS = [
    (0.0, 0.0),    # 정지
    (0.0, -1.0),   # ↑
    (0.707, -0.707),  # ↗
    (1.0, 0.0),    # →
    (0.707, 0.707),   # ↘
    (0.0, 1.0),    # ↓
    (-0.707, 0.707),  # ↙
    (-1.0, 0.0),   # ←
    (-0.707, -0.707), # ↖
]


class PlayerEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, boss_model_path: str | None = None,
                 action_repeat: int = PLAYER_ACTION_REPEAT):
        super().__init__()
        self.render_mode = render_mode
        self.action_repeat = action_repeat

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(PLAYER_OBS_DIM,),
            dtype=np.float32,
        )
        # MultiDiscrete: [이동(9), 공격(2), 회피(2)]
        self.action_space = spaces.MultiDiscrete([9, 2, 2])

        self.world = GameWorld()
        self.boss_opponent = BossOpponent(boss_model_path)

        self._prev_player_hp = 0
        self._prev_boss_hp = 0
        self._was_dodging = False

        self._renderer = None
        if render_mode == 'human':
            from rendering.renderer import GameRenderer
            self._renderer = GameRenderer(self.world)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.boss_opponent.reset()
        self._prev_player_hp = self.world.player.health.current
        self._prev_boss_hp = self.world.boss.health.current
        self._was_dodging = False
        obs = build_player_observation(self.world)
        return obs, {}

    def step(self, action):
        acc = PlayerRewardAccumulator()

        prev_player_hp = self.world.player.health.current
        prev_boss_hp = self.world.boss.health.current

        # 액션 디코딩
        move_idx, attack, dodge = int(action[0]), bool(action[1]), bool(action[2])
        move_dir = MOVE_DIRS[move_idx]

        # aim_angle: 보스 방향으로 자동 조준
        aim_angle = angle_between(
            self.world.player.transform.x, self.world.player.transform.y,
            self.world.boss.transform.x, self.world.boss.transform.y,
        )

        for _ in range(self.action_repeat):
            # 플레이어 액션 적용
            self.world.apply_player_action(move_dir, aim_angle, attack, dodge)

            # 보스 액션
            boss_action = self.boss_opponent.get_action(self.world)
            self.world.apply_boss_action(boss_action)

            # tick
            self.world.tick()

            if self._renderer and self.render_mode == 'human':
                self._renderer.sync(self.world)
                self._renderer.render()

            if self.world.done:
                break

        # 보상 계산
        acc.damage_to_boss = max(0, prev_boss_hp - self.world.boss.health.current)
        acc.damage_from_boss = max(0, prev_player_hp - self.world.player.health.current)
        acc.boss_killed = self.world.boss.health.current <= 0
        acc.player_died = self.world.player.health.current <= 0
        acc.distance_to_boss = self.world.get_distance()

        # 회피 성공 감지: 이전에 회피 중이었고 데미지 안 받았으면
        if self._was_dodging and acc.damage_from_boss == 0:
            acc.dodged_attack = True
        self._was_dodging = self.world.player.is_dodging

        reward = compute_player_reward(acc)

        terminated = (self.world.player.health.current <= 0
                      or self.world.boss.health.current <= 0)
        truncated = self.world.step_count >= MAX_EPISODE_STEPS

        obs = build_player_observation(self.world)
        info = {
            'player_hp': self.world.player.health.ratio,
            'boss_hp': self.world.boss.health.ratio,
            'player_won': self.world.boss.health.current <= 0,
            'step': self.world.step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer:
            return self._renderer.render()

    def close(self):
        if self._renderer:
            self._renderer.close()
