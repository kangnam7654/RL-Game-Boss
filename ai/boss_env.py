"""Gymnasium 환경 래퍼 - 보스 AI 학습용"""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import OBSERVATION_DIM, NUM_BOSS_ACTIONS, MAX_EPISODE_STEPS, ACTION_REPEAT
from core.game_world import GameWorld
from core.physics import distance
from ai.observation import build_observation
from ai.reward import RewardAccumulator, compute_reward
from ai.action_mapping import get_action_mask
from ai.opponent_policy import ScriptedPlayerPolicy


class BossEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, render_mode=None, player_policy='medium',
                 action_repeat=ACTION_REPEAT):
        super().__init__()
        self.render_mode = render_mode
        self.action_repeat = action_repeat

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_BOSS_ACTIONS)

        self.world = GameWorld()
        self.player_policy = ScriptedPlayerPolicy(player_policy)
        self.reward_acc = RewardAccumulator()

        self._renderer = None
        if render_mode == 'human':
            from rendering.renderer import GameRenderer
            self._renderer = GameRenderer(self.world)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.world.reset()
        self.reward_acc = RewardAccumulator()
        obs = build_observation(self.world)
        info = {'action_mask': get_action_mask(self.world)}
        return obs, info

    def step(self, action: int):
        self.reward_acc = RewardAccumulator()

        prev_player_hp = self.world.player.health.current
        prev_boss_hp = self.world.boss.health.current
        prev_boss_phase = self.world.boss.phase

        # 보스 액션 적용
        executed = self.world.apply_boss_action(int(action))
        self.reward_acc.action_executed = executed

        # action_repeat 프레임 시뮬레이션
        for _ in range(self.action_repeat):
            p_action = self.player_policy.get_action(self.world)
            self.world.apply_player_action(**p_action)
            self.world.tick()

            if self._renderer and self.render_mode == 'human':
                self._renderer.sync(self.world)
                self._renderer.render()

            if self.world.done:
                break

        # 보상 계산
        self.reward_acc.damage_dealt = max(0, prev_player_hp - self.world.player.health.current)
        self.reward_acc.damage_taken = max(0, prev_boss_hp - self.world.boss.health.current)
        self.reward_acc.phase_transition = self.world.boss.phase > prev_boss_phase
        self.reward_acc.player_killed = self.world.player.health.current <= 0
        self.reward_acc.boss_died = self.world.boss.health.current <= 0
        self.reward_acc.distance_to_player = self.world.get_distance()
        self.reward_acc.player_hp_ratio = self.world.player.health.ratio
        self.reward_acc.boss_hp_ratio = self.world.boss.health.ratio
        self.reward_acc.action_id = int(action)

        reward = compute_reward(self.reward_acc, self.world.boss.recent_actions)

        terminated = (self.world.player.health.current <= 0
                      or self.world.boss.health.current <= 0)
        truncated = self.world.step_count >= MAX_EPISODE_STEPS

        obs = build_observation(self.world)
        info = {
            'action_mask': get_action_mask(self.world),
            'boss_hp': self.world.boss.health.ratio,
            'player_hp': self.world.player.health.ratio,
            'phase': self.world.boss.phase,
            'step': self.world.step_count,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer:
            return self._renderer.render()

    def close(self):
        if self._renderer:
            self._renderer.close()
