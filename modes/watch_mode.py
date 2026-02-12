"""AI vs AI 관전 모드 - 학습된 보스와 플레이어가 자동 대전"""
from __future__ import annotations
import pygame

from config import (
    WORLD_WIDTH, WORLD_HEIGHT, ACTION_REPEAT, PLAYER_ACTION_REPEAT,
)
from core.game_world import GameWorld
from core.physics import angle_between
from rendering.renderer import GameRenderer
from ai.observation import build_observation
from ai.player_observation import build_player_observation
from ai.action_mapping import get_action_mask

# PlayerEnv와 동일한 이동 방향
MOVE_DIRS = [
    (0.0, 0.0), (0.0, -1.0), (0.707, -0.707), (1.0, 0.0),
    (0.707, 0.707), (0.0, 1.0), (-0.707, 0.707), (-1.0, 0.0),
    (-0.707, -0.707),
]


class WatchMode:
    """학습된 보스 모델 vs 학습된 플레이어 모델 관전"""

    def __init__(self, boss_model_path: str, player_model_path: str):
        self.world = GameWorld()
        self.renderer = GameRenderer(self.world)

        self.boss_model_path = boss_model_path
        self.player_model_path = player_model_path
        self.boss_model = None
        self.player_model = None

        self._boss_action_counter = 0
        self._boss_current_action = 0
        self._player_action_counter = 0
        self._player_current_action = None

        # 통계
        self.stats = {'games': 0, 'player_wins': 0, 'boss_wins': 0, 'draws': 0}

    def _load_models(self):
        from stable_baselines3 import PPO

        try:
            self.boss_model = PPO.load(self.boss_model_path)
            print(f"Boss model loaded: {self.boss_model_path}")
        except FileNotFoundError:
            print(f"Boss model not found: {self.boss_model_path}")

        try:
            self.player_model = PPO.load(self.player_model_path)
            print(f"Player model loaded: {self.player_model_path}")
        except FileNotFoundError:
            print(f"Player model not found: {self.player_model_path}")

    def _get_boss_action(self) -> int:
        if self._boss_action_counter <= 0:
            if self.boss_model is not None:
                obs = build_observation(self.world)
                action, _ = self.boss_model.predict(obs, deterministic=False)
                self._boss_current_action = int(action)
            else:
                import random
                mask = get_action_mask(self.world)
                available = [i for i in range(len(mask)) if mask[i] > 0]
                self._boss_current_action = random.choice(available) if available else 0
            self._boss_action_counter = ACTION_REPEAT

        self._boss_action_counter -= 1
        return self._boss_current_action

    def _get_player_action(self):
        if self._player_action_counter <= 0:
            if self.player_model is not None:
                obs = build_player_observation(self.world)
                action, _ = self.player_model.predict(obs, deterministic=False)
                self._player_current_action = action
            else:
                import numpy as np
                self._player_current_action = np.array([
                    np.random.randint(9),
                    np.random.randint(2),
                    np.random.randint(2),
                ])
            self._player_action_counter = PLAYER_ACTION_REPEAT

        self._player_action_counter -= 1

        act = self._player_current_action
        move_dir = MOVE_DIRS[int(act[0])]
        attack = bool(act[1])
        dodge = bool(act[2])
        aim_angle = angle_between(
            self.world.player.transform.x, self.world.player.transform.y,
            self.world.boss.transform.x, self.world.boss.transform.y,
        )
        return move_dir, aim_angle, attack, dodge

    def _reset_round(self):
        self.world.reset()
        self._boss_action_counter = 0
        self._player_action_counter = 0
        self.renderer.sync(self.world)
        self.renderer._prev_player_hp = self.world.player.health.current
        self.renderer._prev_boss_hp = self.world.boss.health.current

    def run(self):
        self._load_models()
        self._reset_round()

        running = True
        paused = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self._reset_round()
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            if not running:
                break

            if paused:
                self._draw_paused()
                self.renderer.clock.tick(30)
                continue

            # 양쪽 AI 행동
            boss_action = self._get_boss_action()
            move_dir, aim_angle, attack, dodge = self._get_player_action()

            self.world.apply_player_action(move_dir, aim_angle, attack, dodge)
            self.world.apply_boss_action(boss_action)
            self.world.tick()

            # 보스 방향 업데이트
            self.world.boss.transform.angle = angle_between(
                self.world.boss.transform.x, self.world.boss.transform.y,
                self.world.player.transform.x, self.world.player.transform.y,
            )

            running = self.renderer.render()

            # 라운드 종료 체크
            if self.world.done:
                self.stats['games'] += 1
                if self.world.boss.health.current <= 0:
                    self.stats['player_wins'] += 1
                    result = "PLAYER WINS"
                elif self.world.player.health.current <= 0:
                    self.stats['boss_wins'] += 1
                    result = "BOSS WINS"
                else:
                    self.stats['draws'] += 1
                    result = "TIME UP"

                running = self._show_result(result)
                if running:
                    self._reset_round()

        self.renderer.close()
        self._print_stats()

    def _draw_paused(self):
        font = pygame.font.SysFont('monospace', 32, bold=True)
        text = font.render("PAUSED", True, (200, 200, 200))
        rect = text.get_rect(center=(WORLD_WIDTH // 2, WORLD_HEIGHT // 2))

        overlay = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.renderer.screen.blit(overlay, (0, 0))
        self.renderer.screen.blit(text, rect)
        pygame.display.flip()

    def _show_result(self, result: str) -> bool:
        font = pygame.font.SysFont('monospace', 36, bold=True)
        small = pygame.font.SysFont('monospace', 18)

        color = (0, 255, 100) if "PLAYER" in result else (255, 80, 80)
        if result == "TIME UP":
            color = (200, 200, 100)

        text = font.render(result, True, color)
        stats_text = small.render(
            f"W:{self.stats['player_wins']} L:{self.stats['boss_wins']} "
            f"D:{self.stats['draws']} "
            f"WR:{self._winrate():.0%}",
            True, (200, 200, 200),
        )
        hint = small.render("Auto-restart in 2s / R=restart / ESC=quit", True, (150, 150, 150))

        text_rect = text.get_rect(center=(WORLD_WIDTH // 2, WORLD_HEIGHT // 2 - 30))
        stats_rect = stats_text.get_rect(center=(WORLD_WIDTH // 2, WORLD_HEIGHT // 2 + 10))
        hint_rect = hint.get_rect(center=(WORLD_WIDTH // 2, WORLD_HEIGHT // 2 + 40))

        overlay = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.renderer.screen.blit(overlay, (0, 0))
        self.renderer.screen.blit(text, text_rect)
        self.renderer.screen.blit(stats_text, stats_rect)
        self.renderer.screen.blit(hint, hint_rect)
        pygame.display.flip()

        # 2초 대기 후 자동 재시작
        wait_ms = 2000
        start = pygame.time.get_ticks()
        while pygame.time.get_ticks() - start < wait_ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_r:
                        return True
            self.renderer.clock.tick(30)

        return True

    def _winrate(self) -> float:
        total = self.stats['player_wins'] + self.stats['boss_wins']
        if total == 0:
            return 0.0
        return self.stats['player_wins'] / total

    def _print_stats(self):
        g = self.stats['games']
        if g == 0:
            return
        print(f"\n=== Watch Mode Stats ===")
        print(f"  Games:       {g}")
        print(f"  Player wins: {self.stats['player_wins']} ({100*self.stats['player_wins']/g:.1f}%)")
        print(f"  Boss wins:   {self.stats['boss_wins']} ({100*self.stats['boss_wins']/g:.1f}%)")
        print(f"  Draws:       {self.stats['draws']}")
        print(f"  Win rate:    {self._winrate():.1%}")
