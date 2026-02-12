"""인간 플레이 모드 - 학습된 AI 보스와 대전"""
from __future__ import annotations
import math
import pygame

from config import (
    WORLD_WIDTH, WORLD_HEIGHT, ACTION_REPEAT,
    PLAYER_ATTACK_RANGE,
)
from core.game_world import GameWorld
from core.physics import angle_between
from rendering.renderer import GameRenderer
from ai.observation import build_observation
from ai.difficulty import DifficultyManager


class PlayMode:
    def __init__(self, model_path: str = 'models/best/best_model.zip',
                 enable_dda: bool = True):
        self.world = GameWorld()
        self.renderer = GameRenderer(self.world)
        self.model_path = model_path
        self.model = None
        self.action_repeat_counter = 0
        self.current_boss_action = 0
        self.dda = DifficultyManager() if enable_dda else None
        self._prev_player_hp = 0

    def _load_model(self):
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(self.model_path)
            print(f"Loaded model from {self.model_path}")
        except FileNotFoundError:
            print(f"No model found at {self.model_path}")
            print("Running with random boss actions.")
            self.model = None

    def run(self):
        self._load_model()
        self.world.reset()
        if self.dda:
            self.dda.reset()
        self._prev_player_hp = self.world.player.health.current
        self.renderer.sync(self.world)
        self.renderer._prev_player_hp = self.world.player.health.current
        self.renderer._prev_boss_hp = self.world.boss.health.current

        running = True
        result = None

        while running and not self.world.done:
            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        self.world.reset()
                        self.renderer.sync(self.world)
                        self.renderer._prev_player_hp = self.world.player.health.current
                        self.renderer._prev_boss_hp = self.world.boss.health.current
                        continue

            if not running:
                break

            # 인간 입력
            keys = pygame.key.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            mouse_click = pygame.mouse.get_pressed()[0]

            move_dir = self._get_move_dir(keys)
            aim_angle = math.atan2(
                mouse_pos[1] - self.world.player.transform.y,
                mouse_pos[0] - self.world.player.transform.x,
            )
            dodge = keys[pygame.K_SPACE]

            self.world.apply_player_action(move_dir, aim_angle, mouse_click, dodge)

            # 보스 AI
            if self.action_repeat_counter <= 0:
                if self.model is not None:
                    obs = build_observation(self.world)
                    action, _ = self.model.predict(obs, deterministic=True)
                    self.current_boss_action = int(action)
                else:
                    # 랜덤 액션
                    import random
                    from ai.action_mapping import get_action_mask
                    mask = get_action_mask(self.world)
                    available = [i for i in range(len(mask)) if mask[i] > 0]
                    self.current_boss_action = random.choice(available) if available else 0
                self.action_repeat_counter = ACTION_REPEAT

            self.world.apply_boss_action(self.current_boss_action)
            self.action_repeat_counter -= 1

            self.world.tick()

            # DDA 업데이트
            if self.dda is not None:
                cur_hp = self.world.player.health.current
                was_hit = cur_hp < self._prev_player_hp
                is_dodging = self.world.player.is_dodging
                self._prev_player_hp = cur_hp

                self.dda.update(
                    player_hp_ratio=self.world.player.health.ratio,
                    player_was_hit=was_hit,
                    player_dodged=is_dodging,
                )
                # 배율 적용
                self.world.damage_mult = self.dda.state.damage_mult
                self.world.speed_mult = self.dda.state.speed_mult
                self.world.cooldown_mult = self.dda.state.cooldown_mult
                self.world.proj_speed_mult = self.dda.state.proj_speed_mult
                self.renderer.dda_percent = self.dda.difficulty_percent

            # 보스 방향 업데이트 (플레이어를 바라보도록)
            self.world.boss.transform.angle = angle_between(
                self.world.boss.transform.x, self.world.boss.transform.y,
                self.world.player.transform.x, self.world.player.transform.y,
            )

            running = self.renderer.render()

        # 결과 표시
        if self.world.player.health.current <= 0:
            result = "DEFEAT"
        elif self.world.boss.health.current <= 0:
            result = "VICTORY"
        else:
            result = "TIME UP"

        if running and result:
            self._show_result(result)

        self.renderer.close()

    def _get_move_dir(self, keys) -> tuple[float, float]:
        dx, dy = 0.0, 0.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy -= 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy += 1.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= 1.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += 1.0
        # 대각선 정규화
        if dx != 0 and dy != 0:
            mag = math.hypot(dx, dy)
            dx /= mag
            dy /= mag
        return (dx, dy)

    def _show_result(self, result: str):
        font = pygame.font.SysFont('monospace', 48, bold=True)
        small_font = pygame.font.SysFont('monospace', 20)

        color = (0, 255, 100) if result == "VICTORY" else (255, 80, 80)
        text = font.render(result, True, color)
        hint = small_font.render("Press R to restart, ESC to quit", True, (200, 200, 200))

        text_rect = text.get_rect(center=(WORLD_WIDTH // 2, WORLD_HEIGHT // 2 - 20))
        hint_rect = hint.get_rect(center=(WORLD_WIDTH // 2, WORLD_HEIGHT // 2 + 30))

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting = False
                    if event.key == pygame.K_r:
                        self.world.reset()
                        self.renderer.sync(self.world)
                        self.renderer._prev_player_hp = self.world.player.health.current
                        self.renderer._prev_boss_hp = self.world.boss.health.current
                        self.run()
                        return

            # 반투명 오버레이
            overlay = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.renderer.screen.blit(overlay, (0, 0))
            self.renderer.screen.blit(text, text_rect)
            self.renderer.screen.blit(hint, hint_rect)
            pygame.display.flip()
            self.renderer.clock.tick(30)
