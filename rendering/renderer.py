"""Pygame 메인 렌더러"""
from __future__ import annotations
import math
import pygame

from config import WORLD_WIDTH, WORLD_HEIGHT, FPS_PLAY, BOSS_HITBOX_RADIUS
from rendering.effects import EffectsManager
from rendering.ui import UIRenderer


# 색상 팔레트
C_BG = (25, 25, 35)
C_ARENA = (35, 35, 48)
C_ARENA_LINE = (50, 50, 65)
C_PLAYER = (0, 180, 255)
C_PLAYER_DODGE = (100, 220, 255)
C_BOSS = (220, 50, 50)
C_BOSS_ENRAGED = (255, 100, 60)
C_BOSS_SHIELD = (150, 180, 255)
C_PROJECTILE = (255, 170, 30)
C_HOMING = (255, 80, 200)
C_AOE = (100, 255, 100)
C_POISON = (80, 200, 80)
C_MINION = (180, 0, 200)
C_LASER_AIM = (255, 255, 100)
C_LASER_FIRE = (255, 50, 50)
C_PLAYER_ATK = (150, 200, 255)


class GameRenderer:
    def __init__(self, world=None):
        if not pygame.get_init():
            pygame.init()
        self.screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
        pygame.display.set_caption("Boss AI - Top-Down Action")
        self.clock = pygame.time.Clock()
        self.world = world
        self.effects = EffectsManager()
        self.ui = UIRenderer()
        self._prev_player_hp = 0
        self._prev_boss_hp = 0
        self.dda_percent: int | None = None  # DDA 표시용

    def sync(self, world):
        self.world = world

    def render(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False

        w = self.world
        if w is None:
            return True

        # 히트 이펙트 트리거
        self._check_hit_effects(w)

        # 화면 흔들림 오프셋
        ox, oy = self.effects.get_shake_offset()

        self.screen.fill(C_BG)

        # 아레나 그리드
        self._draw_arena(ox, oy)

        # AoE 존
        for zone in w.aoe_zones:
            color = C_POISON if zone.duration > 60 else C_AOE
            surf = pygame.Surface((int(zone.hitbox.radius * 2), int(zone.hitbox.radius * 2)), pygame.SRCALPHA)
            alpha = min(120, int(180 * (zone.duration / max(zone.duration + zone.tick_counter, 1))))
            pygame.draw.circle(surf, (*color, alpha),
                               (int(zone.hitbox.radius), int(zone.hitbox.radius)),
                               int(zone.hitbox.radius))
            self.screen.blit(surf,
                             (int(zone.transform.x - zone.hitbox.radius + ox),
                              int(zone.transform.y - zone.hitbox.radius + oy)))

        # 레이저 빔
        for laser in w.laser_beams:
            if laser['active']:
                color = C_LASER_FIRE
                width = 4
            else:
                color = C_LASER_AIM
                width = 1
            pygame.draw.line(self.screen, color,
                             (int(laser['x1'] + ox), int(laser['y1'] + oy)),
                             (int(laser['x2'] + ox), int(laser['y2'] + oy)),
                             width)

        # 미니언
        for minion in w.minions:
            pygame.draw.circle(self.screen, C_MINION,
                               (int(minion.transform.x + ox), int(minion.transform.y + oy)),
                               int(minion.hitbox.radius))
            # HP 표시
            if minion.health.ratio < 1.0:
                bx = int(minion.transform.x - 8 + ox)
                by = int(minion.transform.y - 16 + oy)
                pygame.draw.rect(self.screen, (60, 60, 60), (bx, by, 16, 3))
                pygame.draw.rect(self.screen, (200, 50, 50),
                                 (bx, by, int(16 * minion.health.ratio), 3))

        # 투사체
        for proj in w.projectiles:
            color = C_HOMING if proj.homing else C_PROJECTILE
            pygame.draw.circle(self.screen, color,
                               (int(proj.transform.x + ox), int(proj.transform.y + oy)),
                               int(proj.hitbox.radius))

        # 보스
        self._draw_boss(w.boss, ox, oy)

        # 플레이어
        self._draw_player(w.player, ox, oy)

        # 파티클
        self.effects.update_and_draw(self.screen)

        # UI
        self.ui.draw(self.screen, w, dda_percent=self.dda_percent)

        pygame.display.flip()
        self.clock.tick(FPS_PLAY)
        return True

    def _draw_arena(self, ox, oy):
        # 격자 패턴
        grid = 40
        for x in range(0, WORLD_WIDTH, grid):
            pygame.draw.line(self.screen, C_ARENA_LINE,
                             (x + ox, 0 + oy), (x + ox, WORLD_HEIGHT + oy), 1)
        for y in range(0, WORLD_HEIGHT, grid):
            pygame.draw.line(self.screen, C_ARENA_LINE,
                             (0 + ox, y + oy), (WORLD_WIDTH + ox, y + oy), 1)
        # 경계선
        pygame.draw.rect(self.screen, (80, 80, 100),
                         (ox, oy, WORLD_WIDTH, WORLD_HEIGHT), 3)

    def _draw_player(self, player, ox, oy):
        px = int(player.transform.x + ox)
        py = int(player.transform.y + oy)
        r = int(player.hitbox.radius)

        # 회피 중이면 반투명 + 다른 색
        if player.is_dodging:
            color = C_PLAYER_DODGE
            self.effects.spawn_dash_trail(player.transform.x, player.transform.y, C_PLAYER)
        else:
            color = C_PLAYER

        # 무적 프레임이면 깜빡임
        if player.health.invincible_frames > 0 and player.health.invincible_frames % 4 < 2:
            color = tuple(min(255, c + 100) for c in color)

        pygame.draw.circle(self.screen, color, (px, py), r)

        # 조준선
        aim_len = 25
        ax = px + int(math.cos(player.aim_angle) * aim_len)
        ay = py + int(math.sin(player.aim_angle) * aim_len)
        pygame.draw.line(self.screen, C_PLAYER_ATK, (px, py), (ax, ay), 2)

        # 공격 범위 표시 (공격 쿨다운 준비 시)
        if player.attack_cooldown.ready:
            atk_x = px + int(math.cos(player.aim_angle) * 48)
            atk_y = py + int(math.sin(player.aim_angle) * 48)
            pygame.draw.circle(self.screen, (*C_PLAYER_ATK, 80), (atk_x, atk_y), 5, 1)

    def _draw_boss(self, boss, ox, oy):
        bx = int(boss.transform.x + ox)
        by = int(boss.transform.y + oy)
        r = int(boss.hitbox.radius)

        # 기본 색상
        if boss.enraged:
            color = C_BOSS_ENRAGED
        else:
            color = C_BOSS

        pygame.draw.circle(self.screen, color, (bx, by), r)

        # 내부 눈 (방향 표시)
        eye_dist = r * 0.4
        eye_x = bx + int(math.cos(boss.transform.angle) * eye_dist)
        eye_y = by + int(math.sin(boss.transform.angle) * eye_dist)
        pygame.draw.circle(self.screen, (255, 255, 200), (eye_x, eye_y), 4)

        # 실드
        if boss.is_shielded:
            pygame.draw.circle(self.screen, C_BOSS_SHIELD, (bx, by), r + 8, 3)

        # 액션 실행 중 표시
        if boss.is_acting and boss.hitbox.damage > 0:
            pygame.draw.circle(self.screen, (255, 255, 0), (bx, by),
                               int(boss.hitbox.radius), 2)

        # 스핀 공격 시 회전 표시
        if boss.is_acting and boss.current_action:
            if boss.current_action.name == 'Spin Attack':
                for i in range(4):
                    angle = (boss.action_timer * 0.3) + (i * math.pi / 2)
                    sx = bx + int(math.cos(angle) * (r + 15))
                    sy = by + int(math.sin(angle) * (r + 15))
                    pygame.draw.circle(self.screen, (255, 200, 0), (sx, sy), 4)

    def _check_hit_effects(self, world):
        # HP 변화 감지 -> 이펙트
        if world.player.health.current < self._prev_player_hp:
            self.effects.spawn_hit_particles(
                world.player.transform.x, world.player.transform.y,
                C_PLAYER, count=10)
            self.effects.shake(5, 3.0)
        if world.boss.health.current < self._prev_boss_hp:
            self.effects.spawn_hit_particles(
                world.boss.transform.x, world.boss.transform.y,
                C_BOSS, count=6)

        self._prev_player_hp = world.player.health.current
        self._prev_boss_hp = world.boss.health.current

        # 보스 액션별 이펙트
        for event in world.events:
            if event.get('type') == 'boss_action':
                name = event.get('name', '')
                if name in ('Explosion', 'Ground Slam'):
                    self.effects.spawn_explosion(
                        world.boss.transform.x, world.boss.transform.y, 80)

    def close(self):
        pygame.quit()
