"""UI 렌더링 - HP바, 보스 페이즈, 상태 표시"""
from __future__ import annotations
import pygame

from config import WORLD_WIDTH


class UIRenderer:
    def __init__(self):
        self.font = None
        self._init_done = False

    def _lazy_init(self):
        if not self._init_done:
            self.font = pygame.font.SysFont('monospace', 16)
            self.small_font = pygame.font.SysFont('monospace', 12)
            self._init_done = True

    def draw(self, screen: pygame.Surface, world, dda_percent: int | None = None):
        self._lazy_init()
        self._draw_player_hp(screen, world.player)
        self._draw_boss_hp(screen, world.boss)
        self._draw_boss_status(screen, world.boss)
        self._draw_step_counter(screen, world.step_count)
        if dda_percent is not None:
            self._draw_dda(screen, dda_percent)

    def _draw_player_hp(self, screen, player):
        x, y = 20, 20
        w, h = 200, 18
        ratio = player.health.ratio

        # 배경
        pygame.draw.rect(screen, (60, 60, 60), (x, y, w, h))
        # HP바
        color = (0, 200, 100) if ratio > 0.5 else (200, 200, 0) if ratio > 0.25 else (200, 50, 50)
        pygame.draw.rect(screen, color, (x, y, int(w * ratio), h))
        # 테두리
        pygame.draw.rect(screen, (200, 200, 200), (x, y, w, h), 2)
        # 텍스트
        text = self.font.render(f"PLAYER  {player.health.current}/{player.health.maximum}", True, (255, 255, 255))
        screen.blit(text, (x + 5, y + 1))

        # 회피 쿨다운
        if not player.dodge_cooldown.ready:
            cd_w = 60
            cd_ratio = 1.0 - player.dodge_cooldown.ratio
            pygame.draw.rect(screen, (60, 60, 60), (x, y + h + 4, cd_w, 6))
            pygame.draw.rect(screen, (100, 180, 255), (x, y + h + 4, int(cd_w * cd_ratio), 6))

    def _draw_boss_hp(self, screen, boss):
        w, h = 400, 22
        x = (WORLD_WIDTH - w) // 2
        y = WORLD_WIDTH // 2 + 240  # 화면 하단 근처
        y = min(y, 560)
        ratio = boss.health.ratio

        # 배경
        pygame.draw.rect(screen, (60, 60, 60), (x, y, w, h))
        # HP바 (페이즈별 색상)
        colors = [(200, 40, 40), (200, 120, 0), (180, 0, 180), (200, 0, 0)]
        color = colors[min(boss.phase, 3)]
        pygame.draw.rect(screen, color, (x, y, int(w * ratio), h))
        # 페이즈 구분선
        for threshold in [0.75, 0.50, 0.25]:
            lx = x + int(w * threshold)
            pygame.draw.line(screen, (255, 255, 255), (lx, y), (lx, y + h), 1)
        # 테두리
        pygame.draw.rect(screen, (200, 200, 200), (x, y, w, h), 2)
        # 텍스트
        text = self.font.render(f"BOSS  Phase {boss.phase + 1}  {boss.health.current}/{boss.health.maximum}",
                                True, (255, 255, 255))
        screen.blit(text, (x + 5, y + 2))

    def _draw_boss_status(self, screen, boss):
        x = WORLD_WIDTH - 150
        y = 20
        statuses = []
        if boss.is_shielded:
            statuses.append(("SHIELD", (100, 150, 255)))
        if boss.enraged:
            statuses.append(("ENRAGED", (255, 80, 80)))
        if boss.is_acting and boss.current_action:
            statuses.append((boss.current_action.name, (255, 220, 100)))

        for i, (text, color) in enumerate(statuses):
            surf = self.small_font.render(text, True, color)
            screen.blit(surf, (x, y + i * 16))

    def _draw_step_counter(self, screen, step_count):
        text = self.small_font.render(f"Step: {step_count}", True, (150, 150, 150))
        screen.blit(text, (WORLD_WIDTH - 100, WORLD_WIDTH // 2 + 265))

    def _draw_dda(self, screen, percent: int):
        x, y = 20, 50
        w, h = 80, 10
        # 배경
        pygame.draw.rect(screen, (40, 40, 40), (x, y, w, h))
        # 난이도 바
        ratio = percent / 100.0
        if ratio < 0.4:
            color = (80, 180, 80)     # 쉬움 - 녹색
        elif ratio < 0.7:
            color = (200, 180, 50)    # 보통 - 노란색
        else:
            color = (200, 60, 60)     # 어려움 - 빨간색
        pygame.draw.rect(screen, color, (x, y, int(w * ratio), h))
        pygame.draw.rect(screen, (120, 120, 120), (x, y, w, h), 1)
        text = self.small_font.render(f"DDA {percent}%", True, (150, 150, 150))
        screen.blit(text, (x + w + 5, y - 1))
