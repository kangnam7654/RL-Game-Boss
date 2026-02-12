"""파티클 및 시각 효과"""
from __future__ import annotations
import random
import math
import pygame


class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size')

    def __init__(self, x, y, vx, vy, life, color, size=3):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.color = color
        self.size = size

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vx *= 0.95
        self.vy *= 0.95
        self.life -= 1

    @property
    def alive(self):
        return self.life > 0

    @property
    def alpha(self):
        return max(0, self.life / self.max_life)


class EffectsManager:
    def __init__(self):
        self.particles: list[Particle] = []
        self.screen_shake = 0
        self.shake_intensity = 0.0

    def spawn_hit_particles(self, x: float, y: float, color: tuple, count: int = 8):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 6)
            self.particles.append(Particle(
                x, y,
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                life=random.randint(10, 25),
                color=color,
                size=random.randint(2, 4),
            ))

    def spawn_explosion(self, x: float, y: float, radius: float):
        count = int(radius / 3)
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(0, radius * 0.7)
            speed = random.uniform(1, 4)
            self.particles.append(Particle(
                x + math.cos(angle) * dist * 0.3,
                y + math.sin(angle) * dist * 0.3,
                math.cos(angle) * speed,
                math.sin(angle) * speed,
                life=random.randint(15, 35),
                color=(255, random.randint(100, 200), 0),
                size=random.randint(2, 5),
            ))
        self.shake(8, 5.0)

    def spawn_dash_trail(self, x: float, y: float, color: tuple):
        for _ in range(3):
            self.particles.append(Particle(
                x + random.uniform(-5, 5),
                y + random.uniform(-5, 5),
                random.uniform(-0.5, 0.5),
                random.uniform(-0.5, 0.5),
                life=random.randint(5, 12),
                color=color,
                size=random.randint(2, 3),
            ))

    def shake(self, duration: int, intensity: float):
        self.screen_shake = max(self.screen_shake, duration)
        self.shake_intensity = max(self.shake_intensity, intensity)

    def get_shake_offset(self) -> tuple[int, int]:
        if self.screen_shake <= 0:
            return 0, 0
        ox = int(random.uniform(-self.shake_intensity, self.shake_intensity))
        oy = int(random.uniform(-self.shake_intensity, self.shake_intensity))
        return ox, oy

    def update_and_draw(self, screen: pygame.Surface):
        if self.screen_shake > 0:
            self.screen_shake -= 1
            self.shake_intensity *= 0.85
            if self.screen_shake <= 0:
                self.shake_intensity = 0

        alive = []
        for p in self.particles:
            p.update()
            if p.alive:
                alive.append(p)
                alpha = p.alpha
                r = int(p.color[0] * alpha)
                g = int(p.color[1] * alpha)
                b = int(p.color[2] * alpha)
                pygame.draw.circle(screen, (r, g, b),
                                   (int(p.x), int(p.y)), p.size)
        self.particles = alive
