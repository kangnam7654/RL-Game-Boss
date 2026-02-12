"""게임 엔티티 클래스"""
from __future__ import annotations
import math
from config import (
    PLAYER_MAX_HP, PLAYER_HITBOX_RADIUS, PLAYER_SPEED,
    PLAYER_DODGE_COOLDOWN, PLAYER_ATTACK_COOLDOWN, PLAYER_DODGE_DURATION,
    BOSS_MAX_HP, BOSS_HITBOX_RADIUS, BOSS_SPEED, PHASE_THRESHOLDS,
    WORLD_WIDTH, WORLD_HEIGHT,
)
from core.components import Transform, Health, Hitbox, Cooldown


_next_entity_id = 0


def _get_id() -> int:
    global _next_entity_id
    eid = _next_entity_id
    _next_entity_id += 1
    return eid


def reset_entity_ids():
    global _next_entity_id
    _next_entity_id = 0


class Entity:
    def __init__(self, x: float, y: float, radius: float):
        self.id = _get_id()
        self.transform = Transform(x, y)
        self.hitbox = Hitbox(radius)
        self.alive = True
        self.vx = 0.0
        self.vy = 0.0

    def update(self):
        pass

    def apply_velocity(self):
        self.transform.x += self.vx
        self.transform.y += self.vy

    def clamp_to_world(self):
        r = self.hitbox.radius
        self.transform.x = max(r, min(WORLD_WIDTH - r, self.transform.x))
        self.transform.y = max(r, min(WORLD_HEIGHT - r, self.transform.y))


class Player(Entity):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, PLAYER_HITBOX_RADIUS)
        self.health = Health(PLAYER_MAX_HP, PLAYER_MAX_HP)
        self.speed = PLAYER_SPEED
        self.dodge_cooldown = Cooldown(0, PLAYER_DODGE_COOLDOWN)
        self.attack_cooldown = Cooldown(0, PLAYER_ATTACK_COOLDOWN)
        self.is_dodging = False
        self.dodge_timer = 0
        self.aim_angle = 0.0
        self.dodge_dir_x = 0.0
        self.dodge_dir_y = 0.0

    def update(self):
        self.dodge_cooldown.tick()
        self.attack_cooldown.tick()
        if self.health.invincible_frames > 0:
            self.health.invincible_frames -= 1
        if self.is_dodging:
            self.dodge_timer -= 1
            self.vx = self.dodge_dir_x
            self.vy = self.dodge_dir_y
            if self.dodge_timer <= 0:
                self.is_dodging = False
                self.vx = 0.0
                self.vy = 0.0


class Boss(Entity):
    def __init__(self, x: float, y: float):
        super().__init__(x, y, BOSS_HITBOX_RADIUS)
        self.health = Health(BOSS_MAX_HP, BOSS_MAX_HP)
        self.speed = BOSS_SPEED
        self.phase = 0
        self.current_action = None
        self.action_timer = 0
        self.action_cooldowns: dict[int, Cooldown] = {}
        self.recent_actions: list[int] = []
        self.is_shielded = False
        self.shield_timer = 0
        self.enraged = False
        self.enrage_timer = 0

    @property
    def is_acting(self) -> bool:
        return self.action_timer > 0

    def update(self):
        for cd in self.action_cooldowns.values():
            cd.tick()
        if self.action_timer > 0:
            self.action_timer -= 1
            if self.action_timer <= 0:
                self.current_action = None
                self.vx = 0.0
                self.vy = 0.0
        if self.health.invincible_frames > 0:
            self.health.invincible_frames -= 1
        if self.shield_timer > 0:
            self.shield_timer -= 1
            if self.shield_timer <= 0:
                self.is_shielded = False
        if self.enrage_timer > 0:
            self.enrage_timer -= 1
            if self.enrage_timer <= 0:
                self.enraged = False
        # 페이즈 전환
        new_phase = sum(1 for t in PHASE_THRESHOLDS if self.health.ratio <= t)
        if new_phase > self.phase:
            self.phase = new_phase


class Projectile(Entity):
    def __init__(self, x: float, y: float, vx: float, vy: float,
                 radius: float, damage: int, owner_id: int,
                 lifetime: int = 180, homing: bool = False):
        super().__init__(x, y, radius)
        self.vx = vx
        self.vy = vy
        self.hitbox.damage = damage
        self.hitbox.owner_id = owner_id
        self.lifetime = lifetime
        self.homing = homing
        self.homing_strength = 0.05

    def update(self, target: Entity | None = None):
        if self.homing and target and target.alive:
            dx = target.transform.x - self.transform.x
            dy = target.transform.y - self.transform.y
            dist = max(math.hypot(dx, dy), 1.0)
            self.vx += (dx / dist) * self.homing_strength
            self.vy += (dy / dist) * self.homing_strength
            speed = math.hypot(self.vx, self.vy)
            if speed > 6.0:
                self.vx = (self.vx / speed) * 6.0
                self.vy = (self.vy / speed) * 6.0
        self.apply_velocity()
        self.lifetime -= 1
        if self.lifetime <= 0:
            self.alive = False
        # 월드 밖으로 나가면 제거
        if (self.transform.x < -50 or self.transform.x > WORLD_WIDTH + 50
                or self.transform.y < -50 or self.transform.y > WORLD_HEIGHT + 50):
            self.alive = False


class AoEZone(Entity):
    def __init__(self, x: float, y: float, radius: float, damage: int,
                 duration: int, damage_interval: int = 15, owner_id: int = -1):
        super().__init__(x, y, radius)
        self.hitbox.damage = damage
        self.hitbox.owner_id = owner_id
        self.duration = duration
        self.damage_interval = damage_interval
        self.tick_counter = 0

    @property
    def can_damage(self) -> bool:
        return self.tick_counter % self.damage_interval == 0

    def update(self):
        self.duration -= 1
        self.tick_counter += 1
        if self.duration <= 0:
            self.alive = False


class Minion(Entity):
    def __init__(self, x: float, y: float, owner_id: int = -1):
        super().__init__(x, y, radius=10.0)
        self.health = Health(30, 30)
        self.speed = 2.0
        self.hitbox.damage = 5
        self.hitbox.owner_id = owner_id
        self.attack_cooldown = Cooldown(0, 60)

    def update(self, target: Entity | None = None):
        self.attack_cooldown.tick()
        if self.health.current <= 0:
            self.alive = False
            return
        if target and target.alive:
            dx = target.transform.x - self.transform.x
            dy = target.transform.y - self.transform.y
            dist = max(math.hypot(dx, dy), 1.0)
            self.vx = (dx / dist) * self.speed
            self.vy = (dy / dist) * self.speed
            self.apply_velocity()
            self.clamp_to_world()
