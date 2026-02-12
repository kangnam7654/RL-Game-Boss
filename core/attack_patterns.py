"""16개 보스 공격 패턴 정의"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.game_world import GameWorld
    from core.entities import Boss, Player

from core.entities import Projectile, AoEZone, Minion
from core.physics import direction, angle_between, spread_directions, distance


class ActionID(IntEnum):
    MOVE_TOWARD = 0
    MOVE_AWAY = 1
    CHARGE_DASH = 2
    GROUND_SLAM = 3
    MULTI_HIT_COMBO = 4
    SPIN_ATTACK = 5
    PROJECTILE_BARRAGE = 6
    LASER_BEAM = 7
    HOMING_MISSILES = 8
    EXPLOSION = 9
    SHOCKWAVE = 10
    POISON_CLOUD = 11
    TELEPORT = 12
    SUMMON_MINIONS = 13
    SHIELD = 14
    ENRAGE = 15


@dataclass
class AttackPattern:
    action_id: ActionID
    name: str
    cooldown_frames: int
    duration_frames: int
    damage: int
    range_min: float = 0.0
    range_max: float = 9999.0
    hitbox_radius: float = 0.0
    min_phase: int = 0
    execute: Callable | None = None

    def is_available(self, boss, dist_to_player: float) -> bool:
        if boss.is_acting:
            return False
        cd = boss.action_cooldowns.get(self.action_id)
        if cd and not cd.ready:
            return False
        if boss.phase < self.min_phase:
            return False
        return True


ATTACK_PATTERNS: dict[int, AttackPattern] = {}


def _register(pattern: AttackPattern):
    ATTACK_PATTERNS[pattern.action_id] = pattern


# ============================================================
# 이동 패턴
# ============================================================

def _exec_move_toward(world: GameWorld, boss: Boss, player: Player):
    dx, dy = direction(boss.transform.x, boss.transform.y,
                       player.transform.x, player.transform.y)
    speed = boss.speed * (1.3 if boss.enraged else 1.0)
    boss.vx = dx * speed
    boss.vy = dy * speed
    return []

_register(AttackPattern(
    action_id=ActionID.MOVE_TOWARD, name="Move Toward",
    cooldown_frames=5, duration_frames=20, damage=0,
    execute=_exec_move_toward,
))


def _exec_move_away(world: GameWorld, boss: Boss, player: Player):
    dx, dy = direction(player.transform.x, player.transform.y,
                       boss.transform.x, boss.transform.y)
    speed = boss.speed * (1.3 if boss.enraged else 1.0)
    boss.vx = dx * speed
    boss.vy = dy * speed
    return []

_register(AttackPattern(
    action_id=ActionID.MOVE_AWAY, name="Move Away",
    cooldown_frames=5, duration_frames=20, damage=0,
    execute=_exec_move_away,
))


# ============================================================
# 근접 패턴
# ============================================================

def _exec_charge_dash(world: GameWorld, boss: Boss, player: Player):
    dx, dy = direction(boss.transform.x, boss.transform.y,
                       player.transform.x, player.transform.y)
    charge_speed = 10.0 * (1.3 if boss.enraged else 1.0)
    boss.vx = dx * charge_speed
    boss.vy = dy * charge_speed
    boss.hitbox.damage = 20 if not boss.enraged else 28
    boss.hitbox.active = True
    boss.hitbox.hit_entities.clear()
    return []

_register(AttackPattern(
    action_id=ActionID.CHARGE_DASH, name="Charge Dash",
    cooldown_frames=120, duration_frames=30, damage=20,
    range_min=80.0, range_max=400.0, hitbox_radius=28.0,
    execute=_exec_charge_dash,
))


def _exec_ground_slam(world: GameWorld, boss: Boss, player: Player):
    # 25프레임 준비 후 충격파 AoE 생성 (world.tick에서 타이머 체크)
    world.pending_effects.append({
        'type': 'ground_slam',
        'x': boss.transform.x,
        'y': boss.transform.y,
        'delay': 25,
        'damage': 25 if not boss.enraged else 35,
        'radius': 80.0,
        'owner_id': boss.id,
    })
    boss.vx = 0.0
    boss.vy = 0.0
    return []

_register(AttackPattern(
    action_id=ActionID.GROUND_SLAM, name="Ground Slam",
    cooldown_frames=150, duration_frames=45, damage=25,
    range_min=0.0, range_max=100.0, hitbox_radius=80.0,
    execute=_exec_ground_slam,
))


def _exec_multi_hit_combo(world: GameWorld, boss: Boss, player: Player):
    dx, dy = direction(boss.transform.x, boss.transform.y,
                       player.transform.x, player.transform.y)
    boss.vx = dx * boss.speed * 1.5
    boss.vy = dy * boss.speed * 1.5
    # 3회 연속 히트 (20프레임 간격)
    for i in range(3):
        world.pending_effects.append({
            'type': 'melee_hit',
            'delay': i * 20,
            'damage': 12 if not boss.enraged else 17,
            'radius': 35.0,
            'owner_id': boss.id,
        })
    return []

_register(AttackPattern(
    action_id=ActionID.MULTI_HIT_COMBO, name="Multi-Hit Combo",
    cooldown_frames=130, duration_frames=60, damage=12,
    range_min=0.0, range_max=80.0, hitbox_radius=35.0,
    execute=_exec_multi_hit_combo,
))


def _exec_spin_attack(world: GameWorld, boss: Boss, player: Player):
    boss.vx = 0.0
    boss.vy = 0.0
    boss.hitbox.damage = 15 if not boss.enraged else 22
    boss.hitbox.active = True
    boss.hitbox.hit_entities.clear()
    # 히트박스를 넓힘 (회전 범위)
    boss.hitbox.radius = 50.0
    world.pending_effects.append({
        'type': 'restore_hitbox',
        'delay': 40,
        'owner_id': boss.id,
    })
    return []

_register(AttackPattern(
    action_id=ActionID.SPIN_ATTACK, name="Spin Attack",
    cooldown_frames=100, duration_frames=40, damage=15,
    range_min=0.0, range_max=60.0, hitbox_radius=50.0,
    execute=_exec_spin_attack,
))


# ============================================================
# 원거리 패턴
# ============================================================

def _exec_projectile_barrage(world: GameWorld, boss: Boss, player: Player):
    angle = angle_between(boss.transform.x, boss.transform.y,
                          player.transform.x, player.transform.y)
    count = 7 if not boss.enraged else 10
    spread = math.radians(60) if not boss.enraged else math.radians(90)
    angles = spread_directions(angle, spread, count)
    proj_speed = 5.0
    proj_dmg = 10 if not boss.enraged else 14

    entities = []
    for a in angles:
        vx = math.cos(a) * proj_speed
        vy = math.sin(a) * proj_speed
        p = Projectile(
            boss.transform.x, boss.transform.y,
            vx, vy, radius=6.0, damage=proj_dmg,
            owner_id=boss.id, lifetime=120,
        )
        entities.append(p)
    boss.vx = 0.0
    boss.vy = 0.0
    return entities

_register(AttackPattern(
    action_id=ActionID.PROJECTILE_BARRAGE, name="Projectile Barrage",
    cooldown_frames=90, duration_frames=50, damage=10,
    range_min=60.0, range_max=500.0,
    execute=_exec_projectile_barrage,
))


def _exec_laser_beam(world: GameWorld, boss: Boss, player: Player):
    angle = angle_between(boss.transform.x, boss.transform.y,
                          player.transform.x, player.transform.y)
    beam_length = 600.0
    # 30프레임 조준 + 40프레임 발사
    world.pending_effects.append({
        'type': 'laser',
        'x1': boss.transform.x,
        'y1': boss.transform.y,
        'angle': angle,
        'length': beam_length,
        'delay': 30,
        'fire_duration': 40,
        'damage': 18 if not boss.enraged else 25,
        'owner_id': boss.id,
    })
    boss.vx = 0.0
    boss.vy = 0.0
    return []

_register(AttackPattern(
    action_id=ActionID.LASER_BEAM, name="Laser Beam",
    cooldown_frames=180, duration_frames=70, damage=18,
    range_min=100.0, range_max=600.0,
    min_phase=1,
    execute=_exec_laser_beam,
))


def _exec_homing_missiles(world: GameWorld, boss: Boss, player: Player):
    count = 3 if not boss.enraged else 5
    entities = []
    for i in range(count):
        offset_angle = (2 * math.pi / count) * i
        vx = math.cos(offset_angle) * 3.0
        vy = math.sin(offset_angle) * 3.0
        p = Projectile(
            boss.transform.x + vx * 5, boss.transform.y + vy * 5,
            vx, vy, radius=5.0,
            damage=12 if not boss.enraged else 17,
            owner_id=boss.id, lifetime=180, homing=True,
        )
        entities.append(p)
    boss.vx = 0.0
    boss.vy = 0.0
    return entities

_register(AttackPattern(
    action_id=ActionID.HOMING_MISSILES, name="Homing Missiles",
    cooldown_frames=140, duration_frames=40, damage=12,
    range_min=80.0, range_max=500.0,
    min_phase=1,
    execute=_exec_homing_missiles,
))


# ============================================================
# 범위 패턴
# ============================================================

def _exec_explosion(world: GameWorld, boss: Boss, player: Player):
    boss.vx = 0.0
    boss.vy = 0.0
    zone = AoEZone(
        boss.transform.x, boss.transform.y,
        radius=90.0,
        damage=22 if not boss.enraged else 30,
        duration=20, damage_interval=20,
        owner_id=boss.id,
    )
    return [zone]

_register(AttackPattern(
    action_id=ActionID.EXPLOSION, name="Explosion",
    cooldown_frames=120, duration_frames=35, damage=22,
    range_min=0.0, range_max=100.0, hitbox_radius=90.0,
    execute=_exec_explosion,
))


def _exec_shockwave(world: GameWorld, boss: Boss, player: Player):
    boss.vx = 0.0
    boss.vy = 0.0
    # 확장 원형 충격파: 3단계로 크기 증가
    entities = []
    for i in range(3):
        zone = AoEZone(
            boss.transform.x, boss.transform.y,
            radius=40.0 + i * 30,
            damage=10 if not boss.enraged else 15,
            duration=8, damage_interval=8,
            owner_id=boss.id,
        )
        # 딜레이를 pending_effects로 처리
        world.pending_effects.append({
            'type': 'delayed_aoe',
            'delay': i * 10,
            'zone': zone,
        })
    return entities

_register(AttackPattern(
    action_id=ActionID.SHOCKWAVE, name="Shockwave",
    cooldown_frames=110, duration_frames=30, damage=10,
    range_min=0.0, range_max=150.0, hitbox_radius=100.0,
    execute=_exec_shockwave,
))


def _exec_poison_cloud(world: GameWorld, boss: Boss, player: Player):
    # 플레이어 위치에 독구름 배치
    zone = AoEZone(
        player.transform.x, player.transform.y,
        radius=60.0,
        damage=5 if not boss.enraged else 8,
        duration=120, damage_interval=15,
        owner_id=boss.id,
    )
    boss.vx = 0.0
    boss.vy = 0.0
    return [zone]

_register(AttackPattern(
    action_id=ActionID.POISON_CLOUD, name="Poison Cloud",
    cooldown_frames=160, duration_frames=20, damage=5,
    range_min=0.0, range_max=400.0, hitbox_radius=60.0,
    min_phase=1,
    execute=_exec_poison_cloud,
))


# ============================================================
# 유틸리티 패턴
# ============================================================

def _exec_teleport(world: GameWorld, boss: Boss, player: Player):
    # 플레이어 뒤편 랜덤 위치로 순간이동
    angle = angle_between(player.transform.x, player.transform.y,
                          boss.transform.x, boss.transform.y)
    # 반대편으로
    angle += math.pi + random.uniform(-0.5, 0.5)
    tp_dist = random.uniform(80, 150)
    new_x = player.transform.x + math.cos(angle) * tp_dist
    new_y = player.transform.y + math.sin(angle) * tp_dist
    # 월드 경계 클램프
    from config import WORLD_WIDTH, WORLD_HEIGHT, BOSS_HITBOX_RADIUS
    new_x = max(BOSS_HITBOX_RADIUS, min(WORLD_WIDTH - BOSS_HITBOX_RADIUS, new_x))
    new_y = max(BOSS_HITBOX_RADIUS, min(WORLD_HEIGHT - BOSS_HITBOX_RADIUS, new_y))
    boss.transform.x = new_x
    boss.transform.y = new_y
    boss.vx = 0.0
    boss.vy = 0.0
    boss.health.invincible_frames = 10
    return []

_register(AttackPattern(
    action_id=ActionID.TELEPORT, name="Teleport",
    cooldown_frames=100, duration_frames=15, damage=0,
    execute=_exec_teleport,
))


def _exec_summon_minions(world: GameWorld, boss: Boss, player: Player):
    count = 2 if boss.phase < 2 else 3
    entities = []
    for i in range(count):
        offset_angle = (2 * math.pi / count) * i + random.uniform(-0.3, 0.3)
        mx = boss.transform.x + math.cos(offset_angle) * 50
        my = boss.transform.y + math.sin(offset_angle) * 50
        entities.append(Minion(mx, my, owner_id=boss.id))
    boss.vx = 0.0
    boss.vy = 0.0
    return entities

_register(AttackPattern(
    action_id=ActionID.SUMMON_MINIONS, name="Summon Minions",
    cooldown_frames=200, duration_frames=60, damage=0,
    min_phase=1,
    execute=_exec_summon_minions,
))


def _exec_shield(world: GameWorld, boss: Boss, player: Player):
    boss.is_shielded = True
    boss.shield_timer = 180  # 3초
    boss.vx = 0.0
    boss.vy = 0.0
    return []

_register(AttackPattern(
    action_id=ActionID.SHIELD, name="Shield",
    cooldown_frames=240, duration_frames=45, damage=0,
    min_phase=1,
    execute=_exec_shield,
))


def _exec_enrage(world: GameWorld, boss: Boss, player: Player):
    boss.enraged = True
    boss.enrage_timer = 300  # 5초
    boss.vx = 0.0
    boss.vy = 0.0
    return []

_register(AttackPattern(
    action_id=ActionID.ENRAGE, name="Enrage",
    cooldown_frames=360, duration_frames=30, damage=0,
    min_phase=2,
    execute=_exec_enrage,
))
