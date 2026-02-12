"""충돌 감지 시스템"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.entities import Entity

from config import PLAYER_INVINCIBLE_FRAMES


def circles_collide(x1: float, y1: float, r1: float,
                    x2: float, y2: float, r2: float) -> bool:
    dx = x2 - x1
    dy = y2 - y1
    return dx * dx + dy * dy <= (r1 + r2) ** 2


def line_circle_intersect(lx1: float, ly1: float, lx2: float, ly2: float,
                          cx: float, cy: float, r: float) -> bool:
    """선분-원 충돌 (레이저 빔용)"""
    dx = lx2 - lx1
    dy = ly2 - ly1
    fx = lx1 - cx
    fy = ly1 - cy
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False
    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 and t2 > 1)


class HitEvent:
    __slots__ = ('attacker_id', 'target_id', 'damage', 'target_type')

    def __init__(self, attacker_id: int, target_id: int, damage: int, target_type: str):
        self.attacker_id = attacker_id
        self.target_id = target_id
        self.damage = damage
        self.target_type = target_type  # 'player' or 'boss' or 'minion'


class CollisionSystem:
    def process(self, world) -> list[HitEvent]:
        hits = []
        player = world.player
        boss = world.boss

        # 1. 투사체 vs 플레이어/보스
        for proj in world.projectiles:
            if not proj.alive or not proj.hitbox.active:
                continue
            # 보스의 투사체 -> 플레이어
            if proj.hitbox.owner_id == boss.id:
                if (not player.is_dodging
                        and player.health.invincible_frames <= 0
                        and player.id not in proj.hitbox.hit_entities
                        and circles_collide(
                            proj.transform.x, proj.transform.y, proj.hitbox.radius,
                            player.transform.x, player.transform.y, player.hitbox.radius)):
                    hits.append(HitEvent(proj.hitbox.owner_id, player.id,
                                         proj.hitbox.damage, 'player'))
                    proj.hitbox.hit_entities.add(player.id)
                    proj.alive = False
            # 플레이어의 투사체 -> 보스
            elif proj.hitbox.owner_id == player.id:
                if (boss.id not in proj.hitbox.hit_entities
                        and circles_collide(
                            proj.transform.x, proj.transform.y, proj.hitbox.radius,
                            boss.transform.x, boss.transform.y, boss.hitbox.radius)):
                    dmg = proj.hitbox.damage
                    if boss.is_shielded:
                        dmg = dmg // 3
                    hits.append(HitEvent(proj.hitbox.owner_id, boss.id, dmg, 'boss'))
                    proj.hitbox.hit_entities.add(boss.id)
                    proj.alive = False

        # 2. AoE 존 vs 플레이어
        for zone in world.aoe_zones:
            if not zone.alive or not zone.can_damage:
                continue
            if (not player.is_dodging
                    and player.health.invincible_frames <= 0
                    and circles_collide(
                        zone.transform.x, zone.transform.y, zone.hitbox.radius,
                        player.transform.x, player.transform.y, player.hitbox.radius)):
                hits.append(HitEvent(zone.hitbox.owner_id, player.id,
                                     zone.hitbox.damage, 'player'))

        # 3. 미니언 vs 플레이어
        for minion in world.minions:
            if not minion.alive or not minion.attack_cooldown.ready:
                continue
            if (not player.is_dodging
                    and player.health.invincible_frames <= 0
                    and circles_collide(
                        minion.transform.x, minion.transform.y, minion.hitbox.radius + 5,
                        player.transform.x, player.transform.y, player.hitbox.radius)):
                hits.append(HitEvent(minion.id, player.id,
                                     minion.hitbox.damage, 'player'))
                minion.attack_cooldown.trigger()

        # 4. 보스 자체 히트박스 vs 플레이어 (돌진/회전 공격 등)
        if (boss.is_acting and boss.current_action
                and boss.hitbox.active and boss.hitbox.damage > 0
                and not player.is_dodging
                and player.health.invincible_frames <= 0):
            if circles_collide(
                    boss.transform.x, boss.transform.y, boss.hitbox.radius,
                    player.transform.x, player.transform.y, player.hitbox.radius):
                if player.id not in boss.hitbox.hit_entities:
                    hits.append(HitEvent(boss.id, player.id,
                                         boss.hitbox.damage, 'player'))
                    boss.hitbox.hit_entities.add(player.id)

        # 5. 플레이어 근접 공격 히트박스 처리
        for atk_hb in world.player_attack_hitboxes:
            if not atk_hb.active:
                continue
            if (boss.id not in atk_hb.hit_entities
                    and circles_collide(
                        atk_hb.x, atk_hb.y, atk_hb.radius,
                        boss.transform.x, boss.transform.y, boss.hitbox.radius)):
                dmg = atk_hb.damage
                if boss.is_shielded:
                    dmg = dmg // 3
                hits.append(HitEvent(player.id, boss.id, dmg, 'boss'))
                atk_hb.hit_entities.add(boss.id)

        # 레이저 빔 충돌
        for laser in world.laser_beams:
            if not laser['active']:
                continue
            if (not player.is_dodging
                    and player.health.invincible_frames <= 0
                    and line_circle_intersect(
                        laser['x1'], laser['y1'], laser['x2'], laser['y2'],
                        player.transform.x, player.transform.y, player.hitbox.radius)):
                hits.append(HitEvent(boss.id, player.id, laser['damage'], 'player'))

        # 피격 적용
        for hit in hits:
            if hit.target_type == 'player':
                actual = player.health.take_damage(hit.damage)
                if actual > 0:
                    player.health.invincible_frames = PLAYER_INVINCIBLE_FRAMES
            elif hit.target_type == 'boss':
                boss.health.take_damage(hit.damage)

        return hits
