"""게임 상태 총괄 - 렌더링과 독립적인 핵심 로직"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

from config import (
    WORLD_WIDTH, WORLD_HEIGHT, MAX_EPISODE_STEPS,
    PLAYER_ATTACK_RANGE, PLAYER_ATTACK_DAMAGE, PLAYER_DODGE_SPEED,
    PLAYER_DODGE_DURATION, BOSS_HITBOX_RADIUS,
)
from core.entities import (
    Player, Boss, Projectile, AoEZone, Minion, reset_entity_ids,
)
from core.attack_patterns import ATTACK_PATTERNS, ActionID
from core.collision import CollisionSystem, HitEvent
from core.components import Cooldown
from core.physics import direction, distance


@dataclass
class MeleeHitbox:
    """플레이어 근접 공격 히트박스 (1프레임짜리)"""
    x: float
    y: float
    radius: float
    damage: int
    owner_id: int
    hit_entities: set = field(default_factory=set)
    active: bool = True


class GameWorld:
    def __init__(self):
        self.player: Player | None = None
        self.boss: Boss | None = None
        self.projectiles: list[Projectile] = []
        self.aoe_zones: list[AoEZone] = []
        self.minions: list[Minion] = []
        self.player_attack_hitboxes: list[MeleeHitbox] = []
        self.laser_beams: list[dict] = []
        self.pending_effects: list[dict] = []
        self.collision_system = CollisionSystem()
        self.step_count = 0
        self.events: list[dict] = []
        self._last_boss_phase = 0
        self._last_hits: list[HitEvent] = []

        # DDA 배율 (기본값 = 1.0, play_mode에서 조절)
        self.damage_mult: float = 1.0
        self.speed_mult: float = 1.0
        self.cooldown_mult: float = 1.0
        self.proj_speed_mult: float = 1.0

    def reset(self):
        reset_entity_ids()
        self.player = Player(WORLD_WIDTH * 0.25, WORLD_HEIGHT * 0.5)
        self.boss = Boss(WORLD_WIDTH * 0.75, WORLD_HEIGHT * 0.5)
        for action_id, pattern in ATTACK_PATTERNS.items():
            self.boss.action_cooldowns[action_id] = Cooldown(0, pattern.cooldown_frames)
        self.projectiles.clear()
        self.aoe_zones.clear()
        self.minions.clear()
        self.player_attack_hitboxes.clear()
        self.laser_beams.clear()
        self.pending_effects.clear()
        self.step_count = 0
        self.events.clear()
        self._last_boss_phase = 0
        self._last_hits = []

    def apply_player_action(self, move_dir: tuple[float, float],
                            aim_angle: float, attack: bool, dodge: bool):
        p = self.player
        if not p.alive:
            return
        if p.is_dodging:
            return

        if dodge and p.dodge_cooldown.ready and (move_dir[0] != 0 or move_dir[1] != 0):
            p.is_dodging = True
            p.dodge_timer = PLAYER_DODGE_DURATION
            p.dodge_cooldown.trigger()
            mag = max(math.hypot(move_dir[0], move_dir[1]), 0.001)
            p.dodge_dir_x = (move_dir[0] / mag) * PLAYER_DODGE_SPEED
            p.dodge_dir_y = (move_dir[1] / mag) * PLAYER_DODGE_SPEED
        else:
            p.vx = move_dir[0] * p.speed
            p.vy = move_dir[1] * p.speed

        p.aim_angle = aim_angle

        if attack and p.attack_cooldown.ready:
            p.attack_cooldown.trigger()
            atk_x = p.transform.x + math.cos(aim_angle) * PLAYER_ATTACK_RANGE
            atk_y = p.transform.y + math.sin(aim_angle) * PLAYER_ATTACK_RANGE
            self.player_attack_hitboxes.append(MeleeHitbox(
                x=atk_x, y=atk_y, radius=20.0,
                damage=PLAYER_ATTACK_DAMAGE, owner_id=p.id,
            ))

    def apply_boss_action(self, action_id: int) -> bool:
        boss = self.boss
        if not boss.alive or boss.is_acting:
            return False

        pattern = ATTACK_PATTERNS.get(action_id)
        if pattern is None:
            return False

        dist = distance(self.player.transform.x, self.player.transform.y,
                        self.boss.transform.x, self.boss.transform.y)
        if not pattern.is_available(boss, dist):
            return False

        boss.current_action = pattern
        boss.action_timer = pattern.duration_frames
        boss.action_cooldowns[action_id].trigger()
        boss.recent_actions.append(action_id)
        if len(boss.recent_actions) > 10:
            boss.recent_actions.pop(0)

        new_entities = pattern.execute(self, boss, self.player)
        for ent in (new_entities or []):
            if isinstance(ent, Projectile):
                ent.vx *= self.proj_speed_mult
                ent.vy *= self.proj_speed_mult
                ent.hitbox.damage = int(ent.hitbox.damage * self.damage_mult)
                self.projectiles.append(ent)
            elif isinstance(ent, AoEZone):
                ent.hitbox.damage = int(ent.hitbox.damage * self.damage_mult)
                self.aoe_zones.append(ent)
            elif isinstance(ent, Minion):
                self.minions.append(ent)

        # DDA: 보스 속도/데미지 배율
        boss.vx *= self.speed_mult
        boss.vy *= self.speed_mult
        if boss.hitbox.damage > 0:
            boss.hitbox.damage = int(boss.hitbox.damage * self.damage_mult)

        self.events.append({'type': 'boss_action', 'action_id': action_id,
                            'name': pattern.name})
        return True

    def tick(self) -> list[dict]:
        self.events.clear()
        self.step_count += 1

        # 1. 엔티티 업데이트
        self.player.update()
        self.boss.update()
        for p in self.projectiles:
            p.update(target=self.player)
        for z in self.aoe_zones:
            z.update()
        for m in self.minions:
            m.update(target=self.player)

        # 2. 위치 이동
        if not self.player.is_dodging:
            self.player.apply_velocity()
        else:
            self.player.apply_velocity()
        self.player.clamp_to_world()

        if self.boss.is_acting:
            self.boss.apply_velocity()
        self.boss.clamp_to_world()

        # 3. pending_effects 처리
        self._process_pending_effects()

        # 4. 충돌 처리
        self._last_hits = self.collision_system.process(self)
        for hit in self._last_hits:
            self.events.append({
                'type': 'hit',
                'attacker_id': hit.attacker_id,
                'target_type': hit.target_type,
                'damage': hit.damage,
            })

        # 5. 매 프레임 히트박스 클리어
        self.player_attack_hitboxes.clear()

        # 6. 보스 히트박스 초기화 (액션 끝나면)
        if not self.boss.is_acting:
            self.boss.hitbox.damage = 0
            self.boss.hitbox.active = True
            self.boss.hitbox.radius = BOSS_HITBOX_RADIUS

        # 7. 죽은 엔티티 제거
        self.projectiles = [p for p in self.projectiles if p.alive]
        self.aoe_zones = [z for z in self.aoe_zones if z.alive]
        self.minions = [m for m in self.minions if m.alive]
        self.laser_beams = [l for l in self.laser_beams if l['active']]

        # 8. 속도 감쇠 (비행동 시)
        if not self.player.is_dodging:
            self.player.vx = 0.0
            self.player.vy = 0.0

        # 9. 페이즈 전환
        if self.boss.phase > self._last_boss_phase:
            self.events.append({
                'type': 'phase_transition',
                'new_phase': self.boss.phase,
            })
            self._last_boss_phase = self.boss.phase

        return self.events

    def _process_pending_effects(self):
        remaining = []
        for eff in self.pending_effects:
            eff['delay'] -= 1
            if eff['delay'] > 0:
                remaining.append(eff)
                continue

            if eff['type'] == 'ground_slam':
                zone = AoEZone(
                    eff['x'], eff['y'], radius=eff['radius'],
                    damage=eff['damage'], duration=15, damage_interval=15,
                    owner_id=eff['owner_id'],
                )
                self.aoe_zones.append(zone)

            elif eff['type'] == 'melee_hit':
                # 보스 위치에 근접 히트박스 생성
                if self.boss.alive:
                    zone = AoEZone(
                        self.boss.transform.x, self.boss.transform.y,
                        radius=eff['radius'], damage=eff['damage'],
                        duration=5, damage_interval=5,
                        owner_id=eff['owner_id'],
                    )
                    self.aoe_zones.append(zone)

            elif eff['type'] == 'laser':
                angle = eff['angle']
                length = eff['length']
                x1 = self.boss.transform.x if self.boss.alive else eff['x1']
                y1 = self.boss.transform.y if self.boss.alive else eff['y1']
                x2 = x1 + math.cos(angle) * length
                y2 = y1 + math.sin(angle) * length
                self.laser_beams.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'damage': eff['damage'],
                    'duration': eff['fire_duration'],
                    'active': True,
                    'owner_id': eff['owner_id'],
                })

            elif eff['type'] == 'delayed_aoe':
                zone = eff['zone']
                self.aoe_zones.append(zone)

            elif eff['type'] == 'restore_hitbox':
                if self.boss.alive:
                    self.boss.hitbox.radius = BOSS_HITBOX_RADIUS
                    self.boss.hitbox.damage = 0

        self.pending_effects = remaining

        # 레이저 빔 duration 감소
        for laser in self.laser_beams:
            laser['duration'] -= 1
            if laser['duration'] <= 0:
                laser['active'] = False

    @property
    def done(self) -> bool:
        return (self.player.health.current <= 0
                or self.boss.health.current <= 0
                or self.step_count >= MAX_EPISODE_STEPS)

    def get_distance(self) -> float:
        return distance(self.player.transform.x, self.player.transform.y,
                        self.boss.transform.x, self.boss.transform.y)
