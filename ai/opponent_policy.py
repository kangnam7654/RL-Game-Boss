"""학습 시 플레이어 역할을 하는 스크립트 정책"""
from __future__ import annotations
import math
import random

from config import PLAYER_ATTACK_RANGE, PLAYER_SPEED
from core.physics import direction, distance, angle_between


class ScriptedPlayerPolicy:
    def __init__(self, difficulty: str = 'medium'):
        self.difficulty = difficulty
        self.dodge_reaction = {'easy': 30, 'medium': 12, 'hard': 5}[difficulty]
        self.attack_range_mult = {'easy': 1.0, 'medium': 1.3, 'hard': 1.5}[difficulty]
        self.ideal_dist = {'easy': 80, 'medium': 120, 'hard': 150}[difficulty]
        self.strafe_timer = 0
        self.strafe_dir = 1  # 1 or -1

    def get_action(self, world) -> dict:
        p = world.player
        b = world.boss

        if not p.alive or not b.alive:
            return {'move_dir': (0.0, 0.0), 'aim_angle': 0.0,
                    'attack': False, 'dodge': False}

        dist = distance(p.transform.x, p.transform.y,
                        b.transform.x, b.transform.y)
        aim = angle_between(p.transform.x, p.transform.y,
                            b.transform.x, b.transform.y)

        # 이동 결정
        dx, dy = direction(p.transform.x, p.transform.y,
                           b.transform.x, b.transform.y)

        if dist < self.ideal_dist * 0.6:
            # 너무 가까우면 뒤로
            mx, my = -dx, -dy
        elif dist > self.ideal_dist * 1.4:
            # 너무 멀면 접근
            mx, my = dx, dy
        else:
            # 적절한 거리면 옆으로 이동 (스트레이프)
            self.strafe_timer += 1
            if self.strafe_timer > 60:
                self.strafe_timer = 0
                self.strafe_dir *= -1
            mx = -dy * self.strafe_dir
            my = dx * self.strafe_dir

        # 정규화
        mag = max(math.hypot(mx, my), 0.001)
        mx /= mag
        my /= mag

        # 약간의 노이즈
        if self.difficulty != 'hard':
            mx += random.gauss(0, 0.15)
            my += random.gauss(0, 0.15)
            mag = max(math.hypot(mx, my), 0.001)
            mx /= mag
            my /= mag

        # 공격 결정
        atk_range = PLAYER_ATTACK_RANGE * self.attack_range_mult
        attack = dist < atk_range and p.attack_cooldown.ready

        # 회피 결정
        dodge = False
        if b.is_acting and b.current_action:
            action_name = b.current_action.name
            if action_name in ('Charge Dash', 'Explosion', 'Ground Slam',
                               'Spin Attack', 'Shockwave'):
                if dist < 120 and random.random() < 0.7:
                    if self.dodge_reaction <= 0 or random.randint(0, self.dodge_reaction) == 0:
                        dodge = True

        # 투사체 회피 (근처에 투사체가 있으면)
        for proj in world.projectiles:
            if proj.hitbox.owner_id == b.id:
                pd = distance(p.transform.x, p.transform.y,
                              proj.transform.x, proj.transform.y)
                if pd < 50:
                    dodge = True
                    # 투사체 수직 방향으로 회피
                    perp_x = -proj.vy
                    perp_y = proj.vx
                    pmag = max(math.hypot(perp_x, perp_y), 0.001)
                    mx = perp_x / pmag
                    my = perp_y / pmag
                    break

        return {
            'move_dir': (mx, my),
            'aim_angle': aim,
            'attack': attack,
            'dodge': dodge and p.dodge_cooldown.ready,
        }
