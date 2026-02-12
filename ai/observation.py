"""관측 공간 구성 및 정규화"""
import math
import numpy as np

from config import (
    WORLD_WIDTH, WORLD_HEIGHT, OBSERVATION_DIM, NUM_BOSS_ACTIONS,
    PLAYER_SPEED,
)
from core.attack_patterns import ActionID


# 주요 공격 쿨다운 추적 대상 (5개)
KEY_ACTIONS = [
    ActionID.CHARGE_DASH,
    ActionID.PROJECTILE_BARRAGE,
    ActionID.LASER_BEAM,
    ActionID.GROUND_SLAM,
    ActionID.HOMING_MISSILES,
]

DIAG = math.sqrt(WORLD_WIDTH ** 2 + WORLD_HEIGHT ** 2)
MAX_DURATION = 70.0  # 최대 action duration (레이저)


def build_observation(world) -> np.ndarray:
    obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    p = world.player
    b = world.boss
    i = 0

    # 플레이어 상태 (7)
    obs[i] = p.transform.x / WORLD_WIDTH; i += 1
    obs[i] = p.transform.y / WORLD_HEIGHT; i += 1
    obs[i] = p.health.ratio; i += 1
    obs[i] = float(p.is_dodging); i += 1
    obs[i] = p.dodge_cooldown.ratio; i += 1
    obs[i] = np.clip(p.vx / PLAYER_SPEED, -1.0, 1.0) * 0.5 + 0.5; i += 1
    obs[i] = np.clip(p.vy / PLAYER_SPEED, -1.0, 1.0) * 0.5 + 0.5; i += 1

    # 보스 상태 (7)
    obs[i] = b.transform.x / WORLD_WIDTH; i += 1
    obs[i] = b.transform.y / WORLD_HEIGHT; i += 1
    obs[i] = b.health.ratio; i += 1
    obs[i] = b.phase / 3.0; i += 1
    obs[i] = float(b.is_acting); i += 1
    obs[i] = b.action_timer / MAX_DURATION; i += 1
    obs[i] = float(b.is_shielded); i += 1

    # 관계 (4)
    dx = p.transform.x - b.transform.x
    dy = p.transform.y - b.transform.y
    dist = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    obs[i] = min(dist / DIAG, 1.0); i += 1
    obs[i] = math.sin(angle) * 0.5 + 0.5; i += 1
    obs[i] = math.cos(angle) * 0.5 + 0.5; i += 1
    aim_diff = abs(angle - p.aim_angle)
    aim_diff = min(aim_diff, 2 * math.pi - aim_diff)
    obs[i] = 1.0 - aim_diff / math.pi; i += 1

    # 주요 쿨다운 (5)
    for aid in KEY_ACTIONS:
        cd = b.action_cooldowns.get(aid)
        obs[i] = cd.ratio if cd else 0.0; i += 1

    # 최근 행동 (4)
    recent = b.recent_actions[-4:]
    pad = [0] * (4 - len(recent))
    for a in pad + recent:
        obs[i] = a / NUM_BOSS_ACTIONS; i += 1

    # 여유 (1) - enraged 상태
    obs[i] = float(b.enraged); i += 1

    return np.clip(obs, 0.0, 1.0)
