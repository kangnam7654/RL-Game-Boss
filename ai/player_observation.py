"""플레이어 관점 관측 공간 (32차원)"""
import math
import numpy as np

from config import (
    WORLD_WIDTH, WORLD_HEIGHT, PLAYER_OBS_DIM, PLAYER_SPEED,
    NUM_BOSS_ACTIONS,
)

DIAG = math.sqrt(WORLD_WIDTH ** 2 + WORLD_HEIGHT ** 2)
MAX_DURATION = 70.0


def build_player_observation(world) -> np.ndarray:
    obs = np.zeros(PLAYER_OBS_DIM, dtype=np.float32)
    p = world.player
    b = world.boss
    i = 0

    # 자신 상태 (7)
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
    dx = b.transform.x - p.transform.x
    dy = b.transform.y - p.transform.y
    dist = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx)
    obs[i] = min(dist / DIAG, 1.0); i += 1
    obs[i] = math.sin(angle) * 0.5 + 0.5; i += 1
    obs[i] = math.cos(angle) * 0.5 + 0.5; i += 1
    # 보스 현재 액션 ID
    boss_action_id = 0
    if b.current_action is not None:
        boss_action_id = b.current_action.action_id
    obs[i] = boss_action_id / NUM_BOSS_ACTIONS; i += 1

    # 가장 가까운 투사체 2개 (6)
    threats = []
    for proj in world.projectiles:
        if proj.hitbox.owner_id == b.id:
            pdx = proj.transform.x - p.transform.x
            pdy = proj.transform.y - p.transform.y
            pdist = math.sqrt(pdx * pdx + pdy * pdy)
            threats.append((pdist, pdx, pdy, proj.vx, proj.vy))
    threats.sort(key=lambda t: t[0])

    for j in range(2):
        if j < len(threats):
            _, tdx, tdy, tvx, tvy = threats[j]
            obs[i] = np.clip(tdx / DIAG, -1, 1) * 0.5 + 0.5; i += 1
            obs[i] = np.clip(tdy / DIAG, -1, 1) * 0.5 + 0.5; i += 1
            obs[i] = np.clip(tvx / 10.0, -1, 1) * 0.5 + 0.5; i += 1
        else:
            obs[i] = 0.5; i += 1  # 없으면 중립값
            obs[i] = 0.5; i += 1
            obs[i] = 0.5; i += 1

    # 가장 가까운 AoE + 미니언 수 (4)
    closest_aoe_dx, closest_aoe_dy, closest_aoe_r = 0.0, 0.0, 0.0
    min_aoe_dist = float('inf')
    for zone in world.aoe_zones:
        zdx = zone.transform.x - p.transform.x
        zdy = zone.transform.y - p.transform.y
        zdist = math.sqrt(zdx * zdx + zdy * zdy)
        if zdist < min_aoe_dist:
            min_aoe_dist = zdist
            closest_aoe_dx = zdx
            closest_aoe_dy = zdy
            closest_aoe_r = zone.hitbox.radius

    obs[i] = np.clip(closest_aoe_dx / DIAG, -1, 1) * 0.5 + 0.5; i += 1
    obs[i] = np.clip(closest_aoe_dy / DIAG, -1, 1) * 0.5 + 0.5; i += 1
    obs[i] = closest_aoe_r / 100.0; i += 1
    obs[i] = min(len(world.minions) / 5.0, 1.0); i += 1

    # 추가 상태 (4)
    obs[i] = p.attack_cooldown.ratio; i += 1
    obs[i] = float(p.health.invincible_frames > 0); i += 1
    obs[i] = float(b.enraged); i += 1
    obs[i] = min(world.step_count / 3000.0, 1.0); i += 1

    return np.clip(obs, 0.0, 1.0)
