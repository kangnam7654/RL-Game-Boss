"""액션 마스킹"""
import numpy as np

from config import NUM_BOSS_ACTIONS
from core.attack_patterns import ATTACK_PATTERNS, ActionID
from core.physics import distance


def get_action_mask(world) -> np.ndarray:
    mask = np.ones(NUM_BOSS_ACTIONS, dtype=np.float32)
    boss = world.boss
    player = world.player

    if boss.is_acting:
        mask[:] = 0
        mask[ActionID.MOVE_TOWARD] = 1
        return mask

    dist = distance(player.transform.x, player.transform.y,
                    boss.transform.x, boss.transform.y)

    for action_id, pattern in ATTACK_PATTERNS.items():
        if not pattern.is_available(boss, dist):
            mask[action_id] = 0

    return mask
