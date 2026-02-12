"""플레이어 보상 함수"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PlayerRewardAccumulator:
    damage_to_boss: float = 0.0
    damage_from_boss: float = 0.0
    boss_killed: bool = False
    player_died: bool = False
    dodged_attack: bool = False    # 회피 후 무사한 경우
    distance_to_boss: float = 0.0


def compute_player_reward(acc: PlayerRewardAccumulator) -> float:
    reward = 0.0

    # 전투 보상
    reward += acc.damage_to_boss * 0.02
    reward -= acc.damage_from_boss * 0.015

    # 클리어 / 사망
    if acc.boss_killed:
        reward += 3.0
    if acc.player_died:
        reward -= 3.0

    # 생존 보상
    reward += 0.01

    # 회피 성공 보상
    if acc.dodged_attack:
        reward += 0.3

    # 너무 멀리 도망치지 않도록
    if acc.distance_to_boss > 400:
        reward -= 0.005

    return reward
