"""보상 함수 설계 - 엔터테인먼트 지향

설계 원칙:
  보스의 목표는 "플레이어를 죽이는 것"이 아니라
  "긴장감 있는 접전을 만드는 것"이다.

보상 구조:
  1. 전투 보상 (기존, 약화) - 타격/피격
  2. 긴장감 유지 - 플레이어 HP가 스위트스팟(30~70%)에 있을 때 보너스
  3. 패턴 다양성 - 최근 행동의 엔트로피가 높을수록 보상
  4. 페이싱 리듬 - 공격 후 빈틈(이동/유틸)을 주면 보상
  5. 접전 보상 - 양쪽 모두 데미지를 주고받으면 보상
"""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass, field

from config import NUM_BOSS_ACTIONS


@dataclass
class RewardAccumulator:
    damage_dealt: float = 0.0      # 보스 → 플레이어
    damage_taken: float = 0.0      # 플레이어 → 보스
    phase_transition: bool = False
    action_executed: bool = False
    player_killed: bool = False
    boss_died: bool = False
    distance_to_player: float = 0.0
    player_hp_ratio: float = 1.0
    boss_hp_ratio: float = 1.0
    action_id: int = -1            # 이번에 선택한 액션


# 공격 패턴 vs 비공격(이동/유틸) 구분
_AGGRESSIVE_ACTIONS = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}  # 근접+원거리+범위
_BREATHING_ACTIONS = {0, 1, 12, 14}  # 이동, 텔레포트, 실드


def _entropy(actions: list[int], n_actions: int = NUM_BOSS_ACTIONS) -> float:
    """최근 행동 리스트의 정규화된 Shannon 엔트로피 (0~1)"""
    if len(actions) < 2:
        return 0.0
    counts = Counter(actions)
    total = len(actions)
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    # 최대 엔트로피로 정규화
    max_ent = math.log2(min(len(counts), n_actions))
    if max_ent <= 0:
        return 0.0
    return ent / max_ent


def compute_reward(acc: RewardAccumulator, recent_actions: list[int]) -> float:
    reward = 0.0

    # ──────────────────────────────────────
    # 1. 전투 보상 (약화됨 - 순수 전투력보다 재미 우선)
    # ──────────────────────────────────────
    reward += acc.damage_dealt * 0.01       # 기존 0.02 → 0.01
    reward -= acc.damage_taken * 0.01       # 기존 0.015 → 0.01

    # 승리/패배 (대폭 감소 - 킬 자체보다 과정이 중요)
    if acc.player_killed:
        reward += 1.0   # 기존 5.0 → 1.0
    if acc.boss_died:
        reward -= 1.0   # 기존 5.0 → 1.0

    # ──────────────────────────────────────
    # 2. 긴장감 유지 보상 (Tension Sweet Spot)
    #    플레이어 HP가 30~70%일 때 = "아슬아슬한 전투"
    # ──────────────────────────────────────
    hp = acc.player_hp_ratio
    if 0.3 <= hp <= 0.7:
        # 스위트스팟 중앙(50%)에 가까울수록 높은 보상
        tension = 1.0 - abs(hp - 0.5) / 0.2  # 0~1 범위
        reward += tension * 0.3
    elif hp > 0.9:
        # 플레이어가 너무 안전하면 약간의 페널티 (공격 유도)
        reward -= 0.05
    elif hp < 0.15:
        # 플레이어가 거의 죽으면 페널티 (일방적 학살 방지)
        reward -= 0.15

    # ──────────────────────────────────────
    # 3. 패턴 다양성 보상 (Entropy)
    #    최근 8개 행동의 엔트로피가 높을수록 보상
    # ──────────────────────────────────────
    if len(recent_actions) >= 4:
        last8 = recent_actions[-8:]
        ent = _entropy(last8)
        reward += ent * 0.2  # 최대 +0.2/step

        # 강한 반복 페널티 (같은 액션 4연속)
        last4 = recent_actions[-4:]
        most_common = Counter(last4).most_common(1)[0][1]
        if most_common >= 4:
            reward -= 1.0
        elif most_common >= 3:
            reward -= 0.4

    # ──────────────────────────────────────
    # 4. 페이싱 리듬 보상 (Attack Window)
    #    공격 후 비공격(이동/유틸)을 선택하면 보상
    #    → 유저에게 반격 기회를 주는 행동을 학습
    # ──────────────────────────────────────
    if len(recent_actions) >= 2 and acc.action_executed:
        prev = recent_actions[-2] if len(recent_actions) >= 2 else -1
        curr = acc.action_id

        # 공격 → 비공격 전환 (빈틈 제공)
        if prev in _AGGRESSIVE_ACTIONS and curr in _BREATHING_ACTIONS:
            reward += 0.15

        # 비공격 → 공격 전환 (빈틈 후 압박)
        if prev in _BREATHING_ACTIONS and curr in _AGGRESSIVE_ACTIONS:
            reward += 0.1

        # 연속 공격은 약한 페널티 (숨 쉴 틈 없음)
        if prev in _AGGRESSIVE_ACTIONS and curr in _AGGRESSIVE_ACTIONS:
            reward -= 0.05

    # ──────────────────────────────────────
    # 5. 접전 보상 (양쪽이 주고받을 때)
    #    보스가 데미지를 주면서 동시에 받기도 하면 보상
    # ──────────────────────────────────────
    if acc.damage_dealt > 0 and acc.damage_taken > 0:
        reward += 0.2  # 상호 교전 보너스

    # ──────────────────────────────────────
    # 6. 기존 유지 (페이즈, 거리 등)
    # ──────────────────────────────────────
    if acc.phase_transition:
        reward += 0.5

    if not acc.action_executed:
        reward -= 0.1

    if acc.distance_to_player > 500:
        reward -= 0.01

    return reward
