"""Dynamic Difficulty Adjustment (DDA)

플레이 시 유저 퍼포먼스를 추적하고 보스 난이도를 실시간 조절.
학습된 모델의 행동 자체는 바꾸지 않고, 데미지/속도 배율만 조절한다.

핵심 원리:
  - 유저가 잘하면 (회피 많이, HP 높음) → 보스 강화
  - 유저가 힘들어하면 (HP 낮음, 피격 많음) → 보스 약화
  - 목표: 유저 HP가 40~60% 근처에서 오래 머물도록
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque


@dataclass
class DifficultyState:
    """보스에 적용할 배율"""
    damage_mult: float = 1.0      # 보스 공격력 배율
    speed_mult: float = 1.0       # 보스 이동/돌진 속도 배율
    cooldown_mult: float = 1.0    # 쿨다운 배율 (>1이면 더 오래 대기)
    proj_speed_mult: float = 1.0  # 투사체 속도 배율


class DifficultyManager:
    """유저 퍼포먼스를 추적하고 난이도를 조절"""

    def __init__(self, target_hp_ratio: float = 0.5,
                 adjustment_speed: float = 0.002):
        self.target_hp = target_hp_ratio
        self.speed = adjustment_speed

        # 내부 난이도 점수 (0.0 = 최소, 1.0 = 최대)
        self._difficulty = 0.5
        self._min_difficulty = 0.2
        self._max_difficulty = 1.0

        # 추적 데이터
        self._hp_history: deque[float] = deque(maxlen=300)  # 최근 5초 (60fps)
        self._hits_taken: deque[bool] = deque(maxlen=60)    # 최근 1초 피격 여부
        self._dodges: deque[bool] = deque(maxlen=60)        # 최근 회피 성공

        self.state = DifficultyState()

    def update(self, player_hp_ratio: float, player_was_hit: bool,
               player_dodged: bool):
        """매 프레임 호출"""
        self._hp_history.append(player_hp_ratio)
        self._hits_taken.append(player_was_hit)
        self._dodges.append(player_dodged)

        # 평균 HP 비율 계산
        if len(self._hp_history) < 30:
            return  # 데이터 부족

        avg_hp = sum(self._hp_history) / len(self._hp_history)

        # 목표 HP와의 차이로 난이도 조절
        error = avg_hp - self.target_hp
        # error > 0: 유저 HP 높음 → 보스 강화 (difficulty 올림)
        # error < 0: 유저 HP 낮음 → 보스 약화 (difficulty 내림)
        self._difficulty += error * self.speed

        # 급격한 위험 감지: 최근에 많이 맞았으면 빠르게 약화
        recent_hit_rate = sum(self._hits_taken) / max(len(self._hits_taken), 1)
        if recent_hit_rate > 0.3:  # 30% 이상 프레임에서 피격
            self._difficulty -= 0.01

        # 범위 제한
        self._difficulty = max(self._min_difficulty,
                               min(self._max_difficulty, self._difficulty))

        # 난이도 → 배율 변환
        self._update_state()

    def _update_state(self):
        d = self._difficulty

        # damage_mult: 0.5 ~ 1.3
        self.state.damage_mult = 0.5 + d * 0.8

        # speed_mult: 0.7 ~ 1.2
        self.state.speed_mult = 0.7 + d * 0.5

        # cooldown_mult: 1.5 (쉬움) ~ 0.8 (어려움)
        # 높을수록 쿨다운이 길어져서 공격 빈도 낮아짐
        self.state.cooldown_mult = 1.5 - d * 0.7

        # proj_speed_mult: 0.6 ~ 1.2
        self.state.proj_speed_mult = 0.6 + d * 0.6

    def reset(self):
        self._difficulty = 0.5
        self._hp_history.clear()
        self._hits_taken.clear()
        self._dodges.clear()
        self._update_state()

    @property
    def difficulty_percent(self) -> int:
        """현재 난이도를 0~100%로 표시"""
        normalized = (self._difficulty - self._min_difficulty) / \
                     (self._max_difficulty - self._min_difficulty)
        return int(normalized * 100)
