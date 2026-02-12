"""게임 컴포넌트 데이터 클래스"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Transform:
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0  # 라디안


@dataclass
class Health:
    current: int
    maximum: int
    invincible_frames: int = 0

    @property
    def ratio(self) -> float:
        return self.current / max(self.maximum, 1)

    def take_damage(self, amount: int) -> int:
        """데미지 적용. 실제 적용된 데미지 반환."""
        if self.invincible_frames > 0:
            return 0
        actual = min(amount, self.current)
        self.current -= actual
        return actual


@dataclass
class Hitbox:
    radius: float
    active: bool = True
    damage: int = 0
    owner_id: int = -1
    lifetime: int = -1  # -1이면 영구
    hit_entities: set = field(default_factory=set)

    def tick(self):
        if self.lifetime > 0:
            self.lifetime -= 1
        if self.lifetime == 0:
            self.active = False


@dataclass
class Cooldown:
    current: int = 0
    maximum: int = 0

    @property
    def ready(self) -> bool:
        return self.current <= 0

    @property
    def ratio(self) -> float:
        if self.maximum <= 0:
            return 0.0
        return self.current / self.maximum

    def trigger(self):
        self.current = self.maximum

    def tick(self):
        if self.current > 0:
            self.current -= 1
