"""물리 헬퍼 함수"""
import math


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def direction(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    """(x1,y1)에서 (x2,y2) 방향의 단위벡터"""
    dx = x2 - x1
    dy = y2 - y1
    dist = max(math.hypot(dx, dy), 0.001)
    return dx / dist, dy / dist


def normalize(vx: float, vy: float) -> tuple[float, float]:
    mag = max(math.hypot(vx, vy), 0.001)
    return vx / mag, vy / mag


def angle_between(x1: float, y1: float, x2: float, y2: float) -> float:
    """(x1,y1)에서 (x2,y2)로의 각도 (라디안)"""
    return math.atan2(y2 - y1, x2 - x1)


def spread_directions(center_angle: float, spread: float, count: int) -> list[float]:
    """중심 각도를 기준으로 부채꼴로 퍼진 각도 리스트"""
    if count == 1:
        return [center_angle]
    angles = []
    start = center_angle - spread / 2
    step = spread / (count - 1)
    for i in range(count):
        angles.append(start + step * i)
    return angles


def knockback(entity, from_x: float, from_y: float, force: float):
    """entity를 (from_x, from_y)에서 밀어냄"""
    dx, dy = direction(from_x, from_y, entity.transform.x, entity.transform.y)
    entity.vx += dx * force
    entity.vy += dy * force
