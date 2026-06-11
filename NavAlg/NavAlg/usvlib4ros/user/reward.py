import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    reward_arrive_bonus: float = 1000
    reward_collision_penalty: float = -500
    reward_obstacle_penalty: float = -5
    reward_near_target_bonus: float = 1
    reward_weight_distance: float = 0.6
    reward_weight_obstacle: float = 0.2
    reward_weight_heading: float = 0.2
    target_slow_range: float = 3.0
    angular_velocity_max: float = 100
    control_dt: float = 0.1
    has_continuous_action: bool = True
    n_actions: int = 1
    speed_scale: float = 100.0


DEFAULT_REWARD_CONFIG = RewardConfig()


def compute_reward(
    state: list,
    action: int | float | list | tuple,
    max_distance: float,
    degreeAship: float,
    arrive: bool,
    done: bool,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    obstacle_min_range = state[-2]
    current_distance = state[-3]
    heading = state[-4]

    distance_reward = calc_distance_reward(current_distance, max_distance)
    heading_reward = calc_heading_reward(
        action=action,
        heading=heading,
        current_distance=current_distance,
        max_distance=max_distance,
        degreeAship=degreeAship,
        config=config,
    )

    obstacle_reward = 0.0
    if obstacle_min_range < 3:
        obstacle_reward = config.reward_obstacle_penalty
    elif current_distance < config.target_slow_range:
        obstacle_reward = config.reward_near_target_bonus

    reward = (
        distance_reward * config.reward_weight_distance
        + obstacle_reward * config.reward_weight_obstacle
        + heading_reward * config.reward_weight_heading
    )

    if arrive:
        reward += config.reward_arrive_bonus
    elif done:
        reward += config.reward_collision_penalty

    return reward


def _split_action(action: int | float | list | tuple) -> tuple[float, float]:
    """将动作统一拆成(转向, 速度)两维。"""
    if isinstance(action, (list, tuple)):
        if len(action) >= 2:
            return float(action[0]), float(action[1])
        if len(action) == 1:
            return float(action[0]), 0.0
        return 0.0, 0.0
    return float(action), 0.0


def calc_distance_reward(current_distance: float, max_distance: float) -> float:
    if current_distance <= 1 or max_distance <= 0:
        return 0.0
    reward = 1 - (current_distance / max_distance)
    return reward * 2 if reward < 0 else reward * 5


def calc_heading_reward(
    action: int | float | list | tuple,
    heading: float,
    current_distance: float,
    max_distance: float,
    degreeAship: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    distance_rate = 2 ** (current_distance / max_distance) if max_distance > 0 else 1.0
    turn_action, _ = _split_action(action)

    if not config.has_continuous_action:
        yaw_rewards = []
        pi = math.pi
        for i in range(config.n_actions):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * abs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * pi) / pi)[0])
            yaw_rewards.append(tr)
        return round(yaw_rewards[action] * 5, 2) * distance_rate

    angular_speed = turn_action * config.angular_velocity_max
    predicted_heading = (heading + angular_speed * config.control_dt) % 360
    angle_diff = abs(predicted_heading - degreeAship)
    angle_diff = min(angle_diff, 360 - angle_diff)
    heading_reward = 1 - 2 * (angle_diff / 180.0)
    return round(heading_reward, 2)
