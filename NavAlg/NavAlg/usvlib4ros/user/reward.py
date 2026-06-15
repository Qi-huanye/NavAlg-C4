import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    reward_arrive_bonus: float = 1000
    reward_collision_penalty: float = -500
    reward_weight_distance: float = 1.0
    reward_weight_obstacle: float = 1.0
    reward_weight_heading: float = 0.2
    reward_weight_speed: float = 0.2
    progress_scale: float = 10.0
    obstacle_safe_range: float = 3.0
    obstacle_penalty_scale: float = 10.0
    step_penalty: float = -0.01
    max_episode_time: float = 300.0
    min_arrive_time_weight: float = 0.5
    target_slow_range: float = 3.0
    angular_velocity_max: float = 100
    control_dt: float = 0.1
    has_continuous_action: bool = True
    n_actions: int = 1
    speed_scale: float = 100.0
    apf_attractive_gain: float = 1.0
    apf_repulsive_gain: float = 1.0
    apf_obstacle_influence_range: float = 3.0
    apf_heading_repulsive_weight: float = 1.0


DEFAULT_REWARD_CONFIG = RewardConfig()


def compute_reward(
    state: list,
    action: int | float | list | tuple,
    max_distance: float,
    angle_diff: float,
    arrive: bool,
    done: bool,
    prev_distance: float | None = None,
    episode_elapsed_time: float | None = None,
    heading_world: float | None = None,
    target_heading_world: float | None = None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    obstacle_min_range = state[-2]
    current_distance = state[-3]
    distance_reward = calc_distance_reward(
        current_distance=current_distance,
        prev_distance=prev_distance,
        max_distance=max_distance,
        config=config,
    )
    heading_reward = calc_heading_reward(
        action=action,
        angle_diff=angle_diff if angle_diff is not None else state[-4],
        current_distance=current_distance,
        max_distance=max_distance,
        obstacle_min_range=obstacle_min_range,
        obstacle_angle=state[-1],
        heading_world=heading_world,
        target_heading_world=target_heading_world,
        config=config,
    )
    obstacle_reward = calc_obstacle_reward(obstacle_min_range, config)
    speed_reward = calc_speed_reward(
        action=action,
        angle_diff=angle_diff if angle_diff is not None else state[-4],
        obstacle_min_range=obstacle_min_range,
        config=config,
    )

    reward = (
        distance_reward * config.reward_weight_distance
        + obstacle_reward * config.reward_weight_obstacle
        + heading_reward * config.reward_weight_heading
        + speed_reward * config.reward_weight_speed
        + config.step_penalty
    )

    if arrive:
        time_reward_weight = calc_time_reward_weight(episode_elapsed_time, config)
        reward += config.reward_arrive_bonus * time_reward_weight
    elif done:
        reward += config.reward_collision_penalty

    return reward


def _split_action(action: int | float | list | tuple) -> tuple[float, float]:
    """将动作统一拆成(转向, 速度)两维。"""
    if hasattr(action, "tolist"):
        action = action.tolist()
    if isinstance(action, (list, tuple)):
        flat = []
        for item in action:
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                flat.extend(float(x) for x in item)
            else:
                flat.append(float(item))
        if len(flat) >= 2:
            return float(flat[0]), float(flat[1])
        if len(flat) == 1:
            return float(flat[0]), 0.0
        return 0.0, 0.0
    return float(action), 0.0


def calc_distance_reward(
    current_distance: float,
    prev_distance: float | None,
    max_distance: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    if current_distance <= 1:
        return 0.0

    if prev_distance is not None:
        return (prev_distance - current_distance) * config.progress_scale

    if max_distance <= 0:
        return 0.0

    reward = 1 - (current_distance / max_distance)
    return reward * 2 if reward < 0 else reward * 5


def calc_obstacle_reward(
    obstacle_min_range: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    if obstacle_min_range >= config.obstacle_safe_range:
        return 0.0

    danger_ratio = (config.obstacle_safe_range - obstacle_min_range) / config.obstacle_safe_range
    return -(danger_ratio ** 2) * config.obstacle_penalty_scale


def calc_speed_reward(
    action: int | float | list | tuple,
    angle_diff: float,
    obstacle_min_range: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    _, speed_action = _split_action(action)
    normalized_speed = max(-1.0, min(1.0, speed_action))

    if abs(angle_diff) < 30 and obstacle_min_range >= config.obstacle_safe_range:
        return max(0.0, normalized_speed)

    return -abs(normalized_speed)


def calc_time_reward_weight(
    episode_elapsed_time: float | None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    if episode_elapsed_time is None:
        return 1.0

    if config.max_episode_time <= 0:
        return 1.0

    progress = episode_elapsed_time / config.max_episode_time
    progress = max(0.0, min(1.0, progress))

    return config.min_arrive_time_weight + (
        1.0 - config.min_arrive_time_weight
    ) * ((1.0 - progress) ** 2)


def calc_heading_reward(
    action: int | float | list | tuple,
    angle_diff: float,
    current_distance: float,
    max_distance: float,
    obstacle_min_range: float | None = None,
    obstacle_angle: float | None = None,
    heading_world: float | None = None,
    target_heading_world: float | None = None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    distance_rate = 2 ** (current_distance / max_distance) if max_distance > 0 else 1.0
    turn_action, _ = _split_action(action)

    if not config.has_continuous_action:
        yaw_rewards = []
        pi = math.pi
        for i in range(config.n_actions):
            angle = -pi / 4 + angle_diff + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * abs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * pi) / pi)[0])
            yaw_rewards.append(tr)
        return round(yaw_rewards[action] * 5, 2) * distance_rate

    if obstacle_min_range is not None and obstacle_angle is not None:
        apf_heading_diff = calc_apf_heading_diff(
            angle_diff=angle_diff,
            current_distance=current_distance,
            obstacle_min_range=obstacle_min_range,
            obstacle_angle=obstacle_angle,
            heading_world=heading_world,
            target_heading_world=target_heading_world,
            config=config,
        )
    else:
        apf_heading_diff = angle_diff

    predicted_angle_diff = _normalize_signed_angle_diff(
        apf_heading_diff - turn_action * config.angular_velocity_max * config.control_dt
    )
    heading_reward = 1 - 2 * (abs(predicted_angle_diff) / 180.0)
    return round(heading_reward, 2) * distance_rate


def calc_apf_heading_diff(
    angle_diff: float,
    current_distance: float,
    obstacle_min_range: float,
    obstacle_angle: float,
    heading_world: float | None,
    target_heading_world: float | None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    heading_world = 0.0 if heading_world is None else heading_world
    if target_heading_world is None:
        target_heading_world = heading_world + angle_diff

    target_rad = math.radians(target_heading_world)
    attractive_strength = config.apf_attractive_gain * max(float(current_distance), 0.0)
    attractive_x = attractive_strength * math.cos(target_rad)
    attractive_y = attractive_strength * math.sin(target_rad)

    repulsive_x = 0.0
    repulsive_y = 0.0
    rho = max(float(obstacle_min_range), 1e-3)
    rho_0 = config.apf_obstacle_influence_range
    if rho < rho_0:
        force_magnitude = config.apf_repulsive_gain * (1.0 / rho - 1.0 / rho_0) / (rho * rho)
        obstacle_relative_deg = float(obstacle_angle) * 2.0 - 90.0
        obstacle_world_deg = heading_world + obstacle_relative_deg
        obstacle_rad = math.radians(obstacle_world_deg)
        repulsive_strength = force_magnitude * config.apf_heading_repulsive_weight
        repulsive_x = -repulsive_strength * math.cos(obstacle_rad)
        repulsive_y = -repulsive_strength * math.sin(obstacle_rad)

    apf_x = attractive_x + repulsive_x
    apf_y = attractive_y + repulsive_y
    if abs(apf_x) < 1e-6 and abs(apf_y) < 1e-6:
        return _normalize_signed_angle_diff(angle_diff)

    apf_heading_world = math.degrees(math.atan2(apf_y, apf_x))
    return _normalize_signed_angle_diff(apf_heading_world - heading_world)


def _normalize_signed_angle_diff(angle: float) -> float:
    return (angle + 180) % 360 - 180
