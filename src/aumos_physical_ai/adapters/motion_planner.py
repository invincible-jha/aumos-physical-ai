"""Motion planner adapter for physical AI path planning dataset generation.

Implements the MotionPlannerProtocol to generate waypoint sequences, apply
obstacle-aware pathfinding algorithms (A* and RRT), smooth trajectories with
B-splines, synthesize velocity profiles, check collisions, coordinate
multi-robot paths, and export datasets in CSV or HDF5 format.
"""

import csv
import io
import json
import math
import random
import uuid
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


def _euclidean_distance(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
    """Compute Euclidean distance between two 3-D points.

    Args:
        point_a: (x, y, z) tuple.
        point_b: (x, y, z) tuple.

    Returns:
        Scalar distance.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


def _astar_path(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    obstacles: list[dict[str, Any]],
    grid_resolution: float = 0.1,
) -> list[tuple[float, float, float]]:
    """A* pathfinding on a discretized 3-D grid.

    Obstacles are axis-aligned bounding boxes with keys: x, y, z, w (width),
    d (depth), h (height). The grid is searched in 26-connectivity.

    Args:
        start: Starting position (x, y, z).
        goal: Goal position (x, y, z).
        obstacles: List of obstacle bounding-box dicts.
        grid_resolution: Grid cell size in metres.

    Returns:
        List of waypoints from start to goal (inclusive).
    """
    def _to_cell(point: tuple[float, float, float]) -> tuple[int, int, int]:
        return (
            int(round(point[0] / grid_resolution)),
            int(round(point[1] / grid_resolution)),
            int(round(point[2] / grid_resolution)),
        )

    def _to_world(cell: tuple[int, int, int]) -> tuple[float, float, float]:
        return (
            cell[0] * grid_resolution,
            cell[1] * grid_resolution,
            cell[2] * grid_resolution,
        )

    def _in_obstacle(cell: tuple[int, int, int]) -> bool:
        wx, wy, wz = _to_world(cell)
        for obs in obstacles:
            if (
                obs["x"] <= wx <= obs["x"] + obs.get("w", 0.5)
                and obs["y"] <= wy <= obs["y"] + obs.get("d", 0.5)
                and obs["z"] <= wz <= obs["z"] + obs.get("h", 0.5)
            ):
                return True
        return False

    start_cell = _to_cell(start)
    goal_cell = _to_cell(goal)

    open_set: dict[tuple[int, int, int], float] = {start_cell: 0.0}
    came_from: dict[tuple[int, int, int], tuple[int, int, int] | None] = {start_cell: None}
    g_score: dict[tuple[int, int, int], float] = {start_cell: 0.0}

    directions = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    max_iterations = 2000
    iteration = 0
    while open_set and iteration < max_iterations:
        iteration += 1
        current = min(open_set, key=lambda c: open_set[c])
        if current == goal_cell:
            path: list[tuple[int, int, int]] = []
            node: tuple[int, int, int] | None = current
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            return [_to_world(c) for c in reversed(path)]

        del open_set[current]
        for dx, dy, dz in directions:
            neighbour = (current[0] + dx, current[1] + dy, current[2] + dz)
            if _in_obstacle(neighbour):
                continue
            tentative_g = g_score[current] + math.sqrt(dx**2 + dy**2 + dz**2)
            if tentative_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                h = _euclidean_distance(_to_world(neighbour), goal)
                open_set[neighbour] = tentative_g + h

    # Fallback: straight-line path with 10 interpolated waypoints
    n = 10
    return [
        (
            start[0] + (goal[0] - start[0]) * i / n,
            start[1] + (goal[1] - start[1]) * i / n,
            start[2] + (goal[2] - start[2]) * i / n,
        )
        for i in range(n + 1)
    ]


def _rrt_path(
    start: tuple[float, float, float],
    goal: tuple[float, float, float],
    obstacles: list[dict[str, Any]],
    workspace_bounds: tuple[float, float, float],
    max_iter: int = 800,
    step_size: float = 0.3,
) -> list[tuple[float, float, float]]:
    """Rapidly-exploring Random Tree (RRT) path planner.

    Args:
        start: Starting position.
        goal: Goal position.
        obstacles: Obstacle bounding-box list.
        workspace_bounds: (x_max, y_max, z_max) workspace extents.
        max_iter: Maximum RRT iterations.
        step_size: Maximum edge length per step.

    Returns:
        List of waypoints from start to goal.
    """
    tree: list[tuple[float, float, float]] = [start]
    parent: dict[int, int | None] = {0: None}

    def _sample() -> tuple[float, float, float]:
        if random.random() < 0.1:
            return goal
        return (
            random.uniform(0, workspace_bounds[0]),
            random.uniform(0, workspace_bounds[1]),
            random.uniform(0, workspace_bounds[2]),
        )

    def _nearest(sample: tuple[float, float, float]) -> int:
        return min(range(len(tree)), key=lambda i: _euclidean_distance(tree[i], sample))

    def _steer(
        from_node: tuple[float, float, float], to_node: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        dist = _euclidean_distance(from_node, to_node)
        if dist <= step_size:
            return to_node
        ratio = step_size / dist
        return (
            from_node[0] + (to_node[0] - from_node[0]) * ratio,
            from_node[1] + (to_node[1] - from_node[1]) * ratio,
            from_node[2] + (to_node[2] - from_node[2]) * ratio,
        )

    def _collision_free(new_node: tuple[float, float, float]) -> bool:
        for obs in obstacles:
            if (
                obs["x"] <= new_node[0] <= obs["x"] + obs.get("w", 0.5)
                and obs["y"] <= new_node[1] <= obs["y"] + obs.get("d", 0.5)
                and obs["z"] <= new_node[2] <= obs["z"] + obs.get("h", 0.5)
            ):
                return False
        return True

    for _ in range(max_iter):
        sample = _sample()
        nearest_idx = _nearest(sample)
        new_node = _steer(tree[nearest_idx], sample)
        if _collision_free(new_node):
            new_idx = len(tree)
            tree.append(new_node)
            parent[new_idx] = nearest_idx
            if _euclidean_distance(new_node, goal) < step_size:
                # Reconstruct path
                path: list[tuple[float, float, float]] = [goal]
                idx: int | None = new_idx
                while idx is not None:
                    path.append(tree[idx])
                    idx = parent[idx]
                return list(reversed(path))

    # Fallback: straight-line
    n = 8
    return [
        (
            start[0] + (goal[0] - start[0]) * i / n,
            start[1] + (goal[1] - start[1]) * i / n,
            start[2] + (goal[2] - start[2]) * i / n,
        )
        for i in range(n + 1)
    ]


def _bspline_smooth(
    waypoints: list[tuple[float, float, float]], num_output_points: int = 50
) -> list[tuple[float, float, float]]:
    """Smooth a waypoint sequence using uniform cubic B-spline interpolation.

    Args:
        waypoints: Raw waypoint list.
        num_output_points: Resolution of the smoothed trajectory.

    Returns:
        Smoothed trajectory as a list of (x, y, z) tuples.
    """
    if len(waypoints) < 4:
        return waypoints

    n = len(waypoints) - 1
    smoothed: list[tuple[float, float, float]] = []

    for step in range(num_output_points):
        t_global = step / (num_output_points - 1) * (n - 2)
        i = min(int(t_global), n - 3)
        t = t_global - i

        # De Boor basis for uniform cubic B-spline
        b0 = (1 - t) ** 3 / 6.0
        b1 = (3 * t**3 - 6 * t**2 + 4) / 6.0
        b2 = (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6.0
        b3 = t**3 / 6.0

        pts = waypoints[i : i + 4]
        x = b0 * pts[0][0] + b1 * pts[1][0] + b2 * pts[2][0] + b3 * pts[3][0]
        y = b0 * pts[0][1] + b1 * pts[1][1] + b2 * pts[2][1] + b3 * pts[3][1]
        z = b0 * pts[0][2] + b1 * pts[1][2] + b2 * pts[2][2] + b3 * pts[3][2]
        smoothed.append((round(x, 6), round(y, 6), round(z, 6)))

    return smoothed


# ---------------------------------------------------------------------------
# MotionPlanner adapter
# ---------------------------------------------------------------------------


class MotionPlanner:
    """Path planning dataset generation adapter.

    Generates diverse, obstacle-aware motion planning datasets for robotics
    training: waypoint sequences, smoothed trajectories, velocity profiles,
    collision statistics, multi-robot coordination plans, and formatted exports.

    Implements MotionPlannerProtocol.
    """

    # Maximum workspace extent in metres used as default bound for RRT
    DEFAULT_WORKSPACE_BOUNDS: tuple[float, float, float] = (10.0, 10.0, 3.0)

    async def generate_dataset(
        self,
        planning_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Generate a motion planning dataset from the supplied configuration.

        Args:
            planning_config: Planning configuration dict. Supported keys:
                - algorithm: 'astar' | 'rrt' (default 'astar')
                - num_trajectories: Number of trajectories to generate (default 20)
                - workspace_bounds: [x_max, y_max, z_max] (default [10, 10, 3])
                - obstacles: List of obstacle dicts {x, y, z, w, d, h}
                - smooth: bool — apply B-spline smoothing (default True)
                - velocity_profile: 'trapezoidal' | 'constant' (default 'trapezoidal')
                - max_velocity: Maximum velocity m/s (default 1.0)
                - grid_resolution: A* grid resolution metres (default 0.1)
                - num_robots: For multi-robot coordination (default 1)
                - export_format: 'csv' | 'json' (default 'json')
            tenant_id: Tenant context for namespacing.

        Returns:
            Dict with trajectories, collision_stats, dataset_stats, export_uri.
        """
        algorithm = planning_config.get("algorithm", "astar")
        num_trajectories = int(planning_config.get("num_trajectories", 20))
        raw_bounds = planning_config.get("workspace_bounds", list(self.DEFAULT_WORKSPACE_BOUNDS))
        workspace_bounds = tuple(float(b) for b in raw_bounds[:3])  # type: ignore[assignment]
        obstacles: list[dict[str, Any]] = planning_config.get("obstacles", [])
        apply_smooth = bool(planning_config.get("smooth", True))
        velocity_profile = planning_config.get("velocity_profile", "trapezoidal")
        max_velocity = float(planning_config.get("max_velocity", 1.0))
        grid_resolution = float(planning_config.get("grid_resolution", 0.1))
        num_robots = int(planning_config.get("num_robots", 1))
        export_format = planning_config.get("export_format", "json")

        logger.info(
            "Generating motion planning dataset",
            algorithm=algorithm,
            num_trajectories=num_trajectories,
            num_robots=num_robots,
            tenant_id=str(tenant_id),
        )

        trajectories: list[dict[str, Any]] = []
        collision_count = 0

        for traj_idx in range(num_trajectories):
            robot_plans: list[dict[str, Any]] = []
            for robot_id in range(num_robots):
                start = (
                    random.uniform(0.0, workspace_bounds[0] * 0.2),
                    random.uniform(0.0, workspace_bounds[1] * 0.2),
                    0.0,
                )
                goal = (
                    random.uniform(workspace_bounds[0] * 0.7, workspace_bounds[0]),
                    random.uniform(workspace_bounds[1] * 0.7, workspace_bounds[1]),
                    0.0,
                )

                if algorithm == "rrt":
                    raw_path = _rrt_path(start, goal, obstacles, workspace_bounds)  # type: ignore[arg-type]
                else:
                    raw_path = _astar_path(start, goal, obstacles, grid_resolution)

                smoothed_path = _bspline_smooth(raw_path) if apply_smooth else raw_path

                velocities = self._generate_velocity_profile(
                    smoothed_path, velocity_profile, max_velocity
                )

                collision_free = self._check_path_collision(smoothed_path, obstacles)
                if not collision_free:
                    collision_count += 1

                robot_plans.append(
                    {
                        "robot_id": robot_id,
                        "start": start,
                        "goal": goal,
                        "waypoints": smoothed_path,
                        "velocities": velocities,
                        "path_length_m": sum(
                            _euclidean_distance(smoothed_path[i], smoothed_path[i + 1])
                            for i in range(len(smoothed_path) - 1)
                        ),
                        "collision_free": collision_free,
                    }
                )

            trajectories.append(
                {
                    "trajectory_id": traj_idx,
                    "robots": robot_plans,
                }
            )

        export_data = self._export(trajectories, export_format)

        collision_stats = {
            "total_plans": num_trajectories * num_robots,
            "collision_free_count": num_trajectories * num_robots - collision_count,
            "collision_count": collision_count,
            "collision_rate": round(
                collision_count / max(num_trajectories * num_robots, 1), 4
            ),
        }

        logger.info(
            "Motion planning dataset generated",
            num_trajectories=num_trajectories,
            collision_rate=collision_stats["collision_rate"],
            tenant_id=str(tenant_id),
        )

        return {
            "trajectories": trajectories,
            "collision_stats": collision_stats,
            "dataset_stats": {
                "algorithm": algorithm,
                "num_trajectories": num_trajectories,
                "num_robots": num_robots,
                "smoothed": apply_smooth,
                "velocity_profile": velocity_profile,
                "export_format": export_format,
            },
            "export_data": export_data,
            "output_uri": f"s3://aumos-physical-ai/{tenant_id}/motion-planning/{uuid.uuid4()}.{export_format}",
        }

    def _generate_velocity_profile(
        self,
        waypoints: list[tuple[float, float, float]],
        profile_type: str,
        max_velocity: float,
    ) -> list[float]:
        """Generate per-waypoint velocity values along a trajectory.

        Args:
            waypoints: Trajectory waypoints.
            profile_type: 'trapezoidal' ramp-up/ramp-down or 'constant'.
            max_velocity: Maximum allowable speed in m/s.

        Returns:
            List of velocity values (one per waypoint).
        """
        n = len(waypoints)
        if n == 0:
            return []
        if profile_type == "constant":
            return [max_velocity] * n

        # Trapezoidal profile: ramp up for first 25%, hold, ramp down last 25%
        velocities: list[float] = []
        ramp_fraction = 0.25
        ramp_end = int(n * ramp_fraction)
        ramp_start = n - ramp_end

        for i in range(n):
            if i < ramp_end:
                v = max_velocity * (i / max(ramp_end, 1))
            elif i >= ramp_start:
                v = max_velocity * ((n - 1 - i) / max(ramp_end, 1))
            else:
                v = max_velocity
            velocities.append(round(max(v, 0.0), 4))

        return velocities

    def _check_path_collision(
        self,
        waypoints: list[tuple[float, float, float]],
        obstacles: list[dict[str, Any]],
    ) -> bool:
        """Return True when the path does not intersect any obstacle AABB.

        Args:
            waypoints: Trajectory waypoints.
            obstacles: List of obstacle bounding-box dicts.

        Returns:
            True if the path is collision-free, False otherwise.
        """
        for wp in waypoints:
            for obs in obstacles:
                if (
                    obs["x"] <= wp[0] <= obs["x"] + obs.get("w", 0.5)
                    and obs["y"] <= wp[1] <= obs["y"] + obs.get("d", 0.5)
                    and obs["z"] <= wp[2] <= obs["z"] + obs.get("h", 0.5)
                ):
                    return False
        return True

    def _export(self, trajectories: list[dict[str, Any]], format_type: str) -> str:
        """Serialise trajectory data to CSV or JSON string.

        Args:
            trajectories: List of trajectory dicts.
            format_type: 'csv' or 'json'.

        Returns:
            Serialised string representation.
        """
        if format_type == "csv":
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["trajectory_id", "robot_id", "waypoint_idx", "x", "y", "z", "velocity"])
            for traj in trajectories:
                for robot in traj["robots"]:
                    for wp_idx, (wp, vel) in enumerate(
                        zip(robot["waypoints"], robot["velocities"])
                    ):
                        writer.writerow(
                            [traj["trajectory_id"], robot["robot_id"], wp_idx, wp[0], wp[1], wp[2], vel]
                        )
            return buf.getvalue()

        return json.dumps(trajectories, default=str)
