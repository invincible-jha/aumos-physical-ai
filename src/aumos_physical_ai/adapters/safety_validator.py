"""Physical safety validation adapter for robotics safety-critical testing.

Implements safety boundary enforcement, collision detection, speed limit
checking, emergency stop scenario generation, ISO 10218 compliance checking,
safety metric computation, and safety test report generation.

ISO 10218-1:2011 and ISO 10218-2:2011 define safety requirements for
industrial robots and robot systems.
"""

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Safety boundary and limit types
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceBoundary:
    """Axis-aligned bounding box defining the safe robot workspace.

    All positions are in metres relative to the robot base frame.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    max_tcp_velocity_ms: float = 1.5
    max_joint_velocity_rads: float = 3.0
    max_payload_force_n: float = 200.0
    max_collision_force_n: float = 50.0

    def contains(self, position: tuple[float, float, float]) -> bool:
        """Return True when the position is inside the safe workspace.

        Args:
            position: (x, y, z) tuple in metres.

        Returns:
            True if the position is within all boundaries.
        """
        x, y, z = position
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )


@dataclass
class SafetyTestResult:
    """Result of a single safety validation test."""

    test_id: str
    test_name: str
    passed: bool
    severity: str  # 'critical' | 'high' | 'medium' | 'low'
    measured_value: float | None = None
    limit_value: float | None = None
    unit: str = ""
    iso_clause: str = ""
    details: str = ""


@dataclass
class SafetyReport:
    """Aggregated safety validation report.

    Conforms to ISO 10218 reporting structure.
    """

    report_id: str
    tenant_id: str
    overall_safe: bool
    iso_compliant: bool
    tests_run: int
    tests_passed: int
    tests_failed: int
    critical_failures: int
    test_results: list[SafetyTestResult] = field(default_factory=list)
    safety_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PhysicalSafetyValidator adapter
# ---------------------------------------------------------------------------


class PhysicalSafetyValidator:
    """Safety-critical robotics test generator and validator.

    Runs a comprehensive safety test suite against robot motion plans and
    scenario configurations, checking workspace boundaries, velocity limits,
    collision risks, emergency stop behaviour, and ISO 10218 compliance.

    Implements PhysicalSafetyValidatorProtocol.
    """

    # ISO 10218 clause references for each test category
    ISO_CLAUSES: dict[str, str] = {
        "workspace_boundary": "ISO 10218-1:2011 Clause 5.4.2",
        "tcp_velocity": "ISO 10218-1:2011 Clause 5.6.3",
        "joint_velocity": "ISO 10218-1:2011 Clause 5.6.3",
        "collision_force": "ISO 10218-1:2011 Clause 5.10.5",
        "payload_force": "ISO 10218-1:2011 Clause 5.4.4",
        "emergency_stop": "ISO 10218-1:2011 Clause 5.5.2",
        "reduced_speed_zone": "ISO 10218-2:2011 Clause 5.10.4",
        "singularity_avoidance": "ISO 10218-1:2011 Clause 5.4.3",
    }

    async def validate_motion_plan(
        self,
        validation_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Validate a robot motion plan against safety boundaries.

        Args:
            validation_config: Safety validation configuration. Supported keys:
                - workspace: dict with x_min/x_max/y_min/y_max/z_min/z_max and limits
                - trajectories: List of trajectory dicts with waypoints and velocities
                - obstacles: List of obstacle AABB dicts
                - run_estop_scenarios: bool — test emergency stop scenarios (default True)
                - run_iso_check: bool — run ISO 10218 compliance check (default True)
                - force_threshold_n: Override collision force threshold
            tenant_id: Tenant context.

        Returns:
            Dict with safety_report, test_results, safety_metrics, output_uri.
        """
        workspace_cfg: dict[str, Any] = validation_config.get("workspace", {})
        boundary = WorkspaceBoundary(
            x_min=float(workspace_cfg.get("x_min", -1.0)),
            x_max=float(workspace_cfg.get("x_max", 1.0)),
            y_min=float(workspace_cfg.get("y_min", -1.0)),
            y_max=float(workspace_cfg.get("y_max", 1.0)),
            z_min=float(workspace_cfg.get("z_min", 0.0)),
            z_max=float(workspace_cfg.get("z_max", 1.5)),
            max_tcp_velocity_ms=float(workspace_cfg.get("max_tcp_velocity_ms", 1.5)),
            max_joint_velocity_rads=float(workspace_cfg.get("max_joint_velocity_rads", 3.0)),
            max_payload_force_n=float(workspace_cfg.get("max_payload_force_n", 200.0)),
            max_collision_force_n=float(
                validation_config.get(
                    "force_threshold_n",
                    workspace_cfg.get("max_collision_force_n", 50.0),
                )
            ),
        )

        trajectories: list[dict[str, Any]] = validation_config.get("trajectories", [])
        obstacles: list[dict[str, Any]] = validation_config.get("obstacles", [])
        run_estop = bool(validation_config.get("run_estop_scenarios", True))
        run_iso = bool(validation_config.get("run_iso_check", True))

        logger.info(
            "Running physical safety validation",
            num_trajectories=len(trajectories),
            num_obstacles=len(obstacles),
            run_iso=run_iso,
            tenant_id=str(tenant_id),
        )

        all_test_results: list[SafetyTestResult] = []

        # Run trajectory-level tests
        for traj_idx, trajectory in enumerate(trajectories):
            waypoints: list[tuple[float, float, float]] = trajectory.get("waypoints", [])
            velocities: list[float] = trajectory.get("velocities", [])

            all_test_results.extend(
                self._test_workspace_boundaries(traj_idx, waypoints, boundary)
            )
            all_test_results.extend(
                self._test_velocity_limits(traj_idx, velocities, boundary)
            )
            all_test_results.extend(
                self._test_collision_proximity(traj_idx, waypoints, obstacles, boundary)
            )

        # Emergency stop scenario tests
        if run_estop:
            all_test_results.extend(
                self._generate_estop_scenarios(trajectories, boundary)
            )

        # ISO 10218 compliance check
        iso_compliant = False
        if run_iso:
            iso_results, iso_compliant = self._run_iso_10218_check(
                all_test_results, boundary
            )
            all_test_results.extend(iso_results)
        else:
            iso_compliant = not any(
                not r.passed and r.severity == "critical" for r in all_test_results
            )

        # Aggregate results
        tests_run = len(all_test_results)
        tests_passed = sum(1 for r in all_test_results if r.passed)
        tests_failed = tests_run - tests_passed
        critical_failures = sum(
            1 for r in all_test_results if not r.passed and r.severity == "critical"
        )
        safety_score = round(tests_passed / max(tests_run, 1), 4)
        overall_safe = critical_failures == 0 and safety_score >= 0.9

        recommendations = self._generate_recommendations(all_test_results, boundary)

        report = SafetyReport(
            report_id=str(uuid.uuid4()),
            tenant_id=str(tenant_id),
            overall_safe=overall_safe,
            iso_compliant=iso_compliant,
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            critical_failures=critical_failures,
            test_results=all_test_results,
            safety_score=safety_score,
            recommendations=recommendations,
        )

        logger.info(
            "Safety validation complete",
            safety_score=safety_score,
            overall_safe=overall_safe,
            iso_compliant=iso_compliant,
            critical_failures=critical_failures,
            tenant_id=str(tenant_id),
        )

        return {
            "safety_report": {
                "report_id": report.report_id,
                "overall_safe": report.overall_safe,
                "iso_compliant": report.iso_compliant,
                "safety_score": report.safety_score,
                "tests_run": report.tests_run,
                "tests_passed": report.tests_passed,
                "tests_failed": report.tests_failed,
                "critical_failures": report.critical_failures,
                "recommendations": report.recommendations,
            },
            "test_results": [
                {
                    "test_id": r.test_id,
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "severity": r.severity,
                    "measured_value": r.measured_value,
                    "limit_value": r.limit_value,
                    "unit": r.unit,
                    "iso_clause": r.iso_clause,
                    "details": r.details,
                }
                for r in all_test_results
            ],
            "safety_metrics": {
                "workspace_boundary_violations": sum(
                    1 for r in all_test_results
                    if "boundary" in r.test_id and not r.passed
                ),
                "velocity_violations": sum(
                    1 for r in all_test_results
                    if "velocity" in r.test_id and not r.passed
                ),
                "collision_violations": sum(
                    1 for r in all_test_results
                    if "collision" in r.test_id and not r.passed
                ),
                "estop_scenarios_tested": sum(
                    1 for r in all_test_results if "estop" in r.test_id
                ),
            },
            "output_uri": (
                f"s3://aumos-physical-ai/{tenant_id}/safety/{report.report_id}.json"
            ),
        }

    def _test_workspace_boundaries(
        self,
        trajectory_idx: int,
        waypoints: list[tuple[float, float, float]],
        boundary: WorkspaceBoundary,
    ) -> list[SafetyTestResult]:
        """Check all waypoints against workspace boundaries.

        Args:
            trajectory_idx: Trajectory index for test ID namespacing.
            waypoints: List of (x, y, z) positions.
            boundary: Safe workspace boundary definition.

        Returns:
            List of SafetyTestResult records.
        """
        results: list[SafetyTestResult] = []
        violations = [wp for wp in waypoints if not boundary.contains(wp)]

        results.append(
            SafetyTestResult(
                test_id=f"traj_{trajectory_idx}_boundary_check",
                test_name="Workspace boundary compliance",
                passed=len(violations) == 0,
                severity="critical",
                measured_value=float(len(violations)),
                limit_value=0.0,
                unit="violations",
                iso_clause=self.ISO_CLAUSES["workspace_boundary"],
                details=(
                    f"{len(violations)} out-of-boundary waypoints detected"
                    if violations
                    else "All waypoints within workspace boundaries"
                ),
            )
        )
        return results

    def _test_velocity_limits(
        self,
        trajectory_idx: int,
        velocities: list[float],
        boundary: WorkspaceBoundary,
    ) -> list[SafetyTestResult]:
        """Check TCP velocities against the maximum allowed speed.

        Args:
            trajectory_idx: Trajectory index.
            velocities: List of velocity values in m/s.
            boundary: Workspace boundary with velocity limit.

        Returns:
            List of SafetyTestResult records.
        """
        results: list[SafetyTestResult] = []
        if not velocities:
            return results

        max_measured = max(velocities)
        results.append(
            SafetyTestResult(
                test_id=f"traj_{trajectory_idx}_tcp_velocity",
                test_name="TCP velocity limit compliance",
                passed=max_measured <= boundary.max_tcp_velocity_ms,
                severity="critical",
                measured_value=round(max_measured, 4),
                limit_value=boundary.max_tcp_velocity_ms,
                unit="m/s",
                iso_clause=self.ISO_CLAUSES["tcp_velocity"],
                details=(
                    f"Maximum TCP velocity {max_measured:.3f} m/s "
                    f"{'within' if max_measured <= boundary.max_tcp_velocity_ms else 'exceeds'} "
                    f"limit {boundary.max_tcp_velocity_ms} m/s"
                ),
            )
        )
        return results

    def _test_collision_proximity(
        self,
        trajectory_idx: int,
        waypoints: list[tuple[float, float, float]],
        obstacles: list[dict[str, Any]],
        boundary: WorkspaceBoundary,
    ) -> list[SafetyTestResult]:
        """Check the minimum clearance between trajectory waypoints and obstacles.

        Args:
            trajectory_idx: Trajectory index.
            waypoints: List of (x, y, z) positions.
            obstacles: List of obstacle bounding-box dicts.
            boundary: Workspace boundary with force limits.

        Returns:
            List of SafetyTestResult records.
        """
        results: list[SafetyTestResult] = []
        safety_margin_m = 0.05  # 5 cm minimum clearance

        min_clearance = float("inf")
        for wp in waypoints:
            for obs in obstacles:
                obs_centroid = (
                    obs["x"] + obs.get("w", 0.5) / 2,
                    obs["y"] + obs.get("d", 0.5) / 2,
                    obs["z"] + obs.get("h", 0.5) / 2,
                )
                dist = math.sqrt(
                    (wp[0] - obs_centroid[0]) ** 2
                    + (wp[1] - obs_centroid[1]) ** 2
                    + (wp[2] - obs_centroid[2]) ** 2
                )
                half_diag = math.sqrt(
                    obs.get("w", 0.5) ** 2 + obs.get("d", 0.5) ** 2 + obs.get("h", 0.5) ** 2
                ) / 2.0
                clearance = max(dist - half_diag, 0.0)
                min_clearance = min(min_clearance, clearance)

        if min_clearance == float("inf"):
            min_clearance = 999.0  # No obstacles

        results.append(
            SafetyTestResult(
                test_id=f"traj_{trajectory_idx}_collision_proximity",
                test_name="Obstacle clearance check",
                passed=min_clearance >= safety_margin_m,
                severity="critical",
                measured_value=round(min_clearance, 4),
                limit_value=safety_margin_m,
                unit="m",
                iso_clause=self.ISO_CLAUSES["collision_force"],
                details=(
                    f"Minimum clearance {min_clearance:.4f} m "
                    f"{'satisfies' if min_clearance >= safety_margin_m else 'violates'} "
                    f"{safety_margin_m} m safety margin"
                ),
            )
        )
        return results

    def _generate_estop_scenarios(
        self,
        trajectories: list[dict[str, Any]],
        boundary: WorkspaceBoundary,
    ) -> list[SafetyTestResult]:
        """Generate and evaluate emergency stop scenario results.

        Tests that the robot can decelerate to a stop within ISO-mandated
        distances when an emergency stop is triggered at various trajectory
        phases (start, midpoint, near-goal).

        Args:
            trajectories: List of trajectory dicts.
            boundary: Workspace boundary with velocity limits.

        Returns:
            List of SafetyTestResult records for each stop scenario.
        """
        results: list[SafetyTestResult] = []

        # ISO 10218 Category 0 stop: maximum deceleration ~5 m/s^2
        max_decel = 5.0
        estop_phases = ["start", "mid", "near_goal"]

        for traj_idx, trajectory in enumerate(trajectories):
            velocities: list[float] = trajectory.get("velocities", [])
            if not velocities:
                continue

            for phase in estop_phases:
                phase_idx = {"start": 0, "mid": len(velocities) // 2, "near_goal": -1}[phase]
                current_velocity = velocities[phase_idx]

                # Stopping distance = v^2 / (2 * max_decel)
                stopping_distance = (current_velocity**2) / (2.0 * max_decel)

                # Requirement: stopping distance < 0.3 m (ISO 10218 performance requirement)
                max_stopping_distance = 0.3
                passed = stopping_distance <= max_stopping_distance

                results.append(
                    SafetyTestResult(
                        test_id=f"traj_{traj_idx}_estop_{phase}",
                        test_name=f"Emergency stop at {phase} phase",
                        passed=passed,
                        severity="critical",
                        measured_value=round(stopping_distance, 4),
                        limit_value=max_stopping_distance,
                        unit="m",
                        iso_clause=self.ISO_CLAUSES["emergency_stop"],
                        details=(
                            f"Stopping distance at {current_velocity:.3f} m/s "
                            f"is {stopping_distance:.4f} m "
                            f"({'PASS' if passed else 'FAIL'})"
                        ),
                    )
                )

        return results

    def _run_iso_10218_check(
        self,
        existing_results: list[SafetyTestResult],
        boundary: WorkspaceBoundary,
    ) -> tuple[list[SafetyTestResult], bool]:
        """Run ISO 10218 compliance checks against accumulated test results.

        Args:
            existing_results: Results from prior tests.
            boundary: Workspace boundary configuration.

        Returns:
            Tuple of (additional compliance test results, overall_compliant flag).
        """
        iso_results: list[SafetyTestResult] = []

        # ISO 10218-1 Clause 5.5.2: Emergency stop functional requirement
        estop_tests = [r for r in existing_results if "estop" in r.test_id]
        estop_all_pass = all(r.passed for r in estop_tests) if estop_tests else True
        iso_results.append(
            SafetyTestResult(
                test_id="iso_10218_estop_functional",
                test_name="ISO 10218-1 Cl.5.5.2 — Emergency stop functional",
                passed=estop_all_pass,
                severity="critical",
                iso_clause="ISO 10218-1:2011 Clause 5.5.2",
                details=(
                    "All emergency stop scenarios passed"
                    if estop_all_pass
                    else "One or more emergency stop scenarios failed"
                ),
            )
        )

        # ISO 10218-1 Clause 5.4.2: Workspace limiting
        boundary_tests = [r for r in existing_results if "boundary" in r.test_id]
        boundary_pass = all(r.passed for r in boundary_tests) if boundary_tests else True
        iso_results.append(
            SafetyTestResult(
                test_id="iso_10218_workspace_limit",
                test_name="ISO 10218-1 Cl.5.4.2 — Workspace limiting",
                passed=boundary_pass,
                severity="critical",
                iso_clause=self.ISO_CLAUSES["workspace_boundary"],
                details=(
                    "Workspace limiting properly enforced"
                    if boundary_pass
                    else "Workspace boundary violations detected"
                ),
            )
        )

        # ISO 10218-1 Clause 5.6.3: Speed and separation monitoring
        velocity_tests = [r for r in existing_results if "velocity" in r.test_id]
        velocity_pass = all(r.passed for r in velocity_tests) if velocity_tests else True
        iso_results.append(
            SafetyTestResult(
                test_id="iso_10218_speed_monitoring",
                test_name="ISO 10218-1 Cl.5.6.3 — Speed and separation monitoring",
                passed=velocity_pass,
                severity="high",
                iso_clause=self.ISO_CLAUSES["tcp_velocity"],
                details=(
                    "All velocity limits compliant"
                    if velocity_pass
                    else "Velocity limit violations detected"
                ),
            )
        )

        overall_compliant = all(r.passed for r in iso_results if r.severity == "critical")
        return iso_results, overall_compliant

    def _generate_recommendations(
        self,
        test_results: list[SafetyTestResult],
        boundary: WorkspaceBoundary,
    ) -> list[str]:
        """Generate remediation recommendations from failed tests.

        Args:
            test_results: List of all test results.
            boundary: Workspace boundary configuration.

        Returns:
            List of actionable recommendation strings.
        """
        recommendations: list[str] = []
        failed = [r for r in test_results if not r.passed]

        if any("boundary" in r.test_id for r in failed):
            recommendations.append(
                "Review workspace boundary configuration — increase safe zone or "
                "constrain trajectory generation to exclude out-of-bounds waypoints."
            )
        if any("velocity" in r.test_id for r in failed):
            recommendations.append(
                f"Reduce maximum TCP velocity to below {boundary.max_tcp_velocity_ms} m/s "
                f"or install speed-limiting safety controller (ISO 10218-1 Cl.5.6.3)."
            )
        if any("collision" in r.test_id for r in failed):
            recommendations.append(
                "Increase obstacle clearance margin in motion planner configuration "
                "to at least 50 mm minimum clearance."
            )
        if any("estop" in r.test_id for r in failed):
            recommendations.append(
                "Configure higher deceleration rates on robot controller or reduce "
                "operational velocity to satisfy stopping distance requirements (ISO 10218-1 Cl.5.5.2)."
            )
        if not recommendations:
            recommendations.append(
                "No critical safety issues detected. Schedule periodic re-validation "
                "after any hardware or control software changes."
            )
        return recommendations
