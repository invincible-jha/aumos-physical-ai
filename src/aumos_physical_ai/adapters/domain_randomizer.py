"""Domain randomization adapter for sim-to-real transfer.

Applies physics-based parameter randomization — textures, lighting conditions,
camera poses, object scale and mass, friction coefficients, and sensor noise
models — to generate training data robust to domain shift.

Implements the DomainRandomizerProtocol interface defined in core/interfaces.py.
"""

import random
import uuid
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Randomization primitive samplers
# ---------------------------------------------------------------------------


def _sample_uniform(low: float, high: float) -> float:
    """Sample uniformly between low and high.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        Sampled float value.
    """
    return round(random.uniform(low, high), 4)


def _sample_truncated_gaussian(mean: float, std: float, low: float, high: float) -> float:
    """Sample from a Gaussian truncated to [low, high].

    Args:
        mean: Distribution mean.
        std: Standard deviation.
        low: Lower clamp.
        high: Upper clamp.

    Returns:
        Clamped sample.
    """
    value = random.gauss(mean, std)
    return round(max(low, min(high, value)), 4)


def _sample_color_temperature(min_kelvin: float, max_kelvin: float) -> float:
    """Sample a colour temperature in Kelvin.

    Args:
        min_kelvin: Minimum colour temperature.
        max_kelvin: Maximum colour temperature.

    Returns:
        Sampled colour temperature in Kelvin.
    """
    return round(random.uniform(min_kelvin, max_kelvin), 1)


def _kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """Approximate conversion of colour temperature to sRGB.

    Based on Tanner Helland's algorithm (2012).

    Args:
        kelvin: Colour temperature in Kelvin.

    Returns:
        (r, g, b) tuple in [0.0, 1.0].
    """
    temp = kelvin / 100.0
    if temp <= 66.0:
        r = 1.0
        g = max(0.0, min(1.0, (99.4708025861 * (1.0 / (1e-8 + (temp - 2.0))) - 161.1195681661) / 255.0))
    else:
        r = max(0.0, min(1.0, (329.698727446 * ((temp - 60.0) ** -0.1332047592)) / 255.0))
        g = max(0.0, min(1.0, (288.1221695283 * ((temp - 60.0) ** -0.0755148492)) / 255.0))

    if temp >= 66.0:
        b = 1.0
    elif temp <= 19.0:
        b = 0.0
    else:
        b = max(0.0, min(1.0, (138.5177312231 * ((temp - 10.0) ** 0.0549944583) - 305.0447927307) / 255.0))

    return (round(r, 4), round(g, 4), round(b, 4))


# ---------------------------------------------------------------------------
# Noise model generators
# ---------------------------------------------------------------------------


def _generate_lidar_noise_model(dropout_rate: float, range_std_m: float) -> dict[str, Any]:
    """Generate a LiDAR sensor noise configuration.

    Args:
        dropout_rate: Fraction of points to randomly drop [0, 1].
        range_std_m: Standard deviation of range measurement noise in metres.

    Returns:
        Noise model configuration dict.
    """
    return {
        "type": "lidar_noise",
        "dropout_rate": round(dropout_rate, 4),
        "range_std_m": round(range_std_m, 4),
        "angular_std_deg": _sample_uniform(0.01, 0.1),
        "reflectivity_noise": _sample_uniform(0.0, 0.05),
        "scan_frequency_hz": random.choice([10, 20, 40]),
    }


def _generate_camera_noise_model(blur_sigma: float, exposure_variation: float) -> dict[str, Any]:
    """Generate a camera sensor noise configuration.

    Args:
        blur_sigma: Gaussian blur kernel standard deviation in pixels.
        exposure_variation: Multiplicative exposure perturbation fraction.

    Returns:
        Noise model configuration dict.
    """
    return {
        "type": "camera_noise",
        "blur_sigma_px": round(blur_sigma, 4),
        "exposure_variation": round(exposure_variation, 4),
        "gaussian_noise_std": _sample_uniform(0.001, 0.02),
        "chromatic_aberration": _sample_uniform(0.0, 0.005),
        "jpeg_compression_quality": random.randint(70, 99),
        "vignetting_strength": _sample_uniform(0.0, 0.3),
    }


def _generate_imu_noise_model(noise_density: float) -> dict[str, Any]:
    """Generate an IMU sensor noise configuration.

    Args:
        noise_density: IMU noise density in (rad/s)/sqrt(Hz) or (m/s^2)/sqrt(Hz).

    Returns:
        Noise model configuration dict.
    """
    return {
        "type": "imu_noise",
        "gyro_noise_density_rads_sqhz": round(noise_density, 6),
        "accel_noise_density_ms2_sqhz": round(noise_density * 10.0, 6),
        "gyro_bias_instability": _sample_uniform(1e-5, 1e-4),
        "accel_bias_instability": _sample_uniform(1e-4, 1e-3),
        "update_rate_hz": random.choice([100, 200, 400]),
    }


# ---------------------------------------------------------------------------
# DomainRandomizer adapter
# ---------------------------------------------------------------------------


class DomainRandomizer:
    """Domain randomization engine for sim-to-real robustness.

    Generates per-variation randomized scene parameters across textures,
    lighting, camera poses, object properties, physics parameters, and
    sensor noise models, producing a diverse training dataset.

    Implements DomainRandomizerProtocol.
    """

    # Texture categories used when no pool URI is provided
    DEFAULT_TEXTURE_CATEGORIES: list[str] = [
        "wood", "metal", "plastic", "rubber", "ceramic",
        "fabric", "stone", "concrete", "glass", "painted_steel",
    ]

    async def randomize(
        self,
        randomization_params: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Execute domain randomization and return varied scene configurations.

        Args:
            randomization_params: Randomization configuration. Supported keys:
                - lighting: {intensity_range: [min, max], color_temp_range: [min_K, max_K], count: int}
                - textures: {object_classes: [...], texture_pool_uri: str, randomize_floor: bool}
                - camera_poses: {position_noise_m: float, rotation_noise_deg: float}
                - object_scale_mass: {scale_range: [min, max], mass_multiplier_range: [min, max]}
                - friction: {friction_range: [min, max]}
                - sensor_noise: {lidar_dropout_rate: float, camera_blur_sigma: float, imu_noise_density: float}
                - num_variations: Number of randomized variations (default 100)
            tenant_id: Tenant context for namespacing.

        Returns:
            Dict with output_uri, variations_generated, diversity_score, coverage_score,
            and sample_variation for inspection.
        """
        num_variations = int(randomization_params.get("num_variations", 100))
        lighting_cfg: dict[str, Any] = randomization_params.get("lighting", {})
        texture_cfg: dict[str, Any] = randomization_params.get("textures", {})
        camera_cfg: dict[str, Any] = randomization_params.get("camera_poses", {})
        scale_mass_cfg: dict[str, Any] = randomization_params.get("object_scale_mass", {})
        friction_cfg: dict[str, Any] = randomization_params.get("friction", {})
        sensor_cfg: dict[str, Any] = randomization_params.get("sensor_noise", {})

        logger.info(
            "Starting domain randomization",
            num_variations=num_variations,
            tenant_id=str(tenant_id),
        )

        variations: list[dict[str, Any]] = []
        diversity_accumulator: list[float] = []

        for variation_idx in range(num_variations):
            random.seed(variation_idx)

            lighting = self._randomize_lighting(lighting_cfg)
            textures = self._randomize_textures(texture_cfg)
            camera_pose_delta = self._randomize_camera_pose(camera_cfg)
            object_properties = self._randomize_object_properties(scale_mass_cfg)
            friction = self._randomize_friction(friction_cfg)
            sensor_noise = self._randomize_sensor_noise(sensor_cfg)

            variation = {
                "variation_id": variation_idx,
                "lighting": lighting,
                "textures": textures,
                "camera_pose_delta": camera_pose_delta,
                "object_properties": object_properties,
                "friction": friction,
                "sensor_noise": sensor_noise,
            }
            variations.append(variation)

            # Compute per-variation diversity contribution (simple hash-based proxy)
            diversity_accumulator.append(
                self._compute_variation_diversity(variation)
            )

        diversity_score = round(sum(diversity_accumulator) / max(len(diversity_accumulator), 1), 4)
        coverage_score = self._compute_coverage_score(variations)

        logger.info(
            "Domain randomization complete",
            num_variations=num_variations,
            diversity_score=diversity_score,
            coverage_score=coverage_score,
            tenant_id=str(tenant_id),
        )

        return {
            "output_uri": (
                f"s3://aumos-physical-ai/{tenant_id}/randomization/{uuid.uuid4()}.json"
            ),
            "variations_generated": num_variations,
            "diversity_score": diversity_score,
            "coverage_score": coverage_score,
            "sample_variation": variations[0] if variations else {},
            "randomization_summary": {
                "lighting_enabled": bool(lighting_cfg),
                "texture_randomization": bool(texture_cfg),
                "camera_perturbation": bool(camera_cfg),
                "physics_randomization": bool(friction_cfg),
                "sensor_noise_injection": bool(sensor_cfg),
            },
        }

    def _randomize_lighting(self, lighting_cfg: dict[str, Any]) -> dict[str, Any]:
        """Generate randomized lighting parameters.

        Args:
            lighting_cfg: Lighting configuration from randomization_params.

        Returns:
            Dict with num_lights, lights list (intensity, color_temp, rgb, position, type).
        """
        intensity_range = lighting_cfg.get("intensity_range", [500, 3000])
        color_temp_range = lighting_cfg.get("color_temp_range", [2700, 6500])
        num_lights = int(lighting_cfg.get("count", random.randint(1, 6)))

        lights: list[dict[str, Any]] = []
        for light_idx in range(num_lights):
            kelvin = _sample_color_temperature(color_temp_range[0], color_temp_range[1])
            rgb = _kelvin_to_rgb(kelvin)
            lights.append(
                {
                    "light_id": light_idx,
                    "type": random.choice(["point", "spot", "area", "directional"]),
                    "intensity_lux": _sample_uniform(intensity_range[0], intensity_range[1]),
                    "color_temperature_k": kelvin,
                    "rgb": rgb,
                    "position_m": [_sample_uniform(-3.0, 3.0) for _ in range(3)],
                    "shadow_enabled": random.random() > 0.3,
                    "shadow_softness": _sample_uniform(0.0, 1.0),
                }
            )

        return {
            "num_lights": num_lights,
            "ambient_intensity": _sample_uniform(0.1, 0.5),
            "lights": lights,
        }

    def _randomize_textures(self, texture_cfg: dict[str, Any]) -> dict[str, Any]:
        """Generate randomized texture assignments for objects and environment.

        Args:
            texture_cfg: Texture configuration from randomization_params.

        Returns:
            Dict with object texture assignments and floor/wall materials.
        """
        object_classes: list[str] = texture_cfg.get("object_classes", ["object_0"])
        texture_pool_uri: str = texture_cfg.get("texture_pool_uri", "s3://textures/default/")
        randomize_floor: bool = bool(texture_cfg.get("randomize_floor", True))

        categories = self.DEFAULT_TEXTURE_CATEGORIES
        texture_assignments: dict[str, Any] = {}

        for obj_class in object_classes:
            category = random.choice(categories)
            texture_assignments[obj_class] = {
                "category": category,
                "texture_uri": f"{texture_pool_uri}{category}/{random.randint(1, 100):04d}.png",
                "roughness": _sample_uniform(0.0, 1.0),
                "metallic": _sample_uniform(0.0, 0.5),
                "scale": _sample_uniform(0.5, 3.0),
            }

        floor_material = (
            {
                "category": random.choice(["concrete", "wood", "tile", "carpet"]),
                "roughness": _sample_uniform(0.3, 1.0),
                "reflectivity": _sample_uniform(0.0, 0.3),
            }
            if randomize_floor
            else None
        )

        return {
            "object_textures": texture_assignments,
            "floor_material": floor_material,
            "wall_material": {
                "color_rgb": [_sample_uniform(0.5, 1.0) for _ in range(3)],
                "roughness": _sample_uniform(0.5, 1.0),
            },
        }

    def _randomize_camera_pose(self, camera_cfg: dict[str, Any]) -> dict[str, Any]:
        """Generate camera pose perturbation parameters.

        Args:
            camera_cfg: Camera pose configuration from randomization_params.

        Returns:
            Dict with position_offset_m and rotation_offset_deg.
        """
        position_noise = float(camera_cfg.get("position_noise_m", 0.05))
        rotation_noise = float(camera_cfg.get("rotation_noise_deg", 5.0))

        return {
            "position_offset_m": [
                _sample_truncated_gaussian(0.0, position_noise, -position_noise * 3, position_noise * 3)
                for _ in range(3)
            ],
            "rotation_offset_deg": [
                _sample_truncated_gaussian(0.0, rotation_noise, -rotation_noise * 3, rotation_noise * 3)
                for _ in range(3)
            ],
            "fov_perturbation_deg": _sample_truncated_gaussian(0.0, 2.0, -5.0, 5.0),
        }

    def _randomize_object_properties(self, scale_mass_cfg: dict[str, Any]) -> dict[str, Any]:
        """Generate randomized object scale and mass properties.

        Args:
            scale_mass_cfg: Scale/mass configuration from randomization_params.

        Returns:
            Dict with scale_multiplier and mass_multiplier.
        """
        scale_range = scale_mass_cfg.get("scale_range", [0.8, 1.2])
        mass_range = scale_mass_cfg.get("mass_multiplier_range", [0.7, 1.5])

        return {
            "scale_multiplier": _sample_uniform(scale_range[0], scale_range[1]),
            "mass_multiplier": _sample_uniform(mass_range[0], mass_range[1]),
            "inertia_perturbation": _sample_uniform(0.9, 1.1),
            "center_of_mass_offset_m": [_sample_uniform(-0.01, 0.01) for _ in range(3)],
        }

    def _randomize_friction(self, friction_cfg: dict[str, Any]) -> dict[str, Any]:
        """Generate randomized friction and contact physics parameters.

        Args:
            friction_cfg: Friction configuration from randomization_params.

        Returns:
            Dict with static_friction, kinetic_friction, and restitution.
        """
        friction_range = friction_cfg.get("friction_range", [0.2, 0.9])

        return {
            "static_friction": _sample_uniform(friction_range[0], friction_range[1]),
            "kinetic_friction": _sample_uniform(
                friction_range[0] * 0.8, friction_range[1] * 0.9
            ),
            "restitution": _sample_uniform(0.0, 0.5),
            "rolling_friction": _sample_uniform(0.001, 0.05),
        }

    def _randomize_sensor_noise(self, sensor_cfg: dict[str, Any]) -> dict[str, Any]:
        """Inject sensor noise model parameters for each modality.

        Args:
            sensor_cfg: Sensor noise configuration from randomization_params.

        Returns:
            Dict with noise models for lidar, camera, and imu.
        """
        lidar_dropout = float(sensor_cfg.get("lidar_dropout_rate", 0.02))
        camera_blur = float(sensor_cfg.get("camera_blur_sigma", 0.5))
        imu_noise = float(sensor_cfg.get("imu_noise_density", 0.0001))

        return {
            "lidar": _generate_lidar_noise_model(
                dropout_rate=_sample_truncated_gaussian(lidar_dropout, lidar_dropout * 0.5, 0.0, 0.2),
                range_std_m=_sample_uniform(0.001, 0.05),
            ),
            "camera": _generate_camera_noise_model(
                blur_sigma=_sample_truncated_gaussian(camera_blur, camera_blur * 0.5, 0.0, 5.0),
                exposure_variation=_sample_uniform(0.85, 1.15),
            ),
            "imu": _generate_imu_noise_model(
                noise_density=_sample_truncated_gaussian(imu_noise, imu_noise * 0.3, 0.0, 0.001),
            ),
        }

    def _compute_variation_diversity(self, variation: dict[str, Any]) -> float:
        """Compute a single-variation diversity score proxy.

        Higher values indicate the variation is more different from a
        canonical centre configuration.

        Args:
            variation: A single randomized variation dict.

        Returns:
            Diversity contribution score in [0.0, 1.0].
        """
        lighting = variation.get("lighting", {})
        num_lights = lighting.get("num_lights", 1)
        ambient = lighting.get("ambient_intensity", 0.3)
        camera_delta = variation.get("camera_pose_delta", {})
        pos_offset = sum(abs(v) for v in camera_delta.get("position_offset_m", [0.0]))
        scale = variation.get("object_properties", {}).get("scale_multiplier", 1.0)
        friction = variation.get("friction", {}).get("static_friction", 0.5)

        raw = (
            min(num_lights / 6.0, 1.0) * 0.2
            + abs(ambient - 0.3) * 0.2
            + min(pos_offset / 0.3, 1.0) * 0.2
            + abs(scale - 1.0) * 0.2
            + abs(friction - 0.5) * 0.2
        )
        return round(min(raw, 1.0), 4)

    def _compute_coverage_score(self, variations: list[dict[str, Any]]) -> float:
        """Estimate parameter space coverage across all variations.

        Args:
            variations: Full list of randomized variation dicts.

        Returns:
            Coverage score in [0.0, 1.0].
        """
        if not variations:
            return 0.0

        friction_values = [
            v.get("friction", {}).get("static_friction", 0.5) for v in variations
        ]
        friction_range = max(friction_values) - min(friction_values)
        num_lights_values = [v.get("lighting", {}).get("num_lights", 1) for v in variations]
        unique_light_configs = len(set(num_lights_values)) / 6.0

        coverage = (
            min(friction_range / 0.7, 1.0) * 0.5
            + min(unique_light_configs, 1.0) * 0.5
        )
        return round(min(coverage, 1.0), 4)
