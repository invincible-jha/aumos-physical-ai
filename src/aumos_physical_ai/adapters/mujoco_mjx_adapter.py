"""MuJoCo MJX GPU-accelerated physics adapter.

GAP-352: GPU-Accelerated Physics Simulation.
Uses MuJoCo MJX (JAX backend) for massively parallel physics simulation,
enabling 1,000-10,000x real-time performance essential for robot RL training.
Falls back to CPU MuJoCo (capped at 8 envs) when JAX/GPU is unavailable.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class MuJoCoMJXAdapter:
    """GPU-accelerated physics using MuJoCo MJX (JAX backend).

    MuJoCo MJX enables parallel simulation of thousands of environments
    simultaneously on GPU, essential for reinforcement learning.
    See: https://mujoco.readthedocs.io/en/stable/mjx.html

    Falls back to standard CPU MuJoCo when JAX/GPU is unavailable.
    CPU fallback caps at 8 parallel environments to avoid memory issues.

    MuJoCo license: Apache 2.0. JAX license: Apache 2.0.

    Args:
        use_gpu: If True, attempt GPU path via JAX. If False, force CPU.
    """

    def __init__(self, use_gpu: bool = True) -> None:
        self._use_gpu = use_gpu
        self._jax_available = self._check_jax()
        self._gpu_available = self._check_gpu() if self._jax_available else False

        mode = "gpu_jax" if (use_gpu and self._gpu_available) else "cpu_fallback"
        logger.info("mujoco_mjx_mode", mode=mode, jax_available=self._jax_available, gpu_available=self._gpu_available)

    @staticmethod
    def _check_jax() -> bool:
        """Return True if jax is importable."""
        try:
            import jax  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_gpu() -> bool:
        """Return True if JAX sees at least one GPU device."""
        try:
            import jax
            return any(d.platform == "gpu" for d in jax.devices())
        except Exception:
            return False

    async def simulate_parallel(
        self,
        model_xml: str,
        num_envs: int,
        num_steps: int,
        initial_states: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Run massively parallel physics simulation.

        Dispatches to GPU (MuJoCo MJX + JAX vmap) or CPU (standard MuJoCo)
        based on hardware availability and the use_gpu flag.

        Args:
            model_xml: MuJoCo XML model definition string.
            num_envs: Number of parallel environments (e.g., 4096 for RL training).
            num_steps: Simulation steps per environment.
            initial_states: Optional per-environment initial qpos/qvel dicts.

        Returns:
            Dict with:
            - observations: ndarray shape (num_envs, num_steps, obs_dim)
            - num_envs: actual number of environments simulated
            - num_steps: steps per environment
            - simulation_fps: total environment-steps per second
            - backend: "mjx_gpu" or "mujoco_cpu"
        """
        if self._gpu_available and self._use_gpu:
            return await self._simulate_mjx(model_xml, num_envs, num_steps, initial_states)
        return await self._simulate_cpu(model_xml, num_envs, num_steps, initial_states)

    async def _simulate_mjx(
        self,
        model_xml: str,
        num_envs: int,
        num_steps: int,
        initial_states: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """GPU path using MuJoCo MJX + JAX vmap for parallel simulation.

        Args:
            model_xml: MuJoCo XML model string.
            num_envs: Number of parallel environments.
            num_steps: Steps per environment.
            initial_states: Optional per-env initial states (unused in this baseline).

        Returns:
            Simulation result dict.
        """
        import asyncio

        import jax
        import jax.numpy as jnp
        import mujoco
        import mujoco.mjx as mjx

        model = mujoco.MjModel.from_xml_string(model_xml)
        mx = mjx.put_model(model)

        def step_fn(state: mjx.Data, _action: jnp.ndarray) -> tuple[mjx.Data, jnp.ndarray]:
            next_state = mjx.step(mx, state)
            obs = jnp.concatenate([next_state.qpos, next_state.qvel])
            return next_state, obs

        batch_step = jax.vmap(step_fn)
        data = mujoco.MjData(model)
        batch_state = jax.vmap(lambda _: mjx.put_data(model, data))(jnp.arange(num_envs))

        loop = asyncio.get_event_loop()
        start = loop.time()
        all_obs: list[np.ndarray] = []
        for _ in range(num_steps):
            actions = jnp.zeros((num_envs, model.nu))
            batch_state, obs_batch = batch_step(batch_state, actions)
            all_obs.append(np.array(obs_batch))

        elapsed = loop.time() - start
        logger.info(
            "mjx_simulation_complete",
            num_envs=num_envs,
            num_steps=num_steps,
            fps=round((num_envs * num_steps) / elapsed, 0),
        )
        return {
            "observations": np.stack(all_obs, axis=1),
            "num_envs": num_envs,
            "num_steps": num_steps,
            "simulation_fps": (num_envs * num_steps) / elapsed,
            "backend": "mjx_gpu",
        }

    async def _simulate_cpu(
        self,
        model_xml: str,
        num_envs: int,
        num_steps: int,
        initial_states: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        """CPU fallback using standard MuJoCo, capped at 8 parallel environments.

        Args:
            model_xml: MuJoCo XML model string.
            num_envs: Requested environments — capped at 8 for CPU.
            num_steps: Steps per environment.
            initial_states: Optional per-env initial states (unused in baseline).

        Returns:
            Simulation result dict with capped num_envs.
        """
        import asyncio

        import mujoco

        model = mujoco.MjModel.from_xml_string(model_xml)
        capped_envs = min(num_envs, 8)
        if capped_envs < num_envs:
            logger.warning("cpu_env_cap_applied", requested=num_envs, capped_to=capped_envs)

        loop = asyncio.get_event_loop()
        start = loop.time()
        results: list[np.ndarray] = []

        for _ in range(capped_envs):
            data = mujoco.MjData(model)
            env_obs: list[np.ndarray] = []
            for _ in range(num_steps):
                mujoco.mj_step(model, data)
                env_obs.append(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
            results.append(np.stack(env_obs))

        elapsed = loop.time() - start
        logger.info(
            "cpu_simulation_complete",
            num_envs=capped_envs,
            num_steps=num_steps,
            fps=round((capped_envs * num_steps) / elapsed, 0),
        )
        return {
            "observations": np.stack(results, axis=0),
            "num_envs": capped_envs,
            "num_steps": num_steps,
            "simulation_fps": (capped_envs * num_steps) / elapsed,
            "backend": "mujoco_cpu",
        }
