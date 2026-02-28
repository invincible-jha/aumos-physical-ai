"""Foundation model adapter for VLA (Vision-Language-Action) robot control.

GAP-353: Foundation Model Integration.
Implements the VLAModelAdapter Protocol with OctoAdapter (UC Berkeley, Apache 2.0).
Octo is a transformer robot manipulation model trained on 800K+ demonstrations from
the Open X-Embodiment dataset. Delegates to aumos-llm-serving when serving_url is set.

Octo weights: huggingface.co/rail-berkeley/octo-small
Local inference requires GPU with >=8GB VRAM.
"""
from __future__ import annotations

import base64
from typing import Protocol

import httpx
import numpy as np
from pydantic import BaseModel

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ActionPrediction(BaseModel):
    """Predicted robot action from a VLA foundation model.

    Attributes:
        joint_positions: Target joint angles in radians, shape (num_joints,).
        gripper_command: Gripper open/close command, range [0.0 closed, 1.0 open].
        end_effector_pose: [x, y, z, qw, qx, qy, qz] end-effector target pose.
        confidence_score: Model confidence in [0, 1].
        model_id: Identifier of the VLA model that produced this prediction.
    """

    joint_positions: list[float]
    gripper_command: float
    end_effector_pose: list[float]  # [x, y, z, qw, qx, qy, qz]
    confidence_score: float
    model_id: str


class VLAModelAdapter(Protocol):
    """Protocol for vision-language-action foundation model adapters.

    All VLA adapters (Octo, GR00T, RT-2) must conform to this interface.
    """

    async def predict_action(
        self,
        camera_images: list[np.ndarray],
        language_instruction: str,
        proprioceptive_state: np.ndarray,
    ) -> ActionPrediction:
        """Predict robot action from visual observations and language.

        Args:
            camera_images: RGB frames, each shape (H, W, 3) uint8.
            language_instruction: Task description (e.g., "pick up the red cup").
            proprioceptive_state: Current joint state, shape (joint_dim,).

        Returns:
            ActionPrediction with joint targets and gripper command.
        """
        ...


class OctoAdapter:
    """Octo (UC Berkeley) VLA model adapter.

    Octo is a transformer robot manipulation model trained on 800K demonstrations
    from the Open X-Embodiment dataset.

    Weights: huggingface.co/rail-berkeley/octo-small-1.5
    License: Apache 2.0. pip install octo-robot (optional, requires GPU).

    When serving_url is set, inference is delegated to aumos-llm-serving.
    When serving_url is None, runs inference locally (requires GPU with >=8GB VRAM).

    Args:
        model_path: HuggingFace model path or local directory.
        http_client: Shared async HTTP client for serving-mode requests.
        serving_url: Base URL of aumos-llm-serving; enables remote inference.
    """

    MODEL_ID = "octo-small-1.5"

    def __init__(
        self,
        model_path: str = "hf://rail-berkeley/octo-small-1.5",
        http_client: httpx.AsyncClient | None = None,
        serving_url: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._client = http_client
        self._serving_url = serving_url
        self._model: object | None = None

    async def predict_action(
        self,
        camera_images: list[np.ndarray],
        language_instruction: str,
        proprioceptive_state: np.ndarray,
    ) -> ActionPrediction:
        """Predict robot action from camera images and language instruction.

        Routes to remote serving or local GPU inference depending on configuration.

        Args:
            camera_images: RGB frames shape (H, W, 3) each.
            language_instruction: Task description (e.g., "pick up the red cup").
            proprioceptive_state: Current joint state shape (joint_dim,).

        Returns:
            ActionPrediction with joint targets and gripper command.
        """
        if self._serving_url and self._client:
            return await self._predict_via_serving(camera_images, language_instruction, proprioceptive_state)
        return await self._predict_local(camera_images, language_instruction, proprioceptive_state)

    async def _predict_via_serving(
        self,
        images: list[np.ndarray],
        instruction: str,
        state: np.ndarray,
    ) -> ActionPrediction:
        """Delegate inference to aumos-llm-serving over HTTP.

        Args:
            images: Camera frames to encode and send.
            instruction: Language task instruction.
            state: Joint state array.

        Returns:
            ActionPrediction deserialized from serving response.
        """
        assert self._client is not None  # checked by caller
        encoded = [base64.b64encode(img.tobytes()).decode() for img in images]
        response = await self._client.post(
            f"{self._serving_url}/api/v1/physical-ai/foundation-model/infer",
            json={
                "model_id": self.MODEL_ID,
                "camera_images_b64": encoded,
                "language_instruction": instruction,
                "proprioceptive_state": state.tolist(),
            },
            timeout=5.0,
        )
        response.raise_for_status()
        logger.info("octo_serving_inference_complete", model_id=self.MODEL_ID)
        return ActionPrediction(**response.json())

    async def _predict_local(
        self,
        images: list[np.ndarray],
        instruction: str,
        state: np.ndarray,
    ) -> ActionPrediction:
        """Run local Octo inference on GPU.

        Requires octo-robot package and GPU with >=8GB VRAM.

        Args:
            images: Camera frames to stack for batch inference.
            instruction: Language task instruction.
            state: Joint state array.

        Returns:
            ActionPrediction from local model.

        Raises:
            RuntimeError: If octo-robot is not installed.
        """
        if self._model is None:
            try:
                from octo.model.octo_model import OctoModel  # type: ignore[import]
                self._model = OctoModel.load_pretrained(self._model_path)
                logger.info("octo_model_loaded", path=self._model_path)
            except ImportError:
                logger.error("octo_not_installed", advice="pip install octo-robot")
                raise RuntimeError(
                    "Octo not installed. Install octo-robot or configure serving_url for remote inference."
                )

        image_batch = np.stack(images)[np.newaxis]  # (1, num_cams, H, W, 3)
        task = self._model.create_tasks(texts=[instruction])  # type: ignore[union-attr]
        obs = {"image_primary": image_batch, "proprio": state[np.newaxis]}
        actions = self._model.sample_actions(obs, task, rng=None)  # type: ignore[union-attr]
        action_array: np.ndarray = actions[0]

        return ActionPrediction(
            joint_positions=action_array[:7].tolist(),
            gripper_command=float(action_array[7]) if len(action_array) > 7 else 0.0,
            end_effector_pose=action_array[8:15].tolist() if len(action_array) > 15 else [0.0] * 7,
            confidence_score=0.85,  # Octo does not expose a confidence score natively
            model_id=self.MODEL_ID,
        )
