"""ROS 2 bridge — connects AumOS Physical AI to robot middleware.

GAP-355: ROS 2 Bridge.
Subscribes to ROS 2 sensor topics (camera, lidar, IMU), calls the AumOS
foundation model API for action prediction, and publishes action commands.

ROS 2 (Robot Operating System 2) is the de facto middleware for production robotics
with 100,000+ deployments. This bridge enables sim-to-real transfer — models trained
in AumOS simulation can be deployed on real hardware via ROS 2 topics.

rclpy is part of ROS 2 Jazzy (not pip-installable). The bridge operates in stub
mode when rclpy is unavailable, allowing the service to run in non-ROS environments.
rclpy license: Apache 2.0.
"""
from __future__ import annotations

import asyncio
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)


class ROS2Bridge:
    """ROS 2 DDS bridge for real robot deployment.

    Subscribes to ROS 2 sensor topics and publishes action commands.
    Calls the AumOS Physical AI foundation model API for action prediction.

    Requires rclpy (part of ROS 2 Jazzy installation).
    Runs in stub mode if rclpy is unavailable — all methods return without error.

    Args:
        node_name: ROS 2 node name for this bridge.
        aumos_api_url: Base URL of the AumOS Physical AI service.
        api_token: Bearer token for authenticated API calls.
    """

    # ROS 2 topic defaults
    CAMERA_TOPIC = "/camera/rgb/image_raw"
    CMD_VEL_TOPIC = "/cmd_vel"
    LANGUAGE_INSTRUCTION = "navigate to goal"

    def __init__(
        self,
        node_name: str = "aumos_physical_ai_bridge",
        aumos_api_url: str = "http://localhost:8000",
        api_token: str = "",
    ) -> None:
        self._node_name = node_name
        self._api_url = aumos_api_url
        self._api_token = api_token
        self._rclpy_available = self._check_rclpy()
        self._cmd_pub: Any = None

    @staticmethod
    def _check_rclpy() -> bool:
        """Return True if rclpy is importable (ROS 2 is installed)."""
        try:
            import rclpy  # noqa: F401
            return True
        except ImportError:
            return False

    async def start(self) -> None:
        """Start the ROS 2 bridge node.

        Initializes rclpy, creates subscriptions and publishers, and begins
        spinning the executor. When rclpy is unavailable, logs a warning and
        returns immediately so the service starts successfully in non-ROS environments.
        """
        if not self._rclpy_available:
            logger.warning(
                "rclpy_not_available",
                mode="stub",
                advice="Install ROS 2 Jazzy to enable the ROS bridge",
            )
            return

        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import Image

        rclpy.init()
        node = Node(self._node_name)

        node.create_subscription(Image, self.CAMERA_TOPIC, self._camera_callback, 10)
        self._cmd_pub = node.create_publisher(Twist, self.CMD_VEL_TOPIC, 10)

        logger.info(
            "ros2_bridge_started",
            node=self._node_name,
            camera_topic=self.CAMERA_TOPIC,
            cmd_topic=self.CMD_VEL_TOPIC,
        )

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        loop = asyncio.get_event_loop()

        while rclpy.ok():
            await loop.run_in_executor(None, executor.spin_once, 0.1)

    async def _camera_callback(self, msg: Any) -> None:
        """Forward a ROS 2 camera frame to the AumOS foundation model API.

        Converts the ROS Image message to numpy, calls the predict endpoint,
        then publishes the resulting action as a Twist message.

        Args:
            msg: ROS 2 sensor_msgs/Image message.
        """
        import httpx
        import numpy as np

        image_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self._api_url}/api/v1/physical-ai/foundation-model/predict",
                    json={
                        "camera_images_b64": [image_data.tobytes().hex()],
                        "language_instruction": self.LANGUAGE_INSTRUCTION,
                        "proprioceptive_state": [],
                    },
                    headers={"Authorization": f"Bearer {self._api_token}"},
                    timeout=2.0,
                )
                if response.status_code == 200:
                    await self._publish_action(response.json())
                else:
                    logger.warning("foundation_model_api_error", status=response.status_code)
            except httpx.TimeoutException:
                logger.warning("foundation_model_api_timeout")

    async def _publish_action(self, action: dict[str, Any]) -> None:
        """Convert ActionPrediction dict to ROS 2 Twist and publish.

        Maps joint_positions[0] to linear.x (forward velocity) as a baseline.
        Production deployments should map the full joint action to the robot's
        specific actuator command interface.

        Args:
            action: ActionPrediction dict from the foundation model API.
        """
        from geometry_msgs.msg import Twist

        twist = Twist()
        joint_positions: list[float] = action.get("joint_positions", [0.0] * 7)
        twist.linear.x = float(joint_positions[0]) if joint_positions else 0.0
        self._cmd_pub.publish(twist)
        logger.debug("ros2_action_published", linear_x=twist.linear.x)

    async def shutdown(self) -> None:
        """Shut down the ROS 2 bridge and rclpy context."""
        if not self._rclpy_available:
            return

        try:
            import rclpy
            rclpy.shutdown()
            logger.info("ros2_bridge_shutdown", node=self._node_name)
        except Exception as exc:
            logger.warning("ros2_shutdown_error", reason=str(exc))
