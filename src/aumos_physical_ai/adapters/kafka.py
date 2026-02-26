"""Kafka event publisher for physical AI lifecycle events.

Publishes structured Protobuf/JSON events to the AumOS event bus
for downstream services to consume (observability, audit, triggers).
"""

from typing import Any

from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

logger = get_logger(__name__)


class PhysicalAIEventPublisher(EventPublisher):
    """Kafka publisher scoped to physical AI topics.

    Extends the base EventPublisher with physical AI-specific
    topic helpers and structured event payloads.
    """

    async def publish_twin_created(
        self,
        tenant_id: str,
        pipeline_id: str,
        name: str,
    ) -> None:
        """Publish a digital twin pipeline creation event."""
        await self.publish(
            Topics.PHYSICAL_AI_TWIN_CREATED,
            {
                "event_type": "twin_pipeline_created",
                "tenant_id": tenant_id,
                "pipeline_id": pipeline_id,
                "name": name,
            },
        )
        logger.debug("Published twin_pipeline_created", pipeline_id=pipeline_id)

    async def publish_twin_completed(
        self,
        tenant_id: str,
        pipeline_id: str,
        output_uri: str | None,
        fidelity_score: float | None,
    ) -> None:
        """Publish a digital twin pipeline completion event."""
        await self.publish(
            Topics.PHYSICAL_AI_TWIN_COMPLETED,
            {
                "event_type": "twin_pipeline_completed",
                "tenant_id": tenant_id,
                "pipeline_id": pipeline_id,
                "output_uri": output_uri,
                "fidelity_score": fidelity_score,
            },
        )
        logger.debug("Published twin_pipeline_completed", pipeline_id=pipeline_id)

    async def publish_synth_completed(
        self,
        tenant_id: str,
        job_id: str,
        sensor_types: list[str],
        output_uri: str | None,
        realism_score: float | None,
    ) -> None:
        """Publish a sensor synthesis completion event."""
        await self.publish(
            Topics.PHYSICAL_AI_SYNTH_COMPLETED,
            {
                "event_type": "robotics_synth_completed",
                "tenant_id": tenant_id,
                "job_id": job_id,
                "sensor_types": sensor_types,
                "output_uri": output_uri,
                "realism_score": realism_score,
            },
        )
        logger.debug("Published robotics_synth_completed", job_id=job_id)

    async def publish_transfer_completed(
        self,
        tenant_id: str,
        transfer_id: str,
        adapted_model_uri: str | None,
        domain_gap_score: float | None,
    ) -> None:
        """Publish a sim-to-real transfer completion event."""
        await self.publish(
            Topics.PHYSICAL_AI_TRANSFER_COMPLETED,
            {
                "event_type": "sim2real_transfer_completed",
                "tenant_id": tenant_id,
                "transfer_id": transfer_id,
                "adapted_model_uri": adapted_model_uri,
                "domain_gap_score": domain_gap_score,
            },
        )
        logger.debug("Published sim2real_transfer_completed", transfer_id=transfer_id)

    async def publish_fusion_completed(
        self,
        tenant_id: str,
        job_id: str,
        fusion_strategy: str,
        output_uri: str | None,
        fusion_quality_score: float | None,
    ) -> None:
        """Publish a sensor fusion completion event."""
        await self.publish(
            Topics.PHYSICAL_AI_FUSION_COMPLETED,
            {
                "event_type": "sensor_fusion_completed",
                "tenant_id": tenant_id,
                "job_id": job_id,
                "fusion_strategy": fusion_strategy,
                "output_uri": output_uri,
                "fusion_quality_score": fusion_quality_score,
            },
        )
        logger.debug("Published sensor_fusion_completed", job_id=job_id)
