"""SQLAlchemy repository implementations for the physical AI service.

All repositories follow the AumOS pattern:
  - Tenant-scoped queries (RLS enforced at DB level via aumos-common)
  - Async session from FastAPI dependency injection
  - Standard CRUD operations with typed return values
"""

import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.observability import get_logger

from aumos_physical_ai.core.models import (
    JobStatus,
    RandomizationConfig,
    RoboticsJob,
    SensorFusionJob,
    SimToRealTransfer,
    TwinPipeline,
)

logger = get_logger(__name__)


class TwinPipelineRepository:
    """Repository for TwinPipeline persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, pipeline: TwinPipeline) -> TwinPipeline:
        """Persist a new TwinPipeline record."""
        self._session.add(pipeline)
        await self._session.flush()
        await self._session.refresh(pipeline)
        return pipeline

    async def get_by_id(self, pipeline_id: uuid.UUID) -> TwinPipeline | None:
        """Fetch a pipeline by primary key."""
        result = await self._session.execute(
            select(TwinPipeline).where(TwinPipeline.id == pipeline_id)
        )
        return result.scalar_one_or_none()

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TwinPipeline]:
        """List pipelines for a tenant, ordered by created_at descending."""
        result = await self._session.execute(
            select(TwinPipeline)
            .where(TwinPipeline.tenant_id == tenant_id)
            .order_by(TwinPipeline.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(result.scalars().all())

    async def update_status(self, pipeline_id: uuid.UUID, status: JobStatus) -> TwinPipeline:
        """Update only the status field of a pipeline."""
        await self._session.execute(
            update(TwinPipeline)
            .where(TwinPipeline.id == pipeline_id)
            .values(status=status)
        )
        await self._session.flush()
        pipeline = await self.get_by_id(pipeline_id)
        assert pipeline is not None
        return pipeline

    async def update(self, pipeline_id: uuid.UUID, **kwargs: Any) -> TwinPipeline:
        """Update arbitrary fields on a pipeline."""
        await self._session.execute(
            update(TwinPipeline).where(TwinPipeline.id == pipeline_id).values(**kwargs)
        )
        await self._session.flush()
        pipeline = await self.get_by_id(pipeline_id)
        assert pipeline is not None
        return pipeline


class RoboticsJobRepository:
    """Repository for RoboticsJob persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, job: RoboticsJob) -> RoboticsJob:
        """Persist a new RoboticsJob record."""
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def get_by_id(self, job_id: uuid.UUID) -> RoboticsJob | None:
        """Fetch a robotics job by primary key."""
        result = await self._session.execute(
            select(RoboticsJob).where(RoboticsJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def update_status(self, job_id: uuid.UUID, status: JobStatus) -> RoboticsJob:
        """Update only the status field of a job."""
        await self._session.execute(
            update(RoboticsJob).where(RoboticsJob.id == job_id).values(status=status)
        )
        await self._session.flush()
        job = await self.get_by_id(job_id)
        assert job is not None
        return job

    async def update(self, job_id: uuid.UUID, **kwargs: Any) -> RoboticsJob:
        """Update arbitrary fields on a robotics job."""
        await self._session.execute(
            update(RoboticsJob).where(RoboticsJob.id == job_id).values(**kwargs)
        )
        await self._session.flush()
        job = await self.get_by_id(job_id)
        assert job is not None
        return job


class SimToRealTransferRepository:
    """Repository for SimToRealTransfer persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, transfer: SimToRealTransfer) -> SimToRealTransfer:
        """Persist a new SimToRealTransfer record."""
        self._session.add(transfer)
        await self._session.flush()
        await self._session.refresh(transfer)
        return transfer

    async def get_by_id(self, transfer_id: uuid.UUID) -> SimToRealTransfer | None:
        """Fetch a transfer record by primary key."""
        result = await self._session.execute(
            select(SimToRealTransfer).where(SimToRealTransfer.id == transfer_id)
        )
        return result.scalar_one_or_none()

    async def update_status(self, transfer_id: uuid.UUID, status: JobStatus) -> SimToRealTransfer:
        """Update only the status field of a transfer."""
        await self._session.execute(
            update(SimToRealTransfer)
            .where(SimToRealTransfer.id == transfer_id)
            .values(status=status)
        )
        await self._session.flush()
        transfer = await self.get_by_id(transfer_id)
        assert transfer is not None
        return transfer

    async def update(self, transfer_id: uuid.UUID, **kwargs: Any) -> SimToRealTransfer:
        """Update arbitrary fields on a transfer record."""
        await self._session.execute(
            update(SimToRealTransfer).where(SimToRealTransfer.id == transfer_id).values(**kwargs)
        )
        await self._session.flush()
        transfer = await self.get_by_id(transfer_id)
        assert transfer is not None
        return transfer


class RandomizationConfigRepository:
    """Repository for RandomizationConfig persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, config: RandomizationConfig) -> RandomizationConfig:
        """Persist a new RandomizationConfig record."""
        self._session.add(config)
        await self._session.flush()
        await self._session.refresh(config)
        return config

    async def get_by_id(self, config_id: uuid.UUID) -> RandomizationConfig | None:
        """Fetch a config record by primary key."""
        result = await self._session.execute(
            select(RandomizationConfig).where(RandomizationConfig.id == config_id)
        )
        return result.scalar_one_or_none()

    async def update_status(self, config_id: uuid.UUID, status: JobStatus) -> RandomizationConfig:
        """Update only the status field of a config."""
        await self._session.execute(
            update(RandomizationConfig)
            .where(RandomizationConfig.id == config_id)
            .values(status=status)
        )
        await self._session.flush()
        config = await self.get_by_id(config_id)
        assert config is not None
        return config

    async def update(self, config_id: uuid.UUID, **kwargs: Any) -> RandomizationConfig:
        """Update arbitrary fields on a randomization config."""
        await self._session.execute(
            update(RandomizationConfig).where(RandomizationConfig.id == config_id).values(**kwargs)
        )
        await self._session.flush()
        config = await self.get_by_id(config_id)
        assert config is not None
        return config


class SensorFusionJobRepository:
    """Repository for SensorFusionJob persistence."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, job: SensorFusionJob) -> SensorFusionJob:
        """Persist a new SensorFusionJob record."""
        self._session.add(job)
        await self._session.flush()
        await self._session.refresh(job)
        return job

    async def get_by_id(self, job_id: uuid.UUID) -> SensorFusionJob | None:
        """Fetch a fusion job by primary key."""
        result = await self._session.execute(
            select(SensorFusionJob).where(SensorFusionJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def update_status(self, job_id: uuid.UUID, status: JobStatus) -> SensorFusionJob:
        """Update only the status field of a fusion job."""
        await self._session.execute(
            update(SensorFusionJob).where(SensorFusionJob.id == job_id).values(status=status)
        )
        await self._session.flush()
        job = await self.get_by_id(job_id)
        assert job is not None
        return job

    async def update(self, job_id: uuid.UUID, **kwargs: Any) -> SensorFusionJob:
        """Update arbitrary fields on a fusion job."""
        await self._session.execute(
            update(SensorFusionJob).where(SensorFusionJob.id == job_id).values(**kwargs)
        )
        await self._session.flush()
        job = await self.get_by_id(job_id)
        assert job is not None
        return job
