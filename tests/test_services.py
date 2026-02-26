"""Unit tests for physical AI core services.

Tests use mock adapters conforming to the Protocol interfaces,
ensuring services are tested in isolation from real infrastructure.
"""

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aumos_physical_ai.core.models import JobStatus
from aumos_physical_ai.core.services import (
    DigitalTwinPipelineService,
    DomainRandomizationService,
    RoboticsSynthService,
    SensorFusionService,
    SimToRealService,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tenant_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture()
def mock_pipeline_repo() -> MagicMock:
    repo = MagicMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.update_status = AsyncMock()
    repo.list_by_tenant = AsyncMock(return_value=[])
    return repo


@pytest.fixture()
def mock_robotics_repo() -> MagicMock:
    repo = MagicMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.update_status = AsyncMock()
    repo.get_by_id = AsyncMock()
    return repo


@pytest.fixture()
def mock_transfer_repo() -> MagicMock:
    repo = MagicMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.update_status = AsyncMock()
    return repo


@pytest.fixture()
def mock_config_repo() -> MagicMock:
    repo = MagicMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.update_status = AsyncMock()
    return repo


@pytest.fixture()
def mock_fusion_repo() -> MagicMock:
    repo = MagicMock()
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.update_status = AsyncMock()
    return repo


@pytest.fixture()
def mock_event_publisher() -> MagicMock:
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    return publisher


def _make_mock_record(status: JobStatus = JobStatus.COMPLETED, **kwargs: Any) -> MagicMock:
    """Create a mock ORM record with standard fields."""
    record = MagicMock()
    record.id = uuid.uuid4()
    record.status = status
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


# ---------------------------------------------------------------------------
# DigitalTwinPipelineService tests
# ---------------------------------------------------------------------------


class TestDigitalTwinPipelineService:
    """Tests for DigitalTwinPipelineService."""

    @pytest.fixture()
    def mock_backend(self) -> MagicMock:
        backend = MagicMock()
        backend.create_pipeline = AsyncMock(
            return_value={"pipeline_id": "bp-test-001", "status": "provisioned"}
        )
        backend.run_pipeline = AsyncMock(
            return_value={
                "output_uri": "s3://aumos-pai/output/test",
                "fidelity_score": 0.90,
                "sync_lag_ms": 10.0,
                "frame_count": 1000,
            }
        )
        return backend

    @pytest.fixture()
    def service(
        self,
        mock_backend: MagicMock,
        mock_pipeline_repo: MagicMock,
        mock_event_publisher: MagicMock,
    ) -> DigitalTwinPipelineService:
        return DigitalTwinPipelineService(
            twin_backend=mock_backend,
            pipeline_repository=mock_pipeline_repo,
            event_publisher=mock_event_publisher,
        )

    @pytest.mark.asyncio()
    async def test_create_pipeline_success(
        self,
        service: DigitalTwinPipelineService,
        mock_pipeline_repo: MagicMock,
        mock_event_publisher: MagicMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Creating a pipeline with valid scene_config succeeds."""
        pending_record = _make_mock_record(status=JobStatus.PENDING)
        running_record = _make_mock_record(status=JobStatus.RUNNING)
        completed_record = _make_mock_record(
            status=JobStatus.COMPLETED,
            output_uri="s3://aumos-pai/output/test",
            fidelity_score=0.90,
        )

        mock_pipeline_repo.create.return_value = pending_record
        mock_pipeline_repo.update_status.return_value = running_record
        mock_pipeline_repo.update.return_value = completed_record

        result = await service.create_pipeline(
            tenant_id=tenant_id,
            name="Test Pipeline",
            scene_config={"world_model": "warehouse_v1", "simulation_steps": 1000},
        )

        assert result.status == JobStatus.COMPLETED
        assert mock_event_publisher.publish.call_count >= 1

    @pytest.mark.asyncio()
    async def test_create_pipeline_missing_world_model(
        self,
        service: DigitalTwinPipelineService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Creating a pipeline without world_model raises ValidationError."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError, match="world_model"):
            await service.create_pipeline(
                tenant_id=tenant_id,
                name="Bad Pipeline",
                scene_config={"physics_engine": "isaac_sim"},
            )

    @pytest.mark.asyncio()
    async def test_list_pipelines(
        self,
        service: DigitalTwinPipelineService,
        mock_pipeline_repo: MagicMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Listing pipelines returns repository results."""
        pipelines = [_make_mock_record() for _ in range(3)]
        mock_pipeline_repo.list_by_tenant.return_value = pipelines

        result = await service.list_pipelines(tenant_id=tenant_id)
        assert len(result) == 3
        mock_pipeline_repo.list_by_tenant.assert_awaited_once_with(
            tenant_id=tenant_id, limit=50, offset=0
        )


# ---------------------------------------------------------------------------
# RoboticsSynthService tests
# ---------------------------------------------------------------------------


class TestRoboticsSynthService:
    """Tests for RoboticsSynthService."""

    @pytest.fixture()
    def mock_simulator(self) -> MagicMock:
        simulator = MagicMock()
        simulator.synthesize = AsyncMock(
            return_value={
                "output_uri": "s3://aumos-pai/synth/bundle",
                "frame_count": 500,
                "total_points": 32_000_000,
                "realism_score": 0.85,
            }
        )
        return simulator

    @pytest.fixture()
    def service(
        self,
        mock_simulator: MagicMock,
        mock_robotics_repo: MagicMock,
        mock_event_publisher: MagicMock,
    ) -> RoboticsSynthService:
        return RoboticsSynthService(
            sensor_simulator=mock_simulator,
            job_repository=mock_robotics_repo,
            event_publisher=mock_event_publisher,
        )

    @pytest.mark.asyncio()
    async def test_synthesize_success(
        self,
        service: RoboticsSynthService,
        mock_robotics_repo: MagicMock,
        mock_event_publisher: MagicMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Synthesizing valid sensor types succeeds."""
        pending = _make_mock_record(status=JobStatus.PENDING)
        running = _make_mock_record(status=JobStatus.RUNNING)
        completed = _make_mock_record(
            status=JobStatus.COMPLETED,
            frame_count=500,
            realism_score=0.85,
        )

        mock_robotics_repo.create.return_value = pending
        mock_robotics_repo.update_status.return_value = running
        mock_robotics_repo.update.return_value = completed

        result = await service.synthesize(
            tenant_id=tenant_id,
            sensor_types=["lidar", "camera"],
            synthesis_config={"scene": "urban", "num_frames": 500},
        )

        assert result.status == JobStatus.COMPLETED

    @pytest.mark.asyncio()
    async def test_synthesize_invalid_sensor_type(
        self,
        service: RoboticsSynthService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Synthesizing an unsupported sensor type raises ValidationError."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError, match="Unsupported sensor types"):
            await service.synthesize(
                tenant_id=tenant_id,
                sensor_types=["lidar", "telepathy"],
                synthesis_config={},
            )

    @pytest.mark.asyncio()
    async def test_synthesize_empty_sensor_types(
        self,
        service: RoboticsSynthService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Synthesizing with empty sensor_types raises ValidationError."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError):
            await service.synthesize(
                tenant_id=tenant_id,
                sensor_types=[],
                synthesis_config={},
            )


# ---------------------------------------------------------------------------
# SimToRealService tests
# ---------------------------------------------------------------------------


class TestSimToRealService:
    """Tests for SimToRealService."""

    @pytest.fixture()
    def mock_adapter(self) -> MagicMock:
        adapter = MagicMock()
        adapter.transfer = AsyncMock(
            return_value={
                "adapted_model_uri": "s3://aumos-pai/models/adapted_v1",
                "sim_accuracy": 0.91,
                "real_accuracy": 0.84,
                "domain_gap_score": 0.07,
                "adaptation_epochs": 50,
            }
        )
        return adapter

    @pytest.fixture()
    def service(
        self,
        mock_adapter: MagicMock,
        mock_transfer_repo: MagicMock,
        mock_event_publisher: MagicMock,
    ) -> SimToRealService:
        return SimToRealService(
            sim2real_adapter=mock_adapter,
            transfer_repository=mock_transfer_repo,
            event_publisher=mock_event_publisher,
        )

    @pytest.mark.asyncio()
    async def test_transfer_success(
        self,
        service: SimToRealService,
        mock_transfer_repo: MagicMock,
        mock_event_publisher: MagicMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Valid transfer method executes successfully."""
        pending = _make_mock_record(status=JobStatus.PENDING)
        running = _make_mock_record(status=JobStatus.RUNNING)
        completed = _make_mock_record(
            status=JobStatus.COMPLETED,
            real_accuracy=0.84,
            domain_gap_score=0.07,
        )

        mock_transfer_repo.create.return_value = pending
        mock_transfer_repo.update_status.return_value = running
        mock_transfer_repo.update.return_value = completed

        result = await service.transfer(
            tenant_id=tenant_id,
            source_model_id="model-registry/perception-v3",
            transfer_method="domain_adaptation",
            transfer_config={"adaptation_epochs": 50},
        )

        assert result.status == JobStatus.COMPLETED
        assert mock_event_publisher.publish.call_count >= 1

    @pytest.mark.asyncio()
    async def test_transfer_invalid_method(
        self,
        service: SimToRealService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Unsupported transfer method raises ValidationError."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError, match="Unsupported transfer_method"):
            await service.transfer(
                tenant_id=tenant_id,
                source_model_id="model-v1",
                transfer_method="magic_transfer",
                transfer_config={},
            )


# ---------------------------------------------------------------------------
# SensorFusionService tests
# ---------------------------------------------------------------------------


class TestSensorFusionService:
    """Tests for SensorFusionService."""

    @pytest.fixture()
    def mock_engine(self) -> MagicMock:
        engine = MagicMock()
        engine.fuse = AsyncMock(
            return_value={
                "output_uri": "s3://aumos-pai/fusion/result",
                "output_format": "rosbag2",
                "temporal_alignment_score": 0.95,
                "spatial_calibration_score": 0.92,
                "fusion_quality_score": 0.93,
            }
        )
        return engine

    @pytest.fixture()
    def service(
        self,
        mock_engine: MagicMock,
        mock_fusion_repo: MagicMock,
        mock_event_publisher: MagicMock,
    ) -> SensorFusionService:
        return SensorFusionService(
            fusion_engine=mock_engine,
            job_repository=mock_fusion_repo,
            event_publisher=mock_event_publisher,
        )

    @pytest.mark.asyncio()
    async def test_generate_fusion_success(
        self,
        service: SensorFusionService,
        mock_fusion_repo: MagicMock,
        mock_event_publisher: MagicMock,
        tenant_id: uuid.UUID,
    ) -> None:
        """Valid fusion request with 2+ streams succeeds."""
        pending = _make_mock_record(status=JobStatus.PENDING)
        running = _make_mock_record(status=JobStatus.RUNNING)
        completed = _make_mock_record(
            status=JobStatus.COMPLETED,
            fusion_quality_score=0.93,
        )

        mock_fusion_repo.create.return_value = pending
        mock_fusion_repo.update_status.return_value = running
        mock_fusion_repo.update.return_value = completed

        streams = [
            {"sensor_type": "lidar", "data_uri": "s3://aumos-pai/lidar/run1"},
            {"sensor_type": "camera", "data_uri": "s3://aumos-pai/camera/run1"},
        ]

        result = await service.generate_fusion(
            tenant_id=tenant_id,
            sensor_streams=streams,
            fusion_strategy="kalman_filter",
            fusion_config={"temporal_window_ms": 50},
        )

        assert result.status == JobStatus.COMPLETED

    @pytest.mark.asyncio()
    async def test_generate_fusion_single_stream_rejected(
        self,
        service: SensorFusionService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Fusion with fewer than 2 streams raises ValidationError."""
        from aumos_common.errors import ValidationError

        with pytest.raises(ValidationError, match="at least 2"):
            await service.generate_fusion(
                tenant_id=tenant_id,
                sensor_streams=[{"sensor_type": "lidar", "data_uri": "s3://test"}],
                fusion_strategy="kalman_filter",
                fusion_config={},
            )

    @pytest.mark.asyncio()
    async def test_generate_fusion_invalid_strategy(
        self,
        service: SensorFusionService,
        tenant_id: uuid.UUID,
    ) -> None:
        """Unsupported fusion strategy raises ValidationError."""
        from aumos_common.errors import ValidationError

        streams = [
            {"sensor_type": "lidar", "data_uri": "s3://test/lidar"},
            {"sensor_type": "camera", "data_uri": "s3://test/camera"},
        ]

        with pytest.raises(ValidationError, match="Unsupported fusion_strategy"):
            await service.generate_fusion(
                tenant_id=tenant_id,
                sensor_streams=streams,
                fusion_strategy="magic_fusion",
                fusion_config={},
            )
