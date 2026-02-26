"""FastAPI router for physical AI endpoints.

All routes delegate to core services — no business logic lives here.
Auth and tenant context are injected via aumos-common dependencies.

Endpoint map:
  POST   /physical/twin/pipeline         — Create digital twin pipeline
  GET    /physical/twin/pipelines        — List pipelines
  POST   /physical/robotics/synthesize   — Synthesize robotics sensor data
  GET    /physical/robotics/jobs/{id}    — Robotics job status
  POST   /physical/sim2real/transfer     — Sim-to-real transfer
  POST   /physical/randomize             — Domain randomization
  POST   /physical/fusion/generate       — Multi-sensor fusion
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from aumos_common.auth import TenantContext, get_current_tenant
from aumos_common.database import get_db_session
from aumos_common.errors import NotFoundError, ValidationError
from aumos_common.observability import get_logger

from aumos_physical_ai.api.schemas import (
    CreateTwinPipelineRequest,
    CreateTwinPipelineResponse,
    DomainRandomizationRequest,
    DomainRandomizationResponse,
    ListTwinPipelinesResponse,
    RandomizationConfigResponse,
    RoboticsJobResponse,
    SensorFusionRequest,
    SensorFusionResponse,
    SimToRealResponse,
    SimToRealTransferRequest,
    SynthesizeRequest,
    SynthesizeResponse,
    TwinPipelineResponse,
)
from aumos_physical_ai.core.services import (
    DigitalTwinPipelineService,
    DomainRandomizationService,
    RoboticsSynthService,
    SensorFusionService,
    SimToRealService,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/physical", tags=["physical-ai"])


# ---------------------------------------------------------------------------
# Dependency factories
# ---------------------------------------------------------------------------


def get_twin_pipeline_service(session: AsyncSession = Depends(get_db_session)) -> DigitalTwinPipelineService:
    """Wire up DigitalTwinPipelineService with request-scoped session."""
    from aumos_physical_ai.adapters.blenderproc_client import BlenderProcClient
    from aumos_physical_ai.adapters.kafka import PhysicalAIEventPublisher
    from aumos_physical_ai.adapters.repositories import TwinPipelineRepository

    return DigitalTwinPipelineService(
        twin_backend=BlenderProcClient(),
        pipeline_repository=TwinPipelineRepository(session),
        event_publisher=PhysicalAIEventPublisher(),
    )


def get_robotics_synth_service(session: AsyncSession = Depends(get_db_session)) -> RoboticsSynthService:
    """Wire up RoboticsSynthService with request-scoped session."""
    from aumos_physical_ai.adapters.kafka import PhysicalAIEventPublisher
    from aumos_physical_ai.adapters.repositories import RoboticsJobRepository
    from aumos_physical_ai.adapters.sensor_simulator import SensorSimulator

    return RoboticsSynthService(
        sensor_simulator=SensorSimulator(),
        job_repository=RoboticsJobRepository(session),
        event_publisher=PhysicalAIEventPublisher(),
    )


def get_sim2real_service(session: AsyncSession = Depends(get_db_session)) -> SimToRealService:
    """Wire up SimToRealService with request-scoped session."""
    from aumos_physical_ai.adapters.kafka import PhysicalAIEventPublisher
    from aumos_physical_ai.adapters.repositories import SimToRealTransferRepository
    from aumos_physical_ai.adapters.sim2real_adapter import SimToRealAdapter

    return SimToRealService(
        sim2real_adapter=SimToRealAdapter(),
        transfer_repository=SimToRealTransferRepository(session),
        event_publisher=PhysicalAIEventPublisher(),
    )


def get_randomization_service(session: AsyncSession = Depends(get_db_session)) -> DomainRandomizationService:
    """Wire up DomainRandomizationService with request-scoped session."""
    from aumos_physical_ai.adapters.blenderproc_client import BlenderProcClient
    from aumos_physical_ai.adapters.kafka import PhysicalAIEventPublisher
    from aumos_physical_ai.adapters.repositories import RandomizationConfigRepository

    return DomainRandomizationService(
        randomizer=BlenderProcClient(),
        config_repository=RandomizationConfigRepository(session),
        event_publisher=PhysicalAIEventPublisher(),
    )


def get_sensor_fusion_service(session: AsyncSession = Depends(get_db_session)) -> SensorFusionService:
    """Wire up SensorFusionService with request-scoped session."""
    from aumos_physical_ai.adapters.kafka import PhysicalAIEventPublisher
    from aumos_physical_ai.adapters.repositories import SensorFusionJobRepository
    from aumos_physical_ai.adapters.sensor_simulator import SensorFusionEngine

    return SensorFusionService(
        fusion_engine=SensorFusionEngine(),
        job_repository=SensorFusionJobRepository(session),
        event_publisher=PhysicalAIEventPublisher(),
    )


# ---------------------------------------------------------------------------
# Digital Twin Pipeline endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/twin/pipeline",
    response_model=CreateTwinPipelineResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_twin_pipeline(
    request: CreateTwinPipelineRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: DigitalTwinPipelineService = Depends(get_twin_pipeline_service),
) -> CreateTwinPipelineResponse:
    """Create and execute a digital twin data pipeline.

    Provisions the scene in the simulation backend and runs the pipeline.
    The response includes fidelity score and output data URI once complete.
    """
    logger.info(
        "Create twin pipeline request",
        tenant_id=str(tenant.tenant_id),
        name=request.name,
    )
    try:
        pipeline = await service.create_pipeline(
            tenant_id=tenant.tenant_id,
            name=request.name,
            scene_config=request.scene_config,
        )
        return CreateTwinPipelineResponse(pipeline=TwinPipelineResponse.model_validate(pipeline))
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get(
    "/twin/pipelines",
    response_model=ListTwinPipelinesResponse,
)
async def list_twin_pipelines(
    tenant: TenantContext = Depends(get_current_tenant),
    service: DigitalTwinPipelineService = Depends(get_twin_pipeline_service),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ListTwinPipelinesResponse:
    """List digital twin pipelines for the current tenant."""
    logger.info("List twin pipelines request", tenant_id=str(tenant.tenant_id))
    pipelines = await service.list_pipelines(
        tenant_id=tenant.tenant_id,
        limit=limit,
        offset=offset,
    )
    return ListTwinPipelinesResponse(
        pipelines=[TwinPipelineResponse.model_validate(p) for p in pipelines],
        total=len(pipelines),
    )


# ---------------------------------------------------------------------------
# Robotics Sensor Synthesis endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/robotics/synthesize",
    response_model=SynthesizeResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def synthesize_sensor_data(
    request: SynthesizeRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: RoboticsSynthService = Depends(get_robotics_synth_service),
) -> SynthesizeResponse:
    """Synthesize multi-modal robotics sensor data.

    Generates LiDAR point clouds, camera frames, IMU readings, and other
    sensor modalities from simulation environments.
    """
    logger.info(
        "Synthesize sensor data request",
        tenant_id=str(tenant.tenant_id),
        sensor_types=[s.value for s in request.sensor_types],
    )
    try:
        job = await service.synthesize(
            tenant_id=tenant.tenant_id,
            sensor_types=[s.value for s in request.sensor_types],
            synthesis_config=request.synthesis_config,
        )
        return SynthesizeResponse(job=RoboticsJobResponse.model_validate(job))
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get(
    "/robotics/jobs/{job_id}",
    response_model=SynthesizeResponse,
)
async def get_robotics_job(
    job_id: uuid.UUID,
    tenant: TenantContext = Depends(get_current_tenant),
    service: RoboticsSynthService = Depends(get_robotics_synth_service),
) -> SynthesizeResponse:
    """Get the status and results of a robotics synthesis job."""
    logger.info("Get robotics job request", job_id=str(job_id), tenant_id=str(tenant.tenant_id))
    try:
        job = await service.get_job(job_id=job_id, tenant_id=tenant.tenant_id)
        return SynthesizeResponse(job=RoboticsJobResponse.model_validate(job))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Sim-to-Real Transfer endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/sim2real/transfer",
    response_model=SimToRealResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def sim_to_real_transfer(
    request: SimToRealTransferRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: SimToRealService = Depends(get_sim2real_service),
) -> SimToRealResponse:
    """Execute sim-to-real domain adaptation.

    Bridges the domain gap between simulation-trained models and
    real-world deployment using the specified adaptation method.
    """
    logger.info(
        "Sim-to-real transfer request",
        tenant_id=str(tenant.tenant_id),
        transfer_method=request.transfer_method.value,
        source_model_id=request.source_model_id,
    )
    try:
        transfer = await service.transfer(
            tenant_id=tenant.tenant_id,
            source_model_id=request.source_model_id,
            transfer_method=request.transfer_method.value,
            transfer_config=request.transfer_config,
        )
        from aumos_physical_ai.api.schemas import SimToRealTransferResponse

        return SimToRealResponse(transfer=SimToRealTransferResponse.model_validate(transfer))
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Domain Randomization endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/randomize",
    response_model=DomainRandomizationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def domain_randomize(
    request: DomainRandomizationRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: DomainRandomizationService = Depends(get_randomization_service),
) -> DomainRandomizationResponse:
    """Apply domain randomization to generate diverse training data.

    Randomly varies lighting, textures, object poses, sensor noise,
    and physics to reduce the simulation-reality gap.
    """
    logger.info(
        "Domain randomization request",
        tenant_id=str(tenant.tenant_id),
        name=request.name,
        num_variations=request.randomization_params.get("num_variations"),
    )
    try:
        config = await service.randomize(
            tenant_id=tenant.tenant_id,
            name=request.name,
            randomization_params=request.randomization_params,
        )
        return DomainRandomizationResponse(config=RandomizationConfigResponse.model_validate(config))
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Sensor Fusion endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/fusion/generate",
    response_model=SensorFusionResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def generate_sensor_fusion(
    request: SensorFusionRequest,
    tenant: TenantContext = Depends(get_current_tenant),
    service: SensorFusionService = Depends(get_sensor_fusion_service),
) -> SensorFusionResponse:
    """Generate a multi-sensor fused dataset.

    Temporally aligns and spatially calibrates multiple sensor streams
    (LiDAR, camera, IMU, radar) into a unified multi-modal dataset.
    """
    logger.info(
        "Sensor fusion request",
        tenant_id=str(tenant.tenant_id),
        fusion_strategy=request.fusion_strategy.value,
        num_streams=len(request.sensor_streams),
    )
    try:
        job = await service.generate_fusion(
            tenant_id=tenant.tenant_id,
            sensor_streams=request.sensor_streams,
            fusion_strategy=request.fusion_strategy.value,
            fusion_config=request.fusion_config,
        )
        from aumos_physical_ai.api.schemas import SensorFusionJobResponse

        return SensorFusionResponse(job=SensorFusionJobResponse.model_validate(job))
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
