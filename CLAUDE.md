# aumos-physical-ai — Claude Development Guide

## Service Overview

Physical AI service: digital twin pipelines, robotics sensor synthesis,
sim-to-real transfer learning, domain randomization, multi-sensor fusion.

- **Package**: `aumos_physical_ai`
- **Table prefix**: `pai_`
- **Env prefix**: `AUMOS_PHYSICAL_AI_`
- **Port**: 8000

## Architecture

Hexagonal architecture — three layers, strict dependency direction:

```
api/ → core/ → adapters/
```

- `api/`: FastAPI routers + Pydantic schemas. No business logic.
- `core/`: Services + Protocol interfaces + ORM models. No I/O.
- `adapters/`: DB repositories, Kafka, BlenderProc, sensor simulator.

Services depend on Protocol interfaces only — never on concrete adapters.

## Key Services

| Service | Protocol | Adapter |
|---|---|---|
| `DigitalTwinPipelineService` | `DigitalTwinBackendProtocol` | `BlenderProcClient` |
| `RoboticsSynthService` | `SensorSimulatorProtocol` | `SensorSimulator` |
| `SimToRealService` | `SimToRealAdapterProtocol` | `SimToRealAdapter` |
| `DomainRandomizationService` | `DomainRandomizerProtocol` | `BlenderProcClient` |
| `SensorFusionService` | `SensorFusionEngineProtocol` | `SensorFusionEngine` |

## Database Models

All models extend `AumOSModel` (id, tenant_id, created_at, updated_at):

- `TwinPipeline` → `pai_twin_pipelines`
- `RoboticsJob` → `pai_robotics_jobs`
- `SimToRealTransfer` → `pai_sim2real_transfers`
- `RandomizationConfig` → `pai_randomization_configs`
- `SensorFusionJob` → `pai_sensor_fusion_jobs`

## Events Published (Kafka)

- `Topics.PHYSICAL_AI_TWIN_CREATED` / `PHYSICAL_AI_TWIN_COMPLETED`
- `Topics.PHYSICAL_AI_SYNTH_STARTED` / `PHYSICAL_AI_SYNTH_COMPLETED`
- `Topics.PHYSICAL_AI_TRANSFER_COMPLETED`
- `Topics.PHYSICAL_AI_RANDOMIZATION_COMPLETED`
- `Topics.PHYSICAL_AI_FUSION_COMPLETED`

## External Dependencies

- **BlenderProc** (`blenderproc_client.py`): Simulation rendering + domain randomization
- **Sensor Simulator** (`sensor_simulator.py`): Sensor data generation + fusion
- **Model Registry** (`sim2real_adapter.py`): Model I/O for transfer learning

All external calls are behind Protocol interfaces — swap implementations by
changing the dependency factory in `router.py`.

## Coding Conventions

- Python 3.11+ type hints on all signatures
- Pydantic for all API schemas (strict validation at boundary)
- `@field_validator` for cross-field constraints in schemas
- Service methods: create record → update to RUNNING → do work → update to COMPLETED/FAILED
- Always publish Kafka events on state transitions
- `get_logger(__name__)` for structured logging in every module

## Common Tasks

```bash
# Add a new sensor type
1. Add to SensorType enum in core/models.py
2. Add synthesis logic branch in sensor_simulator.py
3. No schema changes needed (sensor_types is a JSON list)

# Add a new transfer method
1. Add to TransferMethod enum in core/models.py
2. Add method-specific logic in sim2real_adapter.py
3. Update supported_methods set in SimToRealService.transfer()

# Add a new fusion strategy
1. Add to FusionStrategy enum in core/models.py
2. Add to strategy_quality_map in SensorFusionEngine.fuse()
3. Update supported_strategies set in SensorFusionService.generate_fusion()
```
