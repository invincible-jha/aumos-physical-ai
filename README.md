# aumos-physical-ai

AumOS Physical AI — digital twin data pipelines, robotics sensor synthesis,
sim-to-real transfer learning, domain randomization, and multi-sensor fusion.

## Overview

`aumos-physical-ai` is the AumOS service for physical world AI workflows.
It bridges the simulation-reality gap by generating high-fidelity synthetic
sensor data, adapting simulation-trained models for real-world deployment,
and fusing multi-modal sensor streams into unified training datasets.

## Key Capabilities

| Capability | Description |
|---|---|
| Digital Twin Pipelines | BlenderProc-backed physics-accurate scene simulation |
| Robotics Sensor Synthesis | LiDAR, camera, IMU, radar synthetic data generation |
| Sim-to-Real Transfer | DANN, fine-tuning, meta-learning, CycleGAN domain adaptation |
| Domain Randomization | Lighting, texture, pose, noise variation for robustness |
| Multi-Sensor Fusion | Temporal alignment + spatial calibration of sensor streams |

## API Endpoints

```
POST   /api/v1/physical/twin/pipeline         Create digital twin pipeline
GET    /api/v1/physical/twin/pipelines        List pipelines
POST   /api/v1/physical/robotics/synthesize   Synthesize robotics sensor data
GET    /api/v1/physical/robotics/jobs/{id}    Robotics job status
POST   /api/v1/physical/sim2real/transfer     Sim-to-real transfer learning
POST   /api/v1/physical/randomize             Domain randomization
POST   /api/v1/physical/fusion/generate       Multi-sensor fusion data
```

## Architecture

Follows AumOS hexagonal architecture:

```
src/aumos_physical_ai/
├── api/
│   ├── router.py       FastAPI routes (no business logic)
│   └── schemas.py      Pydantic request/response models
├── core/
│   ├── models.py       SQLAlchemy ORM models (pai_ prefix)
│   ├── services.py     Business logic (5 services)
│   └── interfaces.py   Protocol interfaces for adapters
├── adapters/
│   ├── repositories.py DB repositories (5 models)
│   ├── kafka.py        Kafka event publisher
│   ├── blenderproc_client.py  BlenderProc HTTP client
│   ├── sensor_simulator.py   Sensor sim + fusion engine
│   └── sim2real_adapter.py   Domain adaptation adapter
├── main.py             FastAPI app + lifespan
└── settings.py         Pydantic settings (AUMOS_PHYSICAL_AI_ prefix)
```

## Database Tables

| Table | Purpose |
|---|---|
| `pai_twin_pipelines` | Digital twin pipeline configs and state |
| `pai_robotics_jobs` | Robotics sensor synthesis jobs |
| `pai_sim2real_transfers` | Sim-to-real transfer learning records |
| `pai_randomization_configs` | Domain randomization configurations |
| `pai_sensor_fusion_jobs` | Multi-sensor fusion jobs |

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy environment config
cp .env.example .env

# Start dependencies
docker compose -f docker-compose.dev.yml up -d

# Run database migrations
make migrate

# Start development server
make run-dev
```

## Environment Variables

See `.env.example` for all configuration options.
Key variables use `AUMOS_PHYSICAL_AI_` prefix.

## Development

```bash
make test      # Run tests with coverage
make lint      # Lint and format checks
make typecheck # mypy type checking
make format    # Auto-fix formatting
```

## Package

- **Python**: `aumos_physical_ai`
- **Table prefix**: `pai_`
- **Env prefix**: `AUMOS_PHYSICAL_AI_`
- **Port**: 8000
- **License**: Apache-2.0
