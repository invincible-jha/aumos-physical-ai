# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

Report security vulnerabilities to **security@muveraai.com**.

Do NOT open public GitHub issues for security vulnerabilities.

**Response SLA:**
- Acknowledgement: 24 hours
- Initial assessment: 72 hours
- Fix timeline communicated: 7 days

## Security Considerations

### Sensor Data Privacy
- Synthetic sensor data pipelines must not process or store PII without explicit consent
- Real-world sensor data (camera feeds, LiDAR scans) may contain identifiable information
- Apply differential privacy or anonymization before storing fusion outputs

### Model Security
- Sim-to-real adapted models should be validated against adversarial inputs
- Model weights stored in object storage must be encrypted at rest
- Access to the model registry is tenant-scoped via JWT claims

### API Security
- All endpoints require valid JWT authentication
- Tenant isolation enforced via Row-Level Security at the database layer
- Rate limiting applied to synthesis and transfer endpoints (resource-intensive)
- Input validation via Pydantic prevents injection attacks

### Infrastructure
- BlenderProc and sensor simulator services run in isolated containers
- MinIO buckets are tenant-namespaced; cross-tenant access is denied
- Kafka topics are ACL-protected per tenant
