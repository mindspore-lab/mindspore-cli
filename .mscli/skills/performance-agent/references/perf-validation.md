# Performance Validation

After building a performance profile, validate the most likely bottlenecks
against real evidence.

Typical groups:

- compute bottleneck
- dataloader or input pipeline bottleneck
- communication bottleneck
- memory bottleneck
- host or framework overhead
- operator hotspot

Every bottleneck claim should carry:

- confidence
- supporting evidence
- a validation check
- one first optimization direction
