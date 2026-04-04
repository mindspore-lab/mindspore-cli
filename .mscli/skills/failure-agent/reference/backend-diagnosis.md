# Backend Diagnosis Reference

Use this file after evidence collection to narrow the likely failing layer before
moving into source-level debugging.

## Current Factory card shortcuts

If `factory_root` is available, try these current real `known_issue` cards
first when the signature aligns:

- `missing-cann-environment`
- `device-out-of-memory`
- `distributed-communication-timeout`
- `ms-context-empty`
- `ms-tbe-operator-compilation-error`
- `stack-version-mismatch`

## Ascend Backend (CANN)

First checks:

- parse runtime/CANN/ACLNN codes with [pta-diagnosis](pta-diagnosis.md)
- inspect CANN logs under `/var/log/npu/slog/*/device-*/plog/`
- confirm CANN environment and version compatibility
- check whether the failure is compile-time (`TBE`, `E9xxxx`, `EBxxxx`) or runtime (`107xxx`, `207xxx`, `507xxx`)

High-value shortcuts:

- missing shared libs or `ASCEND_OPP_PATH` -> `missing-cann-environment`
- `107002`, `context is empty` -> `ms-context-empty`
- `EI0002`, `EI0006`, `107020` -> `distributed-communication-timeout`
- `TBE`, `compile failed`, `UB overflow` -> `ms-tbe-operator-compilation-error`
- `symbol not found`, import failure after upgrade -> `stack-version-mismatch`

Useful debug commands:

```bash
grep -i "error\|fail\|exception\|abort" /var/log/npu/slog/*/device-*/plog/*.log | head -50
grep -i "aclnn\|acl_error\|ret=" /var/log/npu/slog/*/device-*/plog/*.log | tail -30
grep -i "EE\|EI\|EJ\|EZ\|EL" /var/log/npu/slog/*/device-*/plog/*.log | head -20
```

## GPU Backend (CUDA)

First checks:

- distinguish CUDA runtime, cuDNN, and NCCL failures
- verify GPU memory pressure and process contention
- enable synchronous launch for accurate stack location

Useful debug commands:

```bash
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO
nvidia-smi
nvidia-smi topo -m
```

High-value shortcuts:

- `CUDA out of memory` -> `device-out-of-memory`
- NCCL timeout / unhandled system error -> distributed communication path first

## CPU Backend

First checks:

- host memory pressure
- thread oversubscription
- unsupported operator path on CPU
- system library or version mismatch when segfaults are involved

Useful checks:

```bash
free -h
ulimit -a
```

## MindSpore Framework Log Analysis

Look for:

- shape/type inference failures
- graph validation or optimization failures
- unsupported kernel selection
- downstream runtime errors hiding an earlier framework error

Useful commands:

```bash
grep -i "error\|exception\|fail\|abort" mindspore.log | head -30
grep -i "infer shape\|infer type\|abstract" mindspore.log | head -20
grep -i "select kernel\|launch kernel\|not supported" mindspore.log | head -20
```

If graph dumps are enabled, inspect:

- `*_validate.ir`
- `*_optimize.ir`

## torch_npu Log Analysis

Look for:

- `ERRxxxxx` framework errors
- CANN passthrough errors (`107xxx`, `207xxx`, `507xxx`)
- async execution warnings that shift the visible Python stack
- OOM snapshots when enabled

Useful commands:

```bash
grep -i "ERR[0-9][0-9][0-9][0-9][0-9]" pta.log | head -20
grep -i "error\|fail\|exception\|abort" /var/log/npu/slog/*/device-*/plog/*.log | head -50
```

Useful env vars:

```bash
export ASCEND_LAUNCH_BLOCKING=1
export OOM_SNAPSHOT_ENABLE=1
export OOM_SNAPSHOT_PATH=./oom/
```

## Further Location Techniques

When the first pass is inconclusive:

- compare `GRAPH_MODE` vs `PYNATIVE_MODE`
- compare Ascend vs CPU for the same minimal reproducer
- compare `float16` vs `float32`
- reduce from distributed to single-card when possible
- confirm the first failure point before searching historical issues

For deeper backend logging:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0
export ASCEND_SLOG_PRINT_TO_STDOUT=1
```

Routing rule:

- do not jump to backend blame before checking environment, context, dtype,
  shape, and version alignment.
