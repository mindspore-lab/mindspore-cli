---
name: api-helper
description: Auto-invoked when users ask mindspore api questions. such as mint.*, tensor.*, forward, backward, cpu/gpu/npu.
---

# API Helper

This skill helps you understand MindSpore's API call chain and basic knowledage.

## When to Use

Use this skill when:
- Questions about mint.*, or tensor.* operators. 
- Questions about forward/backward operator/API
- Questions about API call chains
- Questions about NPU backend dispatch for a resolved operator

## Instructions

### Step 1: understand api call chain and resolve the CORRECT operator 

MUST read ./reference/api-to-operator.md


### Step 2: resolve the backend dispatch if needed

This step is OPTIONAL.

Only do this step when the user's request is specifically about NPU/ACLNN backend support, backend dispatch, or backend routing for the resolved operator.

Use the CORRECT operator name from Step 1 and read ./reference/operator-to-backend.md for the backend dispatch.

If the user's request is only about API mapping, forward operator identity, or backward/operator call chain, skip this step.


### Step 3: display answer
