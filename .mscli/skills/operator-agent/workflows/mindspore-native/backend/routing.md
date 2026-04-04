# MindSpore Native Backend Routing

## How To Use This Routing File

Do not use this file to redefine framework-common phases. It only governs the
backend implementation slot.

## Backend Lane Selection

For now only aclnn lane is supported.

## ACLNN Lane

Use the ACLNN lane when the selected native MindSpore route targets NPU or
Ascend integration.

Execute ACLNN in this order:

1. `aclnn/aclnn-call-mapping.md`
2. `aclnn/aclnn-path-selection.md`
3. `aclnn/04-aclnn-pyboost.md`
4. `aclnn/05-aclnn-kbk.md`
5. Backfill the Feature document.