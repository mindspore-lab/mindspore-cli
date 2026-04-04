# remote_deploy_and_test Guide

<a id="remote-deploy-overview"></a>
## Overview

This section describes the remote execution workflow for `op-info-test`: submit a test job, wait for it to finish, read the result, and download artifacts for analysis when it fails.

The goal is to complete the "client initiates + server executes" workflow through a unified API and avoid manually logging in to the server for step-by-step operations.

This workflow uses two-phase validation:

1. Run the normal functional test first.
2. After the normal functional test returns `status=success`, append `--count=50` to the same test command and run a stability test.
3. The remote workflow is complete only if the stability test also passes.

<a id="remote-deploy-roles"></a>
## Background and Roles

This workflow involves two roles:

1. `server` (`remote_runner_server.py`): a long-running service responsible for receiving jobs, pulling code into the workspace, executing tests, and producing files such as `summary.json` and logs.
2. `client` (`remote_runner_client.py`): the command-line entry point responsible for submitting jobs, polling status, querying summaries, and downloading failed artifacts locally.

There are two deployment layouts:

1. `server` and `client` are on the same machine (local mode), so `/tmp/op_info_artifacts/<job_id>/` can be accessed directly.
2. `server` runs on a remote machine and `client` runs locally (remote mode). In this case, call the API through `--server http://<server_ip>:18080`, and prefer `status/download` to retrieve results and artifacts.

<a id="remote-deploy-prerequisites"></a>
## 1. Prerequisites

1. Test case generation and modification have already been completed locally, and the branch has been pushed.
2. If `server_ip` is not explicitly provided in the incoming task instruction, use `localhost` as `server_ip`.
3. Set the environment variable `no_proxy=127.0.0.1,localhost`.

<a id="remote-deploy-standard-flow"></a>
## 2. Standard Workflow

<a id="remote-deploy-start-server"></a>
### Step 0: Start the Server (Remote Machine)

```bash
cd $MINDSPORE_ROOT
python $SCRIPT_PATH/remote_runner_server.py \
  --host 0.0.0.0 \
  --port 18080 \
  --state-file /tmp/op_info_state.json \
  --lock-file /tmp/op_info_runner.lock \
  --artifact-root /tmp/op_info_artifacts \
  --workspace-root /tmp/op_info_workspace
```

The server is started independently before the test job begins and does not need attention during the test task itself.

<a id="remote-deploy-submit-job"></a>
### Step 1: Submit a Job (Client Machine)

```bash
cd $MINDSPORE_ROOT
python $SCRIPT_PATH/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  submit \
  --repo <mindspore_root> \
  --branch <your_branch> \
  --test-cmd "pytest tests/st/ops/op_info_tests/*.py -q --maxfail=1 --tb=short" \
  --timeout-sec 3600
```

Record the returned `job_id`.

If the current round is the stability test, keep the branch, repository, and timeout parameters unchanged, and only append `--count=50` to the original `--test-cmd`. Example:

```bash
cd $MINDSPORE_ROOT
python $SCRIPT_PATH/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  submit \
  --repo <mindspore_root> \
  --branch <your_branch> \
  --test-cmd "pytest tests/st/ops/op_info_tests/*.py -q --maxfail=1 --tb=short --count=50" \
  --timeout-sec 3600
```

<a id="remote-deploy-wait-job"></a>
### Step 2: Wait for the Job to Finish

```bash
python $SCRIPT_PATH/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  wait --job-id <job_id> --poll-interval-sec 10 --wait-timeout-sec 7200
```

To query the status manually:

```bash
python $SCRIPT_PATH/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  status --job-id <job_id>
```

<a id="remote-deploy-read-summary"></a>
### Step 3: Read the Test Summary

It is recommended to read the summary through the API on the client side (suitable when the server runs on a remote machine):

```bash
python $SCRIPT_PATH/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  status --job-id <job_id>
```

If the current command is being executed locally on the server machine, you can also read the local artifact file directly:

```bash
cat /tmp/op_info_artifacts/<job_id>/summary.json
```

Key fields:

1. `status`
2. `error_type`
3. `failed_cases`
4. `top_traceback`

If `status=failed`, download the remote artifacts locally:

```bash
python $SCRIPT_PATH/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  download --job-id <job_id> --output ./<job_id>_artifacts.zip
```

<a id="remote-deploy-next-round"></a>
### Step 4: Proceed Based on the Result

1. If the normal functional test returns `status=success`, do not stop. Immediately append `--count=50` to the same test scope, launch the stability test, and then repeat Steps 1 through 4.
2. If the stability test returns `status=success`, the workflow is complete.
3. If any stage returns `error_type=testcase`, analyze the failure using `failed_cases/top_traceback`.
4. If the issue is determined to be a test-case problem, fix the test case and repeat Steps 1 through 4.
5. If the issue is determined not to be a test-case problem, record the conclusion and evidence. Do not fabricate a test-case fix.
6. If `error_type=infra`, stop automatic test-case modifications and resolve the environment issue first.

<a id="remote-deploy-optional-actions"></a>
## 3. Optional Action

Cancel a job:

```bash
python $SCRIPT_PATH/remote_runner_client.py \
  --server http://<server_ip>:18080 \
  cancel --job-id <job_id>
```
