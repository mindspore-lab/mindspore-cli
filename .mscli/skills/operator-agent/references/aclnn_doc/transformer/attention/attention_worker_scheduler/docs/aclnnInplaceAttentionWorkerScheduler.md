# aclnnInplaceAttentionWorkerScheduler

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |    ×     |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：Attention和FFN分离部署场景下，Attention侧数据扫描算子。该算子接收来自FFNToAttention算子的输出数据，并对数据进行逐步扫描，确保数据准备就绪。

  **该算子不建议单独使用，建议与FFNToAttention和AttentionWorkerCombine算子配合使用，形成完整的工作流。**

    1. 接收FFNToAttention算子发送的数据。该数据以ScheduleContext结构体存储。其具体定义参见[调用示例](#调用示例)。该结构体包含CommonArea，ControlArea，AttentionArea，FfnArea域。本接口涉及CommonArea（用于存储配置信息，如session_num，micro_batch_num，micro_batch_size，selected_expert_num），ControlArea（用于上层控制进程是否退出），AttentionArea域（负责管理算子计算过程中所需的核心数据缓冲区与状态信息，其中token_info_buf存储了与输入相关的数据信息）。

    2. 读取ScheduleContext.AttentionArea域中token_info_buf存储的flag信息，查看通信数据是否准备就绪。

    3. 数据全部准备就绪后，后续可供AttentionWorkerCombine算子使用。

- 计算公式：

  $$
  \text{Initialize:} \quad \text{ready\_count} = 0, \quad \text{flag\_num} = \text{micro\_batch\_size} \times \text{selected\_expert\_num}
  $$

  $$
  \text{Check if run\_flag is 0:}
    \quad \text{if run\_flag} = 0, \quad \text{exit and log}
  $$

  $$
  \text{Loop:} \quad \text{while run\_flag} \neq 0:
    \quad \text{ready\_count} = \sum_{i=1}^{\text{flag\_num}} \mathbf{1}\_{\{ \text{flag}[i] = 1 \}}; \quad \text{if ready\_count} = \text{flag\_num}, \quad \text{break}
  $$

  $$
  \text{Reset flags:}
    \quad \text{flag}[i] = 0 \quad \text{for} \quad i = 1, 2, \dots, \text{flag\_num}
  $$

  $$
  \text{Set micro\_batch\_id:} \quad \text{micro\_batch\_id} = (\text{micro\_batch\_id} + 1) \% \text{micro\_batch\_num}
  $$

  备注：micro_batch_size、selected_expert_num、run_flag、micro_batch_id是入参ScheduleContext结构体的参数，该结构体信息在[调用示例](#调用示例)中进行展示说明。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceAttentionWorkerScheduler”接口执行计算。

```Cpp
aclnnStatus aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(
    aclTensor* scheduleContextRef,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```Cpp
aclnnStatus aclnnInplaceAttentionWorkerScheduler(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1458px"><colgroup>
  <col style="width: 239px">
  <col style="width: 132px">
  <col style="width: 318px">
  <col style="width: 178px">
  <col style="width: 123px">
  <col style="width: 119px">
  <col style="width: 204px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>scheduleContextRef</td>
      <td>输入/输出</td>
      <td>Attention侧接收的待处理数据，表示输入scheduleContext信息，详细结构见调用示例。</td>
      <td>不支持空Tensor。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td rowspan="1">输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td rowspan="1">输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>参数scheduleContextRef是空指针</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>参数scheduleContextRef维度不为1。</td>
    </tr>
    <tr>
      <td>161002</td>
      <td>参数scheduleContextRef是空tensor。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceAttentionWorkerScheduler

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

  - aclnnInplaceAttentionWorkerScheduler默认为确定性实现，暂不支持非确定性实现，确定性计算配置也不会生效。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_worker_scheduler.h"
#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr)  \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

int CreateAclTensor(const void *hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的stride
  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

#pragma pack(push, 1)
struct AttentionDataDesc {
  int32_t flag[0];
};

struct ScheduleContext {
  struct CommonArea {
    uint32_t session_num;  // Number of attention nodes
    uint32_t micro_batch_num;
    uint32_t micro_batch_size;
    uint32_t selected_expert_num;
    uint32_t expert_num; // Number of experts per layer, including routing experts and shared experts.
    uint32_t attn_to_ffn_token_size;  // Each token in the Ffn window data area has a space size aligned to 512 bytes.
    uint32_t ffn_to_attn_token_size;  // Each token in the Attention window data area has a space size aligned to 512 bytes.
    int32_t schedule_mode;  // 0: Ffn only 1: Attention only
    int8_t reserve0[96];
  };
  struct ControlArea {
    int32_t run_flag;  // 0 : exited  1 : running
    int8_t reserve2[124];
  };
  struct FfnArea {
    // ffn area
    uint64_t token_info_buf;  // Points to device memory.
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;  // Points to device memory.
    uint64_t token_data_buf_size;
    uint64_t polling_index;  // For synchronous computation only:records the micro-batch ids to be processed internally by the ffn worker scheduler.
    int8_t reserve3[88];

    // ffn out area
    uint64_t layer_ids_buf;  // Points to a device memory region that stores the organized layer ids, with an array size of session_num.
    uint64_t layer_ids_buf_size;  // Total size: session_num * sizeof(int32_t)
    uint64_t session_ids_buf;  //  Points to a device memory region that stores the organized session ids. The array size is session_num.
    uint64_t session_ids_buf_size;  // Total size: session_num * sizeof(int32_t)
    uint64_t micro_batch_ids_buf;  // Points to a device memory region that stores the organized micro batch ids. The array size is session_num.
    uint64_t micro_batch_ids_buf_size;  // Total size: session_num * sizeof(int32_t)
    uint64_t expert_ids_buf;  //  Points to a device memory region that stores the organized expert ids. The tensor dimensions are [session_num, batch_size, selected_expert_num].
    uint64_t expert_ids_buf_size;  // Total memory size: session_num * batch_size * selected_expert_num * sizeof(int32_t)
    uint32_t out_num;  // Indicates the number of sessions that have been processed.
    int8_t reserve4[60];
  };

  struct AttentionArea {
    // attention area
    uint64_t token_info_buf;  // Points to device memory.
    uint64_t token_info_buf_size;
    uint64_t token_data_buf;  // Points to device memory.
    uint64_t token_data_buf_size;
    uint32_t micro_batch_id; // Records the latest ready micro batch id.
    int8_t reserve5[92];
  };

  // common area
  CommonArea common;
  ControlArea control;
  AttentionArea attention;
  FfnArea ffn;
  // reserve area
  int8_t reserve6[384];  // Padding to 1024 bytes.
};

static_assert(sizeof(ScheduleContext) == 1024, "ScheduleContext size must be 1024 bytes");

#pragma pack(pop)

int aclnnAttentionWorkerSchedulerTest(int32_t deviceId, aclrtStream &stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  ScheduleContext hostScheduleContext = {};

  // Initialize  CommonArea
  hostScheduleContext.common.session_num = 1;
  hostScheduleContext.common.micro_batch_num = 2;
  hostScheduleContext.common.micro_batch_size = 48;
  hostScheduleContext.common.selected_expert_num = 9;
  hostScheduleContext.common.expert_num = 16;
  hostScheduleContext.common.attn_to_ffn_token_size = 512;
  hostScheduleContext.common.ffn_to_attn_token_size = 512;
  hostScheduleContext.common.schedule_mode = 1;  // attention

  // Initialize ControlArea
  hostScheduleContext.control.run_flag = 1;  // running

  // Initialize AttentionArea
  hostScheduleContext.attention.micro_batch_id = 1;
  size_t per_data_desc_size = sizeof(AttentionDataDesc) + sizeof(int32_t) * hostScheduleContext.common.micro_batch_size * hostScheduleContext.common.selected_expert_num;
  size_t expect_token_info_buf_size = static_cast<size_t>(hostScheduleContext.common.micro_batch_num) * per_data_desc_size;
  void* tokenBufDeviceAddr = nullptr;
  ret = aclrtMalloc(&tokenBufDeviceAddr, expect_token_info_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  hostScheduleContext.attention.token_info_buf = reinterpret_cast<uint64_t>(tokenBufDeviceAddr);
  hostScheduleContext.attention.token_info_buf_size = expect_token_info_buf_size;
  int target_micro_batch_id = 0;
  auto data_desc_ptr = reinterpret_cast<AttentionDataDesc *>(
      reinterpret_cast<uint8_t *>(hostScheduleContext.attention.token_info_buf) + per_data_desc_size * target_micro_batch_id);
  size_t flag_num =
      static_cast<size_t>(hostScheduleContext.common.micro_batch_size) * hostScheduleContext.common.selected_expert_num;

  // Set all flags to 1.
  std::vector<int32_t> host_flags(flag_num, 1);
  ret = aclrtMemcpy(data_desc_ptr->flag, flag_num * sizeof(int32_t),
                    host_flags.data(), host_flags.size() * sizeof(int32_t),
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy flags from host to device failed. ERROR: %d\n", ret); return ret);

  uint64_t token_data_buf_size = 100;
  void* tokenDataDeviceAddr = nullptr;
  ret = aclrtMalloc(&tokenDataDeviceAddr, token_data_buf_size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  hostScheduleContext.attention.token_data_buf = reinterpret_cast<uint64_t>(tokenDataDeviceAddr);

  // 创建scheduleContext aclTensor
  std::vector<int64_t> scheduleContextShape = {1024};
  void* scheduleContextDeviceAddr = nullptr;
  aclTensor* scheduleContextRef = nullptr;
  ret = CreateAclTensor(&hostScheduleContext, scheduleContextShape, &scheduleContextDeviceAddr, aclDataType::ACL_INT8, &scheduleContextRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceAttentionWorkerScheduler第一段接口
  ret = aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(scheduleContextRef, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }

  // 调用aclnnInplaceAttentionWorkerScheduler第二段接口
  ret = aclnnInplaceAttentionWorkerScheduler(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAttentionWorkerScheduler failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(scheduleContextShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), scheduleContextDeviceAddr, size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  // 打印输出结果
  ScheduleContext *out_schedule_context = reinterpret_cast<ScheduleContext *>(resultData.data());
  LOG_PRINT("micro_batch_id = %u.\n", out_schedule_context->attention.micro_batch_id);

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(scheduleContextRef);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(scheduleContextDeviceAddr);
  aclrtFree(tokenBufDeviceAddr);
  aclrtFree(tokenDataDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnAttentionWorkerSchedulerTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionWorkerSchedulerTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
