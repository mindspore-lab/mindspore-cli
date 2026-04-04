# aclnnInplaceFfnWorkerScheduler

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

- 算子功能：Attention和FFN分离场景下，FFN侧数据扫描算子。该算子接收AttentionToFFN算子发送的数据，进行扫描并完成数据整理。

    **不建议直接使用，需要与AttentionToFFN，FFNWorkerBatching配合使用。**

    1. 接收AttentionToFFN算子发送的数据。该数据以ScheduleContext结构体内存排布方式存储。其具体定义参见[调用示例](#调用示例)。该结构体包含CommonArea，ControlArea，AttentionArea，FfnArea域。本接口涉及CommonArea(用于存储配置信息，如session_num，micro_batch_num，micro_batch_size，selected_expert_num)，ControlArea(用于上层控制进程是否退出)，FfnArea域(负责管理本算子计算过程中所需的输入及输出缓冲区，其中token_info_buf字段用来存储该算子的输入信息)。

    2. 扫描token_info_buf存储的信息，当通信数据准备就绪时，本算子开始进行数据整理。整理如下图所示，将layer id， session id，micro batch id，expert ids分别写入layer_id_buf，session_id_buf，micro_batch_id_buf，expert_ids_buf的device内存上。
    ```mermaid
    graph TB
        %% 输入缓冲区
        A[token_info_buf输入]

        %% Session 层级结构
        A --> Session0
        A --> Session1

        %% Session 0 内部结构
        subgraph Session0[session 0]
            direction TB
            S0_M1[micro batch id 0]:::micro
            S0_L1[layer id 0]:::layer
            S0_S1[session id 0]:::session0
            S0_E1[expert ids 0]:::expert
        end

        %% Session 1 内部结构
        subgraph Session1[session 1]
            direction TB
            S1_M1[micro batch id 0]:::micro
            S1_L1[layer id 0]:::layer
            S1_S1[session id 1]:::session1
            S1_E1[expert ids 0]:::expert
        end

        %% 输出缓冲区索引区域
        subgraph Output[输出区域]
            direction TB
            O1[layer_ids_buf]:::layer
            O2[session_ids_buf]:::output
            O3[micro_batch_ids_buf]:::micro
            O4[expert_ids_buf]:::expert
        end

        %% 数据流向
        S0_L1 -.-> O1
        S0_S1 -.-> O2
        S0_M1 -.-> O3
        S0_E1 -.-> O4

        S1_L1 -.-> O1
        S1_S1 -.-> O2
        S1_M1 -.-> O3
        S1_E1 -.-> O4

        classDef layer fill:#c8e6c9
        classDef session0 fill:#ffcdd2
        classDef session1 fill:#ffccbc
        classDef output fill:#e3f2fd
        classDef micro fill:#e1f5fe
        classDef expert fill:#bbdefd
        
        %% 添加子图背景色样式
        style Session0 fill:#fff3e0,stroke:#ff9800,stroke-width:2px
        style Session1 fill:#fce4ec,stroke:#e91e63,stroke-width:2px
        style Output fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    ```
    3. 完成数据整理后，后续可供FFNWorkerBatching算子使用。

- 计算公式：
    1. 初始化，根据入参ScheduleContext中的session_num和sync_group_size计算分组个数。
    2. 若分组个数为1，表示全同步处理数据，待全部session数据准备就绪后，进行数据整理。
    3. 若分组个数不为1，表示非全同步处理数据，待group内的session数据准备就绪后，进行数据整理。

$$
\text{Initialize:} \quad\text{group\_num} = \frac{\text{session\_num}}{\text{sync\_group\_size}}
$$

$$
\text{Process} = 
\begin{cases}
\text{check\_all\_session\_ready()} \quad \text{data\_reorganization()} & \text{if } \text{group\_num} = 1 \\
\text{check\_all\_sessions\_of\_group\_ready()} \quad \text{data\_reorganization()} & \text{otherwise}
\end{cases} 
$$ 
  
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceFfnWorkerScheduler”接口执行计算。

```Cpp
aclnnStatus aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize(
    aclTensor*      scheduleContextRef,
    int32_t         syncGroupSize,
    int32_t         executeMode,
    uint64_t*       workspaceSize,
    aclOpExecutor** executor)
```

```Cpp
aclnnStatus aclnnInplaceFfnWorkerScheduler(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1540px"><colgroup>
  <col style="width: 210px">
  <col style="width: 135px">
  <col style="width: 320px">
  <col style="width: 320px">
  <col style="width: 130px">
  <col style="width: 120px">
  <col style="width: 160px">
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
      <td>scheduleContextRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>FFN侧接收的待处理数据，表示ScheduleContext信息，详细结构参见调用示例</td>
      <td>不支持空tensor。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1维，(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>syncGroupSize（int32_t）</td>
      <td>输入</td>
      <td>每个同步组处理的session个数。</td>
      <td>取值范围为(0，session_num]，session_num表示待处理数据的最大会话数，即调用示例中结构体ScheduleContext中CommonArea域的session_num字段。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executeMode（int32_t）</td>
      <td>输入</td>
      <td>执行模式。</td>
      <td>只支持模式0， 表示执行完一次退出。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
      <td>参数scheduleContextRef是空指针。</td>
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
    <tr>
      <td>161002</td>
      <td>参数executeMode非0。</td>
    </tr>
  </tbody>
  </table>

## aclnnInplaceFfnWorkerScheduler

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

  - aclnnInplaceFfnWorkerScheduler默认为确定性实现，暂不支持非确定性实现，确定性计算配置也不会生效。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_ffn_worker_scheduler.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
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

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
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

constexpr uint32_t kSuccess = 0;
constexpr uint32_t kFailure = 1;
constexpr uint64_t kBufAlignSize = 512;

inline uint64_t AlignUp(uint64_t num, uint64_t align)
{
  return ((num + align - 1) / align) * align;
}

#pragma pack(push, 1)
struct FfnDataDesc {
  volatile int32_t flag;
  volatile int32_t layer_id;
  volatile int32_t expert_ids[0];
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
    uint64_t polling_index;  // For synchronous computation only: records the micro-batch ids to be processed internally by the ffn worker scheduler.
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

uint64_t CalcFfnTokenInfoSize(ScheduleContext& schedule_context)
{
  // token_info_size = (sizeof(FfnDataDesc) + selected_expert_num * micro_batch_size) * micro_batch_num * session_num
  uint64_t flag_and_layer_id_size = sizeof(FfnDataDesc);
  uint64_t token_info_size =(sizeof(int32_t) * static_cast<uint64_t>(schedule_context.common.selected_expert_num) * schedule_context.common.micro_batch_size + flag_and_layer_id_size) * static_cast<uint64_t>(schedule_context.common.micro_batch_num) * static_cast<uint64_t>(schedule_context.common.session_num);
  return token_info_size;
}

uint32_t InitFfnTokenInfoBuf(ScheduleContext& schedule_context)
{
  std::unique_ptr<uint8_t[]> tmp_buf(new (std::nothrow) uint8_t[schedule_context.ffn.token_info_buf_size]);
  if (tmp_buf == nullptr) {
    LOG_PRINT("alloc token info host tmp buf failed, buf_size=   %lu\n", schedule_context.ffn.token_info_buf_size);
    return kFailure;
  }
  auto buf_int = reinterpret_cast<int32_t*>(tmp_buf.get());
  auto token_buf_int = buf_int;
  for (uint32_t session_id = 0; session_id < schedule_context.common.session_num; ++session_id) {
    for (uint32_t micro_batch_id = 0; micro_batch_id < schedule_context.common.micro_batch_num; ++micro_batch_id) {
      // flag
      *buf_int++ = 1;
      // layer_id
      *buf_int++ = 55;
      for (uint32_t idx = 0;
            idx < schedule_context.common.micro_batch_size * schedule_context.common.selected_expert_num; ++idx) {
        // expert_id
        *buf_int++ = idx;
      }
    }
  }
  void *token_info_buf = nullptr;
  size_t token_info_buf_size = sizeof(uint8_t) * schedule_context.ffn.token_info_buf_size;
  if (token_info_buf_size > 0) {
    auto ret = aclrtMalloc(&token_info_buf, token_info_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate token info buf failed. ERROR: %d\n", ret); return ret;);
  }
  schedule_context.ffn.token_info_buf = reinterpret_cast<uint64_t>(token_info_buf);
  auto token_info_buf_tmp = reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_info_buf));
  auto ret = aclrtMemcpy(token_info_buf_tmp, token_info_buf_size, token_buf_int, token_info_buf_size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("memory copy token info buf failed, size_=  %lu\n", token_info_buf_size); return ret;);
  return ACL_SUCCESS;
}

uint32_t InitFfn(ScheduleContext& schedule_context)
{
  uint64_t token_info_size = CalcFfnTokenInfoSize(schedule_context);
  if (token_info_size == 0U) {
    return ACL_ERROR_INVALID_PARAM;
  }

  uint64_t token_info_aligned_size = AlignUp(token_info_size, kBufAlignSize);
  if (token_info_aligned_size < token_info_size) {
     LOG_PRINT("token_info_size[%lu]  overflow after align with %lu\n", token_info_size, kBufAlignSize);
    return ACL_ERROR_INVALID_PARAM;
  }

  schedule_context.ffn.token_info_buf_size = token_info_size;

  // 申请1024大小作为token data buf
  schedule_context.ffn.token_data_buf_size = 1024;
  void *token_data_buf = nullptr;
  size_t token_data_buf_size = sizeof(uint8_t) * schedule_context.ffn.token_data_buf_size;
  auto ret = aclrtMalloc(&token_data_buf, token_data_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate token data buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.token_data_buf = reinterpret_cast<uint64_t>(token_data_buf);

  ret = InitFfnTokenInfoBuf(schedule_context);
  if (ret != ACL_SUCCESS) {
    return ret;
  }

  // calc output size
  schedule_context.ffn.layer_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
  schedule_context.ffn.session_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
  schedule_context.ffn.micro_batch_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num;
  schedule_context.ffn.expert_ids_buf_size = sizeof(int32_t) * schedule_context.common.session_num *
                                             schedule_context.common.micro_batch_size *
                                             schedule_context.common.selected_expert_num;

  void *layer_ids_buf = nullptr;
  size_t layer_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.layer_ids_buf_size;
  ret = aclrtMalloc(&layer_ids_buf, layer_ids_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate layer ids buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.layer_ids_buf = reinterpret_cast<uint64_t>(layer_ids_buf);

  void *session_ids_buf = nullptr;
  size_t session_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.session_ids_buf_size;
  ret = aclrtMalloc(&session_ids_buf, session_ids_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate session ids buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.session_ids_buf = reinterpret_cast<uint64_t>(session_ids_buf);

  void *micro_batch_ids_buf = nullptr;
  size_t micro_batch_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.micro_batch_ids_buf_size;
  ret = aclrtMalloc(&micro_batch_ids_buf, micro_batch_ids_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate micro batch ids buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.micro_batch_ids_buf = reinterpret_cast<uint64_t>(micro_batch_ids_buf);

  void *expert_ids_buf = nullptr;
  size_t expert_ids_buf_size = sizeof(uint8_t) * schedule_context.ffn.expert_ids_buf_size;
  ret = aclrtMalloc(&expert_ids_buf, expert_ids_buf_size,  ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate expert ids buf failed. ERROR: %d\n", ret); return ret;);
  schedule_context.ffn.expert_ids_buf = reinterpret_cast<uint64_t>(expert_ids_buf);

  LOG_PRINT("Init ffn success, token_info_buf_size=%lu,token_data_buf_size= %lu.\n", schedule_context.ffn.token_info_buf_size, schedule_context.ffn.token_data_buf_size);
  return ACL_SUCCESS;
}

uint32_t UninitFfn(ScheduleContext& schedule_context)
{
  if (schedule_context.ffn.token_info_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_info_buf)));
    schedule_context.ffn.token_info_buf = 0;
  }
  if (schedule_context.ffn.token_data_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.token_data_buf)));
    schedule_context.ffn.token_data_buf = 0;
  }
  if (schedule_context.ffn.layer_ids_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.layer_ids_buf)));
    schedule_context.ffn.layer_ids_buf = 0;
  }
  if (schedule_context.ffn.session_ids_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.session_ids_buf)));
    schedule_context.ffn.session_ids_buf = 0;
  }
  if (schedule_context.ffn.micro_batch_ids_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.micro_batch_ids_buf)));
    schedule_context.ffn.micro_batch_ids_buf = 0;
  }
  if (schedule_context.ffn.expert_ids_buf != 0) {
    aclrtFree(reinterpret_cast<void*>(static_cast<uintptr_t>(schedule_context.ffn.expert_ids_buf)));
    schedule_context.ffn.expert_ids_buf = 0;
  }
  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化, 参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {1024};
  void* selfDeviceAddr = nullptr;
  aclTensor* scheduleContextRef = nullptr;
  std::vector<int8_t> selfHostData(1024);
  ScheduleContext schedule_context = {};
  schedule_context.common.session_num = 2;
  schedule_context.common.micro_batch_num = 2;
  schedule_context.common.micro_batch_size = 2;
  schedule_context.common.selected_expert_num = 5;
  schedule_context.control.run_flag = 1;
  schedule_context.common.schedule_mode = 0;
  schedule_context.ffn.polling_index = 1;
  InitFfn(schedule_context); 
  ret = aclrtMemcpy(selfHostData.data(), sizeof(ScheduleContext), &schedule_context, sizeof(ScheduleContext), ACL_MEMCPY_DEVICE_TO_HOST);
  if (ret != ACL_SUCCESS) {
    LOG_PRINT("copy schedule context to host failed. ERROR: %d\n", ret);
    UninitFfn(schedule_context);
    return ret;
  }

  int32_t syncGroupSize = 1;
  int32_t executeMode = 0;
  // 创建scheduleContext aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT8, &scheduleContextRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceFfnWorkerScheduler第一段接口
  ret = aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize(scheduleContextRef, syncGroupSize, executeMode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnInplaceFfnWorkerScheduler第二段接口
  ret = aclnnInplaceFfnWorkerScheduler(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceFfnWorkerScheduler failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int8_t), selfDeviceAddr, size * sizeof(int8_t),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  
  // 打印输出结果
  ScheduleContext *out_schedule_context = reinterpret_cast<ScheduleContext *>(resultData.data());
  LOG_PRINT("layer_ids_buf_size = %lu.\n", out_schedule_context->ffn.layer_ids_buf_size);
  LOG_PRINT("session_ids_buf_size = %lu.\n", out_schedule_context->ffn.session_ids_buf_size);
  LOG_PRINT("micro_batch_ids_buf_size = %lu.\n", out_schedule_context->ffn.micro_batch_ids_buf_size);
  LOG_PRINT("expert_ids_buf_size = %lu.\n", out_schedule_context->ffn.expert_ids_buf_size);

  // 打印 layer_ids 信息
  std::vector<int32_t> layer_ids_buf(out_schedule_context->ffn.layer_ids_buf_size / sizeof(int32_t), 0);
  ret = aclrtMemcpy(layer_ids_buf.data(), out_schedule_context->ffn.layer_ids_buf_size, reinterpret_cast<void *>(
    static_cast<uintptr_t>(out_schedule_context->ffn.layer_ids_buf)), out_schedule_context->ffn.layer_ids_buf_size,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy layer ids buf from device to host failed. ERROR: %d\n", ret); return ret);
  for (int i = 0; i < out_schedule_context->ffn.layer_ids_buf_size / sizeof(int32_t); i++) {
    LOG_PRINT("layer_ids[%d] is: %d\n", i, layer_ids_buf[i]);
  }

  // 打印 session_ids 信息
  std::vector<int32_t> session_ids_buf(out_schedule_context->ffn.session_ids_buf_size / sizeof(int32_t), 0);
  ret = aclrtMemcpy(session_ids_buf.data(), out_schedule_context->ffn.session_ids_buf_size, reinterpret_cast<void *>(
    static_cast<uintptr_t>(out_schedule_context->ffn.session_ids_buf)), out_schedule_context->ffn.session_ids_buf_size,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy session ids buf from device to host failed. ERROR: %d\n", ret); return ret);
  for (int i = 0; i < out_schedule_context->ffn.session_ids_buf_size / sizeof(int32_t); i++) {
    LOG_PRINT("session_ids[%d] is: %d\n", i, session_ids_buf[i]);
  }

  // 打印 micro_batch_ids 信息
  std::vector<int32_t> micro_batch_ids_buf(out_schedule_context->ffn.micro_batch_ids_buf_size / sizeof(int32_t), 0);
  ret = aclrtMemcpy(micro_batch_ids_buf.data(), out_schedule_context->ffn.micro_batch_ids_buf_size, reinterpret_cast<void *>(
    static_cast<uintptr_t>(out_schedule_context->ffn.micro_batch_ids_buf)), out_schedule_context->ffn.micro_batch_ids_buf_size,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy micro batch ids buf from device to host failed. ERROR: %d\n", ret); return ret);
  for (int i = 0; i < out_schedule_context->ffn.micro_batch_ids_buf_size / sizeof(int32_t); i++) {
    LOG_PRINT("micro_batch_ids[%d] is: %d\n", i, micro_batch_ids_buf[i]);
  }

  // 打印 expert_ids 信息
  std::vector<int32_t> expert_ids_buf(out_schedule_context->ffn.expert_ids_buf_size / sizeof(int32_t), 0);
  ret = aclrtMemcpy(expert_ids_buf.data(), out_schedule_context->ffn.expert_ids_buf_size, reinterpret_cast<void *>(
    static_cast<uintptr_t>(out_schedule_context->ffn.expert_ids_buf)), out_schedule_context->ffn.expert_ids_buf_size,
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy expert ids buf from device to host failed. ERROR: %d\n", ret); return ret);
  for (int i = 0; i < out_schedule_context->ffn.expert_ids_buf_size / sizeof(int32_t); i++) {
    LOG_PRINT("expert_ids[%d] is: %d\n", i, expert_ids_buf[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(scheduleContextRef);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  UninitFfn(schedule_context);
  aclrtFree(selfDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
