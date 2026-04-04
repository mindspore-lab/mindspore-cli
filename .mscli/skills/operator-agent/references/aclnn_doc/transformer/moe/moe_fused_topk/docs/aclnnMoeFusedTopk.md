# aclnnMoeFusedTopk

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

-   接口功能：MoE计算中，对输入x做Sigmoid计算，对计算结果分组进行排序，最后根据分组排序的结果选取前k个专家。
-   计算公式：

    对输入做sigmoid：

    $$
    sigmoidRes=sigmoid(x)
    $$

    加上addNum：

    $$
    normOut = sigmoidRes + addNum
    $$

    对计算结果按照groupNum进行分组，每组按照topN的sum值对group进行排序，取前groupTopk个组：

    $$
    groupOut, groupId = TopK(ReduceSum(TopK(Split(normOut, groupCount), k=2, dim=-1), dim=-1),k=kGroup)
    $$

    根据上一步的groupId获取normOut中对应的元素，将数据再做TopK，得到indices的结果：

    $$
    normY,indices=TopK(normOut[groupId, :],k=k)
    $$

    根据indices从sigmoidRes中选出y:

    $$
    y = gather(sigmoidRes, indices)
    $$

    如果isNorm为true，对y按照输入的scale参数进行计算，得到y的结果：

    $$
    y = y / (ReduceSum(y, dim=-1))*scale
    $$
    
    如果enableExpertMapping为true，再将indices中的物理专家按照输入的mappingNum和mappingTable映射到逻辑专家，得到输出的indices。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMoeFusedTopkGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeFusedTopk”接口执行计算。

```Cpp
aclnnMoeFusedTopkGetWorkspaceSize(
  const aclTensor* x, 
  const aclTensor* addNum, 
  const aclTensor* mappingNum, 
  const aclTensor* mappingTable, 
  uint32_t         groupNum, 
  uint32_t         groupTopk, 
  uint32_t         topN, 
  uint32_t         topK, 
  uint32_t         activateType, 
  bool             isNorm, 
  float            scale, 
  bool             enableExpertMapping, 
  aclTensor*       y, 
  aclTensor*       indices, 
  uint64_t*        workspaceSize, 
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnMoeFusedTopk(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnMoeFusedTopkGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1517px"><colgroup>
  <col style="width: 216px">
  <col style="width: 134px">
  <col style="width: 321px">
  <col style="width: 204px">
  <col style="width: 165px">
  <col style="width: 130px">
  <col style="width: 196px">
  <col style="width: 151px">
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
      <td>x</td>
      <td>输入</td>
      <td>每个token对应各个专家的分数。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(numToken, expertNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>addNum</td>
      <td>输入</td>
      <td>与输入x进行计算的偏置值。</td>
      <td>-</td>
      <td>与x一致</td>
      <td>ND</td>
      <td>(expertNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mappingNum</td>
      <td>输入</td>
      <td>每个物理专家被实际映射到的逻辑专家数量。</td>
      <td>enableExpertMapping为false时不启用。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(expertNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mappingTable</td>
      <td>输入</td>
      <td>每个物理专家/逻辑专家映射表。</td>
      <td>enableExpertMapping为false时不启用。<br>小于等于128。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(expertNum, maxMappingNum)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupNum</td>
      <td>输入</td>
      <td>分组数量，必须大于0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupTopk</td>
      <td>输入</td>
      <td>被选择的组的数量，必须大于0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topN</td>
      <td>输入</td>
      <td>组内选取的用于求和的专家数量，必须大于0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topK</td>
      <td>输入</td>
      <td>最终选取的专家数量，必须大于0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateType</td>
      <td>输入</td>
      <td>激活类型，当前只支持0(ACTIVATION_SIGMOID)。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>isNorm</td>
      <td>输入</td>
      <td>是否对输出进行归一化。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>归一化后的系数乘。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>enableExpertMapping</td>
      <td>输入</td>
      <td>是否使能物理专家到逻辑专家的映射。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>Device侧的aclTensor。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(numToken, topK)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>Device侧的aclTensor。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(numToken, topK)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 287px">
  <col style="width: 119px">
  <col style="width: 743px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="3">161001</td>
      <td>输入x或者addNum为空指针。</td>
    </tr>
    <tr>
      <td>输出y或者indices为空指针。</td>
    </tr>
    <tr>
      <td>当enableExpertMapping为true时，输入mappingNum或者mappingTable为空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>输入或者输出的数据类型或数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>输入的参数不满足约束。</td>
    </tr>
    <tr>
      <td>输入输出的Shape不满足约束。</td>
    </tr>
  </tbody>
  </table>

## aclnnMoeFusedTopk

- **参数说明**：

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeFusedTopkGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnMoeFusedTopk默认确定性实现。

- expertNum必须为groupNum的整数倍。
- groupTopk小于等于groupNum。
- maxMappingNum小于等于128。
- TopK小于等于expertNum。
- TopN小于等于expertNum / groupNum。
- expertNum小于等于1024。
- groupNum小于等于256。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_fused_topk.h"

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

int aclnnMoeFusedTopkTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造

  int64_t num_token = 16;
  int64_t expert_num = 32;
  int64_t max_mapping_num = 16;

  uint32_t groupNum = 2;
  uint32_t groupTopk = 2;
  uint32_t topN = 2;
  uint32_t topK = 4;
  uint32_t activateType = 0;
  bool isNorm = false;
  float scale = 1.0;
  bool enableExpertMapping = true;

  std::vector<int64_t> xShape = {num_token, expert_num};
  std::vector<int64_t> addNumShape = {expert_num};
  std::vector<int64_t> mappingNumShape = {expert_num};
  std::vector<int64_t> mappingTableShape = {expert_num, max_mapping_num};

  std::vector<int64_t> yShape = {num_token, topK};
  std::vector<int64_t> indicesShape = {num_token, topK};

  void* xDeviceAddr = nullptr;
  void* addNumDeviceAddr = nullptr;
  void* mappingNumDeviceAddr = nullptr;
  void* mappingTableDeviceAddr = nullptr;

  void* yDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* addNum = nullptr;
  aclTensor* mappingNum = nullptr;
  aclTensor* mappingTable = nullptr;

  aclTensor* y = nullptr;
  aclTensor* indices = nullptr;

  std::vector<float> xHostData(GetShapeSize(xShape), 1);
  std::vector<float> addNumHostData(GetShapeSize(addNumShape), 1);
  std::vector<int32_t> mappingNumHostData(GetShapeSize(mappingNumShape), 1);
  std::vector<int32_t> mappingTableHostData(GetShapeSize(mappingTableShape), 1);

  std::vector<float> yHostData(GetShapeSize(yShape), 0);
  std::vector<int32_t> indicesHostData(GetShapeSize(indicesShape), 0);

  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);

  // 创建addNum aclTensor
  ret = CreateAclTensor(addNumHostData, addNumShape, &addNumDeviceAddr, aclDataType::ACL_FLOAT, &addNum);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> addNumTensorPtr(addNum, aclDestroyTensor);

  // 创建mappingNum aclTensor
  ret = CreateAclTensor(mappingNumHostData, mappingNumShape, &mappingNumDeviceAddr, aclDataType::ACL_INT32, &mappingNum);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> mappingNumTensorPtr(mappingNum, aclDestroyTensor);

  // 创建mappingTable aclTensor
  ret = CreateAclTensor(mappingTableHostData, mappingTableShape, &mappingTableDeviceAddr, aclDataType::ACL_INT32, &mappingTable);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> mappingTableTensorPtr(mappingTable, aclDestroyTensor);

  // 创建y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(y, aclDestroyTensor);

  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> indicesTensorPtr(indices, aclDestroyTensor);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMoeFusedTopk第一段接口
  ret = aclnnMoeFusedTopkGetWorkspaceSize(x,
                                       addNum,
                                       mappingNum,
                                       mappingTable,
                                       groupNum,
                                       groupTopk,
                                       topN,
                                       topK,
                                       activateType,
                                       isNorm,
                                       scale,
                                       enableExpertMapping,
                                       y,
                                       indices,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopkGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // 调用aclnnMoeFusedTopk第二段接口
  ret = aclnnMoeFusedTopk(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopk failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(yShape);
  std::vector<float> yData(size, 0);
  ret = aclrtMemcpy(yData.data(), yData.size() * sizeof(yData[0]), yDeviceAddr,
                    size * sizeof(yData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, yData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnMoeFusedTopkTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeFusedTopkTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
