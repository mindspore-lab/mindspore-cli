# aclnnApplyTopKTopP

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：对原始输入logits进行top-k和top-p采样过滤。

- 计算公式：
  - 对输入logits按最后一轴进行升序排序，得到对应的排序结果sortedValue和sortedIndices。
  $$sortedValue, sortedIndices = sort(logits, dim=-1, descending=false, stable=true)$$
  - 计算保留的阈值（第k大的值）。
  $$topKValue[b][v] = sortedValue[b][sortedValue.size(1) - k[b]]$$
  - 生成top-k需要过滤的mask。
  $$topKMask = sortedValue < topKValue$$
  - 通过topKMask将小于阈值的部分置为-Inf。

  $$
  sortedValue[b][v] =
  \begin{cases}
  -Inf & \text{topKMask[b][v]=true}\\
  sortedValue[b][v] & \text{topKMask[b][v]=false}
  \end{cases}
  $$

  - 通过softmax将经过top-k过滤后的数据按最后一轴转换为概率分布。
  $$probsValue = softmax(sortedValue, dim=-1)$$
  - 按最后一轴计算累计概率（从最小的概率开始累加）
  $$probsSum = cumsum(probsValue, dim=-1)$$
  - 生成top-p的mask，累计概率小于等于1-p的位置需要过滤掉，并保证每个batch至少保留一个元素。
  $$topPMask[b][v] = probsSum[b][v] <= 1-p[b]$$
  $$topPMask[b][-1] = false$$
  - 通过topPMask将小于阈值的部分置为-Inf。

  $$
  sortedValue[b][v] =
  \begin{cases}
  -Inf & \text{topPMask[b][v]=true}\\
  sortedValue[b][v] & \text{topPMask[b][v]=false}
  \end{cases}
  $$

  - 将过滤后的结果按sortedIndices还原到原始顺序。
  $$out[b][v] = sortedValue[b][sortedIndices[b][v]]$$
  其中$0 \le b \lt logits.size(0), 0 \le v \lt logits.size(1)$。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnApplyTopKTopPGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnApplyTopKTopP”接口执行计算。
```Cpp
aclnnStatus aclnnApplyTopKTopPGetWorkspaceSize(
  const aclTensor* logits,
  const aclTensor* p,
  const aclTensor* k,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnApplyTopKTopP(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnApplyTopKTopPGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 146px">
  <col style="width: 110px">
  <col style="width: 301px">
  <col style="width: 359px">
  <col style="width: 328px">
  <col style="width: 101px">
  <col style="width: 143px">
  <col style="width: 146px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>logits</td>
      <td>输入</td>
      <td>表示需要处理的数据，公式中的logits。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>p</td>
      <td>输入</td>
      <td>表示top-p的阈值，公式中的p。</td>
      <td>值域为[0, 1]。<br>数据类型需要与logits一致。<br>shape需要与logits.size(0)一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>表示top-k的阈值，公式中的k。</td>
      <td>值域为[1, logits.size(1)]。<br>shape需要与logits.size(0)一致。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示过滤后的数据，公式中的out。</td>
      <td>数据类型需要与logits一致。<br>shape需要与logits一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的logits、out是空指针或p与k同为空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>logits、p、k或out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>logits、p或out的数据类型不匹配。</td>
    </tr>
    <tr>
      <td>logits、p、k或out的shape不匹配。</td>
    </tr>
  </tbody></table>

## aclnnApplyTopKTopP

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnApplyTopKTopPGetWorkspaceSize获取。</td>
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
  - aclnnApplyTopKTopP默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_top_k_top_p.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
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
  // 固定写法，资源初始化
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

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> logitsShape = {3, 4};
  std::vector<int64_t> pShape = {3};
  std::vector<int64_t> kShape = {3};
  std::vector<int64_t> outShape = {3, 4};
  void* logitsDeviceAddr = nullptr;
  void* pDeviceAddr = nullptr;
  void* kDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* logits = nullptr;
  aclTensor* p = nullptr;
  aclTensor* k = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> logitsHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> pHostData = {0.2, 0.4, 0.6};
  std::vector<int32_t> kHostData = {1, 2, 3};
  std::vector<float> outHostData(12, 0);
  // 创建logits aclTensor
  ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr, aclDataType::ACL_FLOAT, &logits);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建p aclTensor
  ret = CreateAclTensor(pHostData, pShape, &pDeviceAddr, aclDataType::ACL_FLOAT, &p);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建k aclTensor
  ret = CreateAclTensor(kHostData, kShape, &kDeviceAddr, aclDataType::ACL_INT32, &k);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnApplyTopKTopP第一段接口
  ret = aclnnApplyTopKTopPGetWorkspaceSize(logits, p, k, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyTopKTopPGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnApplyTopKTopP第二段接口
  ret = aclnnApplyTopKTopP(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyTopKTopP failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(logits);
  aclDestroyTensor(p);
  aclDestroyTensor(k);
  aclDestroyTensor(out);

  // 7. 释放Device资源，需要根据具体API的接口定义修改
  aclrtFree(logitsDeviceAddr);
  aclrtFree(pDeviceAddr);
  aclrtFree(kDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
