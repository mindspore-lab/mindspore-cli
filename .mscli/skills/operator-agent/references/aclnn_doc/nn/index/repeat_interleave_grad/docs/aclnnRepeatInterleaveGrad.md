# aclnnRepeatInterleaveGrad

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/repeat_interleave_grad)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                          |    √  |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×   |


## 功能说明

  - 接口功能：算子repeatInterleave的反向, 将yGrad tensor的axis维度按repeats进行ReduceSum。

  - 示例：
    假设tensor yGrad是 ([[a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>, d<sub>1</sub>, e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub>, c<sub>2</sub>, d<sub>2</sub>, e<sub>2</sub>, f<sub>2</sub>]]), repeats为([1, 2, 2, 1])，axis为1。
    那么最后生成的tensor为 tensor([[a<sub>1</sub>, b<sub>1</sub> + c<sub>1</sub>, d<sub>1</sub> + e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub> + c<sub>2</sub>, d<sub>2</sub> + e<sub>2</sub>, f<sub>2</sub>]])。将tensor yGrad的axis轴按照repeats做ReduceSum。

    假设tensor yGrad是 ([[a<sub>1</sub>, b<sub>1</sub>, c<sub>1</sub>, d<sub>1</sub>, e<sub>1</sub>, f<sub>1</sub>], [a<sub>2</sub>, b<sub>2</sub>, c<sub>2</sub>, d<sub>2</sub>, e<sub>2</sub>, f<sub>2</sub>]]), repeats为([2])，axis为1。
    那么最后生成的tensor为 tensor([[a<sub>1</sub> + b<sub>1</sub>， c<sub>1</sub> + d<sub>1</sub>， e<sub>1</sub> + f<sub>1</sub>], [a<sub>2</sub> + b<sub>2</sub>, c<sub>2</sub> + d<sub>2</sub>, e<sub>2</sub> + f<sub>2</sub>]])。将tensor yGrad的axis轴按照repeats的值，每2根轴做ReduceSum。
    注意：该场景等效于 repeats为(2)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRepeatInterleaveGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRepeatInterleaveGrad”接口执行计算。

 ```cpp
  aclnnStatus aclnnRepeatInterleaveGradGetWorkspaceSize(
    const aclTensor *yGrad, 
    const aclTensor *repeats, 
    int64_t          axis, 
    const aclTensor *out, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
  ```

  ```cpp
  aclnnStatus aclnnRepeatInterleaveGrad(
    void          *workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
  ```

## aclnnRepeatInterleaveGradGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1714px"><colgroup>
  <col style="width: 138px">
  <col style="width: 120px">
  <col style="width: 304px">
  <col style="width: 424px">
  <col style="width: 291px">
  <col style="width: 132px">
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
      <th>维度（shape）</th>
      <th>非连续张量Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>yGrad</td>
      <td>输入</td>
      <td>功能说明中待被ReduceSum的输入tensor。</td>
      <td>支持空tensor。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>repeats</td>
      <td>输入</td>
      <td>重复的次数。</td>
      <td>repeats只能为0D / 1D Tensor。如果repeats为1D Tensor且size为1，那么repeats支持广播。如果repeats为1D Tensor且size大于1，那么repeats的元素之和等于yGrad的axis维数。不支持空tensor。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td></td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>输入</td>
      <td>ReduceSum作用的维度。</td>
      <td>axis的取值范围为[-n, n), 其中n为yGrad的维度。</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>功能说明中ReduceSum完成的输出tensor。</td>
      <td>shape约束参见约束说明。</td>
      <td>与yGrad一致</td>
      <td>ND</td>
      <td>-</td>
      <td></td>
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
  </tbody></table>

  - <term>Ascend 950PR/Ascend 950DT</term> :yGrad的数据类型支持FLOAT16、BFLOAT16、FLOAT。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1039px"><colgroup>
  <col style="width: 292px">
  <col style="width: 138px">
  <col style="width: 609px">
  </colgroup>
  <thead>
    <tr>
      <th>561002</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、repeats或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>self、repeats的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>self、out的数据类型不一样。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>当self不为空tensor，但repeats为空tensor时。</td>
    </tr>
  </tbody>
  </table>

## aclnnRepeatInterleaveGrad

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1045px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 794px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRepeatInterleaveGradGetWorkspaceSize获取。</td>
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
  - aclnnRepeatInterleaveGrad默认确定性实现。

在计算时需要满足以下要求：

  - 如果repeats为0D tensor或者size为1的1D tensor，那么repeats的元素值必须为yGrad在axis上维数的约数。
    如果为1D tensor且size大于1，那么repeats的元素之和必须为yGrad在axis上的维数。
    repeats tensor中的值必须为自然数。
  - out的shape大小满足yGrad的axis轴ReduceSum后的shape大小。
    例如，yGrad的shape为[64]、repeat为[2]、axis为0，out的shape可以为[32]、[2, 16]、[2, 4, 4]等，只需满足out的shape大小为32即可。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_repeat_interleave_grad.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> yGradShape = {4, 3};
  std::vector<int64_t> repeatsShape = {2};
  std::vector<int64_t> outShape = {2, 3};
  void* yGradDeviceAddr = nullptr;
  void* repeatsDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* yGrad = nullptr;
  aclTensor* repeats = nullptr;
  aclTensor* out = nullptr;
  int64_t axis = 0;
  std::vector<float> yGradHostData = {3, 4, 5, 3, 4, 5, -3, -4, -5, -3, -4, -5};
  std::vector<int64_t> repeatsHostData = {2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0};

  // 创建yGrad aclTensor
  ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT, &yGrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建repeats aclTensor
  ret = CreateAclTensor(repeatsHostData, repeatsShape, &repeatsDeviceAddr, aclDataType::ACL_INT64, &repeats);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnRepeatInterleaveGrad第一段接口
  ret = aclnnRepeatInterleaveGradGetWorkspaceSize(yGrad, repeats, axis, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnRepeatInterleaveGrad第二段接口
  ret = aclnnRepeatInterleaveGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRepeatInterleaveGrad failed. ERROR: %d\n", ret); return ret);

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

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(yGrad);
  aclDestroyTensor(repeats);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(yGradDeviceAddr);
  aclrtFree(repeatsDeviceAddr);
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

