# aclnnCdist

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/cdist)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

- 算子功能：计算两个向量集合中每个点之间的p范数距离。
- 计算公式：
  $$
  y = \sqrt[p]{\sum |x1 - x2|^p}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnCdistGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCdist”接口执行计算。

```cpp
aclnnStatus aclnnCdistGetWorkspaceSize(
  const aclTensor* x1,
  const aclTensor* x2,
  float p,
  int64_t compute_mode,
  aclTensor* out,
  uint64_t* workspaceSize,
  aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnCdist(
  void* workspace,
  uint64_t workspaceSize,
  aclOpExecutor* executor,
  aclrtStream stream)
```

## aclnnCdistGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 200px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>x1</td>
      <td>输入</td>
      <td>表示Cdist的第一个输入，对应公式中的x1，Device侧的aclTensor。</td>
      <td><ul><li>支持空Tensor。</li><li>shape除倒数两维，其他维度需要与x2 shape除倒数两维的其他维度满足<a href="../../../docs/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li><li>shape的尾轴大小需要和x2 shape的尾轴大小相同</li></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示Cdist的第二个输入，对应公式中的x2，Device侧的aclTensor。</td>
      <td><ul><li>支持空Tensor。</li><li>shape除倒数两维，其他维度需要与x1 shape除倒数两维的其他维度满足<a href="../../../docs/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li><li>shape的尾轴大小需要和x1 shape的尾轴大小相同</li></td>
      <td>数据类型与x1保持一致。</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>p</td>
      <td>输入</td>
      <td>表示范数的系数，对应公式中的p。</td>
      <td><ul><li>常用0、1.0、2.0、inf范数。</li><li>取值范围[0, +Inf]。</li></td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
   <tr>
      <td>compute_mode</td>
      <td>输入</td>
      <td>表示计算模式，预留参数，暂无作用。</td>
      <td><ul><li>预留参数，当前无作用，当p为2.0时，此参数只支持2。</li></td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示Cdist的输出，对应公式中的y，Device侧的aclTensor。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与x1和x2相同。</li><li>若x1的shape为[D, P, M]，x2的shape为[D, R, M]，则out的shape为[D, P, R]，其中D为输入输出除倒数两维其他维度broadcast并合轴后的维度。</li></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

   ```
  第一段接口完成入参校验，出现如下场景时报错：
  返回161001（ACLNN_ERR_PARAM_NULLPTR）：1. 传入的x1或x2或out是空指针。
  返回161002（ACLNN_ERR_PARAM_INVALID）：1. x1或x2或out的数据类型不在支持的范围之内。
                                       2. x1或x2和out的数据类型不一致。
                                       3. x1或x2或out的维度小于2维。
                                       4. x1或x2或out的维度大于8维。
                                       5. x1的点特征维度和x2的不一致。
                                       6. p为负数或nan。
   ```

## aclnnCdist

- **参数说明：**

- <table>
      <thead>
            <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
        </thead>
        <tbody>
            <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
            <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceAddGetWorkspaceSize获取。</td></tr>
            <tr><td>executor</td><td>输入</td><td> op执行器，包含了算子计算流程。 </td></tr>
            <tr><td>stream</td><td>输入</td><td> 指定执行任务的Stream。 </td></tr>
        </tbody>
    </table>
  
- **返回值：**

- aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- **功能维度**
  * 数据类型支持
    * <term>Ascend 950PR/Ascend 950DT</term>：x1、x2、y支持FLOAT32、FLOAT16、BFLOAT16。<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x1、x2、y支持FLOAT32、FLOAT16。
    * p 支持：float。
    * compute_mode 支持：int64_t。
  * 数据格式支持：ND。
- **支持类型说明**
  是否支持空Tensor：支持空进空出。
- **确定性计算**
  aclnnCdist默认为确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cdist.h"

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
  std::vector<int64_t> x1Shape = {3, 2};
  std::vector<int64_t> x2Shape = {2, 2};
  std::vector<int64_t> outShape = {3, 2};
  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> x1HostData = {0.9041, 0.0196, -0.3108, -2.4423, -0.4821, 1.059};
  std::vector<float> x2HostData = {-2.1763, -0.4713, -0.6986, 1.3702};
  std::vector<float> outHostData = {3.1193, 2.0959, 2.7138, 3.8322, 2.2830, 0.3791};
  float p = 2.0f;
  int64_t compute_mode = 0;

  // 创建x1 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2 aclTensor
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnCdist第一段接口
  ret = aclnnCdistGetWorkspaceSize(x1, x2, p, compute_mode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCdistGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnCdist第二段接口
  ret = aclnnCdist(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCdist failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
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