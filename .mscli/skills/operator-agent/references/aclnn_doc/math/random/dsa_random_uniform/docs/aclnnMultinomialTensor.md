# aclnnMultinomialTensor

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/random/dsa_random_uniform)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

在输入张量中根据每个对象分布的概率，抽取numsamples个样本，并将这些样本的索引存储在输出张量中。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMultinomialTensorGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMultinomialTensor”接口执行计算。

```Cpp
aclnnStatus aclnnMultinomialTensorGetWorkspaceSize(
  const aclTensor*      self, 
  int64_t               numsamples, 
  bool                  replacement, 
  const aclTensor*      seedTensor, 
  const aclTensor*      offsetTensor, 
  int64_t               offset, 
  aclTensor*            out, 
  uint64_t*             workspaceSize, 
  aclOpExecutor**       executor)
```

```Cpp
aclnnStatus aclnnMultinomialTensor(
  void*                 workspace, 
  uint64_t              workspaceSize, 
  aclOpExecutor*        executor, 
  aclrtStream           stream)
```

## aclnnMultinomialTensorGetWorkspaceSize

- **参数说明：**

  <table class="tg" style="undefined;table-layout: fixed; width: 1571px"><colgroup>
  <col style="width: 233px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 230px">
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 150px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-5agr">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入tensor，表示每个对象分布的概率。</td>
      <td class="tg-0pky">self的取值范围需要大于等于0。</td>
      <td class="tg-0pky">BFLOAT16、FLOAT16、FLOAT、DOUBLE</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">为(N, C)或(C)，且self与out的维度一致。</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">numsamples（int64_t）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">从每个多项分布中抽取的样本数。</td>
      <td class="tg-0pky">numsamples为正数，当replacement为false时，numsamples不大于C。</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">replacement（bool）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">决定了抽样时元素是否有放回。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">BOOL</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">seedTensor（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">设置随机数生成器的种子值，它影响生成的随机数序列。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">为[1]。</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">offsetTensor（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">与标量offset的累加结果作为随机数算子的偏移量。随机数的偏移量影响生成的随机数序列的位置。设置偏移量后，生成的随机数序列会从指定位置开始。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">为[1]。</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0lax">offset（int64_t）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">作为offsetTensor的累加量。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">out（aclTensor*）</td>
      <td class="tg-0lax">输出</td>
      <td class="tg-0lax">输出tensor。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">INT64</td>
      <td class="tg-0lax">ND</td>
      <td class="tg-0lax">为(N, numsamples)或(numsamples)，self与out的维度一致。</td>
      <td class="tg-0lax">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize（uint64_t*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor（aclOpExecutor**）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回op执行器，包括了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **返回值：**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 288px">
  <col style="width: 114px">
  <col style="width: 747px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>self的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>out的数据类型不是INT64。</td>
    </tr>
    <tr>
      <td>self的维度不是1维或2维。</td>
    </tr>
    <tr>
      <td>self与out的维度不一致。</td>
    </tr>
    <tr>
      <td>numsamples的值小于等于0。</td>
    </tr>
    <tr>
      <td>replacement为false且numsamples大于self最后一个维度的大小。</td>
    </tr>
    <tr>
      <td>self最后一个维度的大小不能超过2^24。</td>
    </tr>
    <tr>
      <td>out的最后一个维度要与numsamples的大小保持一致。</td>
    </tr>
    <tr>
      <td>当self为2维Tensor时，out的第一个维度要与self第一个维度的大小相等。</td>
    </tr>
  </tbody>
  </table>

## aclnnMultinomialTensor

- **参数描述：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 153px">
  <col style="width: 124px">
  <col style="width: 872px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMultinomialTensorGetWorkspaceSize获取。</td>
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
  - aclnnMultinomialTensor默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_multinomial.h"

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
  std::vector<int64_t> selfShape = {8};
  std::vector<int64_t> outShape = {4};
  std::vector<int64_t> seedShape = {1};
  std::vector<int64_t> offsetShape = {1};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  void* seedDeviceAddr = nullptr;
  aclTensor* seed = nullptr;
  void* offsetDeviceAddr = nullptr;
  aclTensor* offset = nullptr;
  int64_t offset2 = 102;
  std::vector<float> selfHostData = {0, 10, 3, 0, 1, 2, 1, 1};
  std::vector<int64_t> outHostData = {2, 0, 1, 2};
  std::vector<int64_t> seedHostData = {0};
  std::vector<int64_t> offsetHostData = {392};
  int64_t numsamples = 4;
  bool replacement = false;

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建seed aclTensor
  ret = CreateAclTensor(seedHostData, seedShape, &seedDeviceAddr, aclDataType::ACL_INT64, &seed);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建offset aclTensor
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_INT64, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMultinomialTensor第一段接口
  ret = aclnnMultinomialTensorGetWorkspaceSize(self, numsamples, replacement, seed, offset, offset2, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultinomialTensorGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMultinomialTensor第二段接口
  ret = aclnnMultinomialTensor(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultinomialTensor failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<int64_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(seed);
  aclDestroyTensor(offset);

  // 7. 释放Device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(seedDeviceAddr);
  aclrtFree(offsetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```