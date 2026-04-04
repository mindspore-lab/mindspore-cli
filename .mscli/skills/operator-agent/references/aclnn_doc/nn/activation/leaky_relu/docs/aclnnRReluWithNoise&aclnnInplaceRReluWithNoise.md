# aclnnRReluWithNoise&aclnnInplaceRReluWithNoise

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/leaky_relu)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：实现了带噪声的随机修正线性单元激活函数，它在输入小于等于0时，斜率为a；输入大于0时斜率为1

- 计算公式：

  $$
  RReluWithNoise(self)=\begin{cases}
  self, & self\gt0 \\
  a*self, & self\le 0
  \end{cases}
  $$

  其中a是随机变量，服从均匀分布$U$(lower,upper)。
  如果是训练模式（training == true），noise计算公式如下：
  
  $$
  noise_i = \begin{cases}
  1, & self_i \gt 0 \\
  a, & self_i \le 0
  \end{cases}
  $$

## 函数原型

- aclnnRReluWithNoise和aclnnInplaceRReluWithNoise实现相同的功能，使用区别如下，请根据自身实际场景选择合适的算子。
  - aclnnRReluWithNoise：需新建一个输出张量对象存储计算结果。
  - aclnnInplaceRReluWithNoise：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。
- 每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRReluWithNoiseGetWorkspaceSize”或”aclnnInplaceRReluWithNoiseGetWorkspaceSize“接口获取入参并根据流程计算所需workspace大小，再调用“aclnnRReluWithNoise”或”aclnnInplaceRReluWithNoise“接口执行计算。

```Cpp
aclnnStatus aclnnRReluWithNoiseGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *noise,
  const aclScalar *lower,
  const aclScalar *upper,
  bool             training,
  int64_t          seed,
  int64_t          offset,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRReluWithNoise(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

```Cpp
aclnnStatus aclnnInplaceRReluWithNoiseGetWorkspaceSize(
  const aclTensor* self,
  const aclTensor* noise,
  const aclScalar* lower,
  const aclScalar* upper,
  bool             training,
  int64_t          seed,
  int64_t          offset,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceRReluWithNoise(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnRReluWithNoiseGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1360px"><colgroup>
  <col style="width: 211px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>待进行RReluWithNoise计算的入参，公式中的self。</td>
      <td><ul><li>shape支持的维度不超过32。</li><li>数据类型需要和out的数据类型保持一致。</li><li>shape需要和out的shape保持一致。</li><li>数据格式需要和out的数据格式类型保持一致。</li><li>支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-32</td>
      <td>√</td>
    </tr>
    <tr>
      <td>noise（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的noise_i。</td>
      <td><ul><li>Size需要不小于self（shape建议与self一致）。</li><li>数据类型需要和self的数据类型保持一致。</li><li>shape需要和self的shape保持一致。</li><li>shape支持的维度不超过32。</li><li>支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-32</td>
      <td>√</td>
    </tr>
      <tr>
      <td>lower（aclScalar*）</td>
      <td>输入</td>
      <td>均匀分布U中的lower。</td>
      <td>数据类型需要与self、out满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>upper（aclScalar*）</td>
      <td>输入</td>
      <td>均匀分布U中的upper。</td>
      <td>数据类型需要与self、out满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>training（bool）</td>
      <td>输入</td>
      <td>区分是训练还是推理。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>seed（int64_t）</td>
      <td>输入</td>
      <td>随机数生成器的种子，影响生成的随机数序列。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>offset（int64_t）</td>
      <td>输入</td>
      <td>随机数生成器的偏移量，影响生成的随机数序列的位置。</td>
      <td>偏移量设置后，生成的随机数序列会从指定位置开始。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>均匀分布U中的upper。</td>
      <td><ul><li>数据类型需要和self的数据类型保持一致。</li><li>shape需要和self的shape保持一致。</li><li>数据格式需要和self的数据格式类型保持一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
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
  
   - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT。 

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  第一段接口会完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的self、noise或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>self或noise的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self的Size大于noise的Size。</td>
    </tr>
    <tr>
      <td>self、noise、out的数据类型、数据格式不一致。</td>
    </tr>
       <tr>
      <td>self、out的shape不一致。</td>
    </tr>
    <tr>
      <td>self或noise的shape维度超过32。</td>
    </tr>
  </tbody></table>

## aclnnRReluWithNoise

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRReluWithNoiseGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnInplaceRReluWithNoiseGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1360px"><colgroup>
  <col style="width: 111px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>self</td>
      <td>输入</td>
      <td>公式中的self。</td>
      <td>shape支持的维度不超过32。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-32</td>
      <td>√</td>
    </tr>
    <tr>
      <td>noise</td>
      <td>输入</td>
      <td>公式中的noise_i。</td>
      <td><ul><li>Size需要不小于self（shape建议与self一致）。</li><li>数据类型需要和self的数据类型保持一致。</li><li>数据格式需要和self的数据格式保持一致。</li><li>shape支持的维度不超过32。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-32</td>
      <td>√</td>
    </tr>
      <tr>
      <td>lower</td>
      <td>输入</td>
      <td>均匀分布U中的lower。</td>
      <td>数据类型需要与self一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>upper</td>
      <td>输入</td>
      <td>均匀分布U中的upper。</td>
      <td>数据类型需要与self一致。</td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>training</td>
      <td>输入</td>
      <td>区分是训练还是推理。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>seed</td>
      <td>输入</td>
      <td>随机数生成器的种子，影响生成的随机数序列。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>offset</td>
      <td>输入</td>
      <td>随机数生成器的偏移量，影响生成的随机数序列的位置。</td>
      <td>偏移量设置后，生成的随机数序列会从指定位置开始。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
  
   - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的self或noise是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>self或noise的数据类型不在支持的范围之内。</td>
    </tr>
       <tr>
      <td>self的Size大于noise的Size。</td>
    </tr>
       <tr>
      <td>self、noise的数据类型、数据格式不一致。</td>
    </tr>
    <tr>
      <td>self或noise的shape维度超过32。</td>
    </tr>
  </tbody></table>

## aclnnInplaceRReluWithNoise

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceRReluWithNoiseGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnRReluWithNoise&aclnnInplaceRReluWithNoise默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rrelu_with_noise.h"

#define CHECK_RET(cond, return_expr) \
 do {                                \
    if (!(cond)) {                   \
        return_expr;                 \
    }                                \
 } while (0)

#define LOG_PRINT(message, ...)      \
 do {                                \
    printf(message, ##__VA_ARGS__);  \
 } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i: shape) {
        shape_size *= i;
    }
    return shape_size;
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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  std::vector<int64_t> noiseShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* noiseDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* noise = nullptr;
  aclScalar* lower = nullptr;
  aclScalar* upper = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> outHostData = {0, 0, 0, 0};
  std::vector<float> noiseHostData = {4, 3, 2, 1};
  float lowerValue = 0.1f;
  float upperValue = 0.3f;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建noise aclTensor
  ret = CreateAclTensor(noiseHostData, noiseShape, &noiseDeviceAddr, aclDataType::ACL_FLOAT, &noise);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建lower aclScalar
  lower = aclCreateScalar(&lowerValue, aclDataType::ACL_FLOAT);
  CHECK_RET(lower != nullptr, return ret);
  // 创建upper aclScalar
  upper = aclCreateScalar(&upperValue, aclDataType::ACL_FLOAT);
  CHECK_RET(upper != nullptr, return ret);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // aclnnRReluWithNoise接口调用示例
  // 3. 调用aclnnRReluWithNoise第一段接口
  ret = aclnnRReluWithNoiseGetWorkspaceSize(self, noise, lower, upper, training, seed, offset, 
                                            out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRReluWithNoiseGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnRReluWithNoise第二段接口
  ret = aclnnRReluWithNoise(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRReluWithNoise failed. ERROR: %d\n", ret); return ret);

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

  // aclnnInplaceRReluWithNoise接口调用示例
  // step3.调用aclnnInplaceRReluWithNoise第一段接口
  ret = aclnnInplaceRReluWithNoiseGetWorkspaceSize(self, noise, lower, upper, training, seed, offset, 
                                                   &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRReluWithNoiseGetWorkspaceSize failed. ERROR: %d\n", ret); 
                                          return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceRReluWithNoise第二段接口
  ret = aclnnInplaceRReluWithNoise(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRReluWithNoise failed. ERROR: %d\n", ret); return ret);

  // step4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // step5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(noise);
  aclDestroyScalar(lower);
  aclDestroyScalar(upper);

  // 7. 释放device资源，需要根据具体API的接口定义参数
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(noiseDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
