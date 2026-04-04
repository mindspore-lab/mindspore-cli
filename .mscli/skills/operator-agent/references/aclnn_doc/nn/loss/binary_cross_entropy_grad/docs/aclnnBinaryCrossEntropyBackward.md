# aclnnBinaryCrossEntropyBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/binary_cross_entropy_grad)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 接口功能：求二元交叉熵反向传播的梯度值。

- 计算公式：

  二元交叉熵的计算公式如下：

  $$
  y_i = - weight_i \cdot (target_i \cdot ln(x_i) + (1 - target_i) \cdot ln(1- x_i))
  $$

  其中$x$表示网络前一层的输出，即正向的预测值；$target$表示样本的标签值，求二元交叉熵对$x$的偏导：

  $$
  \begin{aligned}
  \frac {\partial y}{\partial x} &= -weight \cdot \frac {\partial ((target_i \cdot ln(x_i) + (1-target_i) \cdot ln(1-x_i))}{\partial x} \\
  &= -weight \cdot (\frac {\partial (target \cdot ln(x))}{\partial x} + \frac {\partial ((1-target) \cdot ln(1-x))}{\partial x}) \\
  &= -weight \cdot (\frac {target}{x} - \frac {(1-target)}{1-x}) \\
  &= -weight \cdot \frac {target(1-x)-x(1-target)}{x(1-x)} \\
  &= -weight \cdot \frac {target-x}{x(1-x)} \\
  &= weight \cdot \frac {x-target}{x(1-x)}
  \end{aligned}
  $$

  $$
  out = grad\_output \cdot weight \cdot \frac {x-target}{x(1-x)} \\
  out = mean(grad\_input) \ if \ reduction = mean
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBinaryCrossEntropyBackwardGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBinaryCrossEntropyBackward”接口执行计算。

```Cpp
aclnnStatus aclnnBinaryCrossEntropyBackwardGetWorkspaceSize(
    const aclTensor *gradOutput, 
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weightOptional,
    int64_t          reduction,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBinaryCrossEntropyBackward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnBinaryCrossEntropyBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1537px"><colgroup>
  <col style="width: 161px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 260px">
  <col style="width: 252px">
  <col style="width: 166px">
  <col style="width: 149px">
  <col style="width: 141px">
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
      <td>gradOutput（aclTensor*）</td>
      <td>输入</td>
      <td>网络反向传播前一步的梯度值。数据类型需要与其它参数一起转换到promotion类型。</td>
      <td>shape可以<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>到self的shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>网络正向前一层的计算结果。</td>
      <td>数据类型需要与其它参数一起转换到promotion类型。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr> 
      <tr>
      <td>target（aclTensor*）</td>
      <td>输入</td>
      <td>样本的标签值。数据类型需要与其它参数一起转换到promotion类型。</td>
      <td>shape可以<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>到self的shape。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>需要与self保持一致。</td>
      <td>0-8</td>
      <td>√</td>
    </tr> 
       <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入</td>
      <td>结果的权重。</td>
      <td><ul><li>shape可以<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>到self的shape。</li><li>当weightOptional为空时，需要以self的shape创建一个全1的Tensor。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>需要与self保持一致。</td>
      <td>0-8</td>
      <td>√</td>
    </tr> 
    <tr>
      <td>reduction（int64_t）</td>
      <td>输入</td>
      <td>表示对二元交叉熵反向求梯度计算结果做的reduce操作。</td>
      <td><ul>仅支持0(none),1(mean),2(sum)三个值：<li>0表示不做任何操作。</li><li>1表示对结果取平均值。</li><li>2表示对结果求和。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>存储梯度计算结果，shape与self相同。</td>
      <td>-</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>-</td>
      <td>0-8</td>
      <td>√</td>
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

    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型不支持BFLOAT16。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>传入的gradOutput、self、target、out为空指针。</td>
      </tr>
      <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>gradOutput、self、target、weightOptional的数据类型不在支持的范围内。</td>
      </tr>
      <tr>
      <td>self为标量，gradOutput为非标量的场景不在支持范围内。</td>
      </tr>
      <tr>
      <td>gradOutput、target和weightOptional的shape不能<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>到self的shape。</td>
      </tr>
      <tr>
      <td>reduction的值非0、1、2三值之一。</td>
      </tr>
      <tr>
      <td>gradOutput、self、target、weightOptional的shape维度大于8。</td>
      </tr>
    </tbody>
    </table>

## aclnnBinaryCrossEntropyBackward

- **参数说明：**

   <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
          <td>在Device侧申请的workspace大小，由第一段接口aclnnBinaryCrossEntropyBackwardGetWorkspaceSize获取。</td>
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
  - aclnnBinaryCrossEntropyBackward默认确定性实现。  

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_binary_cross_entropy_backward.h"

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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> weightShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutputHostData = {0.1, 0.1, 0.1, 0.1};
  std::vector<float> selfHostData = {0.3, 0.3, 0.3, 0.3};
  std::vector<float> targetHostData = {0.5, 0.5, 0.5, 0.5};
  std::vector<float> weightHostData = {1, 1, 1, 1};
  std::vector<float> outHostData = {0, 0, 0, 0};
  int64_t reduction = 0;
  // 创建gradOutputaclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBinaryCrossEntropyBackward接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnBinaryCrossEntropyBackward第一段接口
  ret = aclnnBinaryCrossEntropyBackwardGetWorkspaceSize(gradOutput, self, target, weight, reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBinaryCrossEntropyBackward第二段接口
  ret = aclnnBinaryCrossEntropyBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
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