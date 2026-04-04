# aclnnGroupNormBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/group_norm_grad)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |


## 功能说明

- 接口功能：[aclnnGroupNorm](../../group_norm/docs/aclnnGroupNorm.md)的反向计算。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。
- 计算公式：
  
  $$
  gradBetaOut = \sum_{i=1}^n gradOut
  $$

  $$
  gradGammaOut = \sum_{i=1}^n (gradOut \cdot \hat{x})
  $$
  
  $$
  gradInput = mean \cdot rstd \cdot gamma \begin{bmatrix}
  gradOut - \frac{1}{N}  (gradBetaOut + \hat{x} \cdot gradGammaOut)
  \end{bmatrix}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupNormBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupNormBackward”接口执行计算。

```Cpp
aclnnStatus aclnnGroupNormBackwardGetWorkspaceSize(
  const aclTensor*     gradOut,
  const aclTensor*     input,
  const aclTensor*     mean,
  const aclTensor*     rstd,
  const aclTensor*     gamma,
  int64_t              N,
  int64_t              C,
  int64_t              HxW,
  int64_t              group,
  const aclBoolArray*  outputMask,
  aclTensor*           gradInput,
  aclTensor*           gradGammaOut,
  aclTensor*           gradBetaOut,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnGroupNormBackward(
  void                *workspace,
  uint64_t             workspaceSize,
  aclOpExecutor       *executor,
  const aclrtStream    stream)
```

## aclnnGroupNormBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>gradOut（aclTensor*）</td>
      <td>输入</td>
      <td>表示反向计算的梯度Tensor，对应公式中的`gradOut`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`相同。</li><li>元素个数需要等于N*C*HxW。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向计算的首个输入，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`gradOut`相同。</li><li>元素个数需要等于N*C*HxW。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向计算的第二个输出，表示input分组后每个组的均值，对应公式中的`mean`。</td>
      <td><ul><li>支持空Tensor。</li><li>元素个数需要等于N*group。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向计算的第三个输出，表示input分组后每个组的标准差倒数，对应公式中的`rstd`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`mean`相同。</li><li>元素个数需要等于N*group。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示每个channel的缩放系数，对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`mean`相同。</li><li>元素数量需与`C`相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>N（int64_t）</td>
      <td>输入</td>
      <td>表示输入`gradOut`在N维度上的空间大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>C（int64_t）</td>
      <td>输入</td>
      <td>表示输入`gradOut`在C维度上的空间大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>HxW（int64_t）</td>
      <td>输入</td>
      <td>表示输入`gradOut`在除N、C维度外的空间大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group（int64_t）</td>
      <td>输入</td>
      <td>表示将输入`gradOut`的C维度分为group组。</td>
      <td><ul><li>group需大于0，且C必须可以被group整除。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask（aclBoolArray*）</td>
      <td>输入</td>
      <td>表示输出掩码，标识是否输出`gradInput`，`gradGammaOut`，`gradBetaOut`。</td>
      <td><ul><li>size为3。分别表示是否输出`gradInput`，`gradGammaOut`，`gradBetaOut`，若为true则输出，否则输出对应位置返回空。</li></ul></td>
      <td>BoolArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput（aclTensor*）</td>
      <td>输出</td>
      <td>表示计算输出的梯度，用于更新输入数据的梯度。对应公式中的`gradInput`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`gradOut`相同。</li><li>shape与`input`相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradGammaOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示计算输出的梯度，用于更新缩放参数的梯度。对应公式中的`gradGammaOut`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`mean`相同。</li><li>shape与`gamma`相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBetaOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示计算输出的梯度，用于更新偏置参数的梯度。对应公式中的`gradBetaOut`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`mean`相同。</li><li>shape与`gamma`相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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

  - <term>Atlas 训练系列产品</term>：
  
    - 参数`gradOut`、`input`、`mean`、`rstd`、`gamma`、`gradInput`、`gradGammaOut`、`gradBetaOut`的数据类型不支持BFLOAT16。
    - 参数`mean`和`gradOut`的数据类型相同。

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

    参数`mean`和`gradOut`的数据类型相同。

  - <term>Ascend 950PR/Ascend 950DT</term>：

    参数`mean`与`gradOut`支持的数据类型对应关系如下：
    |输入参数 | gradOut | mean |
    |--|--|--|
    |数据类型 | FLOAT32 | FLOAT32 |
    |数据类型 | FLOAT16 | FLOAT16 |
    |数据类型 | FLOAT16 | FLOAT32 |
    |数据类型 | BFLOAT16 | BFLOAT16 |
    |数据类型 | BFLOAT16 | FLOAT32 |

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="4">161001</td>
      <td>传入的gradOut、input、mean、rstd是空指针时。</td>
    </tr>
    <tr>
      <td>当outputMask[0]为true，传入的gradInput是空指针时。</td>
    </tr>
    <tr>
      <td>当outputMask[1]为true，传入的gradGammaOut是空指针时。</td>
    </tr>
    <tr>
      <td>当outputMask[2]为true，传入的gradBetaOut是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>gradOut数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input、mean、gamma、rstd和gradOut的数据类型不满足参数说明的约束。</td>
    </tr>
    <tr>
      <td>outputMask的长度不为3。</td>
    </tr>
    <tr>
      <td>当outputMask[0]为true，gradInput的shape与input的shape不相同。</td>
    </tr>
    <tr>
      <td>当outputMask[1]为true，gradGammaOut的shape与gamma的shape不相同。</td>
    </tr>
    <tr>
      <td>当outputMask[2]为true，gradBetaOut的shape与gamma的shape不相同。</td>
    </tr>
    <tr>
      <td>group不大于0。</td>
    </tr>
    <tr>
      <td>C不能被group整除。</td>
    </tr>
    <tr>
      <td>input的元素个数不等于N*C*HxW。</td>
    </tr>
    <tr>
      <td>mean的元素个数不等于N*group。</td>
    </tr>
    <tr>
      <td>rstd的元素个数不等于N*group。</td>
    </tr>
    <tr>
      <td>gamma不为空指针且gamma的元素数量不为C。</td>
    </tr>
  </tbody></table>

## aclnnGroupNormBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupNormBackwardGetWorkspaceSize获取。</td>
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
  - aclnnGroupNormBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> gradOutShape = {2, 3, 4};
    std::vector<int64_t> inputShape = {2, 3, 4};
    std::vector<int64_t> meanShape = {2, 1};
    std::vector<int64_t> rstdShape = {2, 1};
    std::vector<int64_t> gammaShape = {3};
    std::vector<int64_t> gradInputShape = {2, 3, 4};
    std::vector<int64_t> gradGammaOutShape = {3};
    std::vector<int64_t> gradBetaOutShape = {3};
    void* gradOutDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    void* gradGammaOutDeviceAddr = nullptr;
    void* gradBetaOutDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* input = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* gradInput = nullptr;
    aclTensor* gradGammaOut = nullptr;
    aclTensor* gradBetaOut = nullptr;
    std::vector<float> gradOutHostData = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> inputHostData = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                        13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> meanHostData = {6.5, 18.5};
    std::vector<float> rstdHostData = {0.2896827, 0.2896827};
    std::vector<float> gammaHostData = {1.0, 1.0, 1.0};
    std::vector<float> gradInputHostData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<float> gradGammaOutHostData = {0.0, 0.0, 0.0};
    std::vector<float> gradBetaOutHostData = {0.0, 0.0, 0.0};
    int64_t N = 2;
    int64_t C = 3;
    int64_t HxW = 4;
    int64_t group = 1;
    std::array<bool, 3> outputMaskData = {true, true, true};
    // 创建gradOut aclTensor
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mean aclTensor
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rstd aclTensor
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gamma aclTensor
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    auto outputMask = aclCreateBoolArray(outputMaskData.data(), outputMaskData.size());
    CHECK_RET(outputMask != nullptr, return ACL_ERROR_INTERNAL_ERROR);
    // 创建gradInput aclTensor
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradGammaOut aclTensor
    ret = CreateAclTensor(
        gradGammaOutHostData, gradGammaOutShape, &gradGammaOutDeviceAddr, aclDataType::ACL_FLOAT, &gradGammaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradBetaOut aclTensor
    ret = CreateAclTensor(
        gradBetaOutHostData, gradBetaOutShape, &gradBetaOutDeviceAddr, aclDataType::ACL_FLOAT, &gradBetaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的HostApi
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnGroupNormBackward第一段接口
    ret = aclnnGroupNormBackwardGetWorkspaceSize(
        gradOut, input, mean, rstd, gamma, N, C, HxW, group, outputMask, gradInput, gradGammaOut, gradBetaOut,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnGroupNormBackward第二段接口
    ret = aclnnGroupNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> gradInputResultData(size, 0);
    ret = aclrtMemcpy(
        gradInputResultData.data(), gradInputResultData.size() * sizeof(gradInputResultData[0]), gradInputDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradInputResultData[%ld] is: %f\n", i, gradInputResultData[i]);
    }

    size = GetShapeSize(gradGammaOutShape);
    std::vector<float> gradGammaOutResultData(size, 0);
    ret = aclrtMemcpy(
        gradGammaOutResultData.data(), gradGammaOutResultData.size() * sizeof(gradGammaOutResultData[0]),
        gradGammaOutDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradGammaOutResultData[%ld] is: %f\n", i, gradGammaOutResultData[i]);
    }

    size = GetShapeSize(gradBetaOutShape);
    std::vector<float> gradBetaOutResultData(size, 0);
    ret = aclrtMemcpy(
        gradBetaOutResultData.data(), gradBetaOutResultData.size() * sizeof(gradBetaOutResultData[0]),
        gradBetaOutDeviceAddr, size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradBetaOutResultData[%ld] is: %f\n", i, gradBetaOutResultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOut);
    aclDestroyTensor(input);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(gamma);
    aclDestroyTensor(gradInput);
    aclDestroyTensor(gradGammaOut);
    aclDestroyTensor(gradBetaOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(gradGammaOutDeviceAddr);
    aclrtFree(gradBetaOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```