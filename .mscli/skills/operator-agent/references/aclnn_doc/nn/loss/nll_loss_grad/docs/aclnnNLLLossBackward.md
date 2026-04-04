# aclnnNLLLossBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/nll_loss_grad)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：负对数似然损失函数的反向传播。
- 计算公式：
  - reduction=mean：

    $$
    x\_grad_{target(t)} =\begin{cases}
    (-gradOutput * w_{(target(t))}) / totalweight &, target(t)=1 \\
    0 &, target(t)=0
    \end{cases}
    $$

  - reduction=sum：

    $$
    x\_grad_{target(t)} =\begin{cases}
    -gradOutput * w_{(target(t))} &, target(t)=1 \\
    0 &, target(t)=0
    \end{cases}
    $$

  - reduction=none：

    $$
    x\_grad_{target(t)} =\begin{cases}
    -gradOutput_t * w_{(target(t))} &, target(t)=1 \\
    0 &, target(t)=0
    \end{cases}
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNLLLossBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNLLLossBackward”接口执行计算。

```cpp
aclnnStatus aclnnNLLLossBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weight,
    int64_t          reduction,
    int64_t          ignoreIndex,
    const aclTensor *totalWeight,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnNLLLossBackward(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnNLLLossBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 250px">
      <col style="width: 350px">
      <col style="width: 220px">
      <col style="width: 115px">
      <col style="width: 120px">
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
        <td>gradOutput（aclTensor*）</td>
        <td>输入</td>
        <td>输入aclTensor。</td>
        <td>shape为(N)、一维（且元素个数为1）或()。<ul><li>当reduction为0，且self的shape为(N,C)时，shape支持(N)。</li><li>当reduction为0，且self的shape为(C)时，shape支持一维（且元素个数为1）或()。</li><li>当reduction不为0，shape支持一维（且元素个数为1）或()。</li></ul></td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>self（aclTensor*）</td>
        <td>输入</td>
        <td>输入aclTensor。</td>
        <td>数据类型与出参out的数据类型一致，shape与出参out的shape一致。<ul><li>shape为(N,C)或者(C)，其中N表示batch size，C表示类别数。</li><li>当target的shape为(N)时，self的shape需为(N,C)。</li><li>当target的shape为()，self的shape需为(C)。</li></ul></td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target（aclTensor*）</td>
        <td>输入</td>
        <td>表示真实标签。</td>
        <td>shape为(N) 或者()，其中每个元素的取值范围是[0, C - 1]。</td>
        <td>INT64、UINT8、INT32</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight（aclTensor*）</td>
        <td>输入</td>
        <td>表示每个类别的缩放权重。</td>
        <td>公式中的w，shape为(C,)。</td>
        <td>数据类型和self保持一致。</td>
        <td>ND</td>
        <td>(C)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction（int64_t）</td>
        <td>输入</td>
        <td>指定损失函数的计算方式。</td>
        <td><ul>支持0(none)|1(mean)|2(sum)。<li>'none'表示不应用缩减</li><li>'mean'表示输出的总和将除以输出中的元素数</li><li>'sum'表示输出将被求和</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignoreIndex（int64_t）</td>
        <td>输入</td>
         <td>指定一个被忽略且不影响输入梯度的目标值。
        </td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
          <tr>
        <td>totalWeight（aclTensor*）</td>
        <td>输入</td>
        <td>公式中的totalWeight。</td>
        <td>仅当reduction为mean时，totalWeight是通过target取相应位置的weight，然后去除掉ignoreIndex对应的weight，将剩下的weight求和；当reduction为其他值时，该参数默认不处理。</td>
        <td>数据类型与weight相同。</td>
        <td>ND</td>
        <td>(1,)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out（aclTensor*）</td>
        <td>输出</td>
        <td>公式中的out。</td>
        <td>shape与self相同。</td>
        <td>数据类型和self一致。</td>
        <td>ND</td>
        <td>-</td>
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


- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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
      <td>传入的gradOutput、self、target、weight、out、reduction、totalWeight为空指针。</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td> gradOutput、self、target、weight、totalWeight的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
      <td>gradOutput、self、weight、totalWeight的数据类型不一致。</td>
      </tr>
       <tr>
      <td>gradOutput、self、weight、out、totalWeight的shape不正确。</td>
      </tr>
      <tr>
      <td> reduction值不在0~2范围之内。</td>
      </tr>
      </tbody>
      </table>

## aclnnNLLLossBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNLLLossBackwardGetWorkspaceSize获取。</td>
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
    - aclnnNLLLossBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nll_loss_backward.h"

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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  std::vector<int64_t> gradOutputShape = {2};
  std::vector<int64_t> selfShape = {2, 3};
  std::vector<int64_t> targetShape = {2};
  std::vector<int64_t> weightShape = {3};
  std::vector<int64_t> totalWeightShape = {1};
  std::vector<int64_t> outShape = {2, 3};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* totalWeightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* totalWeight = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutputHostData = {3.1, 6.5};
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5};
  std::vector<int32_t> targetHostData = {0, 2};
  std::vector<float> weightHostData = {1.1, 1.2, 1.3};
  std::vector<float> totalWeightHostData = {0};
  std::vector<float> outHostData(6, 0);
  int64_t reduction = 0;
  int64_t ignoreIndex = -100;
  // 创建gradOutput aclTensor
  ret =
      CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT32, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建totalWeight aclTensor
  ret = CreateAclTensor(totalWeightHostData, totalWeightShape, &totalWeightDeviceAddr, aclDataType::ACL_FLOAT,
                        &totalWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnNLLLossBackward第一段接口
  ret = aclnnNLLLossBackwardGetWorkspaceSize(gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight, out,
                                             &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnNLLLossBackward第二段接口
  ret = aclnnNLLLossBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLossBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
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
  aclDestroyTensor(totalWeight);
  aclDestroyTensor(out);

  // 7. 释放device 资源
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(totalWeightDeviceAddr);
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

