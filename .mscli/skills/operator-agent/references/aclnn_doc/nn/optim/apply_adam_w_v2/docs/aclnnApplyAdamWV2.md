# aclnnApplyAdamWV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/optim/apply_adam_w_v2)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能： 实现adamW优化器功能。

- 计算公式：

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$

  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$

  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$

  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnApplyAdamWV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnApplyAdamWV2”接口执行计算。

```cpp
aclnnStatus aclnnApplyAdamWV2GetWorkspaceSize(
    aclTensor       *varRef,
    aclTensor       *mRef,
    aclTensor       *vRef,
    aclTensor       *maxGradNormOptionalRef,
    const aclTensor *grad,
    const aclTensor *step,
    float            lr,
    float            beta1,
    float            beta2,
    float            weightDecay,
    float            eps,
    bool             amsgrad,
    bool             maximize,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnApplyAdamWV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnApplyAdamWV2GetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1520px"><colgroup>
    <col style="width: 230px">
    <col style="width: 120px">
    <col style="width: 330px">
    <col style="width: 220px">
    <col style="width: 230px">
    <col style="width: 115px">
    <col style="width: 130px">
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
        <td>varRef（aclTensor*）</td>
        <td>输入/输出</td>
        <td>待计算的权重输入同时也是输出，公式中的θ。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mRef（aclTensor*）</td>
        <td>输入/输出</td>
        <td>adamw优化器中m参数，公式中的m。</td>
        <td>-</td>
        <td>与“varRef”参数一致。</td>
        <td>ND</td>
        <td>与“varRef”参数一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>vRef（aclTensor*）</td>
        <td>输入/输出</td>
        <td>adamw优化器中v参数，公式中的v。</td>
        <td>-</td>
        <td>与“varRef”参数一致。</td>
        <td>ND</td>
        <td>与“varRef”参数一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maxGradNormOptionalRef（aclTensor*）</td>
        <td>输入/输出</td>
        <td>输入maxGradNormOptionalRef与更新后的vRef比较后，得到的最大值，在maxGradNormOptionalRef输出。</td>
        <td>此参数在amsgrad参数为true时必选，在amsgrad参数为false时可选。</td>
        <td>与“varRef”参数一致。</td>
        <td>ND</td>
        <td>与“varRef”参数一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>grad（aclTensor*）</td>
        <td>输入</td>
        <td>梯度数据，公式中的g<sub>t</sub>。</td>
        <td>-</td>
        <td>与“varRef”参数一致。</td>
        <td>ND</td>
        <td>与“varRef”参数一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>step（aclTensor*）</td>
        <td>输入</td>
        <td>迭代次数，公式中的t。</td>
        <td>元素个数为1。</td>
        <td>INT64、FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>x</td>
      </tr>
      <tr>
        <td>lr（double）</td>
        <td>输入</td>
        <td>学习率，公式中的η。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta1（double）</td>
        <td>输入</td>
        <td>β<sub>1</sub>参数。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta2（double）</td>
        <td>输入</td>
        <td>β<sub>2</sub>参数。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightDecay（double）</td>
        <td>输入</td>
        <td>权重衰减系数。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eps（double）</td>
        <td>输入</td>
        <td>防止除数为0。</td>
        <td>-</td>
        <td>FLOAT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>amsgrad（bool）</td>
        <td>输入</td>
        <td>是否使用算法的AMSGrad变量。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maximize（bool）</td>
        <td>输入</td>
        <td>是否最大化参数。</td>
        <td>-</td>
        <td>BOOL</td>
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
    </tbody></table>

- **返回值：**

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
      <td>传入的varRef、mRef、vRef、maxGradNormOptionalRef、grad、step是空指针时。</td>
      </tr>
      <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>varRef、mRef、vRef、maxGradNormOptionalRef、grad、step的数据类型不在支持的范围内时。</td>
      </tr>
      <tr>
      <td>varRef、mRef、vRef、maxGradNormOptionalRef、grad、step的数据格式不在支持的范围内时。</td>
      </tr>
      <tr>
      <td>mRef、vRef、grad和varRef的shape不一致时。</td>
      </tr>
      <tr>
      <td>当amsgrad为true时，maxGradNormOptionalRef和varRef的shape不一致时。</td>
      </tr>
      <tr>
      <td>step的shape大小不为1时。</td>
      </tr>
    </tbody>
    </table>

## aclnnApplyAdamWV2

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnApplyAdamWV2GetWorkspaceSize获取。</td>
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
- 输入张量中varRef、mRef、vRef、grad的数据类型必须一致时，数据类型支持FLOAT16、BFLOAT16、FLOAT32。
- 输入张量中varRef、mRef、vRef、grad的shape必须保持一致。
- 确定性计算：
  - aclnnApplyAdamWV2默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_w_v2.h"

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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> varShape = {2, 2};
  std::vector<int64_t> mShape = {2, 2};
  std::vector<int64_t> vShape = {2, 2};
  std::vector<int64_t> gradShape = {2, 2};
  std::vector<int64_t> maxgradShape = {2, 2};
  std::vector<int64_t> stepShape = {1};
  void* varDeviceAddr = nullptr;
  void* mDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  void* gradDeviceAddr = nullptr;
  void* maxgradDeviceAddr = nullptr;
  void* stepDeviceAddr = nullptr;
  aclTensor* var = nullptr;
  aclTensor* m = nullptr;
  aclTensor* v = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* maxgrad = nullptr;
  aclTensor* step = nullptr;
  std::vector<float> varHostData = {0, 1, 2, 3};
  std::vector<float> mHostData = {0, 1, 2, 3};
  std::vector<float> vHostData = {0, 1, 2, 3};
  std::vector<float> gradHostData = {0, 1, 2, 3};
  std::vector<float> maxgradHostData = {0, 1, 2, 3};
  std::vector<float> stepHostData = {1};
  bool amsgrad = true;
  bool maximize = true;
  float lr = 1e-3;
  float beta1 = 0.9;
  float beta2 = 0.999;
  float weightDecay = 1e-2;
  float eps = 1e-8;
  // 创建var aclTensor
  ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建m aclTensor
  ret = CreateAclTensor(mHostData, mShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建v aclTensor
  ret = CreateAclTensor(vHostData, vShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建grad aclTensor
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建maxgrad aclTensor
  ret = CreateAclTensor(maxgradHostData, maxgradShape, &maxgradDeviceAddr, aclDataType::ACL_FLOAT, &maxgrad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建step aclTensor
  ret = CreateAclTensor(stepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_FLOAT, &step);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnApplyAdamWV2第一段接口
  ret = aclnnApplyAdamWV2GetWorkspaceSize(var, m, v, maxgrad, grad, step, lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnApplyAdamWV2第二段接口
  ret = aclnnApplyAdamWV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(varShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), varDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(grad);
  aclDestroyTensor(maxgrad);
  aclDestroyTensor(step);

  // 7. 释放device 资源
  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(gradDeviceAddr);
  aclrtFree(maxgradDeviceAddr);
  aclrtFree(stepDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
