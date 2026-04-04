# aclnnBatchNorm

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/batch_norm_v3)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |


## 功能说明

- 接口功能：对一个批次的数据做批量归一化处理，正则化之后生成的数据的统计结果为0均值、1标准差。

- 计算公式：

  $$
  y = \frac{(x - E(x))}{\sqrt{Var(x) + eps}} * weight + bias
  $$
  E(x)表示均值，Var(x)表示方差，均需要在算子内部计算得到；ε表示一个极小的浮点数，防止分母为0的情况。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBatchNormGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBatchNorm”接口执行计算。

```Cpp
aclnnStatus aclnnBatchNormGetWorkspaceSize(
  const aclTensor *input,
  const aclTensor *weight,
  const aclTensor *bias,
  aclTensor       *runningMean,
  aclTensor       *runningVar,
  bool             training,
  double           momentum,
  double           eps,
  aclTensor       *output,
  aclTensor       *saveMean,
  aclTensor       *saveInvstd,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBatchNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBatchNormGetWorkspaceSize

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
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行BatchNorm计算的输入，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>支持的shape和格式有：2维（对应的格式为NC），3维（对应的格式为NCL），4维（对应的格式为NCHW、NHWC），5维（对应的格式为NCDHW、NDHWC），6-8维（对应的格式为ND，其中第2维固定为channel轴）。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NC、NCL、NCHW、NHWC、NCDHW、NDHWC、ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行BatchNorm计算的权重Tensor，对应公式中的`weight`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选参数。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行BatchNorm计算的偏置Tensor，对应公式中的`bias`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选参数。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningMean（aclTensor*）</td>
      <td>输入</td>
      <td>表示推理期间使用的平均值，对应公式中的`E(x)`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选参数。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>runningVar（aclTensor*）</td>
      <td>输入</td>
      <td>表示推理期间使用的方差，对应公式中的`Var(x)`。</td>
      <td><ul><li>支持空Tensor。</li><li>可选参数，数值为非负数。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    </tr>
    <tr>
      <td>training（bool）</td>
      <td>输入</td>
      <td>表示标记是否训练场景。</td>
      <td>true表示训练场景，false表示推理场景。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>momentum（double）</td>
      <td>输入</td>
      <td>表示用于更新运行均值和方差。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps（double）</td>
      <td>输入</td>
      <td>表示添加到方差中的值，以避免出现除以零的情况。对应公式中的`eps`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output（aclTensor*）</td>
      <td>输出</td>
      <td>表示BatchNorm的输出结果，对应公式中的`y`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、数据格式和shape与入参`input`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NC、NCL、NCHW、NHWC、NCDHW、NDHWC、ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveMean（aclTensor*）</td>
      <td>输出</td>
      <td>只在训练场景输出，表示保存的均值，对应公式中的`E(x)`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>saveInvstd（aclTensor*）</td>
      <td>输出</td>
      <td>只在训练场景输出，表示保存的input方差或者input标准差倒数，分别对应公式中的`Var(x)`、(Var(x) + eps)开平方的倒数。</td>
      <td><ul><li>支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
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

  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：
    - 参数`input`、`weight`、`bias`、`runningMean`、`runningVar`、`output`、`saveMean`、`saveInvstd`的数据类型不支持BFLOAT16。
    - 参数`input`、`output`的数据格式不支持NHWC、NDHWC。
    - 参数`saveInvstd`表示input的方差。
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 参数`input`、`output`的数据格式不支持NHWC、NDHWC。
    - 参数`saveInvstd`表示input的方差。
  - <term>Ascend 950PR/Ascend 950DT</term>：参数`saveInvstd`表示input标准差的倒数。
  
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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的指针类型入参是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>输入或输出Tensor的数据类型和数据格式不在支持的范围之内。</td>
    <tr>
      <td>输入或输出Tensor的shape不在支持的范围内。</td>
    </tr>
    <tr>
      <td>weight，bias，runningMean，runningVar、saveMean（训练场景下）、saveInvstd（训练场景下）的shape长度与input shape中channel轴的长度不相等。</td>
    </tr>
    <tr>
      <td>input和output的数据格式不一致。</td>
    </tr>
    <tr>
      <td>input和output的shape不一致，或者shape的维度大于8维或者小于2维。</td>
    </tr>
    </tr>
  </tbody></table>

## aclnnBatchNorm

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBatchNormGetWorkspaceSize获取。</td>
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
  - aclnnBatchNorm默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_norm.h"

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
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {1, 2, 4};
    std::vector<int64_t> weightShape = {2};
    std::vector<int64_t> biasShape = {2};
    std::vector<int64_t> rMeanShape = {2};
    std::vector<int64_t> rVarShape = {2};
    std::vector<int64_t> outShape = {1, 2, 4};
    std::vector<int64_t> meanShape = {2};
    std::vector<int64_t> varShape = {2};
    void* selfDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* rMeanDeviceAddr = nullptr;
    void* rVarDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* varDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* rMean = nullptr;
    aclTensor* rVar = nullptr;
    aclTensor* out = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* var = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> biasHostData = {0, 0};
    std::vector<float> rMeanHostData = {0, 0};
    std::vector<float> rVarHostData = {1, 1};
    std::vector<float> outHostData(8, 0);
    std::vector<float> meanHostData = {1, 1};
    std::vector<float> varHostData = {1, 1};
    bool training = true;
    double momentum = 0.1;
    double eps = 1e-5;

    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rMean aclTensor
    ret = CreateAclTensor(rMeanHostData, rMeanShape, &rMeanDeviceAddr, aclDataType::ACL_FLOAT, &rMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rVar aclTensor
    ret = CreateAclTensor(rVarHostData, rVarShape, &rVarDeviceAddr, aclDataType::ACL_FLOAT, &rVar);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mean aclTensor
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建var aclTensor
    ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnBatchNorm接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // 调用aclnnBatchNorm第一段接口
    ret = aclnnBatchNormGetWorkspaceSize(
        self, weight, bias, rMean, rVar, training, momentum, eps, out, mean, var, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnBatchNorm第二段接口
    ret = aclnnBatchNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchNorm failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(rMean);
    aclDestroyTensor(rVar);
    aclDestroyTensor(out);
    aclDestroyTensor(mean);
    aclDestroyTensor(var);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(rMeanDeviceAddr);
    aclrtFree(rVarDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(varDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```