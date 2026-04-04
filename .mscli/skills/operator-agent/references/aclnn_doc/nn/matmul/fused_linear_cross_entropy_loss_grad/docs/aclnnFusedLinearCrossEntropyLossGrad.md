# aclnnFusedLinearCrossEntropyLossGrad

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/fused_linear_cross_entropy_loss_grad)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：本算子是词汇表并行场景下交叉熵损失计算模块中的一部分，解决超大规模词汇表下的显存和计算效率问题，当前部分为梯度计算实现，用于计算叶子节点`input`和`weight`的梯度。
  需要获得`aclnnFusedLinearOnlineMaxSum`、`aclnnFusedCrossEntropyLossWithMaxSum`的相关输出，以及`logits`相关的全局通信结果作为本接口输入。
- 计算公式：

&emsp;&emsp;高性能模式，softmaxOptional非nullptr：

$$
\text{softmax} \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = \mathbf{1} - \text{target\_mask}.view(-1) \in \mathbb{R}^{BT}
$$

$$
\text{softmax}[\text{arange\_1d}, \text{masked\_target}] \leftarrow \text{softmax}[\text{arange\_1d}, \text{masked\_target}] - \text{softmax\_update}
$$

$$
\text{softmax} \leftarrow \text{softmax} \odot \text{grad}.unsqueeze(-1) \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{softmax} \cdot \text{weight}^T \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{softmax}^T \cdot \text{input} \in \mathbb{R}^{V \times H}
$$

</br>
&emsp;&emsp;省显存模式，softmaxOptional为nullptr：

$$
\text{vocab\_parallel\_logits} = \text{input} \cdot \text{weight}^T \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{logits\_sub} = \text{vocab\_parallel\_logits} - \text{logits\_max}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} = \exp(\text{logits\_sub}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{exp\_logits} \gets \frac{\text{exp\_logits}}{\text{sum\_exp\_logits}.unsqueeze(-1)} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_logits} = \text{exp\_logits} \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_2d} = \text{grad\_logits}.view(-1, \text{partition\_vocab\_size}) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{arange\_1d} = [0, 1, \dots, BT-1] \quad \in \mathbb{N}^{BT}
$$

$$
\text{softmax\_update} = 1 - \text{target\_mask}.view(-1) \quad \in \mathbb{R}^{BT}
$$

$$
\text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] \gets \text{grad\_2d}[\text{arange\_1d}, \text{masked\_target\_1d}] - \text{softmax\_update}
$$

$$
\text{grad\_logits} \gets \text{grad\_logits} \odot \text{grad}.unsqueeze(-1) \quad \in \mathbb{R}^{BT \times V}
$$

$$
\text{grad\_input} = \text{grad\_logits} \cdot \text{weight} \quad \in \mathbb{R}^{BT \times H}
$$

$$
\text{grad\_weight} = \text{grad\_logits}^T \cdot \text{input} \quad \in \mathbb{R}^{V \times H}
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnFusedLinearCrossEntropyLossGrad`接口执行计算。

  ```Cpp
  aclnnStatus aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(
    const aclTensor   *grad,
    const aclTensor   *input,
    const aclTensor   *weight,
    const aclTensor   *targetMask,
    const aclTensor   *maskedTarget,
    float              labelSmoothing,
    const aclTensor   *logitsMaxOptional,
    const aclTensor   *sumExpLogitsOptional,
    const aclTensor   *softmaxOptional,
    aclTensor         *inputGradOut,
    aclTensor         *weightGradOut,
    uint64_t          *workspaceSize,
    aclOpExecutor    **executor)
  ```
  ```Cpp
  aclnnStatus aclnnFusedLinearCrossEntropyLossGrad(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    aclrtStream       stream)
  ```

## aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
    <col style="width: 169px">
    <col style="width: 121px">
    <col style="width: 264px">
    <col style="width: 253px">
    <col style="width: 242px">
    <col style="width: 148px">
    <col style="width: 135px">
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
        <th>非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>grad</td>
        <td>输入</td>
        <td>当前节点的梯度，公式中的输入grad。</td>
        <td>支持空tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>input</td>
        <td>输入</td>
        <td>矩阵乘的输入矩阵，公式中的输入input。</td>
        <td>第1维长度与输入grad的长度相同，支持空tensor。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>矩阵乘的权重矩阵，公式中的weight。</td>
        <td>数据类型与输入input相同，shape第1维长度不支持小于128，第2维长度与输入input的第2维长度相同。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>targetMask</td>
        <td>输入</td>
        <td>中间变量，代表对应词ID是否在目标范围内，公式中的target_mask。</td>
        <td><ul><li>每1bit数据代表1个布尔值，0代表false，1代表true。</li><li>shape长度乘以8须不小于输入grad的长度。</li></ul></td>
        <td>UINT8</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maskedTarget</td>
        <td>输入</td>
        <td>中间变量，代表对应词ID映射到当前设备词汇表分片的局部索引，无效目标会被掩码targetMask处理，公式中的masked_target。</td>
        <td>shape长度与输入grad相同。</td>
        <td>INT64、INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>labelSmoothing</td>
        <td>输入</td>
        <td>标签平滑系数，用于缓解过拟合。</td>
        <td>目前仅支持取值为0。</td>
        <td>FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>logitsMaxOptional</td>
        <td>可选输入</td>
        <td>中间变量，全局logits的最大值，公式中的logits_max。</td>
        <td><ul><li>支持输入nullptr，输入nullptr的场景需要提供有效的softmaxOptional输入。</li><li>shape长度与输入grad相同。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sumExpLogitsOptional</td>
        <td>可选输入</td>
        <td>中间变量，处理后的logits，公式中的sum_exp_logits。</td>
        <td><ul><li>支持输入nullptr，输入nullptr的场景需要提供有效的softmaxOptional输入。</li><li>shape长度与输入grad相同。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>softmaxOptional</td>
        <td>可选输入</td>
        <td>中间变量，矩阵乘的结果，公式中的softmax。</td>
        <td><ul><li>支持输入nullptr，输入nullptr时须提供有效的logitsMaxOptional、sumExpLogitsOptional输入；输入非nullptr时，logitsMaxOptional、sumExpLogitsOptional输入无效。</li><li>shape第1维长度与输入grad长度相同，第二维长度与输入weight的第1维长度相同。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>inputGradOut</td>
        <td>输出</td>
        <td>对应叶子节点input的梯度，公式中的grad_input。</td>
        <td><ul><li>数据类型与输入input相同。</li><li>shape第1维长度与输入grad长度相同，第2维长度与输入weight的第2维长度相同。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightGradOut</td>
        <td>输出</td>
        <td>对应叶子节点weight的梯度，公式中的grad_weight。</td>
        <td><ul><li>数据类型与输入input相同。</li><li>shape第1维长度与输入weight的第1维长度相同，第2维长度与输入grad长度相同。</li></ul></td>
        <td>FLOAT16、BFLOAT16</td>
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
    </tbody></table>

- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>传入的非可选输入是空指针。</td>
    </tr>
    <tr>
      <td>传入的softmaxOptional为空指针的场景下，logitsMaxOptional或sumExpLogitsOptional为空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>输入的labelSmoothing取值不支持。</td>
    </tr>
    <tr>
      <td>输入的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>输入的数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>输入的shape不在支持的范围内，不满足长度要求。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedLinearCrossEntropyLossGrad

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize获取。</td>
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
  
- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性说明：
  - aclnnFusedLinearCrossEntropyLossGrad默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_linear_cross_entropy_loss_grad.h"

#define CHECK_RET(cond, return_expr) \
    do                               \
    {                                \
        if (!(cond))                 \
        {                            \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do                                  \
    {                                   \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
std::vector<T> GenZeroVector(const std::vector<int64_t>& shape) {
    // 1. 计算总元素数量
    size_t total = 1;
    for (auto dim : shape) {
        total *= dim;
    }

    // 2. 填充0
    std::vector<T> vec(total);
    for (auto& elem : vec) {
        elem = 0;
    }
    return vec;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateEmptyAclTensor(const std::vector<int64_t> &shape, void **deviceAddr,
                         aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

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

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t BT = 1024;
    int64_t V = 1024;
    int64_t H = 1024;
    std::vector<int64_t> gradShape = {BT};
    std::vector<int64_t> inputShape = {BT, H};
    std::vector<int64_t> weightShape = {V, H};
    std::vector<int64_t> targetMaskShape = {BT};
    std::vector<int64_t> maskedTargetShape = {BT};
    std::vector<int64_t> softmaxOptionalShape = {BT, V};
    std::vector<int64_t> inputGradOutShape = {BT, H};
    std::vector<int64_t> weightGradOutShape = {V, H};
    void *gradDeviceAddr = nullptr;
    void *inputDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *targetMaskDeviceAddr = nullptr;
    void *maskedTargetDeviceAddr = nullptr;
    void *softmaxOptionalDeviceAddr = nullptr;
    void *inputGradOutDeviceAddr = nullptr;
    void *weightGradOutDeviceAddr = nullptr;
    aclTensor *grad = nullptr;
    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *targetMask = nullptr;
    aclTensor *maskedTarget = nullptr;
    float labelSmoothing = 0.0;
    aclTensor *logitsMaxOptional = nullptr;
    aclTensor *sumExpLogitsOptional = nullptr;
    aclTensor *softmaxOptional = nullptr;
    aclTensor *inputGradOut = nullptr;
    aclTensor *weightGradOut = nullptr;
    // 创建aclTensor
    auto gradData = GenZeroVector<int32_t>(gradShape);
    ret = CreateAclTensor<int32_t>(gradData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto inputData = GenZeroVector<int16_t>(inputShape);
    ret = CreateAclTensor<int16_t>(inputData, inputShape, &inputDeviceAddr, aclDataType::ACL_BF16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto weightData = GenZeroVector<int16_t>(weightShape);
    ret = CreateAclTensor<int16_t>(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto targetMaskData = GenZeroVector<int8_t>(targetMaskShape);
    ret = CreateAclTensor<int8_t>(targetMaskData, targetMaskShape, &targetMaskDeviceAddr, aclDataType::ACL_UINT8, &targetMask);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto maskedTargetData = GenZeroVector<int32_t>(maskedTargetShape);
    ret = CreateAclTensor<int32_t>(maskedTargetData, maskedTargetShape, &maskedTargetDeviceAddr, aclDataType::ACL_INT32, &maskedTarget);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto softmaxOptionalData = GenZeroVector<int32_t>(softmaxOptionalShape);
    ret = CreateAclTensor<int32_t>(softmaxOptionalData, softmaxOptionalShape, &softmaxOptionalDeviceAddr, aclDataType::ACL_FLOAT, &softmaxOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto inputGradOutData = GenZeroVector<int16_t>(inputGradOutShape);
    ret = CreateAclTensor<int16_t>(inputGradOutData, inputGradOutShape, &inputGradOutDeviceAddr, aclDataType::ACL_BF16, &inputGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    auto weightGradOutData = GenZeroVector<int16_t>(weightGradOutShape);
    ret = CreateAclTensor<int16_t>(weightGradOutData, weightGradOutShape, &weightGradOutDeviceAddr, aclDataType::ACL_BF16, &weightGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnFusedLinearCrossEntropyLossGrad第一段接口
    ret = aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(grad, input, weight, targetMask, maskedTarget, labelSmoothing, logitsMaxOptional, sumExpLogitsOptional, softmaxOptional, inputGradOut, weightGradOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnFusedLinearCrossEntropyLossGrad第二段接口
    ret = aclnnFusedLinearCrossEntropyLossGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedLinearCrossEntropyLossGrad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    // inputGradOut
    auto size = GetShapeSize(inputGradOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputGradOutDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < 16; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(grad);
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(targetMask);
    aclDestroyTensor(maskedTarget);
    aclDestroyTensor(softmaxOptional);
    aclDestroyTensor(inputGradOut);
    aclDestroyTensor(weightGradOut);

    // 7. 释放Device资源，需要根据具体API的接口定义修改
    aclrtFree(gradDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(targetMaskDeviceAddr);
    aclrtFree(maskedTargetDeviceAddr);
    aclrtFree(softmaxOptionalDeviceAddr);
    aclrtFree(inputGradOutDeviceAddr);
    aclrtFree(weightGradOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```