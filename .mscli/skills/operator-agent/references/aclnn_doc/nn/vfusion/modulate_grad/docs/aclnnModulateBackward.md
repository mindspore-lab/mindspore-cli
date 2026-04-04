# aclnnModulateBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：完成ModulateBackward反向传播中参数的计算，进行梯度更新。
- 计算公式：

    设输入self的shape为[B, L, D]计算公式如下：
    公式：

    $$
    \begin{cases}
    \text{grad\_input} = \text{grad\_output} \odot \text{scale}^{\uparrow L} \\
    \text{grad\_scale} = \sum_{l=1}^{L} (\text{grad\_output} \odot \text{input})_{b,l,d} \\
    \text{grad\_shift} = \sum_{l=1}^{L} \text{grad\_output}_{b,l,d}
    \end{cases}
    $$

    符号说明：
    - $\odot$: 表示逐元素乘法；
    - $\sum_{l=1}^{L}$: 求和操作，沿序列维度$L$(即dim=1)进行
    -  $b,l,d$：下标，表示张量的维度索引（通常为Batch，Length，Dimension）
    - $\text{scale}^{\uparrow L}$： 表示将scale张量在序列维度 $L$ 上进行广播（扩展）

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnModulateBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnModulateBackward”接口执行计算。

```Cpp
aclnnStatus aclnnModulateBackwardGetWorkspaceSize(
    const aclTensor* grad_output,
    const aclTensor* input,
    const aclTensor* scale,
    const aclTensor* shift,
    const aclTensor* grad_input,
    const aclTensor* grad_scale,
    const aclTensor* grad_shift,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```
```Cpp
aclnnStatus aclnnModulateBackward(
    void*          workspaceAddr,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnModulateBackwardGetWorkspaceSize

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>grad_output</td>
      <td>输入</td>
      <td>表示传入的特征张量，公式中的grad_output。</td>
      <td>-</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, seq_len, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>表示正向传播的特征张量，公式中的input。</td>
      <td>-</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, seq_len, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>可选参数，表示缩放的系数，公式中的scale。</td>
      <td>数据类型需要与grad_output相同且只支持2维输入</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>输入</td>
      <td>可选参数，表示平移的系数，决定是否有对应输出。</td>
      <td>数据类型需要与grad_output相同且只支持2维输入。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>输出</td>
      <td>表示输入张量的梯度，公式中的grad_input。</td>
      <td>数据类型和shape需要与input相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grad_scale</td>
      <td>输出</td>
      <td>表示缩放的梯度，公式中的grad_scale。</td>
      <td>数据类型和shape需要与scale相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
        <tr>
      <td>grad_shift</td>
      <td>输出</td>
      <td>表示平移的梯度，公式中的grad_shift。</td>
      <td>数据类型和shape需要与shift相同。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
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
- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>传入的self或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>self、scaleOptional、shiftOptional的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self、scaleOptional、shiftOptional之间的shape不满足约束。</td>
    </tr>
    <tr>
      <td>self为空tensor，且scale或shift不为空tensor。</td>
    </tr>
  </tbody>
  </table>


## aclnnModulateBackward

- **参数说明**：

  <table><thead>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize获取。</td>
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
- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>传入的grad_output或input是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>输入和输出的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input、scale、shift之间的shape不满足约束。</td>
    </tr>
    <tr>
      <td>输入维度超过3维</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 确定性计算：
  - aclnnModulateBackward默认确定性实现。

- scale和shift是二维向量，第一维需要和input的第一维shape相同，第二维需要和input的第三维shape相同。
- 输入gradoutput的shape需要和输入input的shape保持一致。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_modulate_backward.h"

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

void PrintOutResult(const std::vector<int64_t>& shape, void* deviceAddr, const std::string& name){
    auto size = GetShapeSize(shape);
    std::vector<float>resultData(size,0);
    auto ret = aclrtMemcpy(resultData.data(),resultData.size() * sizeof(float),deviceAddr,
                            size * sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR:%d\n",name.c_str(),ret); return);

    int print_count = std::min(static_cast<int64_t>(10),size);
    LOG_PRINT("%s (first %d elements):\n",name.c_str(),print_count);
    for (int64_t i = 0; i < print_count; i++){
        LOG_PRINT(" [%ld]: %f\n",i,resultData[i]);
    }
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> grad_outputShape = {2, 1, 1};
    std::vector<int64_t> inputShape = {2, 1, 1};
    std::vector<int64_t> scaleShape = {2, 1};
    std::vector<int64_t> shiftShape = {2, 1};
    std::vector<int64_t> grad_inputShape = {2, 1, 1};
    std::vector<int64_t> grad_scaleShape = {2, 1};
    std::vector<int64_t> grad_shiftShape = {2, 1};
    void* grad_outputDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* grad_inputDeviceAddr = nullptr;
    void* grad_scaleDeviceAddr = nullptr;
    void* grad_shiftDeviceAddr = nullptr;
    aclTensor* grad_output = nullptr;
    aclTensor* input = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* shift = nullptr;
    aclTensor* grad_input = nullptr;
    aclTensor* grad_scale = nullptr;
    aclTensor* grad_shift = nullptr;
    std::vector<float> grad_outputHostData{10, 20};
    std::vector<float> inputHostData{10, 20};
    std::vector<float> scaleHostData{20, 30};
    std::vector<float> shiftHostData{30, 40};
    std::vector<float> grad_inputHostData{0, 0};
    std::vector<float> grad_scaleHostData{0, 0};
    std::vector<float> grad_shiftHostData{0, 0};
    // 创建grad_output aclTensor
    ret = CreateAclTensor(grad_outputHostData, grad_outputShape, &grad_outputDeviceAddr, aclDataType::ACL_FLOAT, &grad_output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(
        scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建shift aclTensor
    ret = CreateAclTensor(
        shiftHostData, shiftShape, &shiftDeviceAddr, aclDataType::ACL_FLOAT, &shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建grad_input aclTensor
    ret = CreateAclTensor(grad_inputHostData, grad_inputShape, &grad_inputDeviceAddr, aclDataType::ACL_FLOAT, &grad_input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建grad_scale aclTensor
    ret = CreateAclTensor(grad_scaleHostData, grad_scaleShape, &grad_scaleDeviceAddr, aclDataType::ACL_FLOAT, &grad_scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建grad_shift aclTensor
    ret = CreateAclTensor(grad_shiftHostData, grad_shiftShape, &grad_shiftDeviceAddr, aclDataType::ACL_FLOAT, &grad_shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 调用aclnnModulate第一段接口
    ret = aclnnModulateBackwardGetWorkspaceSize(grad_output, input, scale, shift, grad_input, grad_scale, grad_shift, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnModulaBackwardteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnModulate第二段接口
    ret = aclnnModulateBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnModulateBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
   PrintOutResult(grad_inputShape, grad_inputDeviceAddr,"grad_input");
   PrintOutResult(grad_scaleShape, grad_scaleDeviceAddr,"grad_scale");
   PrintOutResult(grad_shiftShape, grad_shiftDeviceAddr,"grad_shift");

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(grad_output);
    aclDestroyTensor(input);
    aclDestroyTensor(scale);
    aclDestroyTensor(shift);
    aclDestroyTensor(grad_input);
    aclDestroyTensor(grad_scale);
    aclDestroyTensor(grad_shift);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(grad_outputDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(grad_inputDeviceAddr);
    aclrtFree(grad_scaleDeviceAddr);
    aclrtFree(grad_shiftDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    LOG_PRINT("Program completed successfully.\n");
    return 0;
}
```
