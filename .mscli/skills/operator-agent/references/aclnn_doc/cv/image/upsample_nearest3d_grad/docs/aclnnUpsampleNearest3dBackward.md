# aclnnUpsampleNearest3dBackward

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/image/upsample_nearest3d_grad)

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

- 接口功能：[aclnnUpsampleNearest3d](../../upsample_nearest3d/docs/aclnnUpsampleNearest3d.md)的反向计算。
- 计算公式：

  $$
  gradInput(N, C, D, H, W) += gradOut( N, C, ceil ( scalesD * D ), ceil ( scalesH * H  ),  ceil ( scalesW * W ))
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnUpsampleNearest3dBackwardGetWorkspaceSize`接口获取入参并根据计算流程计算所需workspace大小，再调用`aclnnUpsampleNearest3dBackward`接口执行计算。

```Cpp
aclnnStatus aclnnUpsampleNearest3dBackwardGetWorkspaceSize(
  const aclTensor   *gradOut, 
  const aclIntArray *outputSize, 
  const aclIntArray *inputSize, 
  double             scalesD, 
  double             scalesH, 
  double             scalesW, 
  aclTensor         *gradInput, 
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnUpsampleNearest3dBackward(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnUpsampleNearest3dBackwardGetWorkspaceSize

- **参数说明**
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
      <td>不支持空Tensor。</td>
      <td>FLOAT32、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize（aclIntArray*）</td>
      <td>输入</td>
      <td>表示输入`gradOut`在D、H和W维度上的空间大小。</td>
      <td>size为3，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize（aclIntArray*）</td>
      <td>输入</td>
      <td>表示输出`gradInput`的空间大小。</td>
      <td><ul><li>size为5，且最后两个元素均大于零。</li><li>当输入`gradOut`的数据格式为NCDHW时，表示输出`gradInput`分别在N、C、D、H和W维度上的空间大小；当输入`gradOut`的数据格式为NDHWC时，表示输出`gradInput`分别在N、D、H、W和C维度上的空间大小。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesD（double）</td>
      <td>输入</td>
      <td>表示输出`gradInput`的depth维度乘数，对应公式中的`scalesD`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>  
    <tr>
      <td>scalesH（double）</td>
      <td>输入</td>
      <td>表示输出`gradInput`的height维度乘数，对应公式中的`scalesH`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>  
    <tr>
      <td>scalesW（double）</td>
      <td>输入</td>
      <td>表示输出`gradInput`的width维度乘数，对应公式中的`scalesW`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput（aclTensor*）</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`gradInput`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型和数据格式与入参`gradOut`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、DOUBLE、BFLOAT16</td>
      <td>NCDHW、NDHWC</td>
      <td>5</td>
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>gradOut的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradOut和gradInput的数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradOut的维度不为5维。</td>
    </tr>
    <tr>
      <td>outputSize的size不等于3。</td>
    </tr>
    <tr>
      <td>outputSize的某个元素值不大于0。</td>
    </tr>
    <tr>
      <td>inputSize的size不等于5。</td>
    </tr>
    <tr>
      <td>inputSize的某个元素值不大于0。</td>
    </tr>
    <tr>
      <td>gradOut在D、H、W维度上的size与outputSize[0]、outputSize[1]、outputSize[2]不一致。</td>
    </tr>
    <tr>
      <td>gradInput在N、C维度的size与inputSize[0]、inputSize[1]不一致。</td>
    </tr>
    <tr>
      <td>gradInput在D、H、W维度上的size与inputSize[2]、inputSize[3]、inputSize[4]不一致。</td>
    </tr>
  </tbody></table>

## aclnnUpsampleNearest3dBackward

- **参数说明**
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUpsampleNearest3dBackwardGetWorkspaceSize获取。</td>
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

- 参数`gradOut`、`gradInput`的shape约束：
  - 每个维度的取值小于等于2^20。
  - 参数`gradInput`的N轴和C轴与`gradOut`保持一致。
  - 内存占用需小于60G。内存占用的计算公式如下：

    $$
    N * C * (gradOut\_D * gradOut\_H * gradOut\_W + gradInput\_D * gradInput\_H * gradInput\_W + gradOut\_D * gradOut\_H * gradInput\_W + gradOut\_D * gradInput\_H * gradInput\_W) * sizeof(float) < 60 * 1024 * 1024 * 1024
    $$

    其中：
    - N代表输入和输出的N轴。
    - C代表输入和输出的C轴。
  - N \* C \* gradOut_D \* gradOut_H < 2^31
  - gradInput_W * gradInput_H < 2^31
- 参数gradOut、gradInput的数据格式不为NCDHW或NDHWC时，输入其他数据格式默认按NCDHW处理。
- 参数inputSize、outputSize、scalesD、scalesH、scalesW需要满足如下约束：

  $$
  outputSize\_D = floor(inputSize\_D * scalesD)
  $$
  
  $$
  outputSize\_H = floor(inputSize\_H * scalesH)
  $$

  $$
  outputSize\_W = floor(inputSize\_W * scalesW)
  $$

- 确定性计算：
  - aclnnUpsampleNearest3dBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_nearest_3d_backward.h"

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

    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW, shape.data(),
        shape.size(), *deviceAddr);
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
    std::vector<int64_t> gradOutShape = {2, 2, 2, 2, 2};
    std::vector<int64_t> gradInputShape = {2, 2, 1, 1, 1};
    void* gradOutDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    aclTensor* gradOut = nullptr;
    aclTensor* gradInput = nullptr;
    std::vector<float> gradOutHostData = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32};
    std::vector<float> gradInputHostData = {2.0, 2, 2, 2};
    std::vector<int64_t> outputSizeData = {2, 2, 2};
    std::vector<int64_t> inputSizeData = {2, 2, 1, 1, 1};
    double scalesD = 0.0;
    double scalesH = 0.0;
    double scalesW = 0.0;
    // 创建gradOut aclTensor
    ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradInput aclTensor
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const aclIntArray* outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    const aclIntArray* inputSize = aclCreateIntArray(inputSizeData.data(), inputSizeData.size());
    CHECK_RET(inputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnUpsampleNearest3dBackward第一段接口
    ret = aclnnUpsampleNearest3dBackwardGetWorkspaceSize(
        gradOut, outputSize, inputSize, scalesD, scalesH, scalesW, gradInput, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearest3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnUpsampleNearest3dBackward第二段接口
    ret = aclnnUpsampleNearest3dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleNearest3dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOut);
    aclDestroyTensor(gradInput);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
