# aclnnUpsampleBilinear2dAABackward

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/image/upsample_bilinear2d_aa_backward)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：[aclnnUpsampleBilinear2dAA](../../upsample_bilinear2d_aa/docs/aclnnUpsampleBilinear2dAA.md)的反向传播。
- 计算公式：对于一个二维插值点$(N, C, H, W)$, 插值$I(N, C, H, W)$可以表示为：
  
  $$
  {I(N, C, H, W)} = \sum_{i=0}^{kW}\sum_{j=0}^{kH}{w(i) * w(j)} * {f(h_i, w_j)}/\sum_{i=0}^{kW}w(i)/\sum_{j=0}^{kH}w(j)
  $$
  
  $$
  scaleH =\begin{cases}
  (inputSize[2]-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  inputSize[2] / outputSize[0] & otherwise
  \end{cases}
  $$
  
  $$
  scaleW =\begin{cases}
  (inputSize[3]-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  inputSize[3] / outputSize[1] & otherwise
  \end{cases}
  $$
  
  - 其中：
    - $kW$、$kH$分别表示W方向和H方向影响插值点大小的点的数量
    - 如果$scaleH >= 1$，则$kH = floor(scaleH) * 2 + 1$，否则$kH = 3$
    - 如果$scaleW >= 1$，则$kW = floor(scaleW) * 2 + 1$，否则$kW = 3$
    - $f(h_i, w_j)$是原图像在$(h_i, w_j)$的像素值
    - $w(i)$、$w(j)$是双线性抗锯齿插值的W方向和H方向权重，计算公式为：

      $$
        w(i) = \begin{cases}
        1 - |h_i - h| & |h_i -h| < 1 \\
        0 & otherwise
        \end{cases}
      $$

      $$
        w(j) = \begin{cases}
        1 - |w_j - w| & |w_j -w| < 1 \\
        0 & otherwise
        \end{cases}
      $$

  - 假设：正向插值的输出图像out $(h, w)$受原图像input $(h_i, w_j)$影响，则有:
  
    $$
    gradInput(h_i,w_j) += gradOutput(h,w) * f(h_i,w_j)
    $$
  
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleBilinear2dAABackward”接口执行计算。

```Cpp
aclnnStatus aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,
  const aclIntArray *outputSize,
  const aclIntArray *inputSize,
  bool               alignCorners,
  double             scalesH,
  double             scalesW,
  aclTensor         *out,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnUpsampleBilinear2dAABackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize

- **参数说明**：

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
      <td>gradOutput（aclTensor*）</td>
      <td>输入</td>
      <td>表示反向计算的梯度Tensor，对应公式描述中的`gradOutput`。</td>
      <td><ul><li>不支持空Tensor。</li><li>当数据格式为ND时，默认按照NCHW格式处理。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW、ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize（aclIntArray*）</td>
      <td>输入</td>
      <td>表示输入`gradOutput`在H和W维度上的空间大小。对应公式中的`outputSize`。</td>
      <td>size为2，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputSize（aclIntArray*）</td>
      <td>输入</td>
      <td>表示输出`out`分别在N、C、H和W维度上的空间大小。对应公式中的`inputSize`。</td>
      <td>size为4，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners（bool）</td>
      <td>输入</td>
      <td>决定是否对齐角像素点，对应公式中的`alignCorners`。</td>
      <td>如果为true，则输入和输出张量的角像素点会被对齐，否则不对齐。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH（double）</td>
      <td>输入</td>
      <td>表示输出`out`的height维度乘数，对应公式中的`scalesH`。</td>
      <td>值为正数才生效。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW（double）</td>
      <td>输入</td>
      <td>表示输出`out`的width维度乘数，对应公式中的`scalesW`。</td>
      <td>值为正数才生效。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示反向计算的输出张量，对应公式中的`gradInput`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型和数据格式与入参`gradOutput`的数据类型和数据格式保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW、ND</td>
      <td>4</td>
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

* **返回值**：

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
      <td>传入的gradOutput、inputSize或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>gradOutput或out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>gradOutput与out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradOutput、out的shape不是4维。</td>
    </tr>
    <tr>
      <td>outputSize的size不等于2。</td>
    </tr>
    <tr>
      <td>outputSize的某个元素值不大于0。</td>
    </tr>
    <tr>
      <td>inputSize的size不等于4。</td>
    </tr>
    <tr>
      <td>inputSize的某个元素值不大于0。</td>
    </tr>
    <tr>
      <td>scalesH、scalesW的取值为负数。</td>
    </tr>
  </tbody></table>

## aclnnUpsampleBilinear2dAABackward

- **参数说明**：

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize获取。</td>
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

## 约束说明

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

  输入数据缩放场景放大倍数必须小于等于50，即：

  $$
  outputSize\_H / 输出shape的高度H <= 50
  $$
  
  $$
  outputSize\_W / 输出shape的宽度W <=50
  $$

- 参数inputSize、outputSize、scalesH、scalesW需要满足如下约束：

  $$
  outputSize\_H = floor(inputSize\_H * scalesH)
  $$

  $$
  outputSize\_W = floor(inputSize\_W * scalesW)
  $$

- 确定性计算：
  - aclnnUpsampleBilinear2dAABackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_bilinear2d_aa_backward.h"

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
int CreateAclNchTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_NCHW,
        shape.data(),
        shape.size(),
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
    std::vector<int64_t> inputShape = {1, 1, 4, 2};
    std::vector<int64_t> outShape = {1, 1, 8, 4};
    void *inputDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *input = nullptr;
    aclTensor *out = nullptr;
    std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    const size_t kSize32 = 32U;
    std::vector<float> outHostData(kSize32, 0);
    const size_t kSize4 = 4U;
    std::vector<int64_t> outputSize = {kSize4, 2};
    std::vector<int64_t> inputSize = {1, 1, 8, 4};
    bool alignCorners = true;
    double scalesH = 2.0;
    double scalesW = 2.0;
    // 创建input aclTensor
    ret = CreateAclNchTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input aclIntArray
    auto outputSizeArray = aclCreateIntArray(outputSize.data(), 2);
    auto inputSizeArray = aclCreateIntArray(inputSize.data(), 4);
    // 创建out aclTensor
    ret = CreateAclNchTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnUpsampleBilinear2dAABackward第一段接口
    ret = aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize(
        input, outputSizeArray, inputSizeArray, alignCorners, scalesH, scalesW, out, &workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnUpsampleBilinear2dAABackward第二段接口
    ret = aclnnUpsampleBilinear2dAABackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBilinear2dAABackward failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputDeviceAddr);
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
