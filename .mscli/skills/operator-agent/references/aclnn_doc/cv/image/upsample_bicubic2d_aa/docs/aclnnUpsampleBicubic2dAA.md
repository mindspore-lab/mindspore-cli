# aclnnUpsampleBicubic2dAA

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/image/upsample_bicubic2d_aa)

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

- 接口功能：对由多个输入通道组成的输入信号应用双三次抗锯齿算法进行上采样。如果输入Tensor x的shape为(N, C, H, W) ，则输出Tensor out的shape为(N, C, outputSize[0], outputSize[1])。

- 计算公式：对于一个二维插值点$(N, C, h, w)$，插值$out(N, C, h, w)$可以表示为：
  
  $$
  {out(N, C, h, w)}=\sum_{i=0}^{kW}\sum_{j=0}^{kH}{W(i, j)}*{f(h_i, w_j)}
  $$
  
  $$
  scaleH =\begin{cases}
  (x.dim(2)-1) / (outputSize[0]-1) & alignCorners=true \\
  1 / scalesH & alignCorners=false\&scalesH>0\\
  x.dim(2) / outputSize[0] & otherwise
  \end{cases}
  $$
  
  $$
  scaleW =\begin{cases}
  (x.dim(3)-1) / (outputSize[1]-1) & alignCorners=true \\
  1 / scalesW & alignCorners=false\&scalesW>0\\
  x.dim(3) / outputSize[1] & otherwise
  \end{cases}
  $$
  
  其中：
  - i和j是$W(i, j)$的索引变量。
  - 如果$scaleH >= 1$，则$kH = 1/scaleH$，否则$kH = 4$
  - 如果$scaleW >= 1$，则$kW = 1/scaleW$，否则$kW = 4$
  - $h_i = |h| + i$
  - $w_j = |w| + j$
  - $f(h_i, w_j)$是原图像在$(h_i, w_j)$的像素值
  - $W(i, j)$是双三次抗锯齿插值的权重，定义为：

    $$
    W(d) =\begin{cases}
    (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
    a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
    0 & otherwise
    \end{cases}
    $$

    其中：
    - 抗锯齿场景$a=-0.5$。
    - $d = |(h, w) - (h_i, w_j)|$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnUpsampleBicubic2dAAGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleBicubic2dAA”接口执行计算。

```Cpp
aclnnStatus aclnnUpsampleBicubic2dAAGetWorkspaceSize(
  const aclTensor*   x,
  const aclIntArray* outputSize,
  const bool         alignCorners,
  const double       scalesH,
  const double       scalesW,
  aclTensor*         out,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnUpsampleBicubic2dAA(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnUpsampleBicubic2dAAGetWorkspaceSize

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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行上采样的输入张量，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>当数据格式为ND时，默认按照NCHW格式处理。</li></ul></td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>NCHW、ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize（aclIntArray*）</td>
      <td>输入</td>
      <td>表示指定`out`在H和W维度上的空间大小。对应公式中的`outputSize`。</td>
      <td>size为2，且各元素均大于0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners（bool）</td>
      <td>输入</td>
      <td>决定是否对齐角像素点，对应公式中的`alignCorners`。</td>
      <td>alignCorners为True，则输入和输出张量的角像素点会被对齐，否则不对齐。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH（double）</td>
      <td>输入</td>
      <td>指定空间大小的height维度乘数，对应公式中的`scalesH`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW（double）</td>
      <td>输入</td>
      <td>指定空间大小的width维度乘数，对应公式中的`scalesW`。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示采样后的输出张量，对应公式中的`out`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与入参`x`保持一致。</li></ul></td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
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

- **返回值**：

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
      <td>传入的x、outputSize或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>x或out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x和out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>x的shape不是4维。</td>
    </tr>
    <tr>
      <td>outputSize的元素取值不大于0。</td>
    </tr>
  </tbody></table>

## aclnnUpsampleBicubic2dAA

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUpsampleBicubic2dAAGetWorkspaceSize获取。</td>
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

- 参数`x`、`out`的shape约束：
  - 每个维度的取值小于等于2^20。
  - 参数`out`的N轴和C轴与`self`保持一致，且C轴、H轴、W轴大于0。
  - 内存占用需小于60G。内存占用的计算公式如下：

    $$
    (x\_H * x\_W + out\_H * out\_W + x\_H * out\_W) * N * C  * sizeof(float) < 60 * 1024 * 1024 * 1024
    $$

    其中：
    - N代表输入和输出的N轴。
    - C代表输入和输出的C轴。
  - N \* C \* x_H < 2^31
- - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

  输入数据缩放场景缩小倍数必须小于等于50，即：
  
  $$
  输入shape的高度H/outputSize\_H <= 50
  $$
  
  $$
  输入shape的宽度W/outputSize\_W <=50
  $$

- 参数x、outputSize、scalesH、scalesW需要满足如下约束：

  $$
  outputSize\_H = floor(x\_H * scalesH)
  $$

  $$
  outputSize\_W = floor(x\_W * scalesW)
  $$

- 确定性计算：
  - aclnnUpsampleBicubic2dAA默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_bicubic2d_aa.h"

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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
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
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {1, 1, 3, 3};
    std::vector<int64_t> outShape = {1, 1, 5, 5};
    void *selfDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *x = nullptr;
    aclTensor *out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const size_t kSize25 = 25U;
    std::vector<float> outHostData = {kSize25, 0};
    const size_t kSize5 = 5U;
    std::vector<int64_t> outputSizeData = {kSize5, 5};
    bool alignCorners = false;
    double scalesH = 0.0;
    double scalesW = 0.0;
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建outputSize aclIntArray
    const aclIntArray *outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
    CHECK_RET(outputSize != nullptr, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnUpsampleBicubic2dAA第一段接口
    ret = aclnnUpsampleBicubic2dAAGetWorkspaceSize(
        x, outputSize, alignCorners, scalesH, scalesW, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBicubic2dAAGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnUpsampleBicubic2dAA第二段接口
    ret = aclnnUpsampleBicubic2dAA(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleBicubic2dAA failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
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

    // 6. 释放aclTensor和aclIntArray，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyIntArray(outputSize);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
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