# aclnnGridSampler3DBackward

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/image/grid_sampler3_d_grad)

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

- 接口功能：[aclnnGridSampler3D](../../grid_sample/docs/aclnnGridSampler3D.md)的反向传播，完成张量input与张量grid的梯度计算。
- 计算公式：

  - 计算流程：
    1. 根据grid存储的(x, y, z)值，计算出映射到input上的坐标，这些坐标和alignCorners、paddingMode有关。
    2. 坐标根据输入的interpolationMode，选择使用bilinear、nearest、bicubic不同插值模式计算输出值。
    3. 根据grad存储的梯度值乘上对应点的权重值，计算出最终dx、dgrid的结果。
  
  - 其中：
  
      grad、input、grid、dx、dgrid的尺寸如下：
  
      $$
      grad: (N, C, D_{out}, H_{out}, W_{out})\\
      input: (N, C, D_{in}, H_{in}, W_{in})\\
      grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
      dx: (N, C, D_{in}, H_{in}, W_{in})\\
      dgrid: (N, D_{out}, H_{out}, W_{out}, 3)
      $$
  
      其中grad、input、grid、dx、dgrid中的N是一致的，grad、input和dx中的C是一致的，input和dx中的$D_{in}$、$H_{in}$、$W_{in}$是一致的，grad、grid和dgrid中的$D_{out}$、$H_{out}$、$W_{out}$是一致的，grid最后一维大小为3，表示input像素位置信息为(x, y, z)，会将x、y、z的取值范围归一化到[-1, 1]之间。

    - 对于超出范围的坐标，会根据paddingMode进行不同处理：
  
      - paddingMode="zeros"，表示对越界位置用0填充。
      - paddingMode="border"，表示对越界位置用边界值填充。
      - paddingMode="reflection"，表示对越界位置用边界值的对称值填充。
  
    - 对input采样时，会根据interpolationMode进行不同处理：
  
      - interpolationMode="bilinear"，表示取input中(x, y, z)周围八个坐标的加权平均值。
      - interpolationMode="nearest"，表示取input中距离(x, y, z)最近的坐标值。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGridSampler3DBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGridSampler3DBackward”接口执行计算。

```Cpp
aclnnStatus aclnnGridSampler3DBackwardGetWorkspaceSize(
  const aclTensor*    gradOutput,
  const aclTensor*    input,
  const aclTensor*    grid,
  int64_t             interpolationMode,
  int64_t             paddingMode,
  bool                alignCorners,
  const aclBoolArray* outputMask,
  aclTensor*          inputGrad,
  aclTensor*          gridGrad,
  uint64_t*           workspaceSize,
  aclOpExecutor**     executor)
```

```Cpp
aclnnStatus aclnnGridSampler3DBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnGridSampler3DBackwardGetWorkspaceSize

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
      <td>表示反向传播过程中上一层的输出梯度，对应公式描述中的`grad`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`的数据类型一致。</li><li>当数据类型为DOUBLE时，数据格式不支持NDHWC。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>NCDHW、NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>表示反向传播的输入张量，对应公式描述中的`input`。</td>
      <td><ul><li>支持空Tensor。</li><li>当数据类型为DOUBLE时，数据格式不支持NDHWC。</li><li>`input`和`gradOutput`的N轴和C轴的值保持一致，且`input`的D，H，W值不可为0。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>NCDHW、NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grid（aclTensor*）</td>
      <td>输入</td>
      <td>表示采用像素位置的张量，对应公式描述中的`grid`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`input`的数据类型一致。</li><li>`grid`和`gradOutput`的N轴、D轴、H轴、W轴的值保持一致，C轴的值必须为3。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>interpolationMode（int64_t）</td>
      <td>输入</td>
      <td>表示插值模式，对应公式描述中的`interpolationMode`。</td>
     <td>支持0：bilinear（双线性插值）、1：nearest（最邻近插值）两种模式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>paddingMode（int64_t）</td>
      <td>输入</td>
      <td>表示填充模式，即当grid有超过[-1，1]范围的值，则按照paddingMode定义的方式处理相应的输出。对应公式描述中的`paddingMode`。</td>
      <td>支持0：zeros、1：border、2：reflection三种模式。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners（bool）</td>
      <td>输入</td>
      <td>表示设定特征图坐标与特征值的对应方式，对应公式描述中的`alignCorners`。</td>
      <td>如果为True，则将极值-1和1视为参考输入的角像素点的中心点。如果为False，则视为参考输入的角像素点的角点。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputMask（aclBoolArray*）</td>
      <td>输入</td>
      <td>用于表示输出的掩码。</td>
      <td>outputMask[0]为True/False，表示是否获取输出inputGrad；outputMask[1]为True/False，表示是否获取输出gridGrad。</li></ul></td>
      <td>BOOLARRAY</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputGrad（aclTensor*）</td>
      <td>输出</td>
      <td>表示反向传播的输出梯度，对应公式描述中的`dx`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、数据格式和shape与`input`的数据类型、数据格式和shape保持一致。</li><li>当数据类型DOUBLE时，数据格式不支持NDHWC。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>NCDHW、NDHWC</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gridGrad（aclTensor*）</td>
      <td>输出</td>
      <td>表示grid梯度，对应公式描述中的`dgrid`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、数据格式和shape与`grid`的数据类型、数据格式和shape保持一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE</td>
      <td>NDHWC</td>
      <td>5</td>
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

  - <term>Atlas 训练系列产品</term>：
  
    参数`gradOutput`、`input`、`grid`、`inputGrad`、`gridGrad`的数据类型不支持BFLOAT16。

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
      <td>传入的gradOutput、input、grid、inputGrad或gridGrad是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>gradOutput、input、grid、inputGrad或gridGrad的数据类型不在支持的范围之内或数据类型不一致。</td>
    </tr>
    <tr>
      <td>interpolationMode和paddingMode的值不在支持范围内。</td>
    </tr>
    <tr>
      <td>gradOutput、input、grid、inputGrad、gridGrad的维度关系不匹配。</td>
    </tr>
  </tbody></table>

## aclnnGridSampler3DBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGridSampler3DBackwardGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnGridSampler3DBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grid_sampler3d_backward.h"

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
    int64_t interpolationMode = 0;
    int64_t paddingMode = 0;
    bool alignCorners = false;
    aclBoolArray* outputMask = nullptr;
    std::vector<int64_t> gradOutputShape = {1, 1, 1, 2, 2};
    std::vector<int64_t> inputShape = {1, 1, 1, 3, 3};
    std::vector<int64_t> gridShape = {1, 1, 2, 2, 3};
    std::vector<int64_t> inputGradShape = {1, 1, 1, 3, 3};
    std::vector<int64_t> gridGradShape = {1, 1, 2, 2, 3};
    void* gradOutputDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* gridDeviceAddr = nullptr;
    void* inputGradDeviceAddr = nullptr;
    void* gridGradDeviceAddr = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* input = nullptr;
    aclTensor* grid = nullptr;
    aclTensor* inputGrad = nullptr;
    aclTensor* gridGrad = nullptr;

    std::vector<float> gradOutputHostData = {1, 1, 1, 1};
    std::vector<float> inputHostData = {
        1, 2, 3, 4, 5, 6, 7, 8, 9,
    };
    std::vector<float> gridHostData = {-1, -1, 0, -1, 1, -1, -1, 0, 0, 0, 1, 0};
    std::vector<float> inputGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gridGradHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    bool maskValue[2] = {true, true};
    const uint64_t kNum = 2U;
    outputMask = aclCreateBoolArray(&(maskValue[0]), kNum);

    // 创建gradOutput aclTensor
    ret = CreateAclTensor(
        gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建grid aclTensor
    ret = CreateAclTensor(gridHostData, gridShape, &gridDeviceAddr, aclDataType::ACL_FLOAT, &grid);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建inputGrad aclTensor
    ret = CreateAclTensor(inputGradHostData, inputGradShape, &inputGradDeviceAddr, aclDataType::ACL_FLOAT, &inputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gridGrad aclTensor
    ret = CreateAclTensor(gridGradHostData, gridGradShape, &gridGradDeviceAddr, aclDataType::ACL_FLOAT, &gridGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnGridSampler3DBackward第一段接口
    ret = aclnnGridSampler3DBackwardGetWorkspaceSize(
        gradOutput, input, grid, interpolationMode, paddingMode, alignCorners, outputMask, inputGrad, gridGrad,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler3DBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGridSampler3DBackward第二段接口
    ret = aclnnGridSampler3DBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGridSampler3DBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto inputGradSize = GetShapeSize(inputGradShape);
    std::vector<float> inputGradResultData(inputGradSize, 0);
    ret = aclrtMemcpy(
        inputGradResultData.data(), inputGradResultData.size() * sizeof(inputGradResultData[0]), inputGradDeviceAddr,
        inputGradSize * sizeof(inputGradResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy inputGradResultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < inputGradSize; i++) {
        LOG_PRINT("inputGradResultData[%ld] is: %f\n", i, inputGradResultData[i]);
    }

    auto gridGradSize = GetShapeSize(gridGradShape);
    std::vector<float> gridGradResultData(gridGradSize, 0);
    ret = aclrtMemcpy(
        gridGradResultData.data(), gridGradResultData.size() * sizeof(gridGradResultData[0]), gridGradDeviceAddr,
        gridGradSize * sizeof(gridGradResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gridGradResultData from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < gridGradSize; i++) {
        LOG_PRINT("gridGradResultData[%ld] is: %f\n", i, gridGradResultData[i]);
    }

    // 6. 释放aclTensor和aclBoolArray，需要根据具体API的接口定义修改
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(input);
    aclDestroyTensor(grid);
    aclDestroyTensor(inputGrad);
    aclDestroyTensor(gridGrad);
    aclDestroyBoolArray(outputMask);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(gridDeviceAddr);
    aclrtFree(inputGradDeviceAddr);
    aclrtFree(gridGradDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```