# aclnnMaxPool3dWithArgmax

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/pooling/max_pool3d_with_argmax_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

* 算子功能：
  * 对于输入信号的输入通道，提供3维最大池化（Max pooling）操作，输出池化后的值out和索引indices。
  * 输入dims的描述：N - 批次，C - 通道，D - 深度，W - 宽度，H - 高度。
  * 当D * H * W超过int32时，建议在模型尺寸上分割D轴。
* 计算公式：
  
  * output tensor中每个元素的计算公式：
    
    $$
    out(N_i, C_j, d, h, w) = \max\limits_{{k\in[0,k_{D}-1],m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times d + k, stride[1]\times h + m, stride[2]\times w + n)
    $$

  * out tensor的shape推导公式（默认ceilMode=false，即向下取整）：
    
$$
[N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lfloor{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rfloor + 1,\\
\lfloor{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rfloor + 1, \\
\lfloor{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rfloor + 1]
$$

  * out tensor的shape推导公式（默认ceilMode=true，即向上取整）：
    
$$
[N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lceil{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rceil + 1,\\
\lceil{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rceil + 1, \\
\lceil{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rceil + 1]
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMaxPool3dWithArgmaxGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaxPool3dWithArgmax”接口执行计算。

```Cpp
aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *kernelSize, 
  const aclIntArray *stride,
  const aclIntArray *padding,
  const aclIntArray *dilation,
  bool               ceilMode,
  aclTensor         *out,
  aclTensor         *indices,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```
```Cpp
aclnnStatus aclnnMaxPool3dWithArgmax(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```
## aclnnMaxPool3dWithArgmaxGetWorkspaceSize

* **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
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
      <td>self</td>
      <td>输入</td>
      <td>待计算张量。</td>
      <td>-</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCDHW、NDHWC、ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>输入</td>
      <td>最大池化的窗口大小。</td>
      <td>数组长度必须为1或3，且数组元素必须都大于0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>窗口移动的步长。</td>
      <td>数组长度必须为0，1或3，且数组元素必须都大于0。当数组的长度为0时，内部会取kernelSize的值作为strides。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>输入</td>
      <td>每一条边补充的层数。</td>
      <td>补充的位置填写“负无穷”。数组长度必须为1或3，且数组元素必须都大于等于0且小于等于kernelSize/2。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>输入</td>
      <td>控制窗口中元素的步幅。</td>
      <td>数组长度必须为1或3，且数组元素必须都大于0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>输入</td>
      <td>计算输出形状时取整的方法。</td>
      <td>为True时表示计算输出形状时用向上取整的方法，为False时则表示向下取整。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出的tensor，池化后的结果。</td>
      <td>数据类型与self保持一致。shape由上述公式推导出。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>NCDHW、NDHWC、ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>最大值的索引位置组成的Tensor。</td>
      <td>shape和out保持一致。</td>
      <td>INT32、INT64</td>
      <td>NCDHW、NDHWC、ND</td>
      <td>4-5</td>
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

   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：`dilation` 元素值仅支持为1；`indices` 数据类型不支持INT64。输入数据排布不支持NDHWC。depth * height * width 不支持大于 max int32。

* **返回值**：
  
  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="13">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="13">161002</td>
      <td>self的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self的数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self的shape不符合要求。</td>
    </tr>
    <tr>
      <td>通过公式推导出的out的shape的某个轴为0。</td>
    </tr>
    <tr>
      <td>kernelSize、stride、padding、dilation的元素值不符合要求。</td>
    </tr>
    <tr>
      <td>kernelSize、stride、padding、dilation的长度不符合要求。</td>
    </tr>
    <tr>
      <td>平台不支持。</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxPool3dWithArgmax

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMaxPool3dWithArgmaxGetWorkspaceSize获取。</td>
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
-  **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：aclnnMaxPool3dWithArgmax默认确定性实现。

- kernelSize、stride、padding、dilation、ceilMode参数需要保证输出out shape中不存在小于1的轴。

- 当ceilMode为True的时候，如果滑动窗口全部在右侧padding区域上，这个输出结果将被忽略。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool3d_with_argmax.h"

#define CHECK_RET(cond, return_expr)  \
    do {                              \
        if (!(cond)) {                \
            return_expr;              \
        }                             \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {1, 1, 2, 2, 2};
    std::vector<int64_t> outShape = {1, 1, 1, 1, 1};
    std::vector<int64_t> indicesShape = {1, 1, 1, 1, 1};
    std::vector<int64_t> kernelSizeData = {2, 2, 2};
    std::vector<int64_t> strideData = {2, 2, 2};
    std::vector<int64_t> paddingData = {0, 0, 0};
    std::vector<int64_t> dilationData = {1, 1, 1};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclTensor* indices = nullptr;
    std::vector<float> selfHostData = {1, 6, 2, 8, 4, 5, 7, 3};
    std::vector<float> outHostData = {0};
    std::vector<int32_t> indicesHostData = {0};

    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建indices aclTensor
    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输入数组
    aclIntArray* kernelSize = aclCreateIntArray(kernelSizeData.data(), 3);
    aclIntArray* stride = aclCreateIntArray(strideData.data(), 3);
    aclIntArray* padding = aclCreateIntArray(paddingData.data(), 3);
    aclIntArray* dilation = aclCreateIntArray(dilationData.data(), 3);
    const bool ceilMode = false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnMaxPool3dWithArgmax接口调用示例
    // 3. 调用aclnnMaxPool3dWithArgmax第一段接口
    ret = aclnnMaxPool3dWithArgmaxGetWorkspaceSize(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool3dWithArgmaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnMaxPool3dWithArgmax第二段接口
    ret = aclnnMaxPool3dWithArgmax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool3dWithArgmax failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("output[%ld] is: %f\n", i, resultData[i]);
    }

    size = GetShapeSize(indicesShape);
    std::vector<int> indicesResultData(size, 0);
    ret = aclrtMemcpy(indicesResultData.data(), indicesResultData.size() * sizeof(indicesResultData[0]), indicesDeviceAddr,
                      size * sizeof(indicesResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy indices result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("indices[%ld] is: %d\n", i, indicesResultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclDestroyTensor(indices);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```