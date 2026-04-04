# aclnnMaxPool2dWithIndicesBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/pooling/max_pool3d_grad_with_argmax)

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

正向最大池化[aclnnMaxPool2dWithIndices](../../max_pool3d_with_argmax_v2/docs/aclnnMaxPool2dWithIndices.md)的反向传播。
- 输入tensor的推导公式：
  - 当ceilMode=False时，indices tensor的shape中H和W维度推导公式：
    $$
    [H_{out}, W_{out}]=[\lfloor{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rfloor + 1,\lfloor{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rfloor + 1]
    $$

  - 当ceilMode=True时，out tensor的shape中H和W维度推导公式：
    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rceil + 1,\lceil{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rceil + 1]
    $$

  - 滑窗左上角起始位处在下或右侧pad填充位上或者界外（无法取到有效值）时，舍弃该滑窗结果，在上述推导公式基础上对应空间轴shape需减去1：
    $$
    \begin{cases}
    H_{out}=H_{out} - 1& \text{if } (H_{out}-1)*s_h>=H_{in}+padding\_size_{Htop} \\
    W_{out}=W_{out} - 1& \text{if } (W_{out}-1)*s_w>=W_{in}+padding\_size_{Wleft}  \\
    \end{cases}\\
    $$
## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaxPool2dWithIndicesBackward”接口执行计算。

```Cpp
aclnnStatus aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,
  const aclTensor   *self,
  const aclTensor   *indices,
  const aclIntArray *kernelSize, 
  const aclIntArray *stride,
  const aclIntArray *padding,
  const aclIntArray *dilation,
  bool               ceilMode,
  aclTensor         *gradInput,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```
```Cpp
aclnnStatus aclnnMaxPool2dWithIndicesBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize

- **参数说明：**
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
    <td>gradOutput</td>
    <td>输入</td>
    <td>反向传播过程中上一步输出的梯度。</td>
    <td>和正向的输出shape一致，数据格式和self一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND、NCHW、NHWC。</td>
    <td>3-4</td>
    <td>√</td>
  </tr>
  <tr>
    <td>self</td>
    <td>输入</td>
    <td>正向的输入数据。</td>
    <td>-</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND、NCHW、NHWC</td>
    <td>3-4</td>
    <td>√</td>
  </tr>
  <tr>
    <td>indices</td>
    <td>输入</td>
    <td>正向输出的索引。</td>
    <td>shape和输入gradOutput一致，正向输出中最大元素的索引位置，数据格式与self保持一致。</td>
    <td>INT32、INT64</td>
    <td>ND、NCHW、NHWC</td>
    <td>3-4</td>
    <td>√</td>
  </tr>
  <tr>
    <td>kernelSize</td>
    <td>输入</td>
    <td>池化操作中使用的滑动窗口大小。</td>
    <td>长度仅支持1、2。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stride</td>
    <td>输入</td>
    <td>窗口移动的步长。</td>
    <td>长度仅支持0、1、2。stride的长度为0时，stride的数值等于kernelSize的值。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>输入</td>
    <td>输入数据的填充，表示输入每个维度上的填充量，影响池化窗口覆盖整个输入张量的行为。</td>
    <td>长度仅支持1、2。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dilation</td>
    <td>输入</td>
    <td>池化操作的扩张因子，扩张操作增加了池化窗口中元素间的距离。</td>
    <td>-</td>
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
    <td>gradInput</td>
    <td>输出</td>
    <td>反向传播输出的梯度。</td>
    <td>shape、数据格式与self保持一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND、NCHW、NHWC。</td>
    <td>3-4</td>
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
  <term>Ascend 950PR/Ascend 950DT</term>：gradOutput和self数据类型支持FLOAT、FLOAT16、BFLOAT16，数据格式支持ND、NCHW、NHWC。
                                   indices数据类型支持INT32、INT64。
                                   支持dilation中的元素值大于0，支持1维或者2维输入。

  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：gradOutput和self数据类型支持FLOAT。数据格式支持NCHW和CHW。indices数据类型支持INT32。仅支持dilation为（1，1）。

- **返回值：**

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
      <td>传入的self、indices是空指针。</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>gradOutput、self、indices、gradInput的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradOutput、self、indices、gradInput的数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>gradOutput与indices的shape不一致，self和gradInput的shape不一致。</td>
    </tr>
    <tr>
      <td>kernelSize的长度不等于1或者2。</td>
    </tr>
    <tr>
      <td>kernelSize中的数值中存在小于等于0的数值。</td>
    </tr>
    <tr>
      <td>stride的长度不等于0，1或2。</td>
    </tr>
    <tr>
      <td>stride的数值中存在小于等于0的值。</td>
    </tr>
    <tr>
      <td>padding的元素个数不等于1或2</td>
    </tr>
    <tr>
      <td>padding的数值中存在小于0或者大于kernelSize</td>
    </tr>
    <tr>
      <td>dilation的元素数值不符合入参要求。</td>
    </tr>
  </tbody>
  </table>
## aclnnMaxPool2dWithIndicesBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize获取。</td>
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
  - aclnnMaxPool2dWithIndicesBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

- 输入数据暂不支持NaN、-Inf。indices值不能超过公式中的$H\_in*W\_in$，且需要大于等于0。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool2d_with_indices_backward.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
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
  std::vector<int64_t> gradOutShape = {1, 1, 2, 1};
  std::vector<int64_t> selfShape = {1, 1, 4, 3};
  std::vector<int64_t> indicesShape = {1, 1, 2, 1};
  std::vector<int64_t> gradInShape = {1, 1, 4, 3};
  std::vector<int64_t> kernelSizeData = {2, 2};
  std::vector<int64_t> strideData = {2, 2};
  std::vector<int64_t> paddingData = {0, 0};
  std::vector<int64_t> dilationData = {1, 1};
  void* gradOutDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* gradInDeviceAddr = nullptr;
  aclTensor* gradOut = nullptr;
  aclTensor* self = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* gradIn = nullptr;
  std::vector<float> gradOutHostData = {0.4757, 0.1726};
  std::vector<float> selfHostData = {0.0850, -0.5147, -0.0212, -0.5654, -0.3222, 0.5847, 1.7510, 0.9954, 0.1842, 0.8392, 0.4835, 0.9213};
  std::vector<int32_t> indicesHostData = {0, 6};
  std::vector<float> gradInHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // 创建gradOut aclTensor
  ret = CreateAclTensor(gradOutHostData, gradOutShape, &gradOutDeviceAddr, aclDataType::ACL_FLOAT, &gradOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gradIn aclTensor
  ret = CreateAclTensor(gradInHostData, gradInShape, &gradInDeviceAddr, aclDataType::ACL_FLOAT, &gradIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建输入数组
  aclIntArray* kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  aclIntArray* stride = aclCreateIntArray(strideData.data(), 2);
  aclIntArray* padding = aclCreateIntArray(paddingData.data(), 2);
  aclIntArray* dilation = aclCreateIntArray(dilationData.data(), 2);
  const bool ceilMode = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnMaxPool2dWithIndicesBackward接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnMaxPool2dWithIndicesBackward第一段接口
  ret = aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(gradOut, self, indices, kernelSize, stride, padding, dilation, ceilMode, gradIn, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMaxPool2dWithIndicesBackward第二段接口
  ret = aclnnMaxPool2dWithIndicesBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool2dWithIndicesBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradInDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradIn result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOut);
  aclDestroyTensor(self);
  aclDestroyTensor(indices);
  aclDestroyTensor(gradIn);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(gradInDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
