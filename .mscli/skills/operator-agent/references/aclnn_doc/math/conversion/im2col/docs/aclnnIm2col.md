# aclnnIm2col

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/im2col)

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

- 接口功能：

  图像到列，滑动局部窗口数据转为列向量，拼接为大张量。从批处理输入张量中提取滑动窗口。

  考虑一个形状为（N, C, H, W）或 (C, H, W) 的批处理input张量，其中N是批处理维度， C是通道维度， 而 H, W 表示图像大小，此操作将input的空间维度内的每个滑动kernel_size大小的块展平为（N, C $\times \prod$（kernel_size）, L）的3-D 或 （C $\times \prod$（kernel_szie）, L）的2-D 的 output张量的列（即最后一维），而L是这些块的总数。
- 计算公式：

  $L = \prod_{d} \lfloor \frac{spatial\_size[d] + 2 \times padding[d] - dilation[d] \times （kernel\_size[d] -1） -1}{stride[d]} + 1 \rfloor$, 其中spatial_size由上述input张量的H,W构成。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnIm2colGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIm2col”接口执行计算。

```Cpp
aclnnStatus aclnnIm2colGetWorkspaceSize(
    const aclTensor   *self,
    const aclIntArray *kernelSize,
    const aclIntArray *dilation,
    const aclIntArray *padding,
    const aclIntArray *stride,
    const aclTensor   *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnIm2col(
    void          *workspace,
    uint64_t       workspaceSize, 
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnIm2colGetWorkspaceSize

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1580px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 288px">
  <col style="width: 290px">
  <col style="width: 110px">
  <col style="width: 150px">
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
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>待进行im2col计算的入参，对应公式中的self。</td>
      <td>支持空Tensor。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BFLOAT16、FLOAT16、FLOAT、DOUBLE、BOOL、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize（aclIntArray*）</td>
      <td>输入</td>
      <td>卷积核的大小，对应公式中的kernelSize。</td>
      <td>kernelSize[0]表示'H'方向。<br>kernelSize[1]表示'W'方向。</td>
      <td>INT64</td>
      <td>-</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation （aclIntArray*）</td>
      <td>输入</td>
      <td>膨胀参数，对应公式中的dilation。</td>
      <td>dilation[0]表示'H'方向。<br>dilation[1]表示'W'方向。</td>
      <td>INT64</td>
      <td>-</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding（aclIntArray*）</td>
      <td>输入</td>
      <td>卷积的填充大小，对应公式中的padding。</td>
      <td>padding[0]表示'H'方向。<br>padding[1]表示'W'方向。</td>
      <td>INT64</td>
      <td>-</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride （aclIntArray*）</td>
      <td>输入</td>
      <td>卷积的步长，对应公式中的stride。</td>
      <td>stride[0]表示'H'方向。<br>stride[1]表示'W'方向。</td>
      <td>INT64</td>
      <td>-</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>待进行im2col计算的出参，对应公式中的out。</td>
      <td>shape根据上述参数推导。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BFLOAT16、FLOAT16、FLOAT、DOUBLE、BOOL、COMPLEX32、COMPLEX64</td>
      <td>ND</td>
      <td>2维（输入3维）或者3维（输入4维）</td>
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

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：仅支持FLOAT、FLOAT16、BFLOAT16。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
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
      <td>传入的self、kernelSize、dilation、padding、stride或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>self的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self的维度不是3维且不是4维。</td>
    </tr>
    <tr>
      <td>kernelSize、dilation、padding或stride的size不为2。</td>
    </tr>
    <tr>
      <td>kernelSize、dilation或stride存在值等于或小于0的元素。</td>
    </tr>
    <tr>
      <td>padding存在小于0的元素。</td>
    </tr>
    <tr>
      <td>out的数据维度与参数infershape的维度不相同</td>
    </tr>
  </tbody>
  </table>

## aclnnIm2col

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnIm2colGetWorkspaceSize获取。</td>
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
  - aclnnIm2col默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_im2col.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {2, 2, 3};
  std::vector<int64_t> outShape = {8, 4};

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclIntArray* kernelSize = nullptr;
  aclIntArray* dilation = nullptr;
  aclIntArray* padding = nullptr;
  aclIntArray* stride = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
  std::vector<int64_t> kernelSizeData = {2, 2};
  std::vector<int64_t> dilationData = {1, 1};
  std::vector<int64_t> paddingData = {1, 1};
  std::vector<int64_t> strideData = {2, 2};
  std::vector<float> outHostData = {0.0};

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建aclIntArray
  kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  CHECK_RET(kernelSize != nullptr, return ret);
  dilation = aclCreateIntArray(dilationData.data(), 2);
  CHECK_RET(dilation != nullptr, return ret);
  padding = aclCreateIntArray(paddingData.data(), 2);
  CHECK_RET(padding != nullptr, return ret);
  stride = aclCreateIntArray(strideData.data(), 2);
  CHECK_RET(stride != nullptr, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnIm2col第一段接口
  ret = aclnnIm2colGetWorkspaceSize(self, kernelSize, dilation, padding, stride, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIm2colGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnIm2col第二段接口
  ret = aclnnIm2col(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIm2col failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclIntArray，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyIntArray(kernelSize);
  aclDestroyIntArray(dilation);
  aclDestroyIntArray(padding);
  aclDestroyIntArray(stride);
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

