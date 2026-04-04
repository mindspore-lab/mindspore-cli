# aclnnSvd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：计算一个或多个矩阵的奇异值分解。

  当输入张量的维度大于2时，会将高维张量视为一批矩阵进行处理。对于形状为 (..., M, N) 的输入张量，将倒数第二维之前的维度 (...) 视为批处理维度，对每个(M, N) 矩阵独立进行奇异值分解计算。

- 计算公式：

$$
\mathbf{input} = \mathbf{U} \times \mathrm{diag}(\boldsymbol{sigma}) \times \mathbf{V}^T
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSvdGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSvd”接口执行计算。

```c++
aclnnStatus aclnnSvdGetWorkspaceSize(
    const aclTensor *input,
    const bool       fullMatrices,
    const bool       computeUV,
    aclTensor       *sigma,
    aclTensor       *u,
    aclTensor       *v,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnSvd(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnSvdGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1544px"><colgroup>
  <col style="width: 141px">
  <col style="width: 120px">
  <col style="width: 354px">
  <col style="width: 461px">
  <col style="width: 141px">
  <col style="width: 101px">
  <col style="width: 81px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度</th>
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>需要进行奇异值分解的张量，对应公式中的input。</td>
      <td>不支持空Tensor。<br>shape维度至少为2。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>fullMatrices</td>
      <td>输入</td>
      <td>输入参数，表示是否完整计算输出张量u和v。</td>
      <td>控制是否计算完整的SVD分解。<br>当设为true时输出完整的u、v，<br>设为false时只计算经济版本以节省内存和计算资源。<br>当input的shape为[..., M, N]时，记K=min(M, N)，<br>当设为true时，输出shape：<br>u：[..., M, M]，sigma：[..., K]，v：[..., N, N]。<br>当设为false时，输出shape：<br>u：[..., M, K]，sigma：[..., K]，v：[..., N, K]。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>computeUV</td>
      <td>输入</td>
      <td>输入参数，表示是否计算输出张量u和v。</td>
      <td>当设为true时，输出张量sigma、u、v均会被计算。<br>当设为false时，只计算sigma，且不再对u、v的shape进行校验。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sigma</td>
      <td>输出</td>
      <td>输出张量，对应公式中的sigma。</td>
      <td>shape需要根据input的shape及fullMatrices参数进行推导。<br>数据类型需要和input保持一致。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>u</td>
      <td>输出</td>
      <td>输出张量，对应公式中的U。</td>
      <td>shape需要根据input的shape及fullMatrices参数进行推导。<br>数据类型需要和input保持一致。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>v</td>
      <td>输出</td>
      <td>输出张量，对应公式中的V。</td>
      <td>shape需要根据input的shape及fullMatrices参数进行推导。<br>数据类型需要和input保持一致。</td>
      <td>FLOAT、DOUBLE、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
      <td>2-8</td>
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


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1145px"><colgroup>
  <col style="width: 296px">
  <col style="width: 135px">
  <col style="width: 714px">
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
      <td>input、sigma、u、v存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>input、sigma、u、v的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input与sigma、u、v的数据类型不一致。</td>
    </tr>
    <tr>
      <td>input的维度小于2或大于8。</td>
    </tr>
    <tr>
      <td>sigma、u、v的shape和基于input进行推导后的shape不一致。</td>
    </tr>
  </tbody>
  </table>

## aclnnSvd

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSvdGetWorkspaceSize获取。</td>
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
  - aclnnSvd默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_svd.h"

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

struct SVDTensors {
  void* inputDeviceAddr = nullptr;
  void* uDeviceAddr = nullptr;
  void* sigmaDeviceAddr = nullptr;
  void* vDeviceAddr = nullptr;
  aclTensor* input = nullptr;
  aclTensor* u = nullptr;
  aclTensor* sigma = nullptr;
  aclTensor* v = nullptr;
  std::vector<int64_t> uShape = {2, 2};
  std::vector<int64_t> sigmaShape = {2};
  std::vector<int64_t> vShape = {3, 3};
};

struct SVDWorkspace {
  void* addr = nullptr;
  uint64_t size = 0;
};


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



int SetupAndExecuteSVD(aclrtStream stream, SVDTensors& tensors, SVDWorkspace& workspace) {
  auto ret = 0;

  //构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> inputShape = {2, 3};
  std::vector<float> inputHostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> uHostData = {0, 0, 0, 0};
  std::vector<float> sigmaHostData = {0, 0};
  std::vector<float> vHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  bool fullMatrices = true;  
  bool computeUV = true;


  // 创建input aclTensor
  ret = CreateAclTensor(inputHostData, inputShape, &tensors.inputDeviceAddr, aclDataType::ACL_FLOAT, &tensors.input);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建u aclTensor
  ret = CreateAclTensor(uHostData, tensors.uShape, &tensors.uDeviceAddr, aclDataType::ACL_FLOAT, &tensors.u);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建sigma aclTensor
  ret = CreateAclTensor(sigmaHostData, tensors.sigmaShape, &tensors.sigmaDeviceAddr, aclDataType::ACL_FLOAT, &tensors.sigma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建v aclTensor
  ret = CreateAclTensor(vHostData, tensors.vShape, &tensors.vDeviceAddr, aclDataType::ACL_FLOAT, &tensors.v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 调用CANN算子库API，需要修改为具体的Api名称
  aclOpExecutor* executor;
  // 调用aclnnSvdGetWorkspaceSize第一段接口
  ret = aclnnSvdGetWorkspaceSize(tensors.input, fullMatrices, computeUV, tensors.sigma, tensors.u, tensors.v, 
                                 &workspace.size, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSvdGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  
  // 根据第一段接口计算出的workspaceSize申请device内存
  workspace.addr = nullptr;
  if (workspace.size > 0) {
    ret = aclrtMalloc(&workspace.addr, workspace.size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  
  // 调用aclnnSvd第二段接口
  ret = aclnnSvd(workspace.addr, workspace.size, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSvd failed. ERROR: %d\n", ret); return ret);

  // 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
  return 0;
}

int ProcessAndCleanupSVD(SVDTensors& tensors, SVDWorkspace& workspace) {
  auto ret = 0;
  
  // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto uSize = GetShapeSize(tensors.uShape);
  std::vector<float> uData(uSize, 0);
  ret = aclrtMemcpy(uData.data(), uData.size() * sizeof(uData[0]), tensors.uDeviceAddr,
                    uSize * sizeof(uData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor U from device to host failed. ERROR: %d\n", ret); return ret);
  
  for (int64_t i = 0; i < uSize; i++) {
    LOG_PRINT("u[%ld] is: %f\n", i, uData[i]);
  }

  auto sigmaSize = GetShapeSize(tensors.sigmaShape);
  std::vector<float> sigmaData(sigmaSize, 0);
  ret = aclrtMemcpy(sigmaData.data(), sigmaData.size() * sizeof(sigmaData[0]), tensors.sigmaDeviceAddr,
                    sigmaSize * sizeof(sigmaData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor sigma from device to host failed. ERROR: %d\n", ret); return ret);
  
  for (int64_t i = 0; i < sigmaSize; i++) {
    LOG_PRINT("sigma[%ld] is: %f\n", i, sigmaData[i]);
  }
  
  auto vSize = GetShapeSize(tensors.vShape);
  std::vector<float> vData(vSize, 0);
  ret = aclrtMemcpy(vData.data(), vData.size() * sizeof(vData[0]), tensors.vDeviceAddr,
                    vSize * sizeof(vData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy outTensor V from device to host failed. ERROR: %d\n", ret); return ret);
  
  for (int64_t i = 0; i < vSize; i++) {
    LOG_PRINT("v[%ld] is: %f\n", i, vData[i]);
  }  
  
  // 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(tensors.input);
  aclDestroyTensor(tensors.u);
  aclDestroyTensor(tensors.sigma);
  aclDestroyTensor(tensors.v);

  // 释放device 资源
  aclrtFree(tensors.inputDeviceAddr);
  aclrtFree(tensors.uDeviceAddr);
  aclrtFree(tensors.sigmaDeviceAddr);
  aclrtFree(tensors.vDeviceAddr);
  if (workspace.size > 0) {
    aclrtFree(workspace.addr);
  }
  
  return 0;
}

int ExecuteSVDOperator(aclrtStream stream) {
  SVDTensors tensors;
  SVDWorkspace workspace;
  
  auto ret = SetupAndExecuteSVD(stream, tensors, workspace);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  ret = ProcessAndCleanupSVD(tensors, workspace);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  return 0;
}


int main() {
  // device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 执行GtScalar操作
  ret = ExecuteSVDOperator(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ExecuteGtScalarOperator failed. ERROR: %d\n", ret); return ret);

  // 重置设备和终结ACL
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```