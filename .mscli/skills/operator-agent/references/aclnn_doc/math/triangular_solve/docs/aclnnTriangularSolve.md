# aclnnTriangularSolve

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |     ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |   √     |


## 功能说明

- 算子功能：求解一个具有方形上或下三角形可逆矩阵A和多个右侧b的方程组。
- 计算公式：

  $$
  AX = b
  $$
  
  其中$A$是一个上三角方阵（当upper为false时为下三角方阵），其主对角线不含0的元素。$b,A$为二维矩阵或者二维矩阵的batch，当输入为batch时，返回输出的X也为对应的batch。当$A$的主对角线含有0，或元素非常接近0，且unitriangular为false时，输出结果可能包含$NaN$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnTriangularSolveGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnTriangularSolve”接口执行计算。

- `aclnnStatus aclnnTriangularSolveGetWorkspaceSize(const aclTensor *self, const aclTensor *A, bool upper, bool transpose, bool unitriangular, aclTensor *xOut, aclTensor *mOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnTriangularSolve(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnTriangularSolveGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入): 公式中的$b$，数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128, 且数据类型与`A`一致，且数据维度至少为2且不大于8。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。self[-2]=A[-2]。除最后两个维度之外，A和self的其余维度满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)。

  - A(aclTensor*, 计算输入): 公式中的$A$，数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128, 且数据类型与`self`一致，且数据维度至少为2且不大于8。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。最后两个轴相等。除最后两个维度之外，A和self的其余维度满足[broadcast关系](../../../docs/zh/context/broadcast关系.md)。

  - upper(bool, 计算输入)：计算属性，默认为true， `A`为上三角方阵，当upper为false时，`A`为下三角方阵。

  - transpose(bool, 计算输入)：计算属性，默认为false， 当transpose为true时，计算$A^T X=b$。

  - unitriangular(bool, 计算输入)：计算属性，默认为false，当unitriangular为true时，`A`的主对角线元素视为1，而不是从`A`引用，并且unitriangular为true时输入`self`和`A`，输出`xOut`和`mOut`的数据类型只支持FLOAT。

  - xOut(aclTensor *, 计算输出): 公式中的$X$，数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128，且数据类型与self一致，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，且shape需要与broadcast后的`A`,`b`满足$AX=b$约束。A和self满足broadcast关系之后的维度，最后一根轴dim=self[-1]。

  - mOut(aclTensor *, 计算输出): broadcast后`A`的上三角（下三角）拷贝，数据类型支持FLOAT、DOUBLE、COMPLEX64、COMPLEX128，且数据类型与self一致，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。A和self满足broadcast关系之后的维度，最后一根轴dim=A[-1]。

  - workspaceSize(uint64_t *, 出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor \**, 出参)：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1145px"><colgroup>
  <col style="width: 295px">
  <col style="width: 134px">
  <col style="width: 716px">
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
      <td>传入的self、A或xOut,mOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>self、A或xOut,mOut的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self、A或xOut,mOut的shape不符合约束。</td>
    </tr>
  </tbody>
  </table>

## aclnnTriangularSolve

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnTriangularSolveGetWorkspaceSize获取。</td>
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
  - aclnnTriangularSolve默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_triangular_solve.h"

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
  std::vector<int64_t> selfShape = {3, 1};
  std::vector<int64_t> otherShape = {3, 3};
  std::vector<int64_t> xOutShape = {3, 1};
  std::vector<int64_t> mOutShape = {3, 3};
  void* selfDeviceAddr = nullptr;
  void* otherDeviceAddr = nullptr;
  void* xOutDeviceAddr = nullptr;
  void* mOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* other = nullptr;
  aclTensor* xOut = nullptr;
  aclTensor* mOut = nullptr;
  bool upper = true;
  bool transpose = false;
  bool unitriangular = false;
  std::vector<float> selfHostData = {1, 2, 3};
  std::vector<float> otherHostData = {1, 2, 3, 0, 4, 5, 0, 0, 6};
  std::vector<float> xOutHostData = {-0.2500, -0.1250, 0.5000};
  std::vector<float> mOutHostData = {1, 2, 3, 0, 4, 5, 0, 0, 6};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建other aclTensor
  ret = CreateAclTensor(otherHostData, otherShape, &otherDeviceAddr, aclDataType::ACL_FLOAT, &other);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建xOut aclTensor
  ret = CreateAclTensor(xOutHostData, xOutShape, &xOutDeviceAddr, aclDataType::ACL_FLOAT, &xOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mOut aclTensor
  ret = CreateAclTensor(mOutHostData, mOutShape, &mOutDeviceAddr, aclDataType::ACL_FLOAT, &mOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnTriangularSolve第一段接口
  ret = aclnnTriangularSolveGetWorkspaceSize(self, other, upper, transpose, unitriangular, xOut, mOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriangularSolveGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnTriangularSolve第二段接口
  ret = aclnnTriangularSolve(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTriangularSolve failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto xSize = GetShapeSize(xOutShape);
  std::vector<float> resultData(xSize, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), xOutDeviceAddr,
                    xSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < xSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  auto mSize = GetShapeSize(mOutShape);
  std::vector<float> mResultData(mSize, 0);
  ret = aclrtMemcpy(mResultData.data(), mResultData.size() * sizeof(mResultData[0]), mOutDeviceAddr,
                    mSize * sizeof(mResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < mSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, mResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(other);
  aclDestroyTensor(xOut);
  aclDestroyTensor(mOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(otherDeviceAddr);
  aclrtFree(xOutDeviceAddr);
  aclrtFree(mOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```