# aclnnLinalgQr

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 接口功能：对输入Tensor进行正交分解。

- 计算公式：

  $$
  A = QR
  $$

  其中$A$为输入Tensor，维度至少为2， A可以表示为正交矩阵$Q$与上三角矩阵$R$的乘积的形式

- 示例：

  ```
  A = tensor([[1, 2], [3, 4]], dtype=torch.float)
  q,r = linalg_qr(A, mode='reduced')
  q = tensor([[-0.3162, -0.9487],
             [-0.9487, 0.3162]])
  r = tensor([[-3.1623, -4.4272],
             [0.0000, -0.6325]])
  ```

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnLinalgQrGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnLinalgQr”接口执行计算。

- `aclnnStatus aclnnLinalgQrGetWorkspaceSize(const aclTensor *self, int64_t mode, aclTensor *Q, aclTensor *R, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnLinalgQr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnLinalgQrGetWorkspaceSize

- **参数说明：**

  - self(aclTensor*, 计算输入)：公式中的$A$，数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，shape维度至少为2且不大于8, 且shape需要与Q,R满足约束条件。

  - mode(int64_t, 计算输入)：计算属性，当mode为0时，使用'reduced'模式，对于输入A(\*, m, n), 输出简化大小的Q(\*, m, k), R(\*, k, n),其中k为m,n的最小值。当mode为1时，使用'complete'模式，对于输入A(\*, m, n),输出完整大小的Q(\*, m, m), R(\*, m, n), 当mode为2时，使用'r'模式，仅计算reduced场景下的R(\*,k,n),其中k为m,n的最小值，返回Q为空tensor。

  - Q(aclTensor *, 计算输出)：公式中的$Q$，数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，且[数据格式](../../../docs/zh/context/数据格式.md)需要与self, R一致。shape为Q(\*, m, m)或Q(\*, m, k)或为空, 其中k为m, n的最小值。

  - R(aclTensor *, 计算输出): 公式中的$R$，数据类型支持FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，且[数据格式](../../../docs/zh/context/数据格式.md)需要与self, Q一致。shape为R(\*, m, n)或R(\*, k, n), 其中k为m, n的最小值。

  - workspaceSize(uint64_t *, 出参)：返回需要在Device侧申请的workspace大小。

  - executor(aclOpExecutor \**, 出参)：返回op执行器，包含了算子计算流程。


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 287px">
  <col style="width: 124px">
  <col style="width: 740px">
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
      <td>传入的self、Q或R是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>self、Q、R的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self、Q、R的shape不符合约束。</td>
    </tr>
    <tr>
      <td>mode不在可选范围之内。</td>
    </tr>
  </tbody>
  </table>

## aclnnLinalgQr

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnLinalgQrGetWorkspaceSize获取。</td>
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
  - aclnnLinalgQr默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_linalg_qr.h"

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

aclError InitAcl(int32_t deviceId, aclrtStream* stream)
{
  auto ret = Init(deviceId, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

aclError CreateInputs(
    std::vector<int64_t>& selfShape, std::vector<int64_t>& qOutShape, std::vector<int64_t>& rOutShape,
    void** selfDeviceAddr, void** qOutDeviceAddr, void** rOutDeviceAddr, aclTensor** self, aclTensor** qOut,
    aclTensor** rOut)
{
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> qOutHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> rOutHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

  auto ret = CreateAclTensor(selfHostData, selfShape, selfDeviceAddr, aclDataType::ACL_FLOAT, self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(qOutHostData, qOutShape, qOutDeviceAddr, aclDataType::ACL_FLOAT, qOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(rOutHostData, rOutShape, rOutDeviceAddr, aclDataType::ACL_FLOAT, rOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  return ACL_SUCCESS;
}

aclError ExecOpApi(
    aclTensor* self, aclTensor* qOut, aclTensor* rOut, int64_t mode, void** workspaceAddrOut, uint64_t& workspaceSize,
    void* qOutDeviceAddr, void* rOutDeviceAddr, std::vector<int64_t>& qOutShape, std::vector<int64_t>& rOutShape,
    aclrtStream stream)
{
  aclOpExecutor* executor;

  auto ret = aclnnLinalgQrGetWorkspaceSize(self, mode, qOut, rOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgQrGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  *workspaceAddrOut = workspaceAddr;

  ret = aclnnLinalgQr(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLinalgQr failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷贝 qOut
  auto size1 = GetShapeSize(qOutShape);
  std::vector<double> resultData1(size1, 0);
  ret = aclrtMemcpy(
      resultData1.data(), resultData1.size() * sizeof(resultData1[0]), qOutDeviceAddr, size1 * sizeof(resultData1[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size1; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData1[i]);
  }

  // 拷贝 rOut
  auto size2 = GetShapeSize(rOutShape);
  std::vector<float> resultData2(size2, 0);
  ret = aclrtMemcpy(
      resultData2.data(), resultData2.size() * sizeof(resultData2[0]), rOutDeviceAddr, size2 * sizeof(resultData2[0]),
      ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size2; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData2[i]);
  }

  return ACL_SUCCESS;
}

int main()
{
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = InitAcl(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("InitAcl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> selfShape = {1, 1, 4, 4};
  std::vector<int64_t> qOutShape = {1, 1, 4, 4};
  std::vector<int64_t> rOutShape = {1, 1, 4, 4};

  void* selfDeviceAddr = nullptr;
  void* qOutDeviceAddr = nullptr;
  void* rOutDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* qOut = nullptr;
  aclTensor* rOut = nullptr;

  ret = CreateInputs(
      selfShape, qOutShape, rOutShape, &selfDeviceAddr, &qOutDeviceAddr, &rOutDeviceAddr, &self, &qOut, &rOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t mode = 0;
  uint64_t workspaceSize = 0;
  void* workspaceAddr = nullptr;

  ret = ExecOpApi(
      self, qOut, rOut, mode, &workspaceAddr, workspaceSize, qOutDeviceAddr, rOutDeviceAddr, qOutShape, rOutShape,
      stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 释放 Tensor
  aclDestroyTensor(self);
  aclDestroyTensor(qOut);
  aclDestroyTensor(rOut);

  // 释放 Device 内存
  aclrtFree(selfDeviceAddr);
  aclrtFree(qOutDeviceAddr);
  aclrtFree(rOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}

```
