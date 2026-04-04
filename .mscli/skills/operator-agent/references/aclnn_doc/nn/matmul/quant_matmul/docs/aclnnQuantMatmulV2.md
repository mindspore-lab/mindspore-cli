# aclnnQuantMatmulV2

**须知：该接口后续版本会废弃，请使用最新aclnnQuantMatmulV5接口。**


## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                            |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：完成量化的矩阵乘计算，最大支持输入维度为3维。相似接口有[aclnnMm](../../mat_mul_v3/docs/aclnnMm.md)（仅支持2维Tensor作为输入的矩阵乘）和[aclnnBatchMatMul](../../batch_mat_mul_v3/docs/aclnnBatchMatMul.md)（仅支持三维的矩阵乘，其中第一维是Batch维度）。
- 计算公式：

$$
out = (x1@x2 + bias) * deqScale
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 aclnnQuantMatmulV2GetWorkspaceSize 接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用 aclnnQuantMatmulV2 接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulV2GetWorkspaceSize(
  const aclTensor *x1, 
  const aclTensor *x2, 
  const aclTensor *bias, 
  const aclTensor *deqScale, 
  bool             adjX1, 
  bool             adjX2, 
  aclTensor       *out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulV2(
  void              *workspace, 
  uint64_t           workspaceSize, 
  aclOpExecutor     *executor, 
  const aclrtStream  stream)
```

## aclnnQuantMatmulV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1535px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 250px">
  <col style="width: 350px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 130px">
  <col style="width: 153px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入x1。</td>
      <td>在adjX1为false情况下各个维度表示：（batch，m，k）。<br>在adjX1为true情况下各个维度表示：（batch，k，m），batch可不存在。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>在adjX2为false情况下各个维度表示：（batch，k，n）。<br>在adjX2为true情况下各个维度表示：（batch，n，k），batch可不存在，其中k与x1的shape中的k一致。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>公式中的输入bias。</td>
      <td>shape是1维（n，），n与x2的n一致。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale</td>
      <td>输入</td>
      <td>表示量化参数，公式中的输入deqScale。</td>
      <td>shape是1维（t，），t = align（n， 16）， 其中n与x2的n一致。</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>adjX1</td>
      <td>输入</td>
      <td>表示x1的输入shape是否包含transpose。</td>
      <td>在adjX1为false情况下各个维度表示：（batch，m，k）。<br>在adjX1为true情况下各个维度表示：（batch，k，m），batch可不存在。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>adjX2</td>
      <td>输入</td>
      <td>表示x2的输入shape是否包含transpose。</td>
      <td>在adjX2为false情况下各个维度表示：（batch，k，n）。<br>在adjX2为true情况下各个维度表示：（batch，n，k），batch可不存在。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td>（batch，m，n），batch可不存在，支持x1与x2的batch维度broadcast，输出batch与broadcast之后的batch一致，m、n分别与x1的m、x2的n一致。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
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

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
      <td>传入的x1、x2、deqScale或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>x1、x2、bias、deqScale或out的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x1、x2的shape不是2维或者3维。</td>
    </tr>
    <tr>
      <td>bias、deqScale的shape不是1维。</td>
    </tr>
    <tr>
      <td>deqScale的大小不是x2中的n大小按照16向上取整。</td>
    </tr>
    <tr>
      <td>bias存在且m和n均不为0但k为0的空tensor。</td>
    </tr>
  </tbody>
  </table>

## aclnnQuantMatmulV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 230px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulV2GetWorkspaceSize获取。</td>
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

- 确定性说明：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：aclnnQuantMatmulV2默认确定性实现。

该接口迁移到aclnnQuantMatmulV4接口的方法：
- 输入x1，x2，bias，adjX1和adjX2可以直接转为aclnnQuantMatmulV4接口中的x1，x2，bias，transposeX1和transposeX2。
- 输入deqScale为UINT64的aclTensor，数据类型与aclnnQuantMatmulV4接口中的scale一致。aclnnQuantMatmulV2接口的deqScale shape是1维（t，），t = align(n, 16)。aclnnQuantMatmulV4接口中的scale shape是1维（t，），t = 1或n。直接将原始FLOAT型量化参数调用aclnnTransQuantParamV2输出数据类型为UINT64且shape为（n，）的aclTensor（参考[aclnnQuantMatmulV4调用示例](../../quant_batch_matmul_v3/docs/aclnnQuantMatmulV4.md#调用示例)），记为**scale**，对标aclnnQuantMatmulV4接口中的scale。
- aclnnQuantMatmulV4接口中的可选输入offset/pertokenScaleOptional设置为nullptr。
- 接口参数设置为`aclnnQuantMatmulV4GetWorkspaceSize(x1, x2, scale, nullptr, nullptr, bias, adjX1, adjX2, out, workspaceSize, executor)`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <memory>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_matmul.h"
#include "aclnnop/aclnn_trans_quant_param.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      Finalize(deviceId, stream);\
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

void Finalize(int32_t deviceId, aclrtStream stream) {
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnQuantMatmulV2Test(int32_t deviceId, aclrtStream &stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> x1Shape = {5, 2};
  std::vector<int64_t> x2Shape = {2, 3};
  std::vector<int64_t> biasShape = {3};
  std::vector<int64_t> outShape = {5, 3};
  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* out = nullptr;
  std::vector<int8_t> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int8_t> x2HostData = {1, 1, 1, 1, 1, 1};
  std::vector<int32_t> biasHostData = {1, 1, 1};
  std::vector<uint16_t> outHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // 实际上是float16半精度方式
  // 创建x1 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2 aclTensor
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建bias aclTensor
  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  bool adjX1 = false;
  bool adjX2 = false;

  // 将scale数据从float类型转换为硬件需要的uint64_t类型，并创建对应的tensor，作为入参deqScale
  float scaleArray[3] = {1.0, 1.0, 1.0};
  uint64_t scaleSize = 3;
  float *offsetArray = nullptr;
  uint64_t offsetSize = 0;
  uint64_t *deqScaleArray = nullptr;
  uint64_t deqScaleSize = 0;
  ret = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &deqScaleArray, &deqScaleSize);
  std::unique_ptr<uint64_t> deqScaleArrayPtr(deqScaleArray);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParam failed. ERROR: %d\n", ret); return ret);

  std::vector<uint64_t> deqScaleVector(deqScaleArray, deqScaleArray + deqScaleSize);
  std::vector<int64_t> deqScaleShape = {deqScaleSize};
  void* deqScaleDeviceAddr = nullptr;
  aclTensor* deqScale = nullptr;
  ret = CreateAclTensor(deqScaleVector, deqScaleShape, &deqScaleDeviceAddr, aclDataType::ACL_UINT64, &deqScale);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> deqScaleTensorPtr(deqScale, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deqScaleDeviceAddrPtr(deqScaleDeviceAddr, aclrtFree);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnQuantMatmulV2第一段接口
  ret = aclnnQuantMatmulV2GetWorkspaceSize(x1, x2, bias, deqScale, adjX1, adjX2, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // 调用aclnnQuantMatmulV2第二段接口
  ret = aclnnQuantMatmulV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0); // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnQuantMatmulV2Test(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV2Test failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}

```
