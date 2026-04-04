# aclnnScale

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

计算公式：

若不输入bias，则

$$
  y=x*scale
$$

  若输入bias，则

$$
  y=x*scale + bias
$$

  说明：scale/bias支持跟X的broadcast，scale/bias的shape规则如下
  - 当scaleFromBlob为True时（axis转换为正数，numAxes为-1时表示到最后轴）：

    scaleShape为xShape[axis:axis + numAxes]

    biasShape为xShape[axis:axis + numAxes]

  - 当scaleFromBlob为False时（axis转换为正数， numAxes为-1时表示到最前轴）：

    scaleShape为xShape[axis:axis + rank(scaleShape)]

    biasShape为xShape[axis:axis + rank(scaleShape)]

  举例:

  - scaleFromBlob = True：

    xShape = [a, b, c, d, e, f] axis = 3 numAxes = 2  --> scaleShape = [d, e]

    xShape = [a, b, c, d, e, f] axis = 3 numAxes = 3  --> scaleShape = [d, e, f]

    xShape = [a, b, c, d, e, f] axis = 3 numAxes = -1 --> scaleShape = [d, e, f]

  - scaleFromBlob = False：

    xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 2 --> scaleShape = [d, e]

    xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 3 --> scaleShape = [d, e, f]
  
    xShape = [a, b, c, d, e, f] axis = 3 rank(scaleShape) = 1 --> scaleShape = [d]

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnScaleGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScale”接口执行计算。

* `aclnnStatus aclnnScaleGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *bias, int64_t axis, int64_t numAxes, bool scaleFromBlob, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnScale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnScaleGetWorkspaceSize

- **参数说明：**

  - x(aclTensor*, 计算输入): 算子输入的Tensor，Device侧的aclTensor，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。    
  - scale(aclTensor*, 计算输入): 算子输入的Tensor，Device侧的aclTensor，数据类型需要与x的数据类型相同，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND.shape满足broadcast要求，参见[功能说明](#功能说明)。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - bias(aclTensor*, 计算输入): 算子输入的Tensor，Device侧的aclTensor，不为空时数据类型需要与scale的数据类型相同，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND.shape与scale保持一致。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。
  - axis(int64_t, 计算输入) host侧INT64类型，指定进行scale的起始轴, 取值范围 [-x_rank, x_rank)（x_rank表示x的shape维度）。
  - numAxes(int64_t, 计算输入) host侧INT64类型，指定进行scale的轴长度, 取值范围 >= -1, numAxes = -1, 表示从axis轴开始scale到最后一轴。
  - scaleFromBlob(bool, 计算输入) host侧BOOL类型，指定要scaleFromBlob类型, True: scale from blob, 使用numAxes + axis进行scale, False: scale from input scale, 从axis开始 scale input scale长度, 忽略numAxes取值。
  - y(aclTensor*, 计算输出): 输出Tensor，shape维度和x保持一致，Device侧的aclTensor，数据类型需要与x的数据类型相同，支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16、BFLOAT16。 
  - workspaceSize(uint64_t\*, 出参): 返回需要在Device侧申请的workspace大小。
  - executor(aclOpExecutor\*\*, 出参): 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md).

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
  <col style="width: 299px">
  <col style="width: 136px">
  <col style="width: 712px">
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
      <td>传入的x、scale、y是空指针。</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>x的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>bias不为空时，bias与scale的数据类型不一致。</td>
    </tr>
    <tr>
      <td>scale与x的数据类型不一致。</td>
    </tr>
    <tr>
      <td>y与x的数据类型不一致。</td>
    </tr>
    <tr>
      <td>x和y的shape不一致。</td>
    </tr>
    <tr>
      <td>bias不为空时，bias与scale的shape不一致。</td>
    </tr>
    <tr>
      <td>x和scale的shape维度大于8。</td>
    </tr>
    <tr>
      <td>axis的取值不在[-x_rank, x_rank)范围内。</td>
    </tr>
    <tr>
      <td>numAxes的取值小于-1。</td>
    </tr>
    <tr>
      <td>scaleFromBlob为True，numAxes等于0且scale的shape不为[1]。</td>
    </tr>
    <tr>
      <td>axis转换为正数之后与numAxes相加，大于x_rank。</td>
    </tr>
    <tr>
      <td>scale的shape与预期不符（预期shape推导参考功能说明）。</td>
    </tr>
  </tbody>
  </table>

## aclnnScale

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnScaleGetWorkspaceSize获取。</td>
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
  - aclnnScale默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scale.h"

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
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> tensor1Shape = {4};
  std::vector<int64_t> tensor2Shape = {4};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* tensor1DeviceAddr = nullptr;
  void* tensor2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* tensor1 = nullptr;
  aclTensor* tensor2 = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> tensor1HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> tensor2HostData = {2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t axis = 0;
  int64_t numAxes = 1;
  bool fromBlob = true;

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建tensor1 aclTensor
  ret = CreateAclTensor(tensor1HostData, tensor1Shape, &tensor1DeviceAddr, aclDataType::ACL_FLOAT, &tensor1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建tensor2 aclTensor
  ret = CreateAclTensor(tensor2HostData, tensor2Shape, &tensor2DeviceAddr, aclDataType::ACL_FLOAT, &tensor2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnScale第一段接口
  ret = aclnnScaleGetWorkspaceSize(self, tensor1, tensor2, axis, numAxes, fromBlob,  out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnScale第二段接口
  ret = aclnnScale(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScale failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                    outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy resultData from device to host failed. ERROR: %d\n", ret);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("resultData[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(tensor1);
  aclDestroyTensor(tensor2);
  aclDestroyTensor(out);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(tensor1DeviceAddr);
  aclrtFree(tensor2DeviceAddr);
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
