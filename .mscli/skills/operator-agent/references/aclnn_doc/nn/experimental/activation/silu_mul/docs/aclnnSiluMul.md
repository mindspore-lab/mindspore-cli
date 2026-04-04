# aclnnSiluMul

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 接口功能：

  对输入Tensor x进行Silu激活计算，将计算结果与输入Tensor y相乘。

- 计算公式：

  给定输入张量 `x` 和 `y`，函数 `SiluMul` 进行以下计算：

  1. 对 `x` 应用 SiLU (Sigmoid Linear Unit) 激活函数：
     $$
     \text{SiLU}(x) = x \cdot \text{Sigmoid}(x) = \frac{x}{1 + e^{-x}}
     $$

  2. 最终输出是 SiLU(x) 和 y 的逐元素乘积：
     $$
     \text{out} = \text{SiLU}(x) \times y
     $$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSiluMulGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnSiluMul”接口执行计算。

```Cpp
aclnnStatus aclnnSiluMulGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *y,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnSiluMul(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnSiluMulGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1458px"><colgroup>
  <col style="width: 154px">
  <col style="width: 120px">
  <col style="width: 276px">
  <col style="width: 308px">
  <col style="width: 212px">
  <col style="width: 107px">
  <col style="width: 136px">
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
      <td>x</td>
      <td>输入</td>
      <td>输入的张量，公式中的x。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型需与y一致。</li><li>shape需与y一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
      </tr>
      <tr>
      <td>y</td>
      <td>输入</td>
      <td>输入的张量，公式中的y。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型需与x一致。</li><li>shape需与x一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出的张量，公式中的out。</td>
      <td><ul><li>输出的数据类型与输入保持一致。</li><li>输出shape与输入shape一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
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
  </tbody>
  </table>


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。
  第一段接口会完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的x、y或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>input的数据类型不在支持的范围之内。</td>
    </tr>
  </tbody></table>

## aclnnSiluMul

- **参数说明：**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSiluMulGetWorkspaceSize获取。</td>
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
aclnnStatus：返回状态码，具体参见[可疑链接已删除]。

## 约束说明

无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_silu_mul.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  std::vector<int64_t> inputXShape = {2, 4};
  std::vector<int64_t> inputYShape = {2, 4};
  std::vector<int64_t> outShape = {2, 4};

  std::vector<float> inputXHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> inputYHostData = {1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> outHostData(8, 0);

  void* inputXDeviceAddr = nullptr;
  void* inputYDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* inputX = nullptr;
  aclTensor* inputY = nullptr;
  aclTensor* out = nullptr;

  // 创建inputX aclTensor
  ret = CreateAclTensor(inputXHostData, inputXShape, &inputXDeviceAddr, aclDataType::ACL_FLOAT, &inputX);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建inputY aclTensor
  ret = CreateAclTensor(inputYHostData, inputYShape, &inputYDeviceAddr, aclDataType::ACL_FLOAT, &inputY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnSiluMul第一段接口
  ret = aclnnSiluMulGetWorkspaceSize(inputX, inputY, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSiluMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnSiluMul第二段接口
  ret = aclnnSiluMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSiluMul failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. 释放aclTensor和aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(inputX);
  aclDestroyTensor(inputY);
  aclDestroyTensor(out);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(inputXDeviceAddr);
  aclrtFree(inputYDeviceAddr);
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