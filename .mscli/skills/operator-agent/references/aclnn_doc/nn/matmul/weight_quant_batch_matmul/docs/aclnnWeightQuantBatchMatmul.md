# aclnnWeightQuantBatchMatmul

**须知：该接口后续版本会废弃，请使用最新aclnnWeightQuantBatchMatmulV3接口。**

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×     |
| <term>Atlas 推理系列产品 </term>                             |   ×     |
| <term>Atlas 训练系列产品</term>                              |   ×     |

## 功能说明

- 接口功能：伪量化用于对self * mat2（matmul/batchmatmul）中的mat2进行量化。
- 计算公式：

  $$
  result = self@mat2+bias
  $$

## 函数原型

- [两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 aclnnWeightQuantBatchMatmulGetWorkspaceSize 接口获取入参并根据计算流程计算所需workspace大小，再调用 aclnnWeightQuantBatchMatmul 接口执行计算。

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulGetWorkspaceSize(
  const aclTensor *x1, 
  const aclTensor *x2, 
  const aclTensor *diagonalMatrix, 
  const aclTensor *deqOffset, 
  const aclTensor *deqScale, 
  const aclTensor *addOffset, 
  const aclTensor *mulScale, 
  const aclTensor *bias, 
  bool             transposeX1, 
  bool             transposeX2, 
  float            antiquantScale, 
  float            antiquantOffset, 
  aclTensor       *out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```
```cpp
aclnnStatus aclnnWeightQuantBatchMatmul(
  void              *workspace, 
  uint64_t           workspaceSize, 
  aclOpExecutor     *executor, 
  const aclrtStream  stream)
```

## aclnnWeightQuantBatchMatmulGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed;width: 1555px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 350px">
    <col style="width: 150px">
    <col style="width: 120px">
    <col style="width: 200px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>公式中的输入self。</td>
      <td>维度仅支持2维不支持batch轴，与x2需满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>经处理能得到公式中的输入mat2。</td>
      <td>维度仅支持2维不支持batch轴，但与x1需满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diagonalMatrix</td>
      <td>输入</td>
      <td>对x2反量化得到公式中的输入mat2。</td>
      <td>shape为（32, 32），为单位矩阵，m > 64时不参与计算且可以为空。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2维，shape为（32, 32）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqOffset</td>
      <td>输入</td>
      <td>对x2反量化得到公式中的输入mat2，由addOffset、antiquantOffset、antiquantScale计算得到，计算方式见示例代码。</td>
      <td>shape支持1或者n或者（1, 1）或者（1, n）或者（n, 1），需和x2满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>。m > 64时不参与计算且可以为空。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>n或者（1, 1）或者（1, n）或者（n, 1）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale</td>
      <td>输入</td>
      <td>对x2反量化得到公式中的输入mat2，由接口aclnnTransQuantParam计算得到，计算方式见示例代码。</td>
      <td>shape支持 1 或者 n 或者（1, 1） 或者（1, n） 或者（n, 1），需和x2满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>。m > 64时不参与计算且可以为空。</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>1 或者 n 或者（1, 1） 或者（1, n） 或者（n, 1）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>addOffset</td>
      <td>输入</td>
      <td>对x2反量化得到公式中的输入mat2。</td>
      <td>shape支持 1 或者 n 或者（1, 1）或者（1, n）或者（n, 1），需和x2满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>。m < 64时不参与计算, 任意情况都可以为空。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1 或者 n 或者（1, 1）或者（1, n）或者（n, 1）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mulScale</td>
      <td>输入</td>
      <td>对x2反量化得到公式中的输入mat2。</td>
      <td>shape支持 1 或者 n 或者（1, 1）或者（1, n）或者（n, 1），需和x2满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>。m < 64时不参与计算, 任意情况都可以为空。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1 或者 n 或者（1, 1）或者（1, n）或者（n, 1）</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>公式中的输入bias。</td>
      <td>维度为1维且值等于N，可以为空。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX1</td>
      <td>输入</td>
      <td>用于描述x1是否转置。</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>输入</td>
      <td>用于描述x2是否转置。</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantScale</td>
      <td>输入</td>
      <td>对x2反量化得到公式中的输入mat2。</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffset</td>
      <td>输入</td>
      <td>对x2反量化得到公式中的输入mat2。</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的result。</td>
      <td>数据类型需要是x1与x2推导之后可转换的数据类型，shape需要是x1与x2 broadcast之后的shape。</td>
      <td>FLOAT16,INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
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
      <td>传入的x1或x2，diagonalMatrix（m < 64时），deqOffset（m < 64时），deqScale（m < 64时）是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>传入非空tensor的数据类型不在支持的范围之内。</td>
    </tr>
  </tbody>
  </table>

## aclnnWeightQuantBatchMatmul

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1143px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 825px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnWeightQuantBatchMatmulGetWorkspaceSize获取。</td>
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
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：aclnnWeightQuantBatchMatmul默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_quant_param.h"
#include "aclnnop/aclnn_cast.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul.h"

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
  std::vector<int64_t> x1Shape = {128, 4};
  std::vector<int64_t> x2Shape = {4, 4};
  std::vector<int64_t> addOffsetShape = {4};
  std::vector<int64_t> mulScaleShape = {4};
  std::vector<int64_t> diagonalMatrixShape = {32, 32};
  std::vector<int64_t> deqOffsetShape = {4};
  std::vector<int64_t> deqScaleShape = {4};
  std::vector<int64_t> outShape = {128, 4};

  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* addOffsetDeviceAddr = nullptr;
  void* mulScaleDeviceAddr = nullptr;
  void* diagonalMatrixDeviceAddr = nullptr;
  void* deqOffsetDeviceAddr = nullptr;
  void* deqScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* x1Fp16DeviceAddr = nullptr;
  void* addOffsetFp16DeviceAddr = nullptr;
  void* mulScaleFp16DeviceAddr = nullptr;
  void* outFp16DeviceAddr = nullptr;

  std::vector<float> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  std::vector<float> x2HostData = {1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1,
                                    1, 1, 1, 1};

  std::vector<float> outHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  bool transposeX1 = false;
  bool transposeX2 = false;
  float antiquantOffset = 0;
  float antiquantScale = 1;

  std::vector<float> addOffsetHostData = {1.0, 1.0, 1.0, 0.0};
  float* addOffsetDate = addOffsetHostData.data();
  uint64_t addOffsetSize = 4;
  std::vector<float> mulScaleHostData = {2.0, 2.0, 1.0, 1.0};
  float* mulScaleDate = mulScaleHostData.data();
  uint64_t mulScaleSize = 4;

  // diagonalMatrixData
  uint64_t n = 32;
  uint64_t diagonalMatrixSize = n*n;
  int8_t *diagonalMatrixData = (int8_t *)calloc(diagonalMatrixSize, sizeof(int32_t));
  for (int64_t i = 0; i < n; i++) {
    diagonalMatrixData[i * n + i] = 1;
  }
  std::vector<int8_t> diagonalMatrixHostData(diagonalMatrixData, diagonalMatrixData + diagonalMatrixSize);

  // Get deqOffset
  uint64_t deqOffsetSize = addOffsetSize;
  int32_t *deqOffsetData = (int32_t *)calloc(deqOffsetSize, sizeof(int32_t));
  for (int64_t i = 0; i < deqOffsetSize; i++) {
    deqOffsetData[i] = static_cast<int32_t>(round(addOffsetDate[i] / antiquantScale - antiquantOffset));
  }
  std::vector<int32_t> deqOffsetHostData(deqOffsetData, deqOffsetData + deqOffsetSize);

  // Get deqScale
  uint64_t deqScaleSize = mulScaleSize;
  uint64_t *deqScaleData = (uint64_t *)calloc(deqScaleSize, sizeof(uint64_t));
  for (int64_t i = 0; i < deqScaleSize; i++) {
    mulScaleDate[i] = mulScaleDate[i] * antiquantScale;
  }
  std::vector<uint64_t> deqScaleHostData(deqScaleData, deqScaleData + deqScaleSize);

  // creat aclTensor
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* addOffset = nullptr;
  aclTensor* mulScale = nullptr;
  aclTensor* diagonalMatrix = nullptr;
  aclTensor* deqOffset = nullptr;
  aclTensor* deqScale = nullptr;
  aclTensor* out = nullptr;
  aclTensor* x1Fp16 = nullptr;
  aclTensor* addOffsetFp16 = nullptr;
  aclTensor* mulScaleFp16 = nullptr;
  aclTensor* outFp16 = nullptr;

  // 创建x1 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x1Fp16 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1Fp16DeviceAddr, aclDataType::ACL_FLOAT16, &x1Fp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2 aclTensor
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建addOffset aclTensor
  ret = CreateAclTensor(addOffsetHostData, addOffsetShape, &addOffsetDeviceAddr, aclDataType::ACL_FLOAT, &addOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建addOffsetFp16 aclTensor
  ret = CreateAclTensor(addOffsetHostData, addOffsetShape, &addOffsetFp16DeviceAddr, aclDataType::ACL_FLOAT16, &addOffsetFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mulScale aclTensor
  ret = CreateAclTensor(mulScaleHostData, mulScaleShape, &mulScaleDeviceAddr, aclDataType::ACL_FLOAT, &mulScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mulScaleFp16 aclTensor
  ret = CreateAclTensor(mulScaleHostData, mulScaleShape, &mulScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &mulScaleFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建diagonalMatrix aclTensor
  ret = CreateAclTensor(diagonalMatrixHostData, diagonalMatrixShape, &diagonalMatrixDeviceAddr, aclDataType::ACL_INT8, &diagonalMatrix);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建deqOffset aclTensor
  ret = CreateAclTensor(deqOffsetHostData, deqOffsetShape, &deqOffsetDeviceAddr, aclDataType::ACL_INT32, &deqOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建deqScale aclTensor
  ret = CreateAclTensor(deqScaleHostData, deqScaleShape, &deqScaleDeviceAddr, aclDataType::ACL_UINT64, &deqScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建outFp16 aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outFp16DeviceAddr, aclDataType::ACL_FLOAT16, &outFp16);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // aclnnWeightQuantBatchMatmul接口调用示例
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // aclnn cast fp16
  //x1
  ret = aclnnCastGetWorkspaceSize(x1, aclDataType::ACL_FLOAT16, x1Fp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // addOffset
  ret = aclnnCastGetWorkspaceSize(addOffset, aclDataType::ACL_FLOAT16, addOffsetFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // mulScale
  ret = aclnnCastGetWorkspaceSize(mulScale, aclDataType::ACL_FLOAT16, mulScaleFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 调用aclnnWeightQuantBatchMatmul第一段接口
  ret = aclnnWeightQuantBatchMatmulGetWorkspaceSize(x1Fp16, x2, diagonalMatrix, deqOffset, deqScale, addOffsetFp16, mulScaleFp16, nullptr, transposeX1, transposeX2, antiquantScale, antiquantOffset, outFp16, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);


  // 根据第一段接口计算出的workspaceSize申请device内存
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnWeightQuantBatchMatmul第二段接口
  ret = aclnnWeightQuantBatchMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // fp16 to fp 32
  ret = aclnnCastGetWorkspaceSize(outFp16, aclDataType::ACL_FLOAT, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x1);
  aclDestroyTensor(x1Fp16);
  aclDestroyTensor(x2);
  aclDestroyTensor(addOffset);
  aclDestroyTensor(addOffsetFp16);
  aclDestroyTensor(mulScaleFp16);
  aclDestroyTensor(mulScale);
  aclDestroyTensor(out);
  aclDestroyTensor(outFp16);


  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(addOffsetDeviceAddr);
  aclrtFree(deqScaleDeviceAddr);
  aclrtFree(mulScaleDeviceAddr);
  aclrtFree(diagonalMatrixDeviceAddr);
  aclrtFree(deqOffsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(x1Fp16DeviceAddr);
  aclrtFree(addOffsetFp16DeviceAddr);
  aclrtFree(mulScaleFp16DeviceAddr);
  aclrtFree(outFp16DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  free(diagonalMatrixData);
  free(deqOffsetData);
  free(deqScaleData);
  return 0;
}
```