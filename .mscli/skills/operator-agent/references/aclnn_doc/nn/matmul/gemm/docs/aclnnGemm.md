# aclnnGemm

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算α 乘以A与B的乘积，再与β 和input C的乘积求和。
- 计算公式：
  - 若transA非零，计算前会将A进行转置；同样的，若transB非零，则会将B进行转置。

    $$
    out = α  (A @ B) + β  C
    $$

  - 若transA与transB都为非零，则计算公式为：

    $$
    out = α  (A^T @ B^T) + βC
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGemmGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGemm”接口执行计算。

```cpp
aclnnStatus aclnnGemmGetWorkspaceSize(
  const aclTensor *A, 
  const aclTensor *B, 
  const aclTensor *C, 
  float           alpha, 
  float           beta, 
  int64_t         transA, 
  int64_t         transB, 
  aclTensor       *out, 
  int8_t          cubeMathType, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```
```cpp
aclnnStatus aclnnGemm(
  void          *workspace, 
  uint64_t      workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream   stream)
```

## aclnnGemmGetWorkspaceSize

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1587px"><colgroup>
  <col style="width: 159px">
  <col style="width: 127px">
  <col style="width: 230px">
  <col style="width: 400px">
  <col style="width: 249px">
  <col style="width: 117px">
  <col style="width: 117px">
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
      <td>A</td>
      <td>输入</td>
      <td>公式中的输入A，Device侧的aclTensor。</td>
      <td><ul><li>数据类型需要与C、B构成互相推导关系。</li>
      <li>shape（或者转置后shape）需要满足与B相乘条件。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>B</td>
      <td>输入</td>
      <td>公式中的输入B，Device侧的aclTensor。</td>
      <td><ul><li>数据类型需要与C，A构成互相推导关系。</li>
      <li>shape（或者转置后shape）需要满足与A相乘条件。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>C</td>
      <td>输入</td>
      <td>公式中的输入C，Device侧的aclTensor。</td>
      <td><ul><li>数据类型需要与AB计算后的结果构成互相推导关系。</li>
      <li>shape需要与A@B计算后的结果 一致或满足<a href="../../../docs/zh/context/broadcast关系.md">broadcast关系</a>。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>公式中的输入α，Host侧的浮点型，表示A和B乘积的系数。</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>公式中的输入β，Host侧的浮点型，表示C的系数。</td>
      <td>-</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transA</td>
      <td>输入</td>
      <td>公式中的输入transA，Host侧的整型，表示矩阵A是否需要转置，非零表示转置, A矩阵为[K,M]，零表示不需要转置, A矩阵为[M, K]。</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transB</td>
      <td>输入</td>
      <td>公式中的输入transB，Host侧的整型，表示矩阵B是否需要转置，非零表示转置, B矩阵为[N, K]，零表示不需要转置, B矩阵为[K, N]。</td>
      <td>-</td>
      <td>int64_t</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的out，Device侧的aclTensor，数据类型需要与C构成互相推导关系，shape需要A@B计算后的结果一致。</td>
      <td>-</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>输入</td>
      <td>Host侧的整型，判断Cube单元应使用哪种计算逻辑进行运算。</td>
      <td>如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：<ul>
        <li>0：KEEP_DTYPE，保持输入的数据类型进行计算。</li>
        <li>1：ALLOW_FP32_DOWN_PRECISION，允许转换输入数据类型降低精度计算。</li>
        <li>2：USE_FP16，允许转换输入数据类型至FLOAT16计算。</li>
        <li>3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
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

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - A 数据类型支持BFLOAT16、FLOAT16、FLOAT32。
    - B 数据类型支持BFLOAT16、FLOAT16、FLOAT32。
    - C 数据类型支持BFLOAT16、FLOAT16、FLOAT32。
    - out 数据类型支持BFLOAT16、FLOAT16、FLOAT32。
    - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理；
    - cubeMathType=2，当输入数据类型为BFLOAT16时不支持该选项；
    - cubeMathType=3，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不支持该选项。
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：
    - A 数据类型支持FLOAT16、FLOAT32。
    - B 数据类型支持FLOAT16、FLOAT32。
    - C 数据类型支持FLOAT16、FLOAT32。
    - out 数据类型支持FLOAT16、FLOAT32。
    - 不支持BFLOAT16数据类型；
    - 当输入数据类型为FLOAT32时不支持cubeMathType=0；
    - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为FLOAT16计算，当输入为其他数据类型时不做处理；
    - 不支持cubeMathType=3。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 887px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 700px">
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
      <td>传入的A, B，C或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>A或B不是2维，或者进行计算时，shape不满足[m, k]和[k, n]的k维度相等关系。</td>
    </tr>
    <tr>
      <td>self不能与batch1@batch2做broadcast操作。</td>
    </tr>
    <tr>
      <td>C和AB计算后的结果不满足broadcast关系。</td>
    </tr>
    <tr>
      <td>out和AB计算后的shape不一致。</td>
    </tr>
    <tr>
      <td>cubeMathType为非法值。</td>
    </tr>
  </tbody>
  </table>

## aclnnGemm

- **参数说明**：
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGemmGetWorkspaceSize获取。</td>
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
- 确定性说明：
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：aclnnGemm默认确定性实现。

- <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：Cube单元不支持FLOAT32计算。当输入为FLOAT32，可通过设置cubeMathType=1（ALLOW_FP32_DOWN_PRECISION）来允许接口内部cast到FLOAT16进行计算.

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_gemm.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
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
  // 1. （固定写法）device/stream初始化, 参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> AShape = {2, 2};
  std::vector<int64_t> BShape = {2, 2};
  std::vector<int64_t> CShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* ADeviceAddr = nullptr;
  void* BDeviceAddr = nullptr;
  void* CDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* A = nullptr;
  aclTensor* B = nullptr;
  aclTensor* C = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> AHostData = {1, 2, 1, 2};
  std::vector<float> BHostData = {1, 2, 1, 2};
  std::vector<float> CHostData = {1, 1, 1, 1};
  std::vector<float> outHostData = {0, 0, 0, 0};
  float alpha = 1.0f;
  float beta = 2.0f;
  int64_t transA = 0;
  int64_t transB = 0;
  int8_t cubeMathType = 1;

  // 创建A aclTensor
  ret = CreateAclTensor(AHostData, AShape, &ADeviceAddr, aclDataType::ACL_FLOAT, &A);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建B aclTensor
  ret = CreateAclTensor(BHostData, BShape, &BDeviceAddr, aclDataType::ACL_FLOAT, &B);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建C aclTensor
  ret = CreateAclTensor(CHostData, CShape, &CDeviceAddr, aclDataType::ACL_FLOAT, &C);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGemm第一段接口
  ret = aclnnGemmGetWorkspaceSize(A, B, C, alpha, beta, transA, transB, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnGemm第二段接口
  ret = aclnnGemm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemm failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(A);
  aclDestroyTensor(B);
  aclDestroyTensor(C);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(ADeviceAddr);
  aclrtFree(BDeviceAddr);
  aclrtFree(CDeviceAddr);
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
