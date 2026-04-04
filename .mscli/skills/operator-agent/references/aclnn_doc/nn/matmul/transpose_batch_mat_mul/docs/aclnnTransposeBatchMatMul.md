# aclnnTransposeBatchMatMul

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品 </term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |


## 功能说明

- 接口功能：完成张量x1与张量x2的矩阵乘计算。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的数列进行变更。permX1代表张量x1的转置序列，permX2代表张量x2的转置序列，序列值为0的是batch维度，其余两个维度做矩阵乘法。

- 示例：
  - x1的shape是(B, M, K)，x2的shape是(B, K, N)，scale为None，batchSplitFactor等于1时，计算输出out的shape是(M, B, N)。
  - x1的shape是(B, M, K)，x2的shape是(B, K, N)，scale不为None，batchSplitFactor等于1时，计算输出out的shape是(M, 1, B * N)。
  - x1的shape是(B, M, K)，x2的shape是(B, K, N)，scale为None，batchSplitFactor大于1时，计算输出out的shape是(batchSplitFactor, M, B * N / batchSplitFactor)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnTransposeBatchMatMulGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnTransposeBatchMatMul”接口执行计算。

```cpp
aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(
    const aclTensor    *x1,
    const aclTensor    *x2,
    const aclTensor    *bias,
    const aclTensor    *scale,
    const aclIntArray  *permX1,
    const aclIntArray  *permX2,
    const aclIntArray  *permY,
    int8_t             cubeMathType,
    const int32_t      batchSplitFactor,
    aclTensor          *out,
    uint64_t           *workspaceSize,
    aclOpExecutor      **executor)
```

```cpp
aclnnStatus aclnnTransposeBatchMatMul(
    void               *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor      *executor,
    const aclrtStream  stream)
```

## aclnnTransposeBatchMatMulGetWorkSpaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed;width: 1545px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 350px">
    <col style="width: 210px">
    <col style="width: 120px">
    <col style="width: 130px">
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
      <tr>
    </thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>输入</td>
        <td>表示矩阵乘的第一个矩阵。</td>
        <td>
          <ul>
            <li>数据类型需要与x2满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li>
            <li>数据类型支持BFLOAT16、FLOAT16、FLOAT32。</li>
            <li>不支持输入x1,x2分别为BFLOAT16和FLOAT16的数据类型推导。</li>
            <li>不支持输入x1,x2分别为BFLOAT16和FLOAT32的数据类型推导。</li>
          </ul>
        </td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>输入</td>
        <td>表示矩阵乘的第二个矩阵。</td>
        <td>
        <ul>
            <li>数据类型需要与x1满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li>
            <li>x2的Reduce维度需要与x1的Reduce维度大小相等。</li>
            <li>数据类型支持BFLOAT16、FLOAT16、FLOAT32。</li>
            <li>不支持输入x1,x2分别为BFLOAT16和FLOAT16的数据类型推导。</li>
            <li>不支持输入x1,x2分别为BFLOAT16和FLOAT32的数据类型推导。</li>
        </ul>
        </td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>表示矩阵乘的偏置矩阵。</td>
        <td>
        <ul>
            <li>预留参数，当前暂不支持。</li>
        </ul>
        </td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
      <td>scale</td>
        <td>可选输入</td>
        <td>表示输出矩阵的量化系数，可在输入为FLOAT16且输出为INT8时使能。</td>
        <td>
        <ul>
            <li>shape仅支持一维且需要满足且等于[b*n]。</li>
        </ul>
        </td>
        <td>INT64、UINT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>permX1</td>
        <td>输入</td>
        <td>表示矩阵乘的第一个矩阵的转置序列，host侧的aclIntArray。</td>
        <td>
        <ul>
          <li> 支持[0, 1, 2]、[1, 0, 2]。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permX2</td>
        <td>输入</td>
        <td>表示矩阵乘的第二个矩阵的转置序列，host侧的aclIntArray。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permY</td>
        <td>输入</td>
        <td>表示矩阵乘输出矩阵的转置序列，host侧的aclIntArray。</td>
        <td>
        <ul>
            <li>支持[1, 0, 2]。</li>
        </ul>
        </td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cubeMathType</td>
        <td>输入</td>
        <td>用于指定Cube单元的计算逻辑，Host侧的整型。</td>
        <td>如果输入的数据类型存在互相推导关系，该参数默认对推导后的数据类型进行处理。具体的枚举值如下：<ul>
            <li>0：KEEP_DTYPE，保持输入的数据类型进行计算。</li>
            <li>1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当数据为其他数据类型时，保持输入类型计算。</li>
            <li>2：USE_FP16，支持将输入降为FLOAT16精度计算，当输入数据类型为BFLOAT16时不支持该选项。</li>
            <li>3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。</li></ul>
        </td>
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>batchSplitFactor</td>
        <td>输入</td>
        <td>用于指定矩阵乘输出矩阵中B维的切分大小，Host侧的整型。</td>
        <td>
        <ul>
          <li>取值范围为[1, B]且能被B整除。</li>
          <li>当scale不为空时，batchSplitFactor只能等于1。</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>表示矩阵乘的输出矩阵，公式中的out。</td>
        <td>
        <ul>
          <li> 数据类型需要与x1与x2推导之后的数据类型保持一致（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li>
          <li> 当scale有值时，输出shape为(M, 1, B * N)。</li>
        </ul>
        <ul>
          <li>当batchSplitFactor大于1时，out的输出shape为(batchSplitFactor, M, B * N / batchSplitFactor)。</li>
          <ul>
            <li> 示例一: M, K, N, B = 32, 512, 128, 16；batchSplitFactor = 2时，out的输出shape大小为(2, 32, 1024)。</li>
            <li> 示例二: M, K, N, B = 32, 512, 128, 16；batchSplitFactor = 4时，out的输出shape大小为(4, 32, 512)。</li>
          </ul>
        </ul>
        </td>
        <td>BFLOAT16、FLOAT16、FLOAT32、INT8</td>
        <td>ND</td>
        <td>3</td>
        <td>-</td>
      </tr>
      </tbody>
      </table>

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x1、x2或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>x1、x2或out的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>x1的第二维和x2的第一维度不相等。</td>
    </tr>
    <tr>
      <td>x1或x2的维度大小不等于3。</td>
    </tr>
    <tr>
      <td>scale的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>batchSplitFactor的数值大小不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnTransposeBatchMatMul

- **参数说明：**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnTransposeBatchMatMulGetWorkSpaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnTransposeBatchMatMul默认确定性实现。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - B的取值范围为[1, 65536)，N的取值范围为[1, 65536)。
    - 当x1的输入shape为(B, M, K)时，K <= 65535；当x1的输入shape为(M, B, K)时，B * K <= 65535。
    - x2的第二维或x2的第三维不能被16整除。
    - permX2仅支持输入[0, 1, 2]。
    - 当scale不为空时，batchSplitFactor只能等于1，B与N的乘积小于65536, 且仅支持输入为FLOAT16和输出为INT8的类型推导。
- <term>Ascend 950PR/Ascend 950DT</term>：
    - permX2支持输入[0, 1, 2]、[0, 2, 1]。
    - 当scale不为空时，batchSplitFactor只能等于1，且仅支持输入为FLOAT16和输出为INT8的类型推导。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_transpose_batch_mat_mul.h"

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

// 将FP16的uint16_t表示转换为float表示
float Fp16ToFloat(uint16_t h) {
  int s = (h >> 15) & 0x1;              // sign
  int e = (h >> 10) & 0x1F;             // exponent
  int f =  h        & 0x3FF;            // fraction
  if (e == 0) {
    // Zero or Denormal
    if (f == 0) {
      return s ? -0.0f : 0.0f;
    }
    // Denormals
    float sig = f / 1024.0f;
    float result = sig * pow(2, -24);
    return s ? -result : result;
  } else if (e == 31) {
      // Infinity or NaN
      return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
  }
  // Normalized
  float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
  return s ? -result : result;
}

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
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int32_t M = 32;
  int32_t K = 512;
  int32_t N = 128;
  int32_t Batch = 16;
  std::vector<int64_t> x1Shape = {M, Batch, K};
  std::vector<int64_t> x2Shape = {Batch, K, N};
  std::vector<int64_t> outShape = {M, Batch, N};
  std::vector<int64_t> permX1Series = {1, 0, 2};
  std::vector<int64_t> permX2Series = {0, 1, 2};
  std::vector<int64_t> permYSeries = {1, 0, 2};
  void* x1DeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* x1 = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* out = nullptr;
  std::vector<uint16_t> x1HostData(GetShapeSize(x1Shape),0x3C00);
  std::vector<uint16_t> x2HostData(GetShapeSize(x2Shape),0x3C00);
  std::vector<uint16_t> outHostData(GetShapeSize(outShape),0);
  int8_t cubeMathType = 1;
  int8_t batchSplitFactor = 1;

  // 创建x1 aclTensor
  ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2 aclTensor
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *permX1 = aclCreateIntArray(permX1Series.data(), permX1Series.size());
  aclIntArray *permX2 = aclCreateIntArray(permX2Series.data(), permX2Series.size());
  aclIntArray *permY = aclCreateIntArray(permYSeries.data(), permYSeries.size());
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // aclnnTransposeBatchMatMul接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnTransposeBatchMatMul第一段接口
  ret = aclnnTransposeBatchMatMulGetWorkspaceSize(x1, x2, (const aclTensor*)nullptr, (const aclTensor*)nullptr,
                                                  permX1, permX2, permY, cubeMathType, batchSplitFactor, out,
                                                  &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnTransposeBatchMatMul第二段接口
  ret = aclnnTransposeBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeBatchMatMul failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    float fp16Float = Fp16ToFloat(resultData[i]);
    LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(x1DeviceAddr);
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
