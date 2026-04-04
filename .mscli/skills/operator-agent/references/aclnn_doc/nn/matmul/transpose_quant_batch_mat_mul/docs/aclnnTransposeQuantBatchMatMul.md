# aclnnTransposeQuantBatchMatMul

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/transpose_quant_batch_mat_mul)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：完成张量x1与张量x2量化的矩阵乘计算，支持K-C[量化模式](../../../docs/zh/context/量化介绍.md)。仅支持三维的Tensor传入。Tensor支持转置，转置序列根据传入的数列进行变更。permX1代表张量x1的转置序列，permX2代表张量x2的转置序列，序列值为0的是batch维度，其余两个维度做矩阵乘法。

- 示例：
  假设x1的shape是(M, B, K)，x2的shape是(B, K, N)，x1Scale和x2Scale不为None，batchSplitFactor等于1时，计算输出out的shape是(M, B, N)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnTransposeQuantBatchMatMulGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnTransposeQuantBatchMatMul”接口执行计算。

```cpp
aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
    const aclTensor*   x1, 
    const aclTensor*   x2, 
    const aclTensor*   bias, 
    const aclTensor*   x1Scale, 
    const aclTensor*   x2Scale,
    const int32_t      dtype, 
    const int32_t      groupSize, 
    const aclIntArray* permX1, 
    const aclIntArray* permX2,
    const aclIntArray* permY, 
    const int32_t      batchSplitFactor, 
    aclTensor*         out, 
    uint64_t*          workspaceSize,
    aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnTransposeQuantBatchMatMul(
    void               *workspace, 
    uint64_t            workspaceSize,
    aclOpExecutor      *executor,
    const aclrtStream   stream)
```

## aclnnTransposeQuantBatchMatMulGetWorkSpaceSize

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
        <td>x1（aclTensor*）</td>
        <td>输入</td>
        <td>表示矩阵乘的第一个矩阵。</td>
        <td>
          <ul>
            <li>数据类型需要与x2满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li>
            <li>数据类型当前仅支持FLOAT8_E5M2、FLOAT8_E4M3FN。</li>
          </ul>
        </td>
        <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2（aclTensor*）</td>
        <td>输入</td>
        <td>表示矩阵乘的第二个矩阵。</td>
        <td>
        <ul>
            <li>数据类型需要与x1满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li>
            <li>x2的k维度需要与x1的k维度大小相等。</li>
            <li>数据类型当前仅支持FLOAT8_E5M2、FLOAT8_E4M3FN。</li>
        </ul>
        </td>
        <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias（aclTensor*）</td>
        <td>输入</td>
        <td>表示矩阵乘的偏置矩阵。</td>
        <td>预留参数，当前暂不支持。</td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
      <td>x1Scale（aclTensor*）</td>
        <td>输入</td>
        <td>表示左矩阵的量化系数。</td>
        <td>shape仅支持一维且需要满足且等于[m]。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
      <td>x2Scale（aclTensor*）</td>
        <td>输入</td>
        <td>表示右矩阵的量化系数。</td>
        <td>shape仅支持一维且需要满足且等于[n]。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dtype（int32_t）</td>
        <td>输入</td>
        <td>用于指定输出矩阵的数据类型，支持的值为：1、27。</td>
        <td>
        <ul>
          <li>取值为1, 表示输出矩阵类型为FLOAT16。</li>
          <li>取值为27, 表示输出矩阵类型为BFLOAT16。</li>
        </ul>
        </td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize（int32_t）</td>
        <td>输入</td>
        <td>用于指定量化分组大小，预留参数，当前仅支持配置为0，其他取值不生效。</td>
        <td>当前配置非0取值不生效。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permX1（aclIntArray*）</td>
        <td>输入</td>
        <td>表示矩阵乘的第一个矩阵的转置序列，host侧的aclIntArray。</td>
        <td>支持[1, 0, 2]。</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permX2（aclIntArray*）</td>
        <td>输入</td>
        <td>表示矩阵乘的第二个矩阵的转置序列，host侧的aclIntArray。</td>
        <td>支持[0, 1, 2]。</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>permY（aclIntArray*）</td>
        <td>输入</td>
        <td>表示矩阵乘输出矩阵的转置序列，host侧的aclIntArray。</td>
        <td>支持[1, 0, 2]。</td>
        <td>INT64</td>
        <td>-</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>batchSplitFactor（int32_t）</td>
        <td>输入</td>
        <td>用于指定矩阵乘输出矩阵中B维的切分大小，Host侧的整型，当前仅支持取值为1。</td>
        <td>当前取值仅支持为1。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out（aclTensor*）</td>
        <td>输出</td>
        <td>表示矩阵乘的输出矩阵，公式中的out。</td>
        <td>
        <ul>
          <li> 数据类型需要与x1与x2推导之后的数据类型保持一致（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li>
          <li> 当前仅支持x1Scale和x2Scale不为空，groupSize为0，batchSplitFactor为1，输出shape为(M, B, N)。</li>
        </ul>
        </td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>-</td>
      </tr>
      </tbody>
      </table>
  

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  
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
      <td>传入的x1、x2、out、x1Scale、x2Scale、permX1、permX2、permY是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>x1、x2或out的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>x1的第二维和x2的第一维度不相等。</td>
    </tr>
    <tr>
      <td>x1、x2、permX1、permX2、permY的维度大小不等于3。</td>
    </tr>
    <tr>
      <td>x1Scale、x2Scale的维度大小不等于1。</td>
    </tr>
    <tr>
      <td>batchSplitFactor不在支持的范围内</td>
    </tr>
    <tr>
      <td>x1Scale、x2Scale的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>permX1、permX2、permY的取值不在支持的范围内。</td>
    </tr>
    <tr>
      <td>x1Scale、x2Scale的shape不符合要求。</td>
    </tr>
  </tbody>
  </table>

## aclnnTransposeQuantBatchMatMul

- **参数说明：**

  <div style="overflow-x: auto;">
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <table><thead>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnTransposeQuantBatchMatMulGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的stream。</td>
    </tr>
  </tbody>
  </table>
  </div>

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明： aclnnTransposeQuantBatchMatMul默认确定性实现。

- <term>Ascend 950PR/Ascend 950DT</term>：
    - permX1和permY支持[1, 0, 2], permX2支持输入[0, 1, 2]。
    - x1Scale和x2Scale为1维，并且x1Scale为(M,), x2Scale为(N,)。
    - out和dtype支持float16和bfloat16。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include <limits>
#include "acl/acl.h"
#include "aclnnop/aclnn_transpose_quant_batch_mat_mul.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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

// BF16 到 float 的转换函数
float bf16_to_float(uint16_t bf16)
{
    uint16_t sign = (bf16 >> 15) & 0x1;
    uint16_t exp = (bf16 >> 7) & 0xFF; // 8 位指数
    uint16_t mant = bf16 & 0x7F;

    // 特殊值处理
    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        } else {
            // 非规格化 BF16 -> float
            return (sign ? -1.0f : 1.0f) * (float)mant * (1.0f / (1 << 7)) * (1.0f / (1, 127));
        }
    } else if (exp == 255) {
        // 无穷大或 NaN
        if (mant == 0) {
            return sign ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        } else {
            return std::numeric_limits<float>::quiet_NaN();
        }
    } else {
        // 规格化数
        float f_exp = (float)(exp - 127);      // 偏移 127
        float f_mant = (float)mant / (1 << 7); // 7 位小数
        float f = (sign ? -1.0f : 1.0f) * (1.0f + f_mant) * (1 << (int)f_exp);
        return f;
    }
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int AclnnTransposeQuantBatchMatmulTest(int32_t deviceId, aclrtStream& stream)
{
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
    std::vector<int64_t> x1ScaleShape = {M};
    std::vector<int64_t> x2ScaleShape = {N};
    std::vector<int64_t> outShape = {M, Batch, N};
    std::vector<int64_t> permX1Series = {1, 0, 2};
    std::vector<int64_t> permX2Series = {0, 1, 2};
    std::vector<int64_t> permYSeries = {1, 0, 2};
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* x1ScaleDeviceAddr = nullptr;
    void* x2ScaleDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* x1Scale = nullptr;
    aclTensor* x2Scale = nullptr;
    aclTensor* out = nullptr;
    std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 0x38);
    std::vector<int8_t> x2HostData(GetShapeSize(x2Shape), 0x38);
    std::vector<float> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
    std::vector<float> x2ScaleHostData(GetShapeSize(x2ScaleShape), 1);
    std::vector<uint16_t> outHostData(GetShapeSize(outShape), 0); // bf16

    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1deviceAddrPtr(x1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2deviceAddrPtr(x2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x1Scale aclTensor
    ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x1ScaledeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建x2Scale aclTensor
    ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> x2ScaledeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_BF16, &out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> outdeviceAddrPtr(outDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclIntArray* permX1 = aclCreateIntArray(permX1Series.data(), permX1Series.size());
    aclIntArray* permX2 = aclCreateIntArray(permX2Series.data(), permX2Series.size());
    aclIntArray* permY = aclCreateIntArray(permYSeries.data(), permYSeries.size());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> executorAddrPtr(nullptr, aclrtFree);

    int32_t batchSplitFactor = 1;
    int32_t groupSize = 0;
    int32_t dtype = 27; // bf16

    // aclnnTransposeQuantBatchMatMul接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // 调用aclnnTransposeQuantBatchMatMul第一段接口
    ret = aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
        x1, x2, (const aclTensor*)nullptr, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY, batchSplitFactor,
        out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeQuantBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        executorAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnTransposeQuantBatchMatMul第二段接口
    ret = aclnnTransposeQuantBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeQuantBatchMatMul failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0); // bf16
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    float resultDataBF16 = 0;
    for (int64_t i = 0; i < size; i++) {
        resultDataBF16 = bf16_to_float(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataBF16);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = AclnnTransposeQuantBatchMatmulTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransposeQuantBatchMatMulTest failed. ERROR: %d\n", ret);
                   return ret);
    Finalize(deviceId, stream);
    return 0;
}
```
