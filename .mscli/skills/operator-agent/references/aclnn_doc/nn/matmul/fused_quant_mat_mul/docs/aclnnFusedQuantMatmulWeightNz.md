# aclnnFusedQuantMatmulWeightNz

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品 </term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：量化矩阵乘和Gelu计算融合。
- 计算公式：

  - x1Scale， bias INT32（此场景无offset）：

    $$
    qbmmout = (x1@x2 + bias) * x2Scale * x1Scale
    $$

  - x1Scale， bias BFLOAT16/FLOAT16/FLOAT32（此场景无offset）：

    $$
    qbmmout = x1@x2 * x2scale * x1Scale + bias
    $$

  - x1Scale无bias：

    $$
    qbmmout = x1@x2 * x2Scale * x1Scale
    $$

  - OP类型由fusedOpType输入定义，支持如下：

    - gelu_tanh运算：

      $$
      out = gelu\_tanh(qbmmout)
      $$

    - gelu_erf运算：

      $$
      out = gelu\_erf(qbmmout)
      $$

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedQuantMatmulWeightNzGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedQuantMatmulWeightNz”接口执行计算。

```c++
aclnnStatus aclnnFusedQuantMatmulWeightNzGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *x1Scale,
  const aclTensor *x2Scale,
  const aclTensor *yScaleOptional,
  const aclTensor *x1OffsetOptional,
  const aclTensor *x2OffsetOptional,
  const aclTensor *yOffsetOptional,
  const aclTensor *biasOptional,
  const aclTensor *x3Optional,
  const char      *fusedOpType,
  int64_t          groupSizeOptional,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```
```c++
aclnnStatus aclnnFusedQuantMatmulWeightNz(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnFusedQuantMatmulWeightNzGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 198px">
  <col style="width: 121px">
  <col style="width: 220px">
  <col style="width: 397px">
  <col style="width: 220px">
  <col style="width: 115px">
  <col style="width: 138px">
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
        <td>x1</td>
        <td>输入</td>
        <td>公式中的输入x1。</td>
        <td> 
          <ul>
            <li>仅最后m和k轴转置情况下支持<a href="../../../docs/zh/context/非连续的Tensor.md">非连续的Tensor</a>，其他轴方向不支持非连续的Tensor。</li>
            <li>最后一维大小不能超过65535。</li>
          </ul>
        </td>
        <td>INT4、INT8、INT32</td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>输入</td>
        <td>公式中的输入x2。</td>
        <td>
          <ul>
            <li>支持AI处理器亲和数据排布格式。</li>
            <li>最后一维大小不能超过65535。</li>
          </ul>
        </td>
        <td>INT4、INT8、INT32</td>
        <td>NZ</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1Scale</td>
        <td>输入</td>
        <td>表示量化参数，公式中的输入x1Scale。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Scale</td>
        <td>输入</td>
        <td>表示量化参数，公式中的输入x2Scale。</td>
        <td>-</td>
        <td>FLOAT32、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yScaleOptional</td>
        <td>输入</td>
        <td>输出y的量化scale参数，静态量化时使用。</td>
        <td>预留参数，当前版本不支持，需要传入nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x1OffsetOptional</td>
        <td>输入</td>
        <td>公式中的输入x1Offset。</td>
        <td>预留参数，当前版本不支持，需要传入nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2OffsetOptional</td>
        <td>输入</td>
        <td>公式中的输入x2Offset。</td>
        <td>预留参数，当前版本不支持，需要传入nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOffsetOptional</td>
        <td>输入</td>
        <td>公式中的输入yOffset。</td>
        <td>预留参数，当前版本不支持，需要传入nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>公式中的输入bias。</td>
        <td>无bias时传nullptr。</td>
        <td>INT32、FLOAT32、BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>1、3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x3Optional</td>
        <td>输入</td>
        <td>融合二元运算输入。</td>
        <td>当前只支持融合一元运算，预留参数，当前版本不支持，需要传入nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>fusedOpType</td>
        <td>输入</td>
        <td>公式中的输入fusedOpType，表示指定QuantBatchMatmul算子支持的融合模式。</td>
        <td>融合模式取值必须是"gelu_erf"、"gelu_tanh"中的一种。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSizeOptional</td>
        <td>输入</td>
        <td>用于输入m、n、k方向上的量化分组大小。</td>
        <td>预留参数，当前版本不支持，需要传入nullptr。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>公式中的输出out。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2-6</td>
        <td>✓</td>
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
  <table style="undefined;table-layout: fixed; width: 1083px"><colgroup>
  <col style="width: 251px">
  <col style="width: 129px">
  <col style="width: 703px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="1">161001</td>
      <td>传入的x1、x2、x1Scale、x2Scale或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>x1、x2、biasOptional、x1Scale、x2Scale或out的shape不满足校验条件。</td>
    </tr>
    <tr>
      <td>x1、x2、biasOptional、x1Scale、x2Scale或out的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x1、x2、biasOptional、x1Scale、x2Scale或out是空tensor。</td>
    </tr>
    <tr>
      <td>传入的fusedOpType不属于"gelu_tanh"，"gelu_erf"中的一种。</td>
    </tr>
    <tr>
      <td>传入的当前不支持的yScaleOptional、x1OffsetOptional、x2OffsetOptional、yOffsetOptional、x3Optional、groupSizeOptional参数。</td>
    </tr>
  </tbody>
  </table>


## aclnnFusedQuantMatmulWeightNz

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 153px">
    <col style="width: 121px">
    <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedQuantMatmulWeightNzGetWorkspaceSize获取。</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：aclnnFusedQuantMatmulWeightNz默认确定性实现。
  
- 输入和输出支持以下数据类型组合：
  | x1                        | x2                        | x1Scale     | x2Scale         | x2OffsetOptional    | yScaleOptional   | biasOptional         | yOffsetOptional    | out                                    |
  | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
  | INT8                      | INT8                      | FLOAT32| FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
  

- 当前接口支持x1 pertoken量化和x2 perchannel/pertensor量化，不同的[量化模式](../../../docs/zh/context/量化介绍.md)支持的x1、 x2、x1Scale和x2Scale的输入dtype组合约束为：
  - x1数据类型支持INT8、INT32、INT4。
    - 当数据类型为INT32、INT4时，为INT4量化场景：
      - 当前仅支持ND输入。
      - 当前只支持不转置输入。
      - 要求x1内轴为偶数。
    - 当数据类型为INT32时，每个INT32数据存放8个INT4数据，对应维度表示：（batch，m，k // 8），要求k为8的倍数。
  - x2数据类型支持INT8、INT32、INT4。
    - 该接口仅支持x2为NZ格式，此时x2是NZ格式时，k、n不能为1。
    - 数据类型为INT32时，每个INT32数据存放8个INT4数据。
    - 可使用aclnnConvertWeightToINT4Pack接口完成x2从INT32（1个int32在0~3bit位存储1个int4）到INT32（1个int32存储8个int4）或INT4（1个int4表示1个int4）的数据格式转换，具体参见[aclnnConvertWeightToINT4Pack接口](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack.md)。
    - AI处理器亲和数据排布格式下，shape支持4~8维。
      - 转置时维度为：（batch，k1，n1，n0，k0），batch可不存在，其中k0 = 32， n0 = 16， x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 32） = k1。
      - 不转置时维度为：（batch，n1，k1，k0，n0），batch可不存在，其中k0 = 16，n0 = 32，x1 shape中的k和x2 shape中的k1需要满足以下关系：ceil（k / 16） = k1。
      - 可使用aclnnCalculateMatmulWeightSizeV2接口以及aclnnTransMatmulWeight接口完成输入Format从ND到AI处理器亲和数据排布格式的转换。
  - x1Scale约束如下：
    - shape支持1维，形状为（m,），数据类型支持FLOAT32。
  - x2Scale的约束如下：
    - shape支持1维，形状为（n,）或者（1,），其中n与x2的n一致，数据类型支持FLOAT32、BFLOAT16。
  - biasOptional的约束如下：
    - shape支持1、3维，INT4量化场景下只支持biasOptional为1维，shape为(n)，3维时biasOptional shape为(batch, 1, n)。
    - 数据类型支持int32、float32、bfloat16或float16。
  - out的约束如下：
    - shape支持2~6维，（batch，m，n）。数据类型支持FLOAT16、BFLOAT16。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
x1为INT8，x2为INT8，x1Scale为FLOAT32，x2Scale为FLOAT32。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_fused_quant_matmul_weight_nz.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"

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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  template <typename T>
  int CreateAclTensorX2(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret); return ret);
      size *= sizeof(T);

      // 调用aclrtMalloc申请device侧内存
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, storageShape.data(),
          storageShape.size(), *deviceAddr);
      return 0;
  }

  int aclnnFusedQuantMatmulWeightNzTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {5, 32};
      std::vector<int64_t> x2Shape = {32, 32};
      std::vector<int64_t> x1ScaleShape = {5};
      std::vector<int64_t> x2ScaleShape = {32};
      std::vector<int64_t> outShape = {5, 32};

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
      std::vector<int8_t> x1HostData(5 * 32, 1);
      std::vector<int8_t> x2HostData(32 * 32, 1);
      std::vector<float> x1ScaleHostData(5, 1);
      std::vector<float> x2ScaleHostData(32, 1);
      std::vector<uint16_t> outHostData(5 * 32, 1); // 实际上是float16半精度方式
      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建AI处理器亲和数据排布格式的x2 aclTensor
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      int64_t groupSize = 0;
      const char fusedOpType[] = "gelu_tanh";

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // 调用aclnnTransMatmulWeight第一段接口
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // 调用aclnnTransMatmulWeight第二段接口
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnFusedQuantMatmulWeightNz第一段接口
      workspaceSize = 0;
      ret = aclnnFusedQuantMatmulWeightNzGetWorkspaceSize(
          x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, fusedOpType, groupSize, out,
          &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedQuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtrNZ(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrNZ.reset(workspaceAddr);
      }
      // 调用aclnnFusedQuantMatmulWeightNz第二段接口
      ret = aclnnFusedQuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedQuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0); // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnFusedQuantMatmulWeightNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedQuantMatmulWeightNzTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```