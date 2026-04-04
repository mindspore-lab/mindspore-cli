# aclnnConvertWeightToINT4Pack

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/convert_weight_to_int4_pack)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

接口功能：对输入weight数据做预处理，实现低比特数据由稀疏存储到紧密存储的排布转换。输出weightInt4Pack的[数据格式](../../../docs/zh/context/数据格式.md)声明为FRACTAL_NZ时，该算子将[数据格式](../../../docs/zh/context/数据格式.md)从ND转为FRACTAL_NZ。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：将INT32类型的weight输入数据打包为紧密排布的INT4数据。
- <term>Ascend 950PR/Ascend 950DT</term> ：将INT32类型的weight打包为紧密排布的INT4类型，将FLOAT类型的weight打包为紧密排布的FLOAT4_E2M1类型。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnConvertWeightToINT4PackGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConvertWeightToINT4Pack”接口执行计算。

```Cpp
aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(
  const aclTensor *weight,
  aclTensor       *weightInt4Pack,
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```
```Cpp
aclnnStatus aclnnConvertWeightToINT4Pack(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor, 
  aclrtStream      stream)
```

## aclnnConvertWeightToINT4PackGetWorkspaceSize

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1078px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 320px">
  <col style="width: 183px">
  <col style="width: 183px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
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
      <td>weight</td>
      <td>输入</td>
      <td>Matmul类算子的低比特量化后的权重，由32bit类型承载4bit的权重值</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND,FRACTAL_NZ</td>
      <td>2-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightInt4Pack</td>
      <td>输出</td>
      <td>Matmul类算子的低比特量化后的权重，权重值为4bit且紧密排布</td>
      <td>-</td>
      <td>INT4,INT32</td>
      <td>ND,FRACTAL_NZ</td>
      <td>-</td>
      <td>-</td>
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

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
    <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>如果传入的必选输入、输出、属性是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>传入weight、weightInt4Pack的shape维度不符合要求。</td>
    </tr>
    <tr>
      <td>传入weight、weightInt4Pack的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>传入weight、weightInt4Pack的shape大小不符合约束要求。</td>
    </tr>
    <tr>
      <td>传入空tensor场景。</td>
    </tr>
    <tr>
      <td>输入tensor的Format不是ND。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_RUNTIME_ERROR</td>
      <td rowspan="2">361001</td>
      <td>数据从host侧拷贝到device侧异常。</td>
    </tr>
    <tr>
      <td>数据从device侧拷贝到host侧异常。</td>
    </tr>
  </tbody>
  </table>

## aclnnConvertWeightToINT4Pack

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnConvertWeightToINT4PackGetWorkspaceSize获取。</td>
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

  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：aclnnConvertWeightToINT4Pack默认确定性实现。

- 参数间数据类型、数据格式间关系如下：

    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
      <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
      <col style="width: 200px">
      <col style="width: 121px">
      <col style="width: 175px">
      <col style="width: 170px">
      <col style="width: 160px">
      <col style="width: 208px">
      <col style="width: 185px">
      </colgroup>
      <thead>
        <tr>
          <th>weight数据类型</th>
          <th>weight数据格式</th>
          <th>weightInt4Pack数据类型</th>
          <th>weightInt4Pack数据格式</th>
          <th>weight shape</th>
          <th>weightInt4Pack view shape</th>
          <th>weightInt4Pack storage shape</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT4</td>
          <td>ND</td>
          <td>最后一维度为2对齐</td>
          <td>和输入weight保持一致，即(dim0, dim1)</td>
          <td>同view shape</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT4</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为2对齐</td>
          <td>和输入weight保持一致，即(dim0, dim1)</td>
          <td>(⌈dim1/64⌉, ⌈dim0/16⌉, 16, 64)</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT32（1个INT32数据存储8个INT4数据）</td>
          <td>ND</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>同view shape</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT32（1个INT32数据存储8个INT4数据）</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>(⌈dim1/64⌉, ⌈dim0/16⌉, 16, 8)</td>
        </tr>
      </tbody></table>

    - <term>Ascend 950PR/Ascend 950DT</term>：
      <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
      <col style="width: 200px">
      <col style="width: 121px">
      <col style="width: 175px">
      <col style="width: 170px">
      <col style="width: 160px">
      <col style="width: 208px">
      <col style="width: 185px">
      </colgroup>
      <thead>
        <tr>
          <th>weight数据类型</th>
          <th>weight数据格式</th>
          <th>weightInt4Pack数据类型</th>
          <th>weightInt4Pack数据格式</th>
          <th>weight shape</th>
          <th>weightInt4Pack view shape</th>
          <th>weightInt4Pack storage shape</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）或FLOAT（承载FLOAT4_E2M1类型数据，数据表示范围[-6.0, 6.0]）</td>
          <td>ND</td>
          <td>INT4或FLOAT4_E2M1</td>
          <td>ND</td>
          <td>最后一维度为2对齐</td>
          <td>和输入weight保持一致，即(dim0, dim1)</td>
          <td>同view shape</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）或FLOAT（承载FLOAT4_E2M1类型数据，数据表示范围[-6.0, 6.0]）</td>
          <td>ND</td>
          <td>INT4或FLOAT4_E2M1</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为2对齐</td>
          <td>和输入weight保持一致，即(dim0, dim1)</td>
          <td>(⌈dim1/16⌉, ⌈dim0/16⌉, 16, 16)</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）或FLOAT（承载FLOAT4_E2M1类型数据，数据表示范围[-6.0, 6.0]）</td>
          <td>ND</td>
          <td>INT32（1个INT32数据存储8个INT4数据）或FLOAT（1个FLOAT数据存储8个FLOAT4_E2M1数据）</td>
          <td>ND</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>同view shape</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）或FLOAT（承载FLOAT4_E2M1类型数据，数据表示范围[-6.0, 6.0]）</td>
          <td>ND</td>
          <td>INT32（1个INT32数据存储8个INT4数据）或FLOAT（1个FLOAT数据存储8个FLOAT4_E2M1数据）</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>(⌈dim1/16⌉, ⌈dim0/16⌉, 16, 2)</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT4（1个INT32数据存储8个INT4数据）</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>(⌈dim1/32⌉, ⌈dim0/16⌉, 16, 32)</td>
        </tr>
      </tbody></table>

## 调用示例

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
  伪量化有aclnnWeightQuantBatchMatmulV2和aclnnWeightQuantBatchMatmulV3接口，这里以aclnnWeightQuantBatchMatmulV2为例。

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *weightInt4Pack,
      uint64_t *workspaceSize, aclOpExecutor **executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
      aclrtStream stream);

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

  template <typename T>
  int CreateAclTensorInt4(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = hostData.size() * sizeof(T);
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
    if (format == aclFormat::ACL_FORMAT_ND) {
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    } else {
      std::vector<int64_t> nzShape;
      if (dataType == aclDataType::ACL_INT4) {
          nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 64};
      } else {
          nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 8};
      }
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    }

    return 0;
  }

  int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclDataType weightInt4PackDtype = aclDataType::ACL_INT4;
    aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    bool isWeightTransposed = true;

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 16;
    int64_t k = 72;
    int64_t n = 17;
    int64_t weightDim0 = k;
    int64_t weightDim1 = n;
    if (isWeightTransposed) {
      weightDim0 = n;
      weightDim1 = k;
    }
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {weightDim0, weightDim1};
    std::vector<int64_t> weightInt4PackShape;
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      weightInt4PackShape = {weightDim0, weightDim1};
    } else {
      weightInt4PackShape = {weightDim0, weightDim1/8};
    }
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightInt4PackDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightInt4Pack = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(m * k, 1);
    std::vector<int32_t> weightHostData(k * n, 1);
    std::vector<float> yHostData(m * n, 0);

    std::vector<int64_t> antiquantScaleShape = {n};
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<float> antiquantScaleHostData(n, 1);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT32, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      std::vector<int8_t> weightInt4PackHostData(n * k / 2, 0); //一个int8数据存放2个int4数据，所以这里除以2
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/2, 32) * CEIL_ALIGN(weightDim0, 16), 0);
      }
      // 创建weightInt4Pack aclTensor
      ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                weightInt4PackDtype, &weightInt4Pack, weightFormat);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
      std::vector<int32_t> weightInt4PackHostData(n * k / 8, 1); //一个int32数据存放8个int4数据，所以这里除以8
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/8, 8) * CEIL_ALIGN(weightDim0, 16), 0);
        ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                  weightInt4PackDtype, &weightInt4Pack, weightFormat);
      } else {
          // 创建weightInt4Pack aclTensor
          ret = CreateAclTensor(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                weightInt4PackDtype, &weightInt4Pack);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建xFp16 aclTensor
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yFp16 aclTensor
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // 对weight做int32转int4pack
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightInt4Pack, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

    // weight为转置场景，且weightInt4Pack shape为NZ时，需要调用aclInitTensor转换为非连续的tensor
    if (isWeightTransposed && weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
      std::vector<int64_t> strides(weightInt4PackShape.size(), 1);
      for (int64_t i = weightInt4PackShape.size() - 2; i >= 0; i--) {
          strides[i] = weightInt4PackShape[i + 1] * strides[i + 1];
      }
      std::swap(strides[0], strides[1]);
      std::swap(weightInt4PackShape[0], weightInt4PackShape[1]);
      std::vector<int64_t> nzShape = {CEIL_DIV(k, 64), CEIL_DIV(n, 16), 16, 8};
      if (weightInt4PackDtype == aclDataType::ACL_INT4) {
          nzShape[3] = 64;
      }
      aclInitTensor(weightInt4Pack, weightInt4PackShape.data(), weightInt4PackShape.size(), weightInt4PackDtype, strides.data(), 0,
                    weightFormat, nzShape.data(), nzShape.size(), weightInt4PackDeviceAddr);
    }

    // 调用cast生成FP16的输入
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 调用aclnnWeightQuantBatchMatmulV2第一段接口
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weightInt4Pack, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnWeightQuantBatchMatmulV2第二段接口
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 将输出转为FP32
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightInt4Pack);
    aclDestroyTensor(antiquantScale);
    aclDestroyTensor(y);
    aclDestroyTensor(xFp16);
    aclDestroyTensor(antiquantScaleFp16);
    aclDestroyTensor(yFp16);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightInt4PackDeviceAddr);
    aclrtFree(antiquantScaleDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(xFp16DeviceAddr);
    aclrtFree(antiquantScaleFp16DeviceAddr);
    aclrtFree(yFp16DeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }

- <term>Ascend 950PR/Ascend 950DT</term>：
  示例代码如下（INT32输入），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
  伪量化有aclnnWeightQuantBatchMatmulV2和aclnnWeightQuantBatchMatmulV3接口， 这里以aclnnWeightQuantBatchMatmulV2为例

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *weightInt4Pack,
      uint64_t *workspaceSize, aclOpExecutor **executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
      aclrtStream stream);

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

  template <typename T>
  int CreateAclTensorInt4(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = hostData.size() * sizeof(T);
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
    if (format == aclFormat::ACL_FORMAT_ND) {
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    } else {
      std::vector<int64_t> nzShape;
      if (dataType == aclDataType::ACL_INT4) {
          nzShape = {CEIL_DIV(shape[1], 16), CEIL_DIV(shape[0], 16), 16, 16};
      } else {
          nzShape = {CEIL_DIV(shape[1], 2), CEIL_DIV(shape[0], 16), 16, 2};
      }
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    }

    return 0;
  }

  int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclDataType weightInt4PackDtype = aclDataType::ACL_INT4;
    aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    bool isWeightTransposed = false;

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 16;
    int64_t k = 64;
    int64_t n = 64;
    int64_t weightDim0 = k;
    int64_t weightDim1 = n;
    if (isWeightTransposed) {
      weightDim0 = n;
      weightDim1 = k;
    }
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {weightDim0, weightDim1};
    std::vector<int64_t> weightInt4PackShape;
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      weightInt4PackShape = {weightDim0, weightDim1};
    } else {
      weightInt4PackShape = {weightDim0, weightDim1/8};
    }
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightPackDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightPacked = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(m * k, 1);
    std::vector<int32_t> weightHostData(k * n, 1);
    std::vector<float> yHostData(m * n, 0);

    std::vector<int64_t> antiquantScaleShape = {n};
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<float> antiquantScaleHostData(n, 1);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT32, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      std::vector<int8_t> weightInt4PackHostData(n * k / 2, 0); //一个int8数据存放2个int4数据，所以这里除以2
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/2, 8) * CEIL_ALIGN(weightDim0, 16), 0);
      }
      // 创建weightPacked aclTensor
      ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightPackDeviceAddr,
                                weightInt4PackDtype, &weightPacked, weightFormat);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
      std::vector<int32_t> weightInt4PackHostData(n * k / 8, 1); //一个int32数据存放8个int4数据，所以这里除以8
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/8, 2) * CEIL_ALIGN(weightDim0, 16), 0);
        ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightPackDeviceAddr,
                                  weightInt4PackDtype, &weightPacked, weightFormat);
      } else {
          // 创建weightPacked aclTensor
          ret = CreateAclTensor(weightInt4PackHostData, weightInt4PackShape, &weightPackDeviceAddr,
                                weightInt4PackDtype, &weightPacked);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建xFp16 aclTensor
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yFp16 aclTensor
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // 对weight做int32转int4pack
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightPacked, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

    // weight为转置场景，且weightPacked shape为NZ时，需要调用aclInitTensor转换为非连续的tensor
    if (isWeightTransposed && weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
      std::vector<int64_t> strides(weightInt4PackShape.size(), 1);
      for (int64_t i = weightInt4PackShape.size() - 2; i >= 0; i--) {
          strides[i] = weightInt4PackShape[i + 1] * strides[i + 1];
      }
      std::swap(strides[0], strides[1]);
      std::swap(weightInt4PackShape[0], weightInt4PackShape[1]);
      std::vector<int64_t> nzShape = {CEIL_DIV(k, 16), CEIL_DIV(n, 16), 16, 2};
      if (weightInt4PackDtype == aclDataType::ACL_INT4) {
          nzShape[3] = 16;
      }
      aclInitTensor(weightPacked, weightInt4PackShape.data(), weightInt4PackShape.size(), weightInt4PackDtype, strides.data(), 0,
                    weightFormat, nzShape.data(), nzShape.size(), weightPackDeviceAddr);
    }

    // 调用cast生成FP16的输入
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 调用aclnnWeightQuantBatchMatmulV2第一段接口
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weightPacked, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnWeightQuantBatchMatmulV2第二段接口
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 将输出转为FP32
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightPacked);
    aclDestroyTensor(antiquantScale);
    aclDestroyTensor(y);
    aclDestroyTensor(xFp16);
    aclDestroyTensor(antiquantScaleFp16);
    aclDestroyTensor(yFp16);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightPackDeviceAddr);
    aclrtFree(antiquantScaleDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(xFp16DeviceAddr);
    aclrtFree(antiquantScaleFp16DeviceAddr);
    aclrtFree(yFp16DeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```

- <term>Ascend 950PR/Ascend 950DT</term>：
  示例代码如下（FLOAT输入），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
  伪量化有aclnnWeightQuantBatchMatmulV2和aclnnWeightQuantBatchMatmulV3接口， 这里以aclnnWeightQuantBatchMatmulV2为例

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *weightInt4Pack,
      uint64_t *workspaceSize, aclOpExecutor **executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
      aclrtStream stream);

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

  template <typename T>
  int CreateAclTensorB4(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = hostData.size() * sizeof(T);
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
    if (format == aclFormat::ACL_FORMAT_ND) {
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    } else {
      std::vector<int64_t> nzShape;
      if (dataType == aclDataType::ACL_INT4 || dataType == aclDataType::ACL_FLOAT4_E2M1) {
          nzShape = {CEIL_DIV(shape[1], 16), CEIL_DIV(shape[0], 16), 16, 16};
      } else {
          nzShape = {CEIL_DIV(shape[1], 2), CEIL_DIV(shape[0], 16), 16, 2};
      }
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    }

    return 0;
  }

  int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclDataType weightPackedDtype = aclDataType::ACL_FLOAT4_E2M1; // 可选：ACL_FLOAT类型
    aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ; // 可选：ACL_FORMAT_ND
    bool isWeightTransposed = false; // ND：支持true/false。NZ：支持false

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 16;
    int64_t k = 64;
    int64_t n = 64;
    int64_t antiquantGroupSize = 32;
    int64_t weightDim0 = k;
    int64_t weightDim1 = n;
    if (isWeightTransposed) {
      weightDim0 = n;
      weightDim1 = k;
    }
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {weightDim0, weightDim1};
    std::vector<int64_t> weightPackedShape;
    if (weightPackedDtype == aclDataType::ACL_FLOAT4_E2M1) {
      weightPackedShape = {weightDim0, weightDim1};
    } else {
      weightPackedShape = {weightDim0, weightDim1 / 8};
    }
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightPackDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightPacked = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(m * k, 1);
    std::vector<float> weightHostData(k * n, 1);
    std::vector<float> yHostData(m * n, 0);

    std::vector<int64_t> antiquantScaleShape = {k / antiquantGroupSize, n};
    std::vector<uint8_t> antiquantScaleHostData(n * k / antiquantGroupSize, 1); // 使用uint8承载float8_e8m0数据

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    if (weightPackedDtype == aclDataType::ACL_FLOAT4_E2M1) {
      std::vector<int8_t> weightB4PackHostData(n * k / 2, 0); //一个B8数据存放2个B4数据，所以这里除以2
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightB4PackHostData.resize(CEIL_ALIGN(weightDim1 / 2, 8) * CEIL_ALIGN(weightDim0, 16), 0);
      }
      // 创建weightPacked aclTensor
      ret = CreateAclTensorB4(weightB4PackHostData, weightPackedShape, &weightPackDeviceAddr,
                                weightPackedDtype, &weightPacked, weightFormat);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
      std::vector<int32_t> weightB4PackHostData(n * k / 8, 1); //一个int32数据存放8个int4数据，所以这里除以8
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightB4PackHostData.resize(CEIL_ALIGN(weightDim1 / 8, 2) * CEIL_ALIGN(weightDim0, 16), 0);
        ret = CreateAclTensorB4(weightB4PackHostData, weightPackedShape, &weightPackDeviceAddr,
                                  weightPackedDtype, &weightPacked, weightFormat);
      } else {
          // 创建weightPacked aclTensor
          ret = CreateAclTensor(weightB4PackHostData, weightPackedShape, &weightPackDeviceAddr,
                                weightPackedDtype, &weightPacked);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建xFp16 aclTensor
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    void* antiquantScaleFp8DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp8 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp8DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &antiquantScaleFp8);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yFp16 aclTensor
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // 对weight做int32转int4pack
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightPacked, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

    // weight为转置场景，且weightPacked shape为ND时，需要调用aclInitTensor转换为非连续的tensor
    if (isWeightTransposed && weightFormat == aclFormat::ACL_FORMAT_ND) {
      weightPackedShape = {k, n};
      std::vector<int64_t> strides = {1, k};
      std::vector<int64_t> storage_shape = {n, k};
      if (weightPackedDtype == aclDataType::ACL_FLOAT) {
        weightPackedShape = {k / 8, n};
        strides = {1, k / 8};
        storage_shape = {n, k / 8};
      }
      aclInitTensor(weightPacked, weightPackedShape.data(), weightPackedShape.size(), weightPackedDtype, strides.data(), 0,
                    weightFormat, storage_shape.data(), storage_shape.size(), weightPackDeviceAddr);
    }

    // 调用cast生成FP16的输入
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 调用aclnnWeightQuantBatchMatmulV2第一段接口
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weightPacked, antiquantScaleFp8, nullptr, nullptr, nullptr, nullptr, antiquantGroupSize, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnWeightQuantBatchMatmulV2第二段接口
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 将输出转为FP32
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightPacked);
    aclDestroyTensor(y);
    aclDestroyTensor(xFp16);
    aclDestroyTensor(antiquantScaleFp8);
    aclDestroyTensor(yFp16);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightPackDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(xFp16DeviceAddr);
    aclrtFree(antiquantScaleFp8DeviceAddr);
    aclrtFree(yFp16DeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```

