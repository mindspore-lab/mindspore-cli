# aclnnQuantMatmulDequant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：对输入x进行量化，矩阵乘以及反量化。
- 计算公式：  
  1.若输入smoothScaleOptional，则
  
  $$
      x = x\cdot scale_{smooth}
  $$

  2.若不输入xScaleOptional，则为动态量化，需要计算x量化系数。
  
  $$
      scale_{x}=row\_max(abs(x))/max_{quantDataType}
  $$

  3.量化
  
  $$
      x_{quantized}=round(x/scale_{x})
  $$

  4.矩阵乘+反量化
  - 4.1 若输入的$scale_{weight}$数据类型为FLOAT32, 则：

    $$
      out = (x_{quantized}@weight_{quantized} + bias) * scale_{weight} * scale_{x}
    $$

  - 4.2 若输入的$scale_{weight}$数据类型为INT64, 则：

    $$
      scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.int32).tobytes(), dtype=np.float32)) \\
      out = (x_{quantized}@weight_{quantized} + bias) * scale_{weight}
    $$

    特别说明：如果是上述4.2场景，说明$scale_{weight}$输入前已经和$scale_{x}$做过了矩阵乘运算，因此算子内部计算时省略了该步骤，这要求必须要是pertensor静态量化的场景。即输入前要对$scale_{weight}做如下处理得到INT64类型的数据：

    $$
    scale_{weight} = scale_{weight} * scale_{x} \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.float32). \\tobytes(), dtype=np.int32).astype(np.int64))
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantMatmulDequantGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnQuantMatmulDequant”接口执行计算。

```Cpp
aclnnStatus aclnnQuantMatmulDequantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *weightScale,
  const aclTensor *biasOptional,
  const aclTensor *xScaleOptional,
  const aclTensor *xOffsetOptional,
  const aclTensor *smoothScaleOptional,
  char            *xQuantMode,
  bool             transposeWeight,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnQuantMatmulDequant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulDequantGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1491px"><colgroup>
  <col style="width: 201px">
  <col style="width: 120px">
  <col style="width: 240px">
  <col style="width: 350px">
  <col style="width: 177px">
  <col style="width: 120px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>表示输入的左矩阵，公式中的x。</td>
      <td><ul><li>shape支持2维，各个维度表示：（m，k）。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>表示输入的右矩阵。公式中的weight_{quantized}。</td>
      <td>不支持空Tensor。</td>
      <td>INT</td>
      <td>FRACTAL_NZ、ND</td>
      <td>2、4</td>
      <td>√</td>
    </tr>
      <td>weightScale（aclTensor*）</td>
      <td>输入</td>
      <td>表示weight的量化系数，公式中的scale_{weight}。</td>
      <td><ul><li>shape是1维（n，），其中n与weight的n一致。</li><li>支持空Tensor。</li><li>当数据类型为INT64时，必须要求xScaleOptional数据类型为FLOAT16，且xQuantMode值为pertensor。</li></ul></td>
      <td>FLOAT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示计算的偏移量，公式中的bias。</td>
      <td>当前仅支持传入空指针。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>xScaleOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示x的量化系数，公式中的scale_{x}。</td>
      <td><ul><li>当xQuantMode为pertensor时，shape是1维（1，）；当xQuantMode为pertoken时，shape是1维（m，），其中m与输入x的m一致。若为空则为动态量化。</li><li>支持空Tensor。</li><li>当数据类型为FLOAT16时，必须要求weightScale数据类型为INT64，且xQuantMode值为pertensor。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
       <tr>
      <td>xOffsetOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示x的偏移量。</td>
      <td>当前仅支持传入空指针。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>smoothScaleOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示x的平滑系数，x的平滑系数，公式中的scale_{smooth}。</td>
      <td><ul><li>shape是1维（k，），其中k与x的k一致。</li><li>支持空Tensor。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>xQuantMode（char*）</td>
      <td>输入</td>
      <td>指定输入x的量化模式。</td>
      <td>支持取值pertoken/pertensor，动态量化时只支持pertoken。pertoken表示每个token（某一行）都有自己的量化参数；pertensor表示整个张量使用统一的量化参数。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeWeight（bool）</td>
      <td>输入</td>
      <td>表示输入weight是否转置。</td>
      <td>当前只支持true。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>计算结果，公式中的out。</td>
      <td>shape支持2维，各个维度表示：（m，n）。其中m与x的m一致，n与weight的n一致。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
       <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor（aclOpExecutor**）</td>
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

    - weight参数ND格式下，shape支持2维。
      - 在transposeWeight为true情况下各个维度表示：（n，k）。
      - 在transposeWeight为false情况下各个维度表示：（k，n）。
    - weight参数FRACTAL_NZ格式下，shape支持4维。
      - 在transposeWeight为true情况下各个维度表示：（k1，n1，n0，k0），其中k0 = 32，n0 = 16，k1和x的k需要满足以下关系：ceilDiv（k，32）= k1。
      - 在transposeWeight为false情况下各个维度表示：（n1，k1，k0，n0），其中k0 = 16，n0 = 32，k1和x的k需要满足以下关系：ceilDiv（k，16）= k1。
      - 可使用aclnnCalculateMatmulWeightSizeV2接口以及aclnnTransMatmulWeight接口完成输入Format从ND到FRACTAL_NZ格式的转换。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 281px">
  <col style="width: 119px">
  <col style="width: 749px">
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td><ul><li>如果传入参数类型为aclTensor且其数据类型不在支持的范围之内。</li><li>weight的shape中n或者k不能被16整除。</li></ul></td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>如果传入参数类型为aclTensor且其shape与上述参数说明不符。</td>
    </tr>
  </tbody>
  </table>

## aclnnQuantMatmulDequant

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulDequantGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：aclnnQuantMatmulDequant默认确定性实现。

- n，k都需要是16的整数倍。
- 当weightScale数据类型为INT64时，必须要求xScaleOptional数据类型为FLOAT16，且xQuantMode值为pertensor；当xScaleOptional数据类型为FLOAT16时，必须要求weightScale数据类型为INT64，且xQuantMode值为pertensor。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_matmul_dequant.h"

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
  int M = 64;
  int K = 256;
  int N = 512;

  char quantMode[16] = "pertoken";
  bool transposeWeight = true;

  std::vector<int64_t> xShape = {M,K};
  std::vector<int64_t> weightShape = {N,K};
  std::vector<int64_t> weightScaleShape = {N};
  std::vector<int64_t> xScaleShape = {M};
  std::vector<int64_t> smoothScaleShape = {K};
  std::vector<int64_t> outShape = {M,N};

  void* xDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* weightScaleDeviceAddr = nullptr;
  void* xScaleDeviceAddr = nullptr;
  void* smoothScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* weightScale = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* xScale = nullptr;
  aclTensor* xOffset = nullptr;
  aclTensor* smoothScale = nullptr;
  aclTensor* out = nullptr;

  std::vector<uint16_t> xHostData(GetShapeSize(xShape));
  std::vector<uint8_t> weightHostData(GetShapeSize(weightShape));
  std::vector<uint16_t> weightScaleHostData(GetShapeSize(weightScaleShape));
  std::vector<uint16_t> xScaleHostData(GetShapeSize(xScaleShape));
  std::vector<uint16_t> smoothScaleHostData(GetShapeSize(smoothScaleShape));
  std::vector<uint16_t> outHostData(GetShapeSize(outShape));

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, &weightScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xScaleHostData, xScaleShape, &xScaleDeviceAddr, aclDataType::ACL_FLOAT, &xScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(smoothScaleHostData, smoothScaleShape, &smoothScaleDeviceAddr, aclDataType::ACL_FLOAT16, &smoothScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // 调用aclnnQuantMatmulDequant第一段接口
  ret = aclnnQuantMatmulDequantGetWorkspaceSize(x, weight, weightScale, 
                                                bias, xScale, xOffset, smoothScale,
                                                quantMode, transposeWeight, out, 
                                                &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulDequantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnQuantMatmulDequant第二段接口
  ret = aclnnQuantMatmulDequant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulDequant failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(weightScale);
  aclDestroyTensor(xScale);
  aclDestroyTensor(smoothScale);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(weightScaleDeviceAddr);
  aclrtFree(xScaleDeviceAddr);
  aclrtFree(smoothScaleDeviceAddr);
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