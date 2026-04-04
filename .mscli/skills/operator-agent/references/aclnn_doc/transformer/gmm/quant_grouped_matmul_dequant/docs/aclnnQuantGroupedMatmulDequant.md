# aclnnQuantGroupedMatmulDequant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对输入x进行量化，分组矩阵乘以及反量化。
- 计算公式：  
  1.若输入smoothScaleOptional，则
  
  $$
      x = x\cdot scale_{smooth}
  $$

  2.若不输入xScaleOptional，则为动态量化，需要计算x量化系数
  
  $$
      scale_{x}=row\_max(abs(x))/max_{quantDataType}
  $$

  3.量化
  
  $$
      x_{quantized}=round(x/scale_{x})
  $$

  4.分组矩阵乘+反量化
    - 4.1 若输入的$scale_{weight}$数据类型为FLOAT32, 则：

  $$
    \begin{aligned}
      x^{*}_{quantized} &= x_{quantized}[group[i-1]:group[i]]\\
      out^{*}_{quantized} &= out_{quantized}[group[i-1]:group[i]]\\
      scale^{*}_{x} &= \begin{cases}
                      scale_{x}[group[i-1]:group[i]] & pertoken \\
                      scale_{x} & pertensor \\
                      \end{cases} \\
      out^{*}_{quantized} &= (x^{*}_{quantized}@weight_{quantized}[i] + bias) * scale_{weight}[i] * scale^{*}_{x}
    \end{aligned}
  $$

  - 4.2 若输入的$scale_{weight}$数据类型为INT64, 则：

    $$
    x^{*}_{quantized} = x_{quantized}[group[i-1]:group[i]] \\
    out^{*}_{quantized} = out_{quantized}[group[i-1]:group[i]] \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.int32).\\tobytes(), dtype=np.float32)).reshape(scale_{weight}) \\
    out^{*}_{quantized} = (x^{*}_{quantized}@weight_{quantized}[i] + bias) * scale_{weight}[i]
    $$

    特别说明：如果是上述4.2场景，说明$scale_{weight}$输入前已经和$scale_{x}$做过了矩阵乘运算，因此算子内部计算时省略了该步骤，这要求必须是pertensor静态量化的场景。即输入前要对$scale_{weight}做如下处理得到INT64类型的数据：

    $$
    scale_{weight} = scale_{weight} * scale_{x} \\
    scale_{weight} = torch.tensor(np.frombuffer(scale_{weight}.numpy().astype(np.float32). \\tobytes(), dtype=np.int32).astype(np.int64)).reshape(scale_{weight})
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantGroupedMatmulDequantGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnQuantGroupedMatmulDequant”接口执行计算。

```Cpp
aclnnStatus aclnnQuantGroupedMatmulDequantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *weightScale,
  const aclTensor *groupList,
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
aclnnStatus aclnnQuantGroupedMatmulDequant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```


## aclnnQuantGroupedMatmulDequantGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>表示输入的左矩阵，必选参数，公式中的x。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape支持2维，各个维度表示：（m，k）。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示输入的右矩阵，必选参数，公式中的weight_{quantized}。</td>
      <td>不支持空Tensor。</td>
      <td>INT8</td>
      <td>FRACTAL_NZ、ND</td>
      <td>3、5</td>
      <td>√</td>
    </tr>
      <tr>
      <td>weightScale</td>
      <td>输入</td>
      <td>表示weight的量化系数，必选参数，公式中的scale_{weight}。</td>
      <td><ul><li>shape是2维（g，n），其中g，n与weight的g，n一致。</li><li>当数据类型为INT64时，必须要求xScaleOptional数据类型为FLOAT16，且xQuantMode值为pertensor。</li></ul></td>
      <td>FLOAT32、INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>groupList</td>
      <td>输入</td>
      <td>公式中的group。表示moe场景下输入张量x分组的前缀和结果，shape是1维[g]。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>表示计算的偏移量，可选参数，公式中的bias。</td>
      <td>当前仅支持传入空指针。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>xScaleOptional</td>
      <td>输入</td>
      <td>表示x的量化系数，可选参数，公式中的scale_{x}。</td>
      <td><ul><li>当xQuantMode为pertensor时，shape是1维（1，）；当xQuantMode为pertoken时，shape是1维（m，），其中m与输入x的m一致。若为空则为动态量化。</li><li>当数据类型为FLOAT16时，必须要求weightScale数据类型为INT64，且xQuantMode值为pertensor。</li><li>支持空Tensor。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
     <tr>
      <td>xOffsetOptional</td>
      <td>输入</td>
      <td>表示x的偏移量，可选参数。</td>
      <td>当前仅支持传入空指针。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>smoothScaleOptional</td>
      <td>输入</td>
      <td>表示x的平滑系数，可选参数，公式中的scale_{smooth}。</td>
      <td><ul><li>shape是1维（k，），其中k与x的k一致。</li><li>支持空Tensor。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>xQuantMode</td>
      <td>输入</td>
      <td>指定输入x的量化模式。</td>
      <td>支持取值pertoken/pertensor，动态量化时只支持pertoken。pertoken表示每个token（某一行）都有自己的量化参数；pertensor表示整个张量使用统一的量化参数。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>transposeWeight</td>
      <td>输入</td>
      <td>表示输入weight是否转置。</td>
      <td>当前只支持true。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>输出</td>
      <td>计算结果，必选参数，公式中的out。</td>
      <td>shape支持2维，各个维度表示：（m，n）。其中m与x的m一致，n与weight的n一致。</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
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

    - ND格式下，weight的shape支持3维。 
      - 在transposeWeight为true情况下各个维度表示：（g，n，k）。
      - 在transposeWeight为false情况下各个维度表示：（g，k，n）。
    - FRACTAL_NZ格式下，weight的shape支持5维。
      - 在transposeWeight为true情况下各个维度表示：（g，k1，n1，n0，k0），其中k0 = 32，n0 = 16，k1和x的k需要满足以下关系：ceilDiv（k，32）= k1。
      - 在transposeWeight为false情况下各个维度表示：（g，n1，k1，k0，n0），其中k0 = 16，n0 = 32，k1和x的k需要满足以下关系：ceilDiv（k，16）= k1。
      - 可使用aclnnCalculateMatmulWeightSizeV2接口以及aclnnTransMatmulWeight接口完成输入Format从ND到FRACTAL_NZ格式的转换。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
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
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td><li>如果传入参数类型为aclTensor且其数据类型不在支持的范围之内。</li><li>weight的shape中n或者k不能被16整除。</li></td>
    </tr>
    <tr>
      <td rowspan="3">aclnnAdvanceStepGetWorkspaceSize failed</td>
      <td rowspan="3">561002</td>
      <td>如果传入参数类型为aclTensor且其shape与上述参数说明不符。</td>
    </tr>
  </tbody>
  </table>

## aclnnFatreluMul

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantGroupedMatmulDequantGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnQuantGroupedMatmulDequant默认确定性实现。

- n，k都需要是16的整数倍。
- 当weightScale数据类型为INT64时，必须要求xScaleOptional数据类型为FLOAT16，且xQuantMode值为pertensor；当xScaleOptional数据类型为FLOAT16时，必须要求weightScale数据类型为INT64，且xQuantMode值为pertensor。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_grouped_matmul_dequant.h"

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
  // 调用aclrtMemcpy将Host侧数据复制到device侧内存上
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
  int G = 4;
  int M = 64;
  int K = 256;
  int N = 512;

  char quantMode[16] = "pertoken";
  bool transposeWeight = true;

  std::vector<int64_t> xShape = {M,K};
  std::vector<int64_t> weightShape = {G,N,K};
  std::vector<int64_t> weightScaleShape = {G,N};
  std::vector<int64_t> xScaleShape = {M};
  std::vector<int64_t> smoothScaleShape = {K};
  std::vector<int64_t> outShape = {M,N};
  std::vector<int64_t> groupListShape = {G};

  void* xDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* weightScaleDeviceAddr = nullptr;
  void* groupListDeviceAddr = nullptr;
  void* xScaleDeviceAddr = nullptr;
  void* smoothScaleDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* weightScale = nullptr;
  aclTensor* groupList = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* xScale = nullptr;
  aclTensor* xOffset = nullptr;
  aclTensor* smoothScale = nullptr;
  aclTensor* out = nullptr;

  std::vector<uint16_t> xHostData(GetShapeSize(xShape));
  std::vector<uint8_t> weightHostData(GetShapeSize(weightShape));
  std::vector<float> weightScaleHostData(GetShapeSize(weightScaleShape));
  std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
  groupListHostData[0] = 7;
  groupListHostData[0] = 32;
  groupListHostData[0] = 40;
  groupListHostData[0] = 64;
  std::vector<float> xScaleHostData(GetShapeSize(xScaleShape));
  std::vector<uint16_t> smoothScaleHostData(GetShapeSize(smoothScaleShape));
  std::vector<uint16_t> outHostData(GetShapeSize(outShape));

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, &weightScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
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

  // 调用aclnnQuantGroupedMatmulDequant第一段接口
  ret = aclnnQuantGroupedMatmulDequantGetWorkspaceSize(x, weight, weightScale, groupList,
                                                bias, xScale, xOffset, smoothScale,
                                                quantMode, transposeWeight, out, 
                                                &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulDequantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnQuantGroupedMatmulDequant第二段接口
  ret = aclnnQuantGroupedMatmulDequant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulDequant failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &outDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(weight);
  aclDestroyTensor(weightScale);
  aclDestroyTensor(groupList);
  aclDestroyTensor(xScale);
  aclDestroyTensor(smoothScale);
  aclDestroyTensor(out);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(weightScaleDeviceAddr);
  aclrtFree(groupListDeviceAddr);
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
