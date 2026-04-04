# aclnnSwinTransformerLnQkvQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：Swin Transformer 网络模型 完成 Q、K、V 的计算。  
- 计算公式：  

  q/k/v = (Quant(Layernorm(x).transpose)  * weight).dequant.transpose.split
  其中，weight 是 Q、K、V 三个矩阵权重的拼接。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSwinTransformerLnQkvQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwinTransformerLnQkvQuant”接口执行计算。

```cpp
aclnnStatus aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *gamma, 
  const aclTensor *beta, 
  const aclTensor *weight, 
  const aclTensor *bias, 
  const aclTensor *quantScale, 
  const aclTensor *quantOffset, 
  const aclTensor *dequantScale, 
  int64_t          headNum, 
  int64_t          seqLength, 
  double           epsilon, 
  int64_t          oriHeight, 
  int64_t          oriWeight, 
  int64_t          WinSize, 
  int64_t          wWinSize, 
  bool             weightTranspose, 
  const aclTensor *queryOutputOut, 
  const aclTensor *keyOutputOut, 
  const aclTensor *valueOutputOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnSwinTransformerLnQkvQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnSwinTransformerLnQkvQuantGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1524px"><colgroup>
  <col style="width: 166px">
  <col style="width: 121px">
  <col style="width: 336px">
  <col style="width: 250px">
  <col style="width: 149px">
  <col style="width: 128px">
  <col style="width: 230px">
  <col style="width: 144px">
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
      <td>x（aclTensor *）</td>
      <td>输入</td>
      <td>表示待进行归一化计算的目标张量。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>只支持维度为[B,S,H]，其中B为batch size且只支持[1,32],S为原始图像长宽的乘积，H为序列长度和通道数的乘积且小于等于1024</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gamma（aclTensor *）</td>
      <td>输入</td>
      <td>表示layernorm计算中尺度缩放的大小。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>只支持1维且为[H]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta（aclTensor *）</td>
      <td>输入</td>
      <td>表示layernorm计算中尺度偏移的大小。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>只支持1维且为[H]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight（aclTensor *）</td>
      <td>输入</td>
      <td>表示目标张量转换使用的权重矩阵。</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>只支持2维且维度为[H, 3 * H]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias（aclTensor *）</td>
      <td>输入</td>
      <td>表示目标张量转换使用的偏移矩阵。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>只支持1维且维度为[3 * H]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantScale（aclTensor *）</td>
      <td>输入</td>
      <td>表示目标张量量化使用的缩放参数。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>只支持1维且维度为[H]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantOffset（aclTensor *）</td>
      <td>输入</td>
      <td>表示目标张量量化使用的偏移参数。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>只支持1维且维度为[H]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dequantScale（aclTensor *）</td>
      <td>输入</td>
      <td>表示目标张量乘以权重矩阵之后反量化使用的缩放参数。</td>
      <td>-</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>只支持1维且维度为[3 * H]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headNum（int64_t）</td>
      <td>输入</td>
      <td>表示转换使用的通道数。</td>
      <td>支持取值为[1,32]。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>seqLength（int64_t）</td>
      <td>输入</td>
      <td>表示转换使用的通道深度。</td>
      <td>支持取值为32/64。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>layernorm 计算除0保护值。</td>
      <td>为了保证精度，取值建议小于等于1e-4。</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriHeight（int64_t）</td>
      <td>输入</td>
      <td>layernorm 中S轴transpose的维度；oriHeight*oriWeight需等于输入x的第二维S的大小，且为hWinSize的整数倍。</td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>oriWeight（int64_t）</td>
      <td>输入</td>
      <td>layernorm 中S轴transpose的维度；oriHeight*oriWeight需等于输入x的第二维S的大小，且为wWinSize的整数倍。</td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>hWinSize（int64_t）</td>
      <td>输入</td>
      <td>使用的特征窗高度大小。</td>
      <td>支持取值为[7,32]。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>wWinSize（int64_t）</td>
      <td>输入</td>
      <td>使用的特征窗宽度大小。</td>
      <td>支持取值为[7,32]。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightTranspose（bool）</td>
      <td>输入</td>
      <td>weight矩阵是否转置。</td>
      <td>当前不支持取值为False场景。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>queryOutputOut（aclTensor *）</td>
      <td>输出</td>
      <td>表示转换之后的张量，公式中的Q。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keyOutputOut（aclTensor *）</td>
      <td>输出</td>
      <td>表示转换之后的张量，公式中的K。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>valueOutputOut（aclTensor *）</td>
      <td>输出</td>
      <td>表示转换之后的张量，公式中的V。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t）</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor **）</td>
      <td>出参</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>  

- **返回值**：

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
      <td>传入的输入tensor是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出参数的数据类型/数据格式不在支持的范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnSwinTransformerLnQkvQuant

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwinTransformerLnQkvQuantGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnSwinTransformerLnQkvQuant默认确定性实现。
- seqLength只支持32/64。
- oriHeight*oriWeight=输入x Tensor的第二维度，且oriHeight为hWinSize的整数倍，oriWeight为wWinSize的整数倍。
- hWinSize和wWinSize范围只支持7~32。
- 输入x Tensor的第一维度B只支持1~32。
- weight需要转置。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swin_transformer_ln_qkv_quant.h"

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
  std::vector<int64_t> selfShape = {1, 49, 32};
  std::vector<int64_t> gammaShape = {32};
  std::vector<int64_t> weightShape = {32*3, 32};
  std::vector<int64_t> biasShape = {3 * 32};
  std::vector<int64_t> outShape = {1,1,49, 32};
  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* dequantDeviceAddr = nullptr;

  void* outqDeviceAddr = nullptr;
  void* outkDeviceAddr = nullptr;
  void* outvDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* quantScale = nullptr;
  aclTensor* quantOffset = nullptr;
  aclTensor* dequantScale = nullptr;
  aclTensor* queryOutput = nullptr;
  aclTensor* keyOutput = nullptr;
  aclTensor* valueOutput = nullptr;

  std::vector<uint16_t> selfHostData(49*32, 0x1);
  std::vector<int32_t> biasHostData(3*32, 0x1);
  std::vector<uint16_t> gammaHostData(32, 0x1);
  std::vector<uint16_t> betaHostData(32, 0x1);
  std::vector<int8_t> weightHostData(3*32*32, 0x1);
  std::vector<uint16_t> scaleHostData(32, 0x1);
  std::vector<uint16_t> offsetHostData(32, 0x1);
  std::vector<uint64_t> dequantHostData(3*32, 0x1);

  std::vector<uint16_t> outqHostData(49*32, 0x0);
  std::vector<uint16_t> outkHostData(49*32, 0x0);
  std::vector<uint16_t> outvHostData(49*32, 0x0);

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(betaHostData, gammaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT8, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(scaleHostData, gammaShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT16, &quantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(offsetHostData, gammaShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT16, &quantOffset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(dequantHostData, biasShape, &dequantDeviceAddr, aclDataType::ACL_UINT64, &dequantScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outqHostData, outShape, &outqDeviceAddr, aclDataType::ACL_FLOAT16, &queryOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outkHostData, outShape, &outkDeviceAddr, aclDataType::ACL_FLOAT16, &keyOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outvHostData, outShape, &outvDeviceAddr, aclDataType::ACL_FLOAT16, &valueOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  float epsilon = 0.0001;
  int64_t oriHeight = 7;
  int64_t oriWeight = 7;
  int64_t hWinSize = 7;
  int64_t wWinSize = 7;
  int64_t headNum = 1;
  int64_t seqLength = 32;
  bool weightTranspose = true;

  // 调用aclnnSwinTransformerLnQkvQuant第一段接口
  ret = aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(x,gamma,beta,weight, bias, quantScale, quantOffset, dequantScale, headNum, seqLength, epsilon, oriHeight, oriWeight, hWinSize, wWinSize, weightTranspose, queryOutput, keyOutput, valueOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinTransformerLnQkvQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSwinTransformerLnQkvQuant第二段接口
  ret = aclnnSwinTransformerLnQkvQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwinTransformerLnQkvQuant failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outqDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(queryOutput);
  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(outqDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```