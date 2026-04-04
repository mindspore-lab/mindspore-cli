# aclnnAdaptiveMaxPool3d
## 产品支持情况
[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/pooling/adaptive_max_pool3d)


| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：根据输入的outputSize计算每次kernel的大小，对输入self进行3维最大池化操作，输出池化后的值outputOut和索引indicesOut。aclnnAdaptiveMaxPool3d与aclnnMaxPool3d的区别在于，只需指定outputSize大小，并按outputSize的大小来划分pooling区域。

- 计算公式：
  N（Batch）表示批量大小、C（Channels）表示特征图通道、D（Depth）表示特征图深度、H（Height）表示特征图高度、W（Width）表示特征图宽度。
  对于输入self维度$[N,C,D,H,W]$，outputSize值为$[D_o,H_o,W_o]$的场景，其输出output维度为$[N,C,D_o,H_o,W_o]$，索引indices维度为$[N,C,D_o,H_o,W_o]$，相应tensor中每个元素$(l,m,n)$的计算公式如下：

  $$
  D^{l}_{left} = \lfloor(l*D)/D_o\rfloor
  $$

  $$
  D^{l}_{right} = \lceil((l+1)*D)/D_o\rceil
  $$

  $$
  H^{m}_{left} = \lfloor(m*H)/H_o\rfloor
  $$

  $$
  H^{m}_{right} = \lceil((m+1)*H)/H_o\rceil
  $$

  $$
  W^{n}_{left} = \lfloor(n*W)/W_o\rfloor
  $$

  $$
  W^{n}_{right} = \lceil((n+1)*W)/W_o\rceil
  $$

  $$
  output(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{max} input(N,C,i,j,k)
  $$

  $$
  indices(N,C,l,m,n)=\underset {i \in [D^{l}_{left}, D^{l}_{right}],j\in [H^m_{left},H^m_{right}], k \in [W^n_{left},W^n_{right}] }{argmax} input(N,C,i,j,k)
  $$


## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAdaptiveMaxPool3dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAdaptiveMaxPool3d”接口执行计算。

```Cpp
aclnnStatus aclnnAdaptiveMaxPool3dGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *outputSize,
  aclTensor         *outputOut,
  aclTensor         *indicesOut,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```
```Cpp
aclnnStatus aclnnAdaptiveMaxPool3d(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAdaptiveMaxPool3dGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
    <col style="width: 149px">
    <col style="width: 121px">
    <col style="width: 264px">
    <col style="width: 253px">
    <col style="width: 262px">
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
        <td>self</td>
        <td>输入</td>
        <td>待计算张量。</td>
        <td>D轴H轴W轴3个维度的乘积D*H*W不能大于int32的最大表示，且与outputOut的数据类型保持一致。</td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>NCHW、NCDHW</td>
        <td>4-5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>outputSize</td>
        <td>输入</td>
        <td>输出窗口大小。</td>
        <td>表示输出结果在Dout，Hout，Wout维度上的空间大小。</td>
        <td>-</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outputOut</td>
        <td>输出</td>
        <td>池化后的结果。</td>
        <td>与self的数据类型一致，shape与indicesOut一致。</td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>NCHW、NCDHW</td>
        <td>4-5</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indicesOut</td>
        <td>输出</td>
        <td>outputOut元素在输入self中的索引位置。</td>
        <td>shape与outputOut一致。</td>
        <td>INT32、INT64</td>
        <td>NCHW、NCDHW</td>
        <td>4-5</td>
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
    </tbody></table>

  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>： 参数`indicesOut`的数据类型不支持INT64，

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
      <td>传入的self、outputSize、outputOut或indicesOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>self、outputOut、indicesOut的数据类型、shape、format、参数取值不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>outputSize的shape、参数取值不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self和outputOut数据类型不一致。</td>
    </tr>
    <tr>
      <td>outputOut和indicesOut的shape不一致</td>
    </tr>
    <tr>
      <td>平台不支持。</td>
    </tr>
    <tr>
      <td>depth * height * width > max int32，超出了indices的表示范围。</td>
    </tr>
  </tbody>
  </table>

## aclnnAdaptiveMaxPool3d

- **参数说明：**

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnAdaptiveMaxPool3dGetWorkspaceSize获取。</td>
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
- 确定性计算：
  - aclnnAdaptiveMaxPool3d默认确定性实现。

- Shape描述：
  - self.shape = (N, C, Din, Hin, Win)
  - outputSize = [Dout, Hout, Wout]
  - outputOut.shape = (N, C, Dout, Hout, Wout)
  - indicesOut.shape = (N, C, Dout, Hout, Wout)

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_adaptive_max_pool3d.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
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
  std::vector<int64_t> selfShape = {1, 1, 1, 4, 4};
  std::vector<int64_t> outShape = {1, 1, 1, 2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* indDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4.1, 5, 6, 7,
                                     8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData = {0, 0, 0, 0.0};
  std::vector<int64_t> indicesHostData = {0, 0, 0, 0};

  //创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  //创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, outShape, &indDeviceAddr, aclDataType::ACL_INT32, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> arraySize = {1, 2, 2};
  const aclIntArray *outputSize = aclCreateIntArray(arraySize.data(), arraySize.size());
  CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnAdaptiveMaxPool3d第一段接口
  ret = aclnnAdaptiveMaxPool3dGetWorkspaceSize(self, outputSize, out, indices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool3dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnAdaptiveMaxPool3d第二段接口
  ret = aclnnAdaptiveMaxPool3d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdaptiveMaxPool3d failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  std::vector<int32_t> indicesData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(indicesData.data(), indicesData.size() * sizeof(indicesData[0]), indDeviceAddr,
                    size * sizeof(indicesData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out[%ld] is: %f\n", i, outData[i]);
  }

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);
  aclDestroyIntArray(outputSize);

  // 7. 释放Device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(indDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```

