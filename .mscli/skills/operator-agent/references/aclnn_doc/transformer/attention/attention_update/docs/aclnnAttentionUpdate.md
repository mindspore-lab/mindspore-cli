# aclnnAttentionUpdate

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：将各SP域PA算子的输出的中间结果lse，localOut两个局部变量结果更新成全局结果。
- 计算公式：

$$
lse_{max} = \text{max}lse_i
$$

$$
lse = \sum_i \text{exp}(lse_i - lse_{max})
$$

$$
lse_m = lse_{max} + \text{log}(lse)
$$

$$
O = \sum_i O_i \cdot \text{exp}(lse_i - lse_m)
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAttentionUpdateGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAttentionUpdate”接口执行计算。

```c++
aclnnStatus aclnnAttentionUpdateGetWorkspaceSize(
   const aclTensorList  *lse, 
   const aclTensorList  *localOut, 
   int64_t              updateType, 
   aclTensor            *out, 
   aclTensor            *lseOut, 
   uint64_t             *workspaceSize, 
   aclOpExecutor        **executor)
```

```c++
aclnnStatus aclnnAttentionUpdate(
   void          *workspace,
   uint64_t       workspaceSize,
   aclOpExecutor *executor,
   aclrtStream    stream)
```

## aclnnAttentionUpdateGetWorkspaceSize

- **参数说明**
  
  <table style="undefined; table-layout: fixed; width: 1567px">
    <colgroup>
      <col style="width: 170px"> <!-- 参数名 -->
      <col style="width: 120px"> <!-- 输入/输出 -->
      <col style="width: 300px"> <!-- 描述 -->
      <col style="width: 330px"> <!-- 使用说明 -->
      <col style="width: 212px"> <!-- 数据类型 -->
      <col style="width: 100px"> <!-- 数据格式 -->
      <col style="width: 190px"> <!-- 维度(shape) -->
      <col style="width: 145px"> <!-- 非连续Tensor -->
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>lse</td>
        <td>输入</td>
        <td>各SP域的局部lse。</td>
        <td>tensorList长度为sp。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>localOut</td>
        <td>输入</td>
        <td>各SP域的局部attentionout。</td>
        <td>tensorList长度为sp。</td>
        <td>FLOAT32，FLOAT16，BFLOAT16</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum, headDim]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>updateType</td>
        <td>输入</td>
        <td>控制lseOut是否输出。</td>
        <td>支持0、1，分别表示不输出lseOut，输出lseOut。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>输出的tensor。</td>
        <td>
          - 
        </td>
        <td>与localOut一致</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum, headDim]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>lseOut</td>
        <td>可选输出</td>
        <td>作为lse_m可选输出。</td>
        <td>不输出lseOut可传入nullptr。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>[batch * seqLen * headNum]</td>
        <td>x</td>
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
  <ul>
    <li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持FLOAT32、FLOAT16、BFLOAT16的localOut和out。</li>
    <li><term>Ascend 950PR/Ascend 950DT</term>：支持FLOAT32、FLOAT16、BFLOAT16的localOut和out。</li>
  </ul>

- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <div style="overflow-x: auto;">
    <table style="table-layout: fixed; width: 1100px">
      <colgroup>
        <col style="width: 250px">
        <col style="width: 130px">
        <col style="width: 720px">
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
          <td>传入的lse、localOut或者out是空指针。</td>
        </tr>
        <tr>
          <!-- 合并单元格添加 merged-cell 类，与模板结构保持一致 -->
          <td class="merged-cell" rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
          <td class="merged-cell" rowspan="5">161002</td>
          <td>传入的lse、localOut或者out的数据类型/数据格式不在支持的范围之内。</td>
        </tr>
        <tr>
          <td>传入的updateType或者sp不在取值范围。</td>
        </tr>
        <tr>
          <td>传入的lse、localOut或者out的shape不满足约束。</td>
        </tr>
        <tr>
          <td>updateType为0时，传入的lseOut不为nullptr。</td>
        </tr>
        <tr>
          <td>updateType为1时，传入的lseOut为nullptr。</td>
        </tr>
      </tbody>
    </table>
  </div>

## aclnnAttentionUpdate

- **参数说明**
  
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAttentionUpdateGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream流。</td>
    </tr>
  </tbody>
  </table>
  
- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnAttentionUpdate默认确定性实现。
- 序列并行的并行度sp取值范围[1, 16]。
- headDim取值范围[8, 512]且是8的倍数。
- 支持空Tensor。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_attention_update.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                     \
      if (!(cond)) {                       \
          Finalize(deviceId, stream);      \
          return_expr;                     \
      }                                    \
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
  // 固定写法，初始化
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

  // 计算连续tensor的stride
  std::vector<int64_t> stride(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    stride[i] = shape[i + 1] * stride[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

aclnnStatus aclnnAttentionUpdateTest(int32_t deviceId, aclrtStream& stream) {
  auto ret = Init(deviceId, &stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造

  std::vector<int64_t> lseShape = {256};
  std::vector<int64_t> localOutShape = {256, 128};
  std::vector<int64_t> outShape = {256, 128};

  int64_t updateType = 0;

  void* lseDeviceAddr[2] = {nullptr, nullptr};
  void* localOutDeviceAddr[2] = {nullptr, nullptr};
  void* outDeviceAddr = nullptr;

  std::vector<aclTensor*> lse = {nullptr, nullptr};
  std::vector<aclTensor*> localOut = {nullptr, nullptr};
  aclTensor* out = nullptr;

  std::vector<float> lse1HostData(GetShapeSize(lseShape), 1);
  std::vector<float> lse2HostData(GetShapeSize(lseShape), 1);
  std::vector<float> localOut1HostData(GetShapeSize(localOutShape), 1);
  std::vector<float> localOut2HostData(GetShapeSize(localOutShape), 1);
  std::vector<float> outHostData(GetShapeSize(outShape), 0);

  // 创建lse aclTensorList
  ret = CreateAclTensor(lse1HostData, lseShape, &(lseDeviceAddr[0]), aclDataType::ACL_FLOAT, &(lse[0]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> lse1TensorPtr(lse[0], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> lse1DeviceAddrPtr(lseDeviceAddr[0], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lse2HostData, lseShape, &(lseDeviceAddr[1]), aclDataType::ACL_FLOAT, &(lse[1]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> lse2TensorPtr(lse[1], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> lse2DeviceAddrPtr(lseDeviceAddr[1], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclTensorList *lseList = aclCreateTensorList(lse.data(), lse.size());

  // 创建localOut aclTensorList
  ret = CreateAclTensor(localOut1HostData, localOutShape, &(localOutDeviceAddr[0]), aclDataType::ACL_FLOAT, &(localOut[0]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> localOut1TensorPtr(localOut[0], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> localOut1DeviceAddrPtr(localOutDeviceAddr[0], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(localOut2HostData, localOutShape, &(localOutDeviceAddr[1]), aclDataType::ACL_FLOAT, &(localOut[1]));
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> localOut2TensorPtr(localOut[1], aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> localOut2DeviceAddrPtr(localOutDeviceAddr[1], aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclTensorList *localOutList = aclCreateTensorList(localOut.data(), localOut.size());

  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnAttentionUpdate第一段接口
  ret = aclnnAttentionUpdateGetWorkspaceSize(lseList,
                                       localOutList,
                                       updateType,
                                       out,
                                       nullptr,
                                       &workspaceSize,
                                       &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // 调用aclnnAttentionUpdate第二段接口
  ret = aclnnAttentionUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnAttentionUpdateTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAttentionUpdateTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```