# aclnnNsaCompressGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|     x      |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|     √      |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |


## 功能说明

- 算子功能：aclnnNsaCompress算子的反向计算。

- 计算公式：
  选择注意力的正向计算公式如下：

    $$
    \text{dw} = \text{dk\_cmp} \cdot K^\top
    $$

    $$
    \text{dk} = W^\top \cdot \text{dk\_cmp}
    $$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNsaCompressGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaCompressGrad”接口执行计算。

```c++
aclnnStatus aclnnNsaCompressGradGetWorkspaceSize(
  const aclTensor   *outputGrad,
  const aclTensor   *input,
  const aclTensor   *weight,
  const aclIntArray *actSeqLenOptionalOptional,
  int64_t            compressBlockSize,
  int64_t            compressStride,
  int64_t            actSeqLenType,
  char              *layoutOptionalOptional,
  const aclTensor   *inputGradOut,
  const aclTensor   *weightGradOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaCompressGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnNsaCompressGradGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1565px">
    <colgroup>
      <col style="width: 146px">
      <col style="width: 135px">
      <col style="width: 326px">
      <col style="width: 246px">
      <col style="width: 275px">
      <col style="width: 101px">
      <col style="width: 190px">
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>query</td>
        <td>输入</td>
        <td>公式中的query。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>key</td>
        <td>输入</td>
        <td>公式中的key。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>value</td>
        <td>输入</td>
        <td>公式中的value。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3-4</td>
        <td>√</td>
      </tr>
      <tr>
        <td>attenMaskOptional</td>
        <td>输入</td>
        <td>公式中的atten_mask。</td>
        <td>
          <ul>
            <li>输入shape需为[S,S]。</li>
            <li>TND场景只支持SS格式，SS分别是max(Sq)和max(CmqSkv)。</li>
          </ul>
        </td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>actualSeqQLenOptional</td>
        <td>输入</td>
        <td>描述每个Batch对应的query S大小(Sq)。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td></td>
      </tr>
      <tr>
        <td>outputGrad</td>
        <td>输入</td>
        <td>正向算子输出的反向梯度。</td>
        <td>-</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>input</td>
        <td>输入</td>
        <td>待压缩张量。</td>
        <td>-</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>压缩的权重，与input的shape满足broadcast关系。</td>
        <td>数据类型与input一致。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>actSeqLenOptional</td>
        <td>输入</td>
        <td>描述每个Batch对应的S大小，各batch长度不等时需要输入。</td>
        <td>-</td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td></td>
      </tr>
      <tr>
        <td>compressBlockSize</td>
        <td>输入</td>
        <td>压缩滑窗大小。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>输入</td>
        <td>两次压缩滑窗间隔大小。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
      <tr>
        <td>actSeqLenType</td>
        <td>输入</td>
        <td>取值0或1，当前仅支持0。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
      <tr>
        <td>layoutOptional</td>
        <td>输入</td>
        <td>代表输入input的数据排布格式，支持TND。</td>
        <td>-</td>
        <td>string</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
      <tr>
        <td>inputGrad</td>
        <td>输出</td>
        <td>input的梯度，与input shape一致。</td>
        <td>数据类型与input一致。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weightGrad</td>
        <td>输出</td>
        <td>weight的梯度，与weight shape一致。</td>
        <td>数据类型与weight一致。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
        <td>2</td>
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
        <td></td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td></td>
      </tr>
    </tbody>
  </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
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
      <td>传入的input、weight、outputGrad、inputGrad或weightGrad是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>input和weight的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input和weight的shape无法做broadcast。</td>
    </tr>
    <tr>
      <td>layoutOptional不合法。</td>
    </tr>
  </tbody>
  </table>


## aclnnNsaCompressGrad

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNsaCompressGradGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnNsaCompressGrad默认确定性实现。
- compressBlockSize和compressStride要是16的整数倍，且compressBlockSize > compressStride


## 调用示例

通过aclnn单算子调用示例代码如下（以Atlas A2 训练系列产品），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_compress_grad.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)

#define CHECK_RET(cond, return_expr)     \
    do {                                 \
        if (!(cond)) {                   \
            return_expr;                 \
        }                                \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
      shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
    // 固定写法，acl初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** xOrResult) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续xOrResult的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
    *xOrResult = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                 shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
    // 1. （固定写法）device/context/stream初始化，参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t headNum = 64;
    int64_t headDim = 128;
    int64_t blockSize = 32;
    int64_t blockStride = 16;
    int64_t blockNum = 15;
    int64_t seqLensSum = 272;
    int64_t seqLen = 3;
    std::vector<int64_t> outputGradShape = {blockNum, headNum, headDim};
    std::vector<int64_t> inputKVShape = {seqLensSum, headNum, headDim};
    std::vector<int64_t> weightShape = {blockSize, headNum};
    std::vector<int64_t> inputGradOutShape = {seqLensSum, headNum, headDim};
    std::vector<int64_t> weightGradOutShape = {blockSize, headNum};
    int64_t SeqLenType = 0;
    char layOut[] = "TND";

    void* outputGradDeviceAddr = nullptr;
    void* inputKVDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* inputGradOutDeviceAddr = nullptr;
    void* weightGradOutDeviceAddr = nullptr;

    aclTensor* outputGrad = nullptr;
    aclTensor* inputKV = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* inputGradOut = nullptr;
    aclTensor* weightGradOut = nullptr;

    std::vector<float> inputGradOutHostData(seqLensSum * headNum * headDim, 0.0);
    std::vector<float> weightGradOutHostData(blockSize * headNum, 0.0);

    std::vector<float> outputGradHostData(blockNum * headNum * headDim, 1.0);
    std::vector<float> inputKVHostData(seqLensSum * headNum * headDim, 1.0);
    std::vector<float> weightHostData(blockSize * headNum, 1.0);
    std::vector<int64_t> actSeqLenOptionalHostData = {0, 128, 272};

    aclIntArray *actSeqLenOptional = aclCreateIntArray(actSeqLenOptionalHostData.data(), actSeqLenOptionalHostData.size());

    // 创建dy aclTensor
    ret = CreateAclTensor(outputGradHostData, outputGradShape, &outputGradDeviceAddr, aclDataType::ACL_FLOAT16,
                          &outputGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(inputKVHostData, inputKVShape, &inputKVDeviceAddr, aclDataType::ACL_FLOAT16, &inputKV);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gelu aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    ret = CreateAclTensor(inputGradOutHostData, inputGradOutShape, &inputGradOutDeviceAddr, aclDataType::ACL_FLOAT16, &inputGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightGradOutHostData, weightGradOutShape, &weightGradOutDeviceAddr, aclDataType::ACL_FLOAT16, &weightGradOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnGeGluBackward第一段接口
    ret = aclnnNsaCompressGradGetWorkspaceSize(
        outputGrad, inputKV, weight, actSeqLenOptional, blockSize, blockStride, SeqLenType, layOut,
        inputGradOut, weightGradOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeGluGradV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnGeGluBackward第二段接口
    ret = aclnnNsaCompressGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGeGluGradV2 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(inputGradOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), inputGradOutDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(outputGrad);
    aclDestroyTensor(inputKV);
    aclDestroyTensor(weight);
    aclDestroyTensor(inputGradOut);
    aclDestroyTensor(weightGradOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(outputGradDeviceAddr);
    aclrtFree(inputKVDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(inputGradOutDeviceAddr);
    aclrtFree(weightGradOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```