# aclnnRopeWithSinCosCacheV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/posembedding/rope_with_sin_cos_cache)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    x     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |


## 功能说明

* 接口功能：推理网络为了提升性能，将sin和cos输入通过cache传入，执行旋转位置编码计算。**该接口相较于[aclnnRopeWithSinCosCache](./aclnnRopeWithSinCosCache.md)接口，新增cacheMode参数，指示拼接cos和sin的方式**：
  - cacheMode=0时，与[aclnnRopeWithSinCosCache](./aclnnRopeWithSinCosCache.md)实现相同，为分段式拼接cos和sin。
  - cacheMode=1时，为交错式拼接cos和sin。
* 计算公式：

    1、**mrope模式**：positions的shape输入是[m, numTokens], m为mropeSection的元素数，支持3或4：
    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    （1）cacheMode为0：
    - mropeSection的元素数为3：
      $$
      cos0 = cos[0, :, :mropeSection[0]]
      $$

      $$
      cos1 = cos[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
      $$

      $$
      cos2 = cos[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
      $$

      $$
      cos = torch.cat((cos0, cos1, cos2), dim=-1)
      $$

      $$
      sin0 = sin[0, :, :mropeSection[0]]
      $$

      $$
      sin1 = sin[1, :, mropeSection[0]:(mropeSection[0] + mropeSection[1])]
      $$

      $$
      sin2 = sin[2, :, (mropeSection[0] + mropeSection[1]):(mropeSection[0] + mropeSection[1] + mropeSection[2])]
      $$

      $$
      sin= torch.cat((sin0, sin1, sin2), dim=-1)
      $$

      $$
      queryRot = query[..., :rotaryDim]
      $$

      $$
      queryPass = query[..., rotaryDim:]
      $$

    - mropeSection的元素数为4：
      $$
      cos = torch.cat([m[i]\ for\ i, m\ in\ enumerate(cos.split(mropeSection, dim=-1))], dim=-1)
      $$

      $$
      sin = torch.cat([m[i]\ for\ i, m\ in\ enumerate(sin.split(mropeSection, dim=-1))], dim=-1)
      $$
      
      $$
      queryRot = query[..., :rotaryDim]
      $$

      $$
      queryPass = query[..., rotaryDim:]
      $$

    （2）cacheMode为1：
    $$
    cosTmp = cos
    $$

    $$
    cos [..., 1:mropeSection[1] * 3:3] = cosTmp[1, ..., 1:mropeSection[1] * 3:3]
    $$

    $$
    cos[..., 2:mropeSection[1] * 3:3] = cosTmp[2, ..., 2:mrope_section[1] * 3:3]
    $$

    $$
    sinTmp = sin
    $$

    $$
    sin[..., 1:mropeSection[1] * 3:3] = sinTmp [1, ..., 1:mropeSection[1] * 3:3]
    $$

    $$
    sin[..., 2:mropeSection[1] * 3:3] = sinTmp [2, ..., 2:mrope_section[1] * 3:3]
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    （1）rotate\_half（GPT-NeoX style）计算模式：
    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    （2）rotate\_interleaved（GPT-J style）计算模式：
    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    2、**rope模式**：positions的shape输入是[numTokens]：
    $$
    cosSin[i] = cosSinCache[positions[i]]
    $$

    $$
    cos, sin = cosSin.chunk(2, dim=-1)
    $$

    $$
    queryRot = query[..., :rotaryDim]
    $$

    $$
    queryPass = query[..., rotaryDim:]
    $$

    （1）rotate\_half（GPT-NeoX style）计算模式：
    $$
    x1, x2 = torch.chunk(queryRot, 2, dim=-1)
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.cat((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

    （2）rotate\_interleaved（GPT-J style）计算模式：
    $$
    x1 = queryRot[..., ::2]
    $$

    $$
    x2 = queryRot[..., 1::2]
    $$

    $$
    o1[i] = x1[i] * cos[i] - x2[i] * sin[i]
    $$

    $$
    o2[i] = x2[i] * cos[i] + x1[i] * sin[i]
    $$

    $$
    queryRot = torch.stack((o1, o2), dim=-1)
    $$

    $$
    query = torch.cat((queryRot, queryPass), dim=-1)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRopeWithSinCosCacheV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnRopeWithSinCosCacheV2”接口执行计算。

```c++
aclnnStatus aclnnRopeWithSinCosCacheV2GetWorkspaceSize(
    const aclTensor   *positions,
    const aclTensor   *queryIn,
    const aclTensor   *keyIn,
    const aclTensor   *cosSinCache,
    const aclIntArray *mropeSection,
    int64_t            headSize,
    bool               isNeoxStyle,
    int64_t            cacheMode,
    aclTensor         *queryOut,
    aclTensor         *keyOut,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnRopeWithSinCosCacheV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnRopeWithSinCosCacheV2GetWorkspaceSize

-   **参数说明**：

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">  
      <col style="width: 550px">  
      <col style="width: 212px">  
      <col style="width: 100px"> 
      <col style="width: 190px">
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
        <td>positions</td>
        <td>输入</td>
        <td>公式中的positions，用于选取位置编码张量。</td>
        <td><ul><li>不支持空tensor。</li><li>rope模式shape为(numTokens)。</li><li>mrope模式shape为(3, numTokens)或(4, numTokens)。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>queryIn</td>
        <td>输入</td>
        <td>公式中的query，要执行旋转位置编码的第一个张量。</td>
        <td><ul><li>不支持空tensor。</li><li>要求是一个2D的Tensor，shape为(numTokens,  numQHeads*headSize)。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>keyIn</td>
        <td>输入</td>
        <td>要执行旋转位置编码的第二个张量。</td>
        <td><ul><li>不支持空tensor。</li><li>要求是一个2D的Tensor，shape为(numTokens,  numKHeads*headSize)。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cosSinCache</td>
        <td>输入</td>
        <td>表示参与计算的位置编码张量。</td>
        <td><ul><li>不支持空tensor。</li><li>要求是一个2D的Tensor，shape为(maxSeqLen, rotaryDim)，maxSeqLen表示模型处理的序列的最大长度，rotaryDim表示旋转位置嵌入的维度大小。</li></ul></td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mropeSection</td>
        <td>输入</td>
        <td>公式中的mropeSection，mrope模式下用于整合输入的位置编码张量信息。</td>
        <td>输入mropeSection属性表示使能mrope模式，不使能mrope模式（即rope模式）输入为nullptr。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>headSize</td>
        <td>输入</td>
        <td>每个注意力头维度大小。</td>
        <td>-</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>isNeoxStyle</td>
        <td>输入</td>
        <td>表示是否使用GPT-NeoX计算模式。</td>
        <td><ul><li>true表示GPT-NeoX style计算模式。</li><li>false表示GPT-J style计算模式。</li></ul></td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cacheMode</td>
        <td>输入</td>
        <td>表示使用分段式或者交错式拼接cos和sin。</td>
        <td><ul><li>0表示分段式。</li><li>1表示交错式。</li></ul></td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>queryOut</td>
        <td>输出</td>
        <td>query执行旋转位置编码后的结果。</td>
        <td><ul><li>数据类型同query。</li><li>要求是一个2D的Tensor，shape为(numTokens,  numQHeads*headSize)。</li></ul></td>
        <td>FLOAT32、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>keyOut</td>
        <td>输出</td>
        <td>key执行旋转位置编码后的结果。</td>
        <td><ul><li>数据类型同key。</li><li>要求是一个2D的Tensor，shape为(numTokens,  numKHeads*headSize)。</li></ul></td>
        <td>FLOAT32、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小。</td>
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

-   **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
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
        <td> ACLNN_ERR_PARAM_NULLPTR </td>
        <td> 161001 </td>
        <td>传入的必选输入、必选输出或者必选属性，是空指针。</td>
        </tr>
        <tr>
        <td> ACLNN_ERR_PARAM_INVALID </td>
        <td> 161002 </td>
        <td>输入和输出的数据类型和数据格式不在支持的范围之内。</td>
        </tr>
        <tr>
        <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
        <td rowspan="2"> 561002 </td>
        <td>多个输入tensor之间的shape信息不匹配。</td>
        </tr>
        <tr>
        <td>输入属性和输入tensor之间的shape信息不匹配。</td>
        </tr>
        <tr>
        <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
        <td rowspan="2"> 361001 </td>
        <td>query或者key非64B对齐。</td>
        </tr>
        <tr>
        <td>rotaryDim>headSize。</td>
        </tr>
    </tbody></table>

## aclnnRopeWithSinCosCacheV2

-   **参数说明：**
    <table style="undefined;table-layout: fixed; width: 1030px"> <colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
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
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnRopeWithSinCosCacheV2GetWorkspaceSize</code>获取。</td>
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

-   **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：
  - aclnnRopeWithSinCosCacheV2默认确定性实现。
- queryIn、keyIn、cosSinCache只支持2维shape输入。
- queryIn、keyIn、cosSinCache输入的数据类型需要保持一致。
- headSize：数据类型为BFLOAT16或FLOAT16时为32的倍数，数据类型为FLOAT32时为16的倍数。
- rotaryDim：始终小于等于headSize；数据类型为BFLOAT16或FLOAT16时为32的倍数，数据类型为FLOAT32时为16的倍数;mrope模式下应满足mropeSection所有元素累加和为rotaryDim值的一半。
- 输入tensor positions的取值应小于cosSinCache的0维maxSeqLen。
- mrope模式下，mropeSection：取值当前仅支持[16, 24, 24]、[24, 20, 20]、[8, 12, 12]和[16, 16, 16, 16]。
- mrope模式下，cacheMode仅支持0和1, 当mropeSection为[16, 16, 16, 16]时，仅支持0。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/level2/aclnn_rope_with_sin_cos_cache_v2.h"
#include <iostream>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> positionsShape = {2};
  std::vector<int64_t> queryInShape = {2, 64};
  std::vector<int64_t> keyInShape = {2, 64};
  std::vector<int64_t> cosSinCacheShape = {2, 32};
  std::vector<int64_t> queryOutShape = {2, 64};
  std::vector<int64_t> keyOutShape = {2, 64};
  void* positionsDeviceAddr = nullptr;
  void* queryInDeviceAddr = nullptr;
  void* keyInDeviceAddr = nullptr;
  void* cosSinCacheDeviceAddr = nullptr;
  void* queryOutDeviceAddr = nullptr;
  void* keyOutDeviceAddr = nullptr;

  aclTensor* positions = nullptr;
  aclTensor* queryIn = nullptr;
  aclTensor* keyIn = nullptr;
  aclTensor* cosSinCache = nullptr;
  int64_t headSize = 32;
  bool isNeoxStyle = true;
  int64_t cacheMode = 1;
  aclTensor *queryOut = nullptr;
  aclTensor *keyOut = nullptr;

  std::vector<int64_t> positionsHostData = {0, 1};
  std::vector<float> queryInHostData = {74, 54, 84, 125, 23, 78, 37, 72, 27, 98, 34, 107, 29, 23, 54, 60, 70, 49,
                                        119, 54, 29, 54, 41, 99, 27, 62, 5, 46, 108, 39, 24, 123, 33, 82, 6, 40, 88,
                                        24, 6, 116, 38, 119, 110, 5, 30, 79, 87, 18, 29, 100, 90, 24, 21, 93, 63, 68,
                                        34, 112, 119, 48, 74, 43, 85, 64, 14, 49, 128, 59, 18, 37, 123, 76, 14, 63, 10,
                                        39, 107, 124, 79, 16, 17, 76, 80, 47, 90, 41, 58, 82, 75, 80, 69, 37, 74, 36, 54,
                                        26, 32, 54, 13, 100, 105, 15, 13, 69, 122, 26, 94, 59, 29, 14, 60, 8, 24, 17, 45,
                                        33, 107, 122, 63, 111, 75, 128, 68, 31, 105, 6, 82, 99};
  std::vector<float> keyInHostData = {112, 32, 66, 114, 69, 31, 117, 122, 77, 57, 78, 119, 115, 25, 54, 27, 122, 65, 15, 85,
                                      33, 16, 36, 6, 95, 15, 43, 6, 66, 91, 14, 101, 78, 51, 110, 74, 56, 30, 127, 61, 53, 29,
                                      32, 65, 114, 77, 26, 116, 89, 38, 75, 14, 96, 91, 87, 34, 25, 42, 57, 26, 51, 43, 23, 42,
                                      40, 17, 98, 117, 53, 75, 68, 75, 38, 41, 115, 76, 67, 22, 76, 10, 24, 46, 85, 54, 61, 114,
                                      10, 59, 6, 123, 58, 10, 115, 9, 13, 58, 66, 120, 23, 30, 83, 13, 11, 76, 18, 82, 57, 4,
                                      117, 105, 8, 73, 127, 5, 91, 56, 12, 125, 20, 3, 104, 40, 46, 18, 89, 63, 99, 104};
  std::vector<float> cosSinCacheHostData = {112, 32, 66, 114, 69, 31, 117, 122, 77, 57, 78, 119, 115, 25, 54, 27, 122, 65, 15, 85,
                                      33, 16, 36, 6, 95, 15, 43, 6, 66, 91, 14, 101, 78, 51, 110, 74, 56, 30, 127, 61, 53, 29,
                                      32, 65, 114, 77, 26, 116, 89, 38, 75, 14, 96, 91, 87, 34, 25, 42, 57, 26, 51, 43, 23, 42};
  std::vector<float> queryOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> keyOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  ret = CreateAclTensor(positionsHostData, positionsShape,
                        &positionsDeviceAddr, aclDataType::ACL_INT64,
                        &positions);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(queryInHostData, queryInShape, &queryInDeviceAddr,
                      aclDataType::ACL_FLOAT, &queryIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(keyInHostData, keyInShape, &keyInDeviceAddr,
                      aclDataType::ACL_FLOAT, &keyIn);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cosSinCacheHostData, cosSinCacheShape, &cosSinCacheDeviceAddr,
                      aclDataType::ACL_FLOAT, &cosSinCache);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(queryOutHostData, queryOutShape, &queryOutDeviceAddr, aclDataType::ACL_FLOAT,
                        &queryOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(keyOutHostData, keyOutShape, &keyOutDeviceAddr, aclDataType::ACL_FLOAT,
                        &keyOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnRopeWithSinCosCacheV2第一段接口
  ret = aclnnRopeWithSinCosCacheV2GetWorkspaceSize(positions, queryIn, keyIn, cosSinCache, nullptr, headSize, isNeoxStyle, cacheMode,
                                               queryOut, keyOut, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnRopeWithSinCosCacheV2GetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnRopeWithSinCosCacheV2第二段接口
  ret = aclnnRopeWithSinCosCacheV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnRopeWithSinCosCacheV2 failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(queryOutShape, &queryOutDeviceAddr);
  PrintOutResult(keyOutShape, &keyOutDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(positions);
  aclDestroyTensor(queryIn);
  aclDestroyTensor(keyIn);
  aclDestroyTensor(cosSinCache);
  aclDestroyTensor(queryOut);
  aclDestroyTensor(keyOut);

  // 7. 释放device资源
  aclrtFree(positionsDeviceAddr);
  aclrtFree(queryInDeviceAddr);
  aclrtFree(keyInDeviceAddr);
  aclrtFree(cosSinCacheDeviceAddr);
  aclrtFree(queryOutDeviceAddr);
  aclrtFree(keyOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
