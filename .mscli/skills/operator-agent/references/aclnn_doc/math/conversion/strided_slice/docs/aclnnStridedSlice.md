# aclnnStridedSlice

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/strided_slice)

## 产品支持情况
| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：按照指定的起始、结束位置和步长，从输入张量中提取一个子张量。
- 计算公式：在指定维度$dim$上，按照指定的起始位置$begin$、结束位置$end$和步长$strides$，从输入张量$self$中提取子张量$out$。
  $begin$和$end$可以取$[0, self.shape[dim]]$以外的值，取值后根据以下公式转换为合法值，假设self.shape[dim] = N：

  $$
  begin = \begin{cases}
  &0, & if\;begin < -N \\
  &N, & if\;begin >= N\\
  &(begin+N) \% N, & otherwise
  \end{cases}
  $$

  $$
  end = \begin{cases}
  &N, & if\; end >= N\\
  &begin, & else\;if\;  (end+N)\%N < begin \\
  &(end+N)\%N, & otherwise\\
  \end{cases}
  $$

  $out.shape$与$self.shape$只有dim轴上不一致，其他轴一致:

  $$
  out.shape[dim] = ⌊\frac{end - begin + strides - 1}{strides}⌋
  $$

  若存在mask参数，则按照以下规则对outshape进行进一步计算，
  $beginMask$指定$bit$位为1对应的索引维度的$begin$被忽略，
  $endMask$指定$bit$位为1对应的索引维度的$end$被忽略，
  $ellipsisMask$从$bit$位为1对应的索引维度开始全选后续维度，直到遇到指定$begin$才退出，
  $newAxisMask$指定$bit$位为1对应的索引维度增加维度为1的$shape$，
  $shrinkAxisMask$指定$bit$位为1对应的索引维度强制降为1 。

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnStridedSliceGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnStridedSlice”接口执行计算。
```Cpp
aclnnStatus aclnnStridedSliceGetWorkspaceSize(
    const aclTensor   *self, 
    const aclIntArray *begin, 
    const aclIntArray *end, 
    const aclIntArray *strides,
    int64_t           beginMask, 
    int64_t           endMask, 
    int64_t           ellipsisMask, 
    int64_t           newAxisMask, 
    int64_t           shrinkAxisMask,
    aclTensor         *out, 
    uint64_t          *workspaceSize, 
    aclOpExecutor     **executor)
```
```Cpp
aclnnStatus aclnnStridedSlice(
    void          *workspace,
    uint64_t       workspaceSize, 
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnStridedSliceGetWorkspaceSize

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 211px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 308px">
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 150px">
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
      <td>self</td>
      <td>输入</td>
      <td>输入的张量。</td>
      <td>self与out数据类型一致。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>begin</td>
      <td>输入</td>
      <td>每个维度的起始值。</td>
      <td>begin/end/strides数组长度需一致。</td>
      <td>INT32、INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>end</td>
      <td>输入</td>
      <td>每个维度的结束值。</td>
      <td>begin/end/strides数组长度需一致。</td>
      <td>INT32、INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>输入</td>
      <td>每个维度上每个点取值的跨度。</td>
      <td>begin/end/strides数组长度需一致，strides值不能有0。</td>
      <td>INT32、INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beginMask</td> 
      <td>输入</td>
      <td>这个值指定bit位为1对应的索引维度的begin被忽略。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>endMask</td>
      <td>输入</td>
      <td>这个值指定bit位为1对应的索引维度的end被忽略。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ellipsisMask</td>
      <td>输入</td>
      <td>从bit位为1对应的索引维度开始全选，直到遇到指定begin才退出。</td>
      <td>ellipsisMask 只能有一个bit位为1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>newAxisMask</td>
      <td>输入</td>
      <td>把bit位为1对应的索引维度增加维度为1的shape。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shrinkAxisMask</td>
      <td>输入</td>
      <td>把bit位为1对应的索引维度强制降为1 。</td> 
      <td>shrinkAxisMask 中bit位为1的索引，对应的strides需要大于0，即正数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量。</td>
      <td>self与out数据类型一致。</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、FLOAT、FLOAT16、BF16、BOOL、COMPLEX32、COMPLEX64、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>-</td>
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
  </tbody>
  </table>

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 724px">
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
      <td>传入的self、begin、end、strides或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>self或out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self或out的数据类型不一致。</td>
    </tr>
    <tr>
      <td>self的维度大于8维。</td>
    </tr>
    <tr>
      <td>begin和end和strides的长度不一致。</td>
    </tr>
    <tr>
      <td>strides存在等于0的元素。</td>
    </tr>
    <tr>
      <td>out的数据维度与infershape的维度不相同。</td>
    </tr>
    <tr>
      <td>产品型号不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>ellipsisMask 不止有一个bit位为1。</td>
    </tr>
    <tr>
      <td>shrinkAxisMask 中bit位为1的索引，对应的strides小于0。</td>
    </tr>
  </tbody>
  </table>

## aclnnStridedSlice

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 832px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnStridedSliceGetWorkspaceSize获取。</td>
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
  - aclnnStridedSlice默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_strided_slice.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
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

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API文档
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {4, 3};
    std::vector<int64_t> outShape = {2, 2};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclIntArray* begin = nullptr;
    aclIntArray* end = nullptr;
    aclIntArray* strides = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int64_t> beginData = {1, 1};
    std::vector<int64_t> endData = {3, 3};
    std::vector<int64_t> stridesData = {1, 1};
    std::vector<float> outHostData(4, 0);
    int64_t beginMask = 0;
    int64_t endMask = 0;
    int64_t ellipsisMask = 0;
    int64_t newAxisMask = 0;
    int64_t shrinkAxisMask = 0;

    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建aclIntArray
    begin = aclCreateIntArray(beginData.data(), 2);
    CHECK_RET(begin != nullptr, return ret);
    end = aclCreateIntArray(endData.data(), 2);
    CHECK_RET(end != nullptr, return ret);
    strides = aclCreateIntArray(stridesData.data(), 2);
    CHECK_RET(strides != nullptr, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnCast第一段接口
    ret = aclnnStridedSliceGetWorkspaceSize(
        self, begin, end, strides, beginMask, endMask, ellipsisMask, newAxisMask, shrinkAxisMask, out, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSliceGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnCast第二段接口
    ret = aclnnStridedSlice(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnStridedSlice failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyIntArray(begin);
    aclDestroyIntArray(end);
    aclDestroyIntArray(strides);
    aclDestroyTensor(out);

    // 7. 释放device 资源
    aclrtFree(selfDeviceAddr);
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

