# aclnnRound

# 产品支持情况

## **产品支持情况**

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | -------- |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √        |

## 功能说明

- 算子功能：对输入张量的每一个元素，四舍五入。

- 计算公式：

  $$out_i=round(input_i)$$

  举例如下：

  Round(3.56) = 4.0

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRoundGetWorkspaceSize”接口获取计算所需workspace大小及包含了算子计算流程的执行器，再调用“aclnnRound”接口执行计算

```c++
aclnnStatus aclnnRoundGetWorkspaceSize(
    const aclTensor* self, 
    aclTensor* out, 
    uint64_t* workspaceSize, 
    aclOpExecutor** executor
)
```

```c++
aclnnStatus aclnnRound(
    void* workspace, 
    uint64_t workspaceSize, 
    aclOpExecutor* executor,
    const aclrtStream stream)
```

## aclnnRoundGetWorkspaceSize

- 参数说明

  | 参数名        | 输入/输出 | 描述                                    | 使用说明        | 数据类型                        | 数据格式 | 维度（shape） |
  | ------------- | --------- | --------------------------------------- | --------------- | ------------------------------- | -------- | ------------- |
  | self          | 输入      | 待进行round计算的入参，公式中的self     | 无              | bfloat16，float16，float，int32 | ND       | 0-8           |
  | out           | 输出      | 待进行round计算的出参，公式中的out      | shape与self相同 | bfloat16，float16，float，int32 | ND       | 0-8           |
  | workspaceSize | 输出      | 返回需要在Device侧申请的workspace大小。 | -               | -                               | -        | -             |
  | executor      | 输出      | 返回op执行器，包含了算子计算流程。      | -               | -                               | -        | -             |

- 返回值：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## aclnnRound

- 参数说明：

  | 参数名        | 输入/输出 | 描述                                                         |
  | ------------- | --------- | ------------------------------------------------------------ |
  | workspace     | 输入      | 在Device侧申请的workspace内存地址。                          |
  | workspaceSize | 输入      | 在Device侧申请的workspace大小，由第一段接口aclnnSWhereGetWorkspaceSize获取。 |
  | executor      | 输入      | op执行器，包含了算子计算流程。                               |
  | stream        | 输入      | 指定执行任务的Stream。                                       |

- 返回值：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

无

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

```c++
/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <iostream>
 #include <vector>
 #include "acl/acl.h"
 #include "aclnn_round.h"
 
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
 
 void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
 {
     auto size = GetShapeSize(shape);
     std::vector<float> resultData(size, 0);
     auto ret = aclrtMemcpy(
         resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
         ACL_MEMCPY_DEVICE_TO_HOST);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
     for (int64_t i = 0; i < size; i++) {
         LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
     }
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
     // 2. 申请device侧内存
     auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
     // 3. 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
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
     // 1. 调用acl进行device/stream初始化
     int32_t deviceId = 0;
     aclrtStream stream;
     auto ret = Init(deviceId, &stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
 
     // 2. 构造输入与输出，需要根据API的接口自定义构造
     aclTensor* selfX = nullptr;
     void* selfXDeviceAddr = nullptr;
     // 构建输入数据x的张量形状
     std::vector<int64_t> selfXShape = {2,8};
     //初始化输入的x数据
     std::vector<float> selfXHostData(16, 1); 
     // 对输入x的数据进行挨个赋值
     for (int i = 0; i < 16; i++) {
        selfXHostData[i] = (static_cast<float>(rand()) / RAND_MAX) * 10.0f; 
     }

     // 打印输入数据，根据前置信息，输入数据维度为16，赋值后打印相应的数据信息
     for (int i = 0; i < 16; i++) {
        std::cout << selfXHostData[i] << " ";
        if ((i + 1) % selfXShape.back() == 0) {
            std::cout << std::endl;  // 每行按最后一维换行（2×8 的话每 8 个值换行）
        }
     }
     std::cout << std::endl;
     ret = CreateAclTensor(selfXHostData, selfXShape, &selfXDeviceAddr, aclDataType::ACL_FLOAT, &selfX);
     CHECK_RET(ret == ACL_SUCCESS, return ret);
     aclTensor* out = nullptr;
     void* outDeviceAddr = nullptr;
     std::vector<int64_t> outShape = {2,8}; // 构造输出数据out的张量形状
     std::vector<float> outHostData(16, 1);  // 初始化输出数据
     ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
     CHECK_RET(ret == ACL_SUCCESS, return ret);
 
     // 3. 调用CANN算子库API，需要修改为具体的Api名称
     uint64_t workspaceSize = 0;
     aclOpExecutor* executor;
 
     // 4. 调用aclnnAddExample第一段接口
     ret = aclnnRoundGetWorkspaceSize(selfX,0, out, &workspaceSize, &executor);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
 
     // 根据第一段接口计算出的workspaceSize申请device内存
     void* workspaceAddr = nullptr;
     if (workspaceSize > static_cast<uint64_t>(0)) {
         ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
     }
 
     // 5. 调用aclnnAddExample第二段接口
     ret = aclnnRound(workspaceAddr, workspaceSize, executor, stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExample failed. ERROR: %d\n", ret); return ret);
 
     // 6. （固定写法）同步等待任务执行结束
     ret = aclrtSynchronizeStream(stream);
     CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
 
     // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
     PrintOutResult(outShape, &outDeviceAddr);
 
     // 7. 释放aclTensor，需要根据具体API的接口定义修改
     aclDestroyTensor(selfX);
     aclDestroyTensor(out);
 
     // 8. 释放device资源
     aclrtFree(selfXDeviceAddr);
     aclrtFree(outDeviceAddr);
     if (workspaceSize > static_cast<uint64_t>(0)) {
         aclrtFree(workspaceAddr);
     }
     aclrtDestroyStream(stream);
     aclrtResetDevice(deviceId);
 
     // 9. acl去初始化
     aclFinalize();
 
     return 0;
 }
```

