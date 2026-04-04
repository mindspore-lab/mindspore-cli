# aclnnTransSparse4to2Para

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 算子功能：

  对结构化稀疏的weight矩阵进行压缩预处理，输出压缩后的稀疏矩阵以及对应的索引矩阵。
  假设原始稀疏矩阵的每4个元素中至少有2个零，压缩后的稀疏矩阵是一个在每4个元素中过滤掉2个零的矩阵。矩阵压缩过程成生成索引矩阵，原始矩阵中的每4个元素，将在index矩阵中生成2个2位索引，并按照规则进行编码。

## 函数原型

```Cpp
aclnnStatus aclnnTransSparse4to2Para(
    const int8_t* weight, 
    aclIntArray*  shape, 
    int8_t**      sparseWeight, 
    int64_t**     sparseWeightDims,
    uint64_t*     sparseWeightDimsNum, 
    uint8_t**     index, 
    int64_t**     indexDims, 
    uint64_t*     indexDimsNum)
```

## aclnnTransSparse4to2Para

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1505px"><colgroup>
  <col style="width: 150px">
  <col style="width: 120px">
  <col style="width: 350px">
  <col style="width: 360px">
  <col style="width: 130px">
  <col style="width: 120px">
  <col style="width: 130px">
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
        <td>weight</td>
        <td>输入</td>
        <td>矩阵乘运算中未压缩的稀疏右矩阵。</td>
        <td><ul><li>shape可表达为（n，k）。</li><li>满足每4个元素至少2个零。</li></ul></td>
        <td>INT8</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>shape</td>
        <td>输入</td>
        <td>矩阵乘运算中未压缩的稀疏右矩阵shape。</td>
        <td><ul><li>通过aclCreateIntArray接口创建。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseWeight</td>
        <td>输出</td>
        <td>矩阵乘运算中压缩后的右矩阵。</td>
        <td><ul><li>内存由调用者释放。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseWeightDims</td>
        <td>输出</td>
        <td>矩阵乘运算中压缩后右矩阵StorageShape数组指针首地址。</td>
        <td><ul><li>内存由调用者释放。</li><li>StorageShape可表达为（ceil（k_half / 32），ceil（n / 16），16，32），其中k_half=ceil（k / 8） * 8 / 2。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>sparseWeightDimsNum</td>
        <td>输出</td>
        <td>矩阵乘运算中压缩后右矩阵StorageShape数组维度。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index</td>
        <td>输出</td>
        <td>矩阵乘运算中的压缩后右矩阵对应的索引矩阵。</td>
        <td><ul><li>内存由调用者释放。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>indexDims</td>
        <td>输出</td>
        <td>矩阵乘运算中的压缩后右矩阵对应的索引矩阵StorageShape数组指针首地址。</td>
        <td><ul><li>内存由调用者释放。</li><li>StorageShape可表达为（ceil（k_half / 32），ceil（n / 16），16，8），其中k_half=ceil（k / 8） * 8 / 2。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>indexDimsNum</td>
        <td>输出</td>
        <td>矩阵乘运算中的压缩后右矩阵对应的索引矩阵StorageShape数组维度。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tr>
  </tbody>
  </table>


- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：
    <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
    <col style="width: 291px">
    <col style="width: 135px">
    <col style="width: 723px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的weight是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>weight的shape、format和数据类型不满足要求；数据不满足每4个元素至少2个为零。</td>
    </tr>
  </tbody></table>


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include <stdlib.h>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_sparse4to2quant_matmul_weight_nz.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  #define CREATE_TENSOR(hostData, shape, deviceAddr, dtype, tensor)                                        \
      ret = CreateAclTensor(hostData, shape, &deviceAddr, dtype, &tensor);                                 \
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> tensor##Ptr(tensor, aclDestroyTensor); \
      std::unique_ptr<void, aclError (*)(void*)> deviceAddr##Ptr(xDeviceAddr, aclrtFree);                  \
      CHECK_RET(ret == ACL_SUCCESS, return ret)

  #define CREATE_SPARSE_TENSOR(hostData, weightShape, storageShape, deviceAddr, dataType, tensor)          \
      ret = CreateSparseTensor(hostData, weightShape, storageShape, &deviceAddr, dataType, &tensor);       \
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> tensor##Ptr(tensor, aclDestroyTensor); \
      std::unique_ptr<void, aclError (*)(void*)> deviceAddr##Ptr(xDeviceAddr, aclrtFree);                  \
      CHECK_RET(ret == ACL_SUCCESS, return ret)

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
  int CreateSparseTensor(
      const T* sparseWeightData, const std::vector<int64_t>& viewShape, const std::vector<int64_t>& storageShape,
      void** deviceAddr, aclDataType dataType, aclTensor** tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(storageShape)) * sizeof(T);

      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, sparseWeightData, size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(viewShape.size(), 1);
      for (int64_t i = viewShape.size() - 2; i >= 0; i--) {
          strides[i] = viewShape[i + 1] * strides[i + 1];
      }

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(
          viewShape.data(), viewShape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, storageShape.data(),
          storageShape.size(), *deviceAddr);
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
      if (hostData.size() > 0) {
          ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      }

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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  void GenRandomMask(std::vector<size_t>& masks)
  {
      masks[0] = random() % 4;
      masks[1] = random() % 4;
      while (masks[1] == masks[0]) {
          masks[1] = random() % 4;
      }
  }

  void GenRandomSparseData(std::vector<int8_t>& weightHostData)
  {
      srandom(233U);
      std::vector<size_t> masks(2, 0UL);
      constexpr size_t step = 4UL;
      for (size_t i = 0; i < weightHostData.size(); i += step) {
          GenRandomMask(masks);
          for (auto mask : masks) {
              weightHostData[i + mask] = 0;
          }
      }
  }

  std::vector<int64_t> GenStorageShape(int64_t* dims, uint64_t dimsNum)
  {
      std::vector<int64_t> storageShape;
      for (uint64_t i = 0UL; i < dimsNum; i++) {
          storageShape.push_back(dims[i]);
      }
      return storageShape;
  }

  int aclnnSparse4to2QuantMatmulTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t m = 64L;
      int64_t k = 512L;
      int64_t n = 128L;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {n, k};
      std::vector<int64_t> indexShape = {n, (k + 7) / 8};
      std::vector<int64_t> biasShape = {n};
      std::vector<int64_t> xScaleShape = {m};
      std::vector<int64_t> weightScaleShape = {n};
      std::vector<int64_t> outShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* sparseWeightDeviceAddr = nullptr;
      void* indexDeviceAddr = nullptr;
      void* biasDeviceAddr = nullptr;
      void* xScaleDeviceAddr = nullptr;
      void* weightScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* sparseWeight = nullptr;
      aclTensor* index = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* xScale = nullptr;
      aclTensor* weightScale = nullptr;
      aclTensor* out = nullptr;
      std::vector<int8_t> xHostData(GetShapeSize(xShape), 1);
      std::vector<int8_t> weightHostData(GetShapeSize(weightShape), 1);
      std::vector<uint16_t> biasHostData(GetShapeSize(biasShape), 1); // 实际上是bfloat16半精度方式
      std::vector<float> xScaleHostData(GetShapeSize(xScaleShape), 1);
      std::vector<float> weightScaleHostData(GetShapeSize(weightScaleShape), 1);
      GenRandomSparseData(weightHostData);

      int8_t* sparseWeightHostData = nullptr;
      uint8_t* indexHostData = nullptr;
      int64_t* sparseWeightDims = nullptr;
      uint64_t sparseWeightDimsNum = 0UL;
      int64_t* indexDims = nullptr;
      uint64_t indexDimsNum = 0UL;
      aclIntArray* weightShapeArray = aclCreateIntArray(weightShape.data(), weightShape.size());
      ret = aclnnTransSparse4to2Para(
          weightHostData.data(), weightShapeArray, &sparseWeightHostData, &sparseWeightDims, &sparseWeightDimsNum,
          &indexHostData, &indexDims, &indexDimsNum);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransSparse4to2Para failed. ERROR: %d\n", ret); return ret);
      std::unique_ptr<int8_t[]> sparseWeightHostDataPtr(sparseWeightHostData);
      std::unique_ptr<uint8_t[]> indexHostDataPtr(indexHostData);
      std::unique_ptr<int64_t[]> sparseWeightDimsPtr(sparseWeightDims);
      std::unique_ptr<int64_t[]> indexDimsPtr(indexDims);

      CREATE_TENSOR(xHostData, xShape, xDeviceAddr, aclDataType::ACL_INT8, x);

      weightShape.back() = (weightShape.back() + 7) / 8 * 8 / 2; // 4选2后 K轴向上8对齐后减半
      auto sparseWeightStorageShape = GenStorageShape(sparseWeightDims, sparseWeightDimsNum);
      CREATE_SPARSE_TENSOR(
          sparseWeightHostData, weightShape, sparseWeightStorageShape, sparseWeightDeviceAddr, aclDataType::ACL_INT8,
          sparseWeight);

      auto indexStorageShape = GenStorageShape(indexDims, indexDimsNum);
      CREATE_SPARSE_TENSOR(indexHostData, indexShape, indexStorageShape, indexDeviceAddr, aclDataType::ACL_UINT8, index);
      CREATE_TENSOR(biasHostData, biasShape, biasDeviceAddr, aclDataType::ACL_BF16, bias);
      CREATE_TENSOR(xScaleHostData, xScaleShape, xScaleDeviceAddr, aclDataType::ACL_FLOAT, xScale);
      CREATE_TENSOR(weightScaleHostData, weightScaleShape, weightScaleDeviceAddr, aclDataType::ACL_FLOAT, weightScale);
      CREATE_TENSOR(std::vector<uint16_t>(), outShape, outDeviceAddr, aclDataType::ACL_BF16, out);

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      void* workspaceAddr = nullptr;

      // 调用aclnnSparse4to2QuantMatmul第一段接口
      ret = aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(
          x, sparseWeight, index, xScale, weightScale, bias, out, &workspaceSize, &executor);

      CHECK_RET(
          ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
          return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // 调用aclnnSparse4to2QuantMatmul第二段接口
      ret = aclnnSparse4to2QuantMatmulWeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      // C语言中无法直接打印bf16的数据，需要用uint16读出来，自行通过二进制转成bf16
      std::vector<uint16_t> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnSparse4to2QuantMatmulTest(deviceId, stream);
      CHECK_FREE_RET(
          ret == ACL_SUCCESS, LOG_PRINT("aclnnSparse4to2QuantMatmulTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }

