# aclnnMatmulCompressDequant

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/conversion/matmul_v2_compress_dequant)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾950 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 接口功能：进行l@r矩阵乘计算时，可先通过msModelSlim工具对r矩阵进行无损压缩，减少r矩阵的内存占用大小，然后通过本接口完成无损解压缩、矩阵乘、反量化计算。
- 计算公式：

  $$
  x2\_unzip = unzip(x2, compressIndex, compressInfo)\\
  result=(x1 @ x2\_unzip + bias)*deqScale
  $$

  其中x2表示r矩阵经过msModelSlim工具进行压缩后的一维数据，compressIndex以及compressInfo表示压缩算法相关的信息，$x2\_unzip$是本接口内部进行无损解压缩后的数据（与原始r矩阵数据一致），压缩和调用本接口的详细使用样例参考[调用示例](#调用示例)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMatmulCompressDequantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMatmulCompressDequant”接口执行计算。

```cpp
aclnnStatus aclnnMatmulCompressDequantGetWorkspaceSize(
  const aclTensor*   x1, 
  const aclTensor*   x2, 
  const aclTensor*   compressIndex, 
  const aclTensor*   bias, 
  const aclTensor*   deqScale, 
  const aclTensor*   offsetW, 
  int                offsetX, 
  const aclIntArray* compressInfo, 
  aclTensor*         out, 
  uint64_t*          workspaceSize, 
  aclOpExecutor**    executor)
```
```cpp
aclnnStatus aclnnMatmulCompressDequant(
  void*           workspace, 
  uint64_t        workspaceSize, 
  aclOpExecutor*  executor, 
  aclrtStream     stream)
```

## aclnnMatmulCompressDequantGetWorkspaceSize

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1475px"><colgroup>
  <col style="width: 167px">
  <col style="width: 123px">
  <col style="width: 325px">
  <col style="width: 230px">
  <col style="width: 128px">
  <col style="width: 118px">
  <col style="width: 239px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>表示矩阵乘的左输入，2维ND格式aclTensor。</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示压缩后的矩阵乘的右输入，1维ND格式aclTensor，为通过msModelSlim工具中weight_compression模块压缩后的输入。</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressIndex</td>
      <td>输入</td>
      <td>表示矩阵乘右输入的压缩索引表，1维ND格式aclTensor。</td>
      <td>通过示例中的msModelSlim工具中获取</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>2维ND格式aclTensor。</td>
      <td>支持空指针传入。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>2维，shape仅支持(1, n)或者(n)，其中n为输出shape(m, n)的n</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deqScale</td>
      <td>输入</td>
      <td>表示反量化参数，2维ND格式aclTensor。</td>
      <td>tensor中的值为float通过下述示例中转换后的UINT64的数据。</td>
      <td>UINT64</td>
      <td>ND</td>
      <td>2维，shape支持(1, n)或者(1, 1), 其中n为输出shape(m, n)中的n。</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetW</td>
      <td>输入</td>
      <td>标量，表示矩阵乘右输入的偏移量。</td>
      <td>当前仅支持空指针传入。</td>
      <td>INT8</td>
      <td>-</td>
      <td>与x2_unzip一致。</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetX</td>
      <td>输入</td>
      <td>标量，表示矩阵乘左输入的偏移量。</td>
      <td>当前仅支持0。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressInfo</td>
      <td>输入</td>
      <td>整型数据列表，数据类型为INT64。其中包括压缩块信息tilingN、tilingK（通过msModelSlim工具中weight_compression模块压缩后获取，分别表示压缩前shape(n, k)在n方向和k方向上一个基本压缩块的大小），压缩前x2矩阵原始shape（shape为2维，用(n, k)表示），以及压缩块遍历方向的标识。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输入</td>
      <td>2维ND格式aclTensor。</td>
      <td>-</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>


- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1475px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 975px">
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
      <td>传入的x1、x2或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>x1或x2的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x1或x2无法做数据类型推导。</td>
    </tr>
    <tr>
      <td>推导出的数据类型无法转换为指定输出out的类型。</td>
    </tr>
  </tbody>
  </table>

## aclnnMatmulCompressDequant

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1475px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 975px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMatmulCompressDequantGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMatmulCompressDequant默认确定性实现。


## 调用示例

- **准备压缩前的数据**
  
  假设通过脚本gen_data.py生成输入数据，示例如下，仅供参考：
  
  ```python
  import numpy as np
  import os
  import sys
  from numpy import random

  def write2file(data, path):
    with open(path, 'wb') as f:
        data.tofile(f)

  if not os.path.exists("./data"):
      os.mkdir("./data")

  if len(sys.argv) != 4:
    print("Usage: python gen_data.py m k n")
    sys.exit(1)

  m = int(sys.argv[1])
  k = int(sys.argv[2])
  n = int(sys.argv[3])

  if m <= 0 or k <= 0 or n <= 0:
    print("Error: m, k and n must be positive integers.")
    sys.exit(1)

  # 随机生成矩阵mat1，shape为(m,k )
  mat1 = random.randn(m, k).astype(np.int8)
  write2file(mat1, "./data/mat1.bin")

  # 随机生成矩阵mat2，shape为(n, k)
  mat2 = random.randint(0, 100, size=(n, k)).astype(np.int8)
  np.save("./data/weight.npy", {'weight': mat2})
  os.chmod("./data/weight.npy", 0o0640)

  # 生成output
  output = np.random.randn(m, n).astype(np.float16)
  write2file(output, "./data/output.bin")

  # 生成bias
  bias = random.randn(n).astype(np.float32)
  write2file(bias, "./data/bias.bin")

  # 生成deq_scale
  deq_scale = random.randn(n).astype(np.float32)
  write2file(deq_scale, "./data/deqScale_ori.bin")
  deq_scale_int64 = np.fromfile("./data/deqScale_ori.bin", dtype=np.int32).astype(np.int64)
  deq_scale_int64.tofile("./data/deqScale.bin")
  ```

  执行gen_data.py，假设mat1和mat2的shape入参为m=512、k=1024、n=1024。
  
  ```shell
  python3 gen_data.py 512 1024 1024
  ```

- **对数据进行预处理**

  - **原始权重通过msModelSlim压缩工具生成压缩后的x2、compressIndex以及compressInfo**
  
    使用以下接口时，需对CANN包中msModelSlim压缩工具进行编译，具体操作参考[Gitee msit仓](https://gitee.com/ascend/msit/tree/master/msmodelslim)中msmodelslim/pytorch/weight_compression目录下的README.md。

    ```python
    from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

    compress_config = CompressConfig(do_pseudo_sparse=False, sparse_ratio=1)
    compressor = Compressor(compress_config, weight_path=weight_path)

    compress_weight, compress_index, compress_info = compressor.run()
    # 压缩后的权重，对应aclnnMatmulCompressDequantGetWorkspaceSize接口的x2
    compressor.export(compress_weight, './data/weight')
    # 压缩权重的索引，对应aclnnMatmulCompressDequantGetWorkspaceSize接口的compressIndex
    compressor.export(compress_index, './data/index')
    # 压缩数据的相关信息，对应aclnnMatmulCompressDequantGetWorkspaceSize接口的compressInfo
    compressor.export(compress_info, './data/compress_info')
    ```
  - **将原始float类型的反量化参数deqscale进行转换， 得到aclnn接口需要的uint64数据**

    deqScale原始为float类型，以int32读取并转换为int64

    ```python
    import numpy as np
    data = np.fromfile('./deqScale_original.bin', dtype=np.int32).astype(np.int64)
    data.tofile('./deqScale.bin')
    ```

- **调用aclnn接口运算**
  
  示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <acl/acl.h>
#include <aclnnop/aclnn_matmul_compress_dequant.h>
#include <fstream>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>
#include <cstdlib>
#include <string>

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

int ReadBinFileNNop(std::string filePath, void* buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    CHECK_RET(fileStatus == ACL_SUCCESS, LOG_PRINT("Failed to get file %s\n", filePath); return -1);

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    CHECK_RET(file.is_open(), LOG_PRINT("Open file failed.\n"); return -1);

    file.seekg(0, file.end);
    uint64_t binFileBufferLen = file.tellg();
    CHECK_RET(binFileBufferLen > 0,
        std::cout<<"File size is 0.\n";
        file.close();
        return -1);

    file.seekg(0, file.beg);
    file.read(static_cast<char *>(buffer), binFileBufferLen);
    file.close();
    return ACL_SUCCESS;
}

int CreateAclTensor(std::string filePath, const std::vector<int64_t>& shape, int typeSize,
                    void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * typeSize;
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMallocHost申请host侧内存
  void* binBufferHost = nullptr;
  ret = aclrtMallocHost(&binBufferHost, size);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMallocHost failed. ERROR: %d\n", ret); return ret);

  // 读取文件
  ret = ReadBinFileNNop(filePath, binBufferHost, size);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("ReadBinFileNNop failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, binBufferHost, size, ACL_MEMCPY_HOST_TO_DEVICE);
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

int main(int argc, char* argv[]) {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  if (argc != 6) {
    std::cerr << "Error: Invalid number of arguments. Usage: <program> m k n wCompressedSize indexSize" << std::endl;
    return -1;
  }

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  int m = atoi(argv[1]);
  int k = atoi(argv[2]);
  int n = atoi(argv[3]);
  // wShape是右矩阵压缩后数据的大小
  int wCompressedSize = atoi(argv[4]);
  // indexShape是压缩索引数据的大小
  int indexSize = atoi(argv[5]);

  if (m <= 0 || k <= 0 || n <= 0 || wCompressedSize <= 0 || indexSize <= 0) {
    std::cerr << "Error: m, k, n, wCompressedSize and indexSize must be positive integers." << std::endl;
    return -1;
  }

  std::vector<int64_t> mat1Shape = {m, k};
  std::vector<int64_t> mat2CompressedShape = {wCompressedSize};
  std::vector<int64_t> indexShape = {indexSize};
  std::vector<int64_t> biasShape = {n};
  std::vector<int64_t> deqScaleShape = {n};
  std::vector<int64_t> outputShape = {m, n};

  std::vector<int64_t> compressInfoHostData = {8, 8, k, n, 1};

  void* mat1DeviceAddr = nullptr;
  void* mat2CompressedDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;
  void* deqScaleDeviceAddr = nullptr;
  void* outputDeviceAddr = nullptr;

  aclTensor* mat1 = nullptr;
  aclTensor* mat2Compressed = nullptr;
  aclTensor* index = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* deqScale = nullptr;
  aclTensor* output = nullptr;
  aclIntArray* compressInfo = nullptr;

  std::string rootPath = "./data/";

  // 创建mat1 aclTensor
  std::string mat1FilePath = rootPath + "mat1.bin";
  ret = CreateAclTensor(mat1FilePath, mat1Shape, sizeof(int8_t), &mat1DeviceAddr, aclDataType::ACL_INT8, &mat1);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create mat1 tensor failed. ERROR: %d\n", ret); return ret);
  // 创建mat2Compressed aclTensor
  std::string mat2FilePath = rootPath + "weight/weight.dat";
  ret = CreateAclTensor(mat2FilePath, mat2CompressedShape, sizeof(int8_t), &mat2CompressedDeviceAddr,
                        aclDataType::ACL_INT8, &mat2Compressed);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create mat2 tensor failed. ERROR: %d\n", ret); return ret);
  // 创建index aclTensor
  std::string indexFilePath = rootPath + "index/weight.dat";
  ret = CreateAclTensor(indexFilePath, indexShape, sizeof(int8_t), &indexDeviceAddr, aclDataType::ACL_INT8, &index);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create index tensor failed. ERROR: %d\n", ret); return ret);
  // 创建bias aclTensor
  std::string biasFilePath = rootPath + "bias.bin";
  ret = CreateAclTensor(biasFilePath, biasShape, sizeof(int32_t), &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create bias tensor failed. ERROR: %d\n", ret); return ret);
  // 创建deqScale aclTensor
  std::string deqScaleFilePath = rootPath + "deqScale.bin";
  ret = CreateAclTensor(deqScaleFilePath, deqScaleShape, sizeof(int32_t), &deqScaleDeviceAddr, aclDataType::ACL_UINT64,
                        &deqScale);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create deqScale tensor failed. ERROR: %d\n", ret); return ret);
  // 创建compressInfo
  compressInfo = aclCreateIntArray(compressInfoHostData.data(), aclDataType::ACL_INT64);
  // 创建out aclTensor
  std::string outputFilePath = rootPath + "output.bin";
  ret = CreateAclTensor(outputFilePath, outputShape, 2, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &output);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create output tensor failed. ERROR: %d\n", ret); return ret);

  int32_t offsetX = 0;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMm第一段接口
  ret = aclnnMatmulCompressDequantGetWorkspaceSize(mat1, mat2Compressed, index, bias, deqScale, nullptr, offsetX, compressInfo, output, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulCompressDequantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMm第二段接口
  ret = aclnnMatmulCompressDequant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulCompressDequant failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outputShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(mat1);
  aclDestroyTensor(mat2Compressed);
  aclDestroyTensor(index);
  aclDestroyTensor(bias);
  aclDestroyTensor(deqScale);
  aclDestroyTensor(output);
  aclDestroyIntArray(compressInfo);

  // 7.释放硬件资源，需要根据具体API的接口定义修改
  aclrtFree(mat1DeviceAddr);
  aclrtFree(mat2CompressedDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(deqScaleDeviceAddr);
  aclrtFree(outputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
