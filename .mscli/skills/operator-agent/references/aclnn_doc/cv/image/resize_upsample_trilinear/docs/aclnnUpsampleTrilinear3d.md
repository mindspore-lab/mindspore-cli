# aclnnUpsampleTrilinear3d

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/image/resize_upsample_trilinear)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |


## 功能说明

- 接口功能：对由多个输入通道组成的输入信号应用三线性插值算法进行上采样。
- 计算公式：
  - 核心算法逻辑：
    1. 将目标图像缩放到和原始图像一样大的尺寸。
    2. 计算缩放之后的目标图像的点，以及前后相邻的原始图像的点。
    3. 分别计算相邻点到对应目标点的权重，按照权重相乘累加即可得到目标点值。
  - 具体计算逻辑：
    缩放方式分为角对齐和边对齐，角对齐表示按照原始图片左上角像素中心点对齐，边对齐表示按照原始图片左上角顶点及两条边对齐，在计算缩放系数和坐标位置时存在差异。对于一个三维插值点$(N, C, D, H, W)$，则有以下公式：

    $$
    scale\_d =\begin{cases}
    (self.dim[2]-1) / (outputSize[0]-1) & alignCorners=true \\
    1 / scales\_d & alignCorners=false\&scales\_d>0\\
    self.dim[2] / outputSize[0] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_h =\begin{cases}
    (self.dim[3]-1) / (outputSize[1]-1) & alignCorners=true \\
    1 / scales\_h & alignCorners=false\&scales\_h>0\\
    self.dim[3] / outputSize[1] & alignCorners=false
    \end{cases}
    $$

    $$
    scale\_w =\begin{cases}
    (self.dim[4]-1) / (outputSize[2]-1) & alignCorners=true \\
    1 / scales\_w & alignCorners=false\&scales\_w>0\\
    self.dim[4] / outputSize[2] & alignCorners=false
    \end{cases}
    $$

    因此，对于output的某个方向上的点p(x,y,z)，映射回原始图像中的点记为q(x',y',z')，则有关系：

    $$
    x' =\begin{cases}
    x * scale\_d & alignCorners=true \\
    MAX(0,{(x+0.5)*scale\_d-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    y' =\begin{cases}
    y * scale\_h & alignCorners=true \\
    MAX(0,{(y+0.5)*scale\_h-0.5}) & alignCorners=false
    \end{cases}
    $$

    $$
    z' =\begin{cases}
    z * scale\_w & alignCorners=true \\
    MAX(0,{(z+0.5)*scale\_w-0.5}) & alignCorners=false
    \end{cases}
    $$

    - 记：

      $$
      x_{0} =int(x'),x_{1} =int(x')+1, lambda_{0} = x_{1}-x', lambda_{1} =   1-lambda_{0}
      $$

      $$
      y_{0} =int(y'),y_{1} =int(y')+1, lambdb_{0} = y_{1}-y', lambdb_{1} =   1-lambdb_{0}
      $$

      $$
      z_{0} =int(z'),z_{1} =int(z')+1, lambdc_{0} = z_{1}-z', lambdc_{1} =   1-lambdc_{0}
      $$

    - 则有以下公式：

      $$
      {V(p_{x, y, z})} = {V(p_{x0, y0, z0})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x0, y0, z1})} * {lambda_{0}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x0, y1, z0})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x0, y1, z1})} * {lambda_{0}} * {lambdb_{1}} * {lambdc_{1}} + {V(p_{x1, y0, z0})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{0}} + {V(p_{x1, y0, z1})} * {lambda_{1}} * {lambdb_{0}} * {lambdc_{1}} + {V(p_{x1, y1, z0})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{0}} + {V(p_{x1, y1, z1})} * {lambda_{1}} * {lambdb_{1}} * {lambdc_{1}}
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnUpsampleTrilinear3dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnUpsampleTrilinear3d”接口执行计算。

```Cpp
aclnnStatus aclnnUpsampleTrilinear3dGetWorkspaceSize(
  const aclTensor   *self, 
  const aclIntArray *outputSize, 
  bool               alignCorners, 
  double             scalesD, 
  double             scalesH, 
  double             scalesW, 
  aclTensor         *out,  
  uint64_t          *workspaceSize, 
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnUpsampleTrilinear3d(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnUpsampleTrilinear3dGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
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
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>表示进行上采样的输入张量，对应公式中的`self`。</td>
      <td><ul><li>不支持空Tensor。</li><li>当数据格式为ND时，默认按照NCDHW格式处理。</li><li>shape的C轴、D轴、H轴、W轴的取值大于0。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>NCDHW、NDHWC、ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputSize（aclIntArray*）</td>
      <td>输入</td>
      <td>表示出参`out`在D、H和W维度上的空间大小，对应公式中的`outputSize`。</td>
      <td>size为3，且各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alignCorners（bool）</td>
      <td>输入</td>
      <td>表示是否对齐角像素点，对应公式中的`alignCorners`。</td>
      <td>如果为true，则输入和输出张量的角像素点会被对齐，否则不对齐。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesD（double）</td>
      <td>输入</td>
      <td>表示输出`out`的depth维度乘数，对应公式中的`scales_d`。</td>
      <td>取值小于等于50。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesH（double）</td>
      <td>输入</td>
      <td>表示输出`out`的height维度乘数，对应公式中的`scales_h`。</td>
      <td>取值小于等于50。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scalesW（double）</td>
      <td>输入</td>
      <td>表示输出`out`的width维度乘数，对应公式中的`scales_w`。</td>
      <td>取值小于等于50。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示采样后的输出张量。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型和数据格式与入参`self`保持一致。</li><li>shape的N轴、C轴与入参`self`保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>NCDHW、NDHWC、ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
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

  - <term>Atlas 推理系列产品</term>：
    - 入参`self`的数据类型仅支持FLOAT32、FLOAT16，不支持inf、-inf输入。
    - 出参`out`的数据类型仅支持FLOAT32、FLOAT16。
  - <term>Atlas 训练系列产品</term>：

    入参`self`和出参`out`的数据类型支持FLOAT32、FLOAT16、DOUBLE。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>传入的self 、outputSize或out是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>self的数据类型不在支持的范围之内或self与out数据类型不同。</td>
    </tr>
    <tr>
      <td>self的shape不是5维。</td>
    </tr>
    <tr>
      <td>outputSize的size不等于3。</td>
    </tr>
    <tr>
      <td>self在C、D、H、W维度上的size不大于0。</td>
    </tr>
    <tr>
      <td>outputSize的某个元素值不大于0。</td>
    </tr>
  </tbody></table>

## aclnnUpsampleTrilinear3d

- **参数说明：**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnUpsampleTrilinear3dGetWorkspaceSize获取。</td>
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

- 输入数据缩放场景缩小倍数必须小于等于50，即：

  $$
  输出shape的深度D/outputSize\_D <= 50
  $$
  
  $$
  输出shape的高度H/outputSize\_H <= 50
  $$
  
  $$
  输出shape的宽度W/outputSize\_W <=50
  $$

- 参数`self`、`out`的shape约束：
  - 每个维度的取值小于等于2^20。
  - 参数`out`的N轴和C轴与`self`保持一致。
  - 内存占用需小于60G。内存占用的计算公式如下：

    $$
    N * C * (self\_D * self\_H * self\_W + out\_D * out\_H * out\_W + self\_D * self\_H * out\_W + self\_D * out\_H * out\_W) * sizeof(float) < 60 * 1024 * 1024 * 1024
    $$

    其中：
    - N代表输入和输出的N轴。
    - C代表输入和输出的C轴。
  - N \* C \* self_D \* self_H < 2^31
  - out_W * out_H < 2^31
- 参数self、outputSize、scalesD、scalesH、scalesW需要满足如下约束：

  $$
  outputSize\_D = floor(self\_D * scalesD)
  $$

  $$
  outputSize\_H = floor(self\_H * scalesH)
  $$

  $$
  outputSize\_W = floor(self\_W * scalesW)
  $$

- 确定性计算：
  - aclnnUpsampleTrilinear3d默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_upsample_trilinear_3d.h"

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

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_NCDHW,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {1, 1, 2, 2, 2};
    std::vector<int64_t> outShape = {1, 1, 4, 4, 4};
    void *selfDeviceAddr = nullptr;
    void *outDeviceAddr = nullptr;
    aclTensor *self = nullptr;
    aclTensor *out = nullptr;
    std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    const size_t kSize64 = 64U;
    std::vector<float> outHostData(kSize64, 0);
    std::vector<int64_t> outputSizeData = {4, 4, 4};
    bool alignCorners = false;
    double scalesD = 0.0;
    double scalesH = 0.0;
    double scalesW = 0.0;
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    const aclIntArray *outputSize = aclCreateIntArray(outputSizeData.data(), outputSizeData.size());
    CHECK_RET(outputSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnUpsampleTrilinear3d第一段接口
    ret = aclnnUpsampleTrilinear3dGetWorkspaceSize(
        self, outputSize, alignCorners, scalesD, scalesH, scalesW, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleTrilinear3dGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnUpsampleTrilinear3d第二段接口
    ret = aclnnUpsampleTrilinear3d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnUpsampleTrilinear3d failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(),
        resultData.size() * sizeof(resultData[0]),
        outDeviceAddr,
        size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
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
