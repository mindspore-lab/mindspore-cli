# aclnnRoiPoolingWithArgMax

[📄 查看源码](https://gitcode.com/cann/ops-cv/tree/master/objdetect/roi_pooling_with_arg_max)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |



## 功能说明

- 接口功能：对输入特征图按 ROI（感兴趣区域）进行池化，在每个 ROI 内按空间划分为 `pooled_h × pooled_w` 个格子，对每个格子做最大池化，并输出池化结果及最大值在通道内的一维索引（argmax）。

- 计算公式：

  输入特征图 $x$ 的 shape 为 $(N, C, H, W)$，ROI 张量 $\text{rois}$ 的 shape 为 $(\text{num\_rois}, 5)$，每行表示 $(b_n, x_1, y_1, x_2, y_2)$。标量参数为 $s_h$、$s_w$（spatial_scale）以及 $\text{pooled\_h}$、$\text{pooled\_w}$。下标 $n$ 表示 ROI 索引，$c$ 表示通道，$(\text{ph}, \text{pw})$ 表示池化格点。

  - **ROI 映射到特征图**：将 ROI 坐标乘以 spatial_scale 得到特征图上的浮点区间：

    $$
    \tilde{x}_1 = x_1 s_w,\quad \tilde{y}_1 = y_1 s_h,\quad \tilde{x}_2 = (x_2+1)s_w,\quad \tilde{y}_2 = (y_2+1)s_h
    $$

    $$
    W_{\text{roi}} = \tilde{x}_2 - \tilde{x}_1,\qquad H_{\text{roi}} = \tilde{y}_2 - \tilde{y}_1
    $$

    若 $W_{\text{roi}} \le 0$ 或 $H_{\text{roi}} \le 0$，该 ROI 的 $y$ 全为 0，$\text{argmax}$ 全为 -1。

  - **Bin 步长与区间**：每个池化格 (ph, pw) 对应 ROI 内一个 bin，步长与浮点区间为：

    $$
    \Delta w = \frac{W_{\text{roi}}}{\text{pooled\_w}},\qquad \Delta h = \frac{H_{\text{roi}}}{\text{pooled\_h}}
    $$

    $$
    \tilde{w}_1 = \text{pw} \cdot \Delta w + \tilde{x}_1,\quad \tilde{w}_2 = (\text{pw}+1) \cdot \Delta w + \tilde{x}_1
    $$

    $$
    \tilde{h}_1 = \text{ph} \cdot \Delta h + \tilde{y}_1,\quad \tilde{h}_2 = (\text{ph}+1) \cdot \Delta h + \tilde{y}_1
    $$

    取整并裁剪到 $[0,W) \times [0,H)$：

    $$
    w_1 = \text{clip}(\lfloor\tilde{w}_1\rfloor,\, 0,\, W),\quad w_2 = \text{clip}(\lceil\tilde{w}_2\rceil,\, 0,\, W)
    $$

    $$
    h_1 = \text{clip}(\lfloor\tilde{h}_1\rfloor,\, 0,\, H),\quad h_2 = \text{clip}(\lceil\tilde{h}_2\rceil,\, 0,\, H)
    $$

    其中 $\text{clip}(a,l,u) = \min(\max(a,l), u)$。若 $w_2 \le w_1$ 或 $h_2 \le h_1$，该 bin 为空：$y=0$，$\text{argmax}=-1$。

  - **池化输出与 Argmax**：记 $b = \text{rois}[n,0]$，bin 区域 $R = \{(h,w) : h_1 \le h < h_2,\, w_1 \le w < w_2\}$，则

    $$
    y[n,c,\text{ph},\text{pw}] = \max_{(h,w) \in R} x[b,c,h,w]
    $$

    （空 $R$ 时为 0。）

    $$
    \text{argmax}[n,c,\text{ph},\text{pw}] = h^* W + w^*
    $$

    $(h^*, w^*)$ 为 bin 内最大值位置（多解取第一个）；空 $R$ 为 -1。

  - **输出 Shape**：

    | 输出 | Shape | 数据类型 |
    |------|--------|----------|
    | $y$ | $(\text{num\_rois},\, C,\, \text{pooled\_h},\, \text{pooled\_w})$ | 与 $x$ 一致 |
    | $\text{argmax}$ | 同上 | INT32 |

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRoiPoolingWithArgMaxGetWorkspaceSize”接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用“aclnnRoiPoolingWithArgMax”接口执行计算。

```Cpp
aclnnStatus aclnnRoiPoolingWithArgMaxGetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *rois,
  int64_t            pooled_h,
  int64_t            pooled_w,
  float              spatial_scale_h,
  float              spatial_scale_w,
  aclTensor         *y,
  aclTensor         *argmax,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```Cpp
aclnnStatus aclnnRoiPoolingWithArgMax(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRoiPoolingWithArgMaxGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 311px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>输入特征图，格式为 NCHW，（N，C，H，W）。</td>
      <td><ul><li>不支持空 Tensor。</li><li>输入维度必须为 4 维。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rois（aclTensor*）</td>
      <td>输入</td>
      <td>ROI 框，每行 5 个元素：batch_idx, x1, y1, x2, y2。</td>
      <td>shape 为（num_rois，5），不支持空 Tensor。</td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>pooled_h（int64_t）</td>
      <td>输入</td>
      <td>池化输出高度。</td>
      <td>必须大于 0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pooled_w（int64_t）</td>
      <td>输入</td>
      <td>池化输出宽度。</td>
      <td>必须大于 0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_h（float）</td>
      <td>输入</td>
      <td>ROI 坐标映射到特征图时在高度方向的缩放比例。</td>
      <td>必须大于 0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>spatial_scale_w（float）</td>
      <td>输入</td>
      <td>ROI 坐标映射到特征图时在宽度方向的缩放比例。</td>
      <td>必须大于 0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>池化结果，shape 为（num_rois，C，pooled_h，pooled_w）。</td>
      <td><ul><li>不支持空 Tensor。</li><li>数据类型与 x 一致。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>argmax（aclTensor*）</td>
      <td>输出</td>
      <td>每个池化格点最大值在通道内的线性偏移索引。</td>
      <td><ul><li>不支持空 Tensor。</li><li>shape 与 y 一致。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

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
      <td>如果传入参数是必选输入、输出或必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
    </tr>
    <tr>
      <td>x、rois、y、argmax 的数据类型或格式不在支持范围内。</td>
    </tr>
    <tr><td>x 的 shape 不是 4 维（NCHW）。</td>
    </tr>
    <tr><td>rois 的 shape 第二维不是 5。</td>
    </tr>
    <tr><td>pooled_h、pooled_w、spatial_scale_h、spatial_scale_w 不大于 0。</td>
    </tr>
  </tbody></table>

## aclnnRoiPoolingWithArgMax

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
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnRoiPoolingWithArgMaxGetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnRoiPoolingWithArgMax 默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。实际调用时需先通过 opgen 生成 `aclnnop/aclnn_roi_pooling_with_arg_max.h`，若生成的头文件或接口签名不同，请以生成接口为准。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_roi_pooling_with_arg_max.h"

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
  int64_t shape_size = 1;
  for (auto i : shape) {
    shape_size *= i;
  }
  return shape_size;
}

int Init(int32_t deviceId, aclrtStream* stream) {
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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorOutput(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                          aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> xShape = {2, 16, 25, 42};
  std::vector<int64_t> roisShape = {2, 5};
  std::vector<int64_t> yShape = {2, 16, 3, 3};
  std::vector<int64_t> argmaxShape = {2, 16, 3, 3};

  void* xDeviceAddr = nullptr;
  void* roisDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* argmaxDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* rois = nullptr;
  aclTensor* y = nullptr;
  aclTensor* argmax = nullptr;

  int64_t xSize = GetShapeSize(xShape);
  int64_t roisSize = GetShapeSize(roisShape);
  std::vector<float> xHostData(xSize, 1.0f);
  std::vector<float> roisHostData(roisSize, 0.0f);
  roisHostData[0] = 0.0f;
  roisHostData[1] = 0.0f;
  roisHostData[2] = 0.0f;
  roisHostData[3] = 24.0f;
  roisHostData[4] = 41.0f;
  roisHostData[5] = 1.0f;
  roisHostData[6] = 0.0f;
  roisHostData[7] = 0.0f;
  roisHostData[8] = 24.0f;
  roisHostData[9] = 41.0f;

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(roisHostData, roisShape, &roisDeviceAddr, aclDataType::ACL_FLOAT, &rois);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorOutput<float>(yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensorOutput<int32_t>(argmaxShape, &argmaxDeviceAddr, aclDataType::ACL_INT32, &argmax);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t pooledH = 3;
  int64_t pooledW = 3;
  float spatialScaleH = 1.0f;
  float spatialScaleW = 1.0f;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnRoiPoolingWithArgMaxGetWorkspaceSize(x, rois, pooledH, pooledW, spatialScaleH, spatialScaleW,
                                                  y, argmax, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingWithArgMaxGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnRoiPoolingWithArgMax(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRoiPoolingWithArgMax failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  int64_t yElem = GetShapeSize(yShape);
  std::vector<float> yResult(yElem, 0.0f);
  ret = aclrtMemcpy(yResult.data(), yElem * sizeof(float), yDeviceAddr, yElem * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);

  int64_t argmaxElem = GetShapeSize(argmaxShape);
  std::vector<int32_t> argmaxResult(argmaxElem, 0);
  ret = aclrtMemcpy(argmaxResult.data(), argmaxElem * sizeof(int32_t), argmaxDeviceAddr,
                    argmaxElem * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy argmax from device to host failed. ERROR: %d\n", ret); return ret);

  aclDestroyTensor(x);
  aclDestroyTensor(rois);
  aclDestroyTensor(y);
  aclDestroyTensor(argmax);
  aclrtFree(xDeviceAddr);
  aclrtFree(roisDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(argmaxDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
