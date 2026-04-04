# aclnnMlaPrologV3WeightNz

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/mla_prolog_v3)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |

## 功能说明

-  **功能更新**：（相对于aclnnMlaPrologV2weightNz的差异）
    -  新增Query与Key的尺度矫正因子，分别对应qcQrScale（$\alpha_q$）与kcScale（$\alpha_{kv}$）。
    -  新增可选输入参数（例如actualSeqLenOptional、kNopeClipAlphaOptional、queryNormFlag、weightQuantMode、kvCacheQuantMode、queryQuantMode、ckvkrRepoMode、quantScaleRepoMode、tileSize、queryNormOutOptional和dequantScaleQNormOptional等），将cache_mode由必选改为可选。
    -  调整cacheIndex参数的名称与位置，对应当前的cacheIndexOptional。
-  **接口功能**：推理场景，Multi-Head Latent Attention前处理的计算。主要计算过程分为五路:
    -  首先对输入$x$乘以$W^{DQ}$进行下采样和RmsNorm后分为两路，第一路乘以$W^{UQ}$和$W^{UK}$经过两次上采样后，再乘以Query尺度矫正因子$\alpha_q$得到$q^N$；第二路乘以$W^{QR}$后经过旋转位置编码（ROPE）得到$q^R$。
    -  第三路是输入$x$乘以$W^{DKV}$进行下采样和RmsNorm后，乘以Key尺度矫正因子$\alpha_{kv}$传入Cache中得到$k^C$；
    -  第四路是输入$x$乘以$W^{KR}$后经过旋转位置编码后传入另一个Cache中得到$k^R$；
    -  第五路是输出$q^N$经过DynamicQuant后得到的量化参数。
    -  权重参数WeightDq、WeightUqQr和WeightDkvKr需要以NZ格式传入

-  **计算公式**：

    RmsNorm公式

    $$
    \text{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\text{RMS}(x)}
    $$

    $$
    \text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
    $$

    Query的计算公式，包括下采样、RmsNorm和两次上采样

    $$
    c^Q = \alpha_q\cdot\mathrm{RmsNorm}(x \cdot W^{DQ})
    $$

    $$
    q^C = c^Q \cdot W^{UQ}
    $$

    $$
    q^N = q^C \cdot W^{UK}
    $$

    对Query进行ROPE旋转位置编码

    $$
    q^R = \mathrm{ROPE}(c^Q \cdot W^{QR})
    $$

    Key的计算公式，包括下采样和RmsNorm，将计算结果存入cache

    $$
    c^{KV} = \alpha_{kv}\cdot\mathrm{RmsNorm}(x \cdot W^{DKV})
    $$

    $$
    k^C = \mathrm{Cache}(c^{KV})
    $$

    对Key进行ROPE旋转位置编码，并将结果存入cache

    $$
    k^R = \mathrm{Cache}(\mathrm{ROPE}(x \cdot W^{KR}))
    $$

    Dequant Scale Query Nope 计算公式

    $$
    \mathrm{dequantScaleQNope} = {\mathrm{RowMax}(\mathrm{abs}(q^{N})) / 127}
    $$

    $$
    q^{N} = {\mathrm{round}(q^{N} / \mathrm{dequantScaleQNope})}
    $$


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMlaPrologV3WeightNzGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnMlaPrologV3WeightNz”接口执行计算。

```cpp
aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(
    const aclTensor *tokenX,
    const aclTensor *weightDq,
    const aclTensor *weightUqQr,
    const aclTensor *weightUk,
    const aclTensor *weightDkvKr,
    const aclTensor *rmsnormGammaCq,
    const aclTensor *rmsnormGammaCkv,
    const aclTensor *ropeSin,
    const aclTensor *ropeCos,
    aclTensor       *kvCacheRef,
    aclTensor       *krCacheRef,
    const aclTensor *cacheIndexOptional,
    const aclTensor *dequantScaleXOptional,
    const aclTensor *dequantScaleWDqOptional,
    const aclTensor *dequantScaleWUqQrOptional,
    const aclTensor *dequantScaleWDkvKrOptional,
    const aclTensor *quantScaleCkvOptional,
    const aclTensor *quantScaleCkrOptional,
    const aclTensor *smoothScalesCqOptional,
    const aclTensor *actualSeqLenOptional,
    const aclTensor *kNopeClipAlphaOptional,
    double           rmsnormEpsilonCq,
    double           rmsnormEpsilonCkv,
    char            *cacheModeOptional,
    int64_t          weightQuantMode,
    int64_t          kvCacheQuantMode,
    int64_t          queryQuantMode,
    int64_t          ckvkrRepoMode,
    int64_t          quantScaleRepoMode,
    int64_t          tileSize,
    double           qcQrScale,
    double           kcScale,
    const aclTensor *queryOut,
    const aclTensor *queryRopeOut,
    const aclTensor *dequantScaleQNopeOutOptional,
    const aclTensor *queryNormOutOptional,
    const aclTensor *dequantScaleQNormOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnMlaPrologV3WeightNz(
  void             *workspace,
  uint64_t          workspaceSize,
  aclOpExecutor    *executor,
  const aclrtStream stream)
```


## aclnnMlaPrologV3WeightNzGetWorkspaceSize

- **参数说明**

  | 参数名                     | 输入/输出 | 描述             | 使用说明       | 数据类型       | Ascend 950PR/Ascend 950DT支持数据类型| 数据格式   | 维度(shape)    |非连续Tensor |
  |----------------------------|-----------|----------------------------------------------|----------------|----------------|-|------------|-----------------|-------|
  | tokenX          | 输入      | 公式中用于计算Query和Key的输入tensor。    | - 支持B=0,S=0,T=0的空Tensor   | BFLOAT16、INT8 | BFLOAT16、FLOAT8_E4M3FN | ND    | - BS合轴：(T,He) <br>- BS非合轴：(B,S,He)         |×   |
  | weightDq        | 输入      | 公式中用于计算Query的下采样权重矩阵$W^{DQ}$。<br>在不转置的情况下各个维度的表示：（k，n）| - 不支持空Tensor      | BFLOAT16、INT8 | BFLOAT16、FLOAT8_E4M3FN | FRACTAL_NZ | (He,Hcq)                      |×   |
  | weightUqQr      | 输入      | 公式中用于计算Query的上采样权重矩阵$W^{UQ}$和位置编码权重矩阵$W^{QR}$。<br>在不转置的情况下各个维度的表示：（k，n） | - 不支持空Tensor  | BFLOAT16、INT8 | BFLOAT16、INT8、FLOAT8_E4M3FN| FRACTAL_NZ | (Hcq,N*(D+Dr))                |×   |
  | weightUk        | 输入      | 公式中用于计算Key的上采样权重$W^{UK}$。           | - 不支持空Tensor     | BFLOAT16       | BFLOAT16 | ND         | (N,D,Hckv)                    |×   |
  | weightDkvKr     | 输入      | 公式中用于计算Key的下采样权重矩阵$W^{DKV}$和位置编码权重矩阵$W^{KR}$。<br>在不转置的情况下各个维度的表示：（k，n） | - 不支持空Tensor  | BFLOAT16、INT8 | BFLOAT16、FLOAT8_E4M3FN |  FRACTAL_NZ | (He,Hckv+Dr)                  |×   |
  | rmsnormGammaCq  | 输入      | 计算$c^Q$的RmsNorm公式中的$\gamma$参数。        | - 不支持空Tensor   | BFLOAT16       | BFLOAT16 | ND         | (Hcq)                         |×   |
  | rmsnormGammaCkv | 输入      | 计算$c^{KV}$的RmsNorm公式中的$\gamma$参数。      | - 不支持空Tensor | BFLOAT16       | BFLOAT16 | ND         | (Hckv)                        |×   |
  | ropeSin         | 输入      | 用于计算旋转位置编码的正弦参数矩阵。              | - 支持B=0,S=0,T=0的空Tensor | BFLOAT16       | BFLOAT16| ND         | - BS合轴：(T,Dr) <br>- BS非合轴：(B,S,Dr)         |×   |
  | ropeCos         | 输入      | 用于计算旋转位置编码的余弦参数矩阵。           | - 支持B=0,S=0,T=0的空Tensor  | BFLOAT16       | BFLOAT16| ND         | - BS合轴：(T,Dr) <br>- BS非合轴：(B,S,Dr)         |×   |
  | kvCacheRef      | 输入      | 用于cache索引的aclTensor，计算结果原地更新（对应公式中的$k^C$）。  | - 支持B=0,Skv=0的空Tensor；Nkv与N关联，N是超参，故Nkv不支持等于0  | BFLOAT16、INT8 | BFLOAT16、INT8、FLOAT8_E4M3FN | ND   | - CacheMode="PA_BSND"/"PA_NZ"/"PA_BLK_BSND"/"PA_BLK_NZ": (BlockNum,BlockSize,Nkv,Dtile) <br> - CacheMode="BSND": (B,S,Nkv,Dtile) <br> - CacheMode="TND": (T,Nkv,Dtile) |×   |
  | krCacheRef      | 输入      | 用于key位置编码的cache，计算结果原地更新（对应公式中的$k^R$）。    | - 支持B=0,Skv=0的空Tensor；Nkv与N关联，N是超参，故Nkv不支持等于0 | BFLOAT16、INT8 | BFLOAT16、INT8 | ND         | - CacheMode="PA_BSND"/"PA_NZ"/"PA_BLK_BSND"/"PA_BLK_NZ": (BlockNum,BlockSize,Nkv,Dr) <br> - CacheMode="BSND": (B,S,Nkv,Dr) <br> - CacheMode="TND"时: (T,Nkv,Dr) <br> - 当ckvkrRepoMode=1时: 维度应包含0，支持维度为(0) |×   |
  | cacheIndexOptional | 输入      | 用于存储kvCache和krCache的索引。| - 支持B=0,S=0,T=0的空Tensor <br>- cacheMode="PA_BSND"/"PA_NZ": 取值范围需在[0,BlockNum*BlockSize]内 <br>- cacheMode="PA_BLK_BSND"/"PA_BLK_NZ": 取值范围需在[0,BlockNum]内 <br>- cacheMode="TND"/"BSND": nullptr | INT64   | INT64 | ND  | - CacheMode="PA_BSND"/"PA_NZ": <br>1. BS合轴：(T) <br>2. BS非合轴：(B,S) <br>- CacheMode="PA_BLK_BSND"/"PA_BLK_NZ": <br> 1. BS合轴：(Sum(Ceil(S_i/BlockSize)))，S_i为每个Batch中的S的长度 <br> 2. BS非合轴：(B,Ceil(S/BlockSize)) <br>- CacheMode="TND"/"BSND": nullptr |×   |
  | dequantScaleXOptional      | 输入      | tokenX的反量化参数。 | - 支持B=0,S=0,T=0的空Tensor（weightQuantMode=2/3的场景需传）   | FLOAT          | FLOAT8_E8M0 | ND         | - weightQuantMode=2： <br>1. BS合轴：(T) <br>2. BS非合轴：(B\*S,1) <br>- weightQuantMode=3： <br>  1. BS合轴：(T, He/32) <br>  2. BS非合轴：(B*S, He/32)                               |×   |
  | dequantScaleWDqOptional    | 输入      | weightDq的反量化参数。   | - 支持非空Tensor（weightQuantMode=2/3的场景需传）    | FLOAT          | FLOAT8_E8M0| ND          | - weightQuantMode=2：(1,Hcq) <br> - weightQuantMode=3：(Hcq, He/32)                                 |×   |
  | dequantScaleWUqQrOptional  | 输入      | 用于MatmulQcQr矩阵乘后反量化操作的per-channel参数。 | - 支持非空Tensor（weightQuantMode=1/2/3的场景需传）  | FLOAT          | FLOAT、FLOAT8_E8M0 | ND         | - weightQuantMode=1/2：(1,N*(D+Dr))<br> - weightQuantMode=3： (N*(D+Dr), Hcq/32)     |×   |
  | dequantScaleWDkvKrOptional | 输入      | weightDkvKr的反量化参数。   | - 支持非空Tensor（weightQuantMode=2/3的场景需传）   | FLOAT          | FLOAT8_E8M0| ND         | - weightQuantMode=2：(1,Hckv+Dr) <br> - weightQuantMode=3：(Hckv+Dr, He/32) |×   |
  | quantScaleCkvOptional      | 输入      | 用于对kvCache输出数据做量化操作的参数。 | - 支持非空Tensor（kvCacheQuantMode=1/2的场景需传）  | FLOAT          | FLOAT| ND         | - kvCacheQuantMode=1：(1) <br> - kvCacheQuantMode=2：(1,Hckv) |×   |
  | quantScaleCkrOptional      | 输入      | 用于对krCache输出数据做量化操作的参数。| - 支持非空Tensor（kvCacheQuantMode=2的场景需传）    | FLOAT    | FLOAT | ND   | (1,Dr)     |×   |
  | smoothScalesCqOptional     | 输入      | 用于对RmsNormCq输出做动态量化操作的参数。   | - 支持非空Tensor（weightQuantMode=1/2的场景可选传）| FLOAT  | FLOAT | ND | (1,Hcq)                       |×   |
  | actualSeqLenOptional     | 输入      | 表示每个batch中的序列长度，以前缀和的形式储存。 | - BS合轴且CacheMode="PA_BLK_BSND"/"PA_BLK_NZ"时需传  | INT32    | INT32 | ND   | (B)     |×   |
  | kNopeClipAlphaOptional     | 输入      | 表示对kvCache做clip操作时的缩放因子。  | - 在半量化和int8全量化的per-tile场景下shape为1，其余场景可不填，不支持空Tensor | FLOAT  | - | ND | (1)    |×   |
  | rmsnormEpsilonCq           | 输入      | 计算$c^Q$的RmsNorm公式中的$\epsilon$参数。        | - 用户未特意指定时，建议传入1e-05 - 仅支持double类型 | DOUBLE         | DOUBLE | -          | - |-   |
  | rmsnormEpsilonCkv          | 输入      | 计算$c^{KV}$的RmsNorm公式中的$\epsilon$参数。   | - 用户未特意指定时，建议传入1e-05 - 仅支持double类型   | DOUBLE         | DOUBLE | -          | -  |-   |
  | cacheModeOptional          | 输入      | 表示kvCache的模式。| - 用户未特意指定时，建议传入"PA_BSND" <br> - 仅支持char*类型 <br> - 可选值为："PA_BSND"、 "PA_NZ"、 "PA_BLK_BSND"、 "PA_BLK_NZ"、 "BSND"、 "TND" | CHAR*          | CHAR* | -          | - |-   |
  | queryNormFlag     | 输入      | 表示是否输出queryNormOutOptional、dequantScaleQNormOutOptional。  | - false表示不输出，true表示输出，默认值为false | BOOL  | BOOL| -- | --    |-   |
  | weightQuantMode     | 输入      | 表示weightDq、weightUqQr、weightUk、weightDkvKr的量化模式。  | - 0表示非量化，1表示weightUqQr量化，2表示weightDq、weightUqQr、weightDkvKr量化，3表示weightDq、weightUqQr、weightDkvKr mxfp8量化，默认值为0 | INT  | INT| -- | --    |-   |
  | kvCacheQuantMode     | 输入      | 表示kvCache的量化模式。  | - 0表示非量化，1表示per-tensor量化，2表示per-channel量化，3-表示per-tile量化，默认值为0| INT64  | INT64| -- | --    |-   |
  | queryQuantMode     | 输入      | 表示query的量化模式。  | - 0表示非量化，1表示per-token-head量化，默认值为0| INT64  | INT64| -- | --    |-   |
  | ckvkrRepoMode     | 输入      | 表示kvCache和krCache的存储模式。  | - 0表示kvCache和krCache分别存储，1表示kvCache和krCache合并存储，默认值为0| INT64  | INT64 | -- | --    |-   |
  | quantScaleRepoMode     | 输入      | 表示量化scale的存储模式。  | - 0表示量化scale和数据分别存储，1表示量化scale和数据合并存储作为kvCacheRef输出，默认值为0| INT64  | INT64 | -- | --    |-   |
  | tileSize     | 输入      | 表示per-tile量化时每个tile的大小，仅在kvCacheQuantMode为3时有效。  | - 默认值为128 | INT64 | INT64 | -- | --    |-   |
  | qcQrScale     | 输入      |   表示Query的尺度矫正系数。  | - 用户不特意指定时需要传入1.0 | DOUBLE | DOUBLE | -   | -  |- |
  | kcScale     | 输入      |   表示Key的尺度矫正系数。  | - 用户不特意指定时需要传入1.0 | DOUBLE | DOUBLE | -    | -  |- |
  | queryOut                   | 输出      | 公式中Query的输出tensor（对应$q^N$）。     | -  | BFLOAT16、INT8 | BFLOAT16、FLOAT8_E4M3FN | ND         | - BS合轴：(T,N,Hckv) <br>- BS非合轴：(B,S,N,Hckv) |×   |
  | queryRopeOut               | 输出      | 公式中Query位置编码的输出tensor（对应$q^R$）。  | - | BFLOAT16       | BFLOAT16 | ND         | - BS合轴：(T,N,Dr) <br>- BS非合轴：(B,S,N,Dr)     |×   |
  | dequantScaleQNopeOutOptional  | 输出           | 公式中Query输出的量化参数。  | -     | FLOAT      | FLOAT| ND   | - BS合轴：(T,N,1) <br>- BS非合轴：(B*S,N,1)   |×   |
  | queryNormOutOptional     | 输出      | 公式中tokenX做rmsNorm后的输出tensor（对应$c^Q$）。  |  | BFLOAT16、INT8  | BFLOAT16、INT8、FLOAT8_E4M3FN | ND | - BS合轴：(T,Hcq) <br> - BS非合轴：(B,S,Hcq)  |×   |
  | dequantScaleQNormOutOptional     | 输出      | queryNormOutOptional的反量化参数。  | - weightQuantMode=0时，queryNormFlag=true仅有queryNormOutOptional输出，dequantScaleQNormOutOptional为nullptr | FLOAT  | FLOAT、FLOAT8_E8M0 | ND | - weightQuantMode=1/2：<br> 1. BS合轴：（T,1）<br> 2. BS非合轴：（B*S,1） <br> -weightQuantMode=3： <br>  1. BS合轴：(T, Hcq/32) <br>  2. BS非合轴：(B\*S, Hcq/32) |×   |
  | workspaceSize              | 输出      | 返回需在Device侧申请的workspace大小。  | - 仅用于输出结果，无需输入配置 - 数据类型为uint64_t* | -              | -| -          | -                                  |-   |
  | executor                   | 输出      | 返回op执行器，包含算子计算流程。        | - 仅用于输出结果，无需输入配置 - 数据类型为aclOpExecutor**    | -              | -| -          | -                                  |-   |

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。</br>

  第一段接口完成入参校验，出现以下场景时报错：
    
    | 返回值                 | 错误码               | 描述                                                                 |
    |------------------------|----------------------|----------------------------------------------------------------------|
    | ACLNN_ERR_PARAM_NULLPTR | 161001               | 必须传入的参数（如接口核心依赖的输入/输出参数）中存在空指针。         |
    | ACLNN_ERR_PARAM_INVALID | 161002               | 输入参数的 shape（维度/尺寸）、dtype（数据类型）不在接口支持的范围内。 |
    | ACLNN_ERR_RUNTIME_ERROR | 361001               | API 内存调用 NPU Runtime 接口时发生异常（如 Runtime 服务未启动、内存申请失败等）。 |
    | ACLNN_ERR_INNER_TILING_ERROR | 561002          | tiling发生异常，入参的dtype类型或者shape错误。 |

## aclnnMlaPrologV3WeightNz

- **参数说明**

  | 参数名        | 参数类型         | 含义                                                                 |
  |---------------|------------------|----------------------------------------------------------------------|
  | workspace     | void\*           | 在Device侧申请的workspace内存地址。                                  |
  | workspaceSize | uint64_t         | 在Device侧申请的workspace大小，由第一段接口aclnnMlaPrologV3WeightNzGetWorkspaceSize获取。 |
  | executor      | aclOpExecutor\*  | op执行器，包含了算子计算流程。                                       |
  | stream        | aclrtStream      | 指定执行任务的Stream。                                   |
      

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnMlaPrologV3WeightNz默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
- shape 格式字段含义说明

  | 字段名       | 英文全称/含义                  | 取值规则与说明                                                                 |
  |--------------|--------------------------------|------------------------------------------------------------------------------|
  | B            | Batch（输入样本批量大小）      | 取值范围：0~65536                                                           |
  | S            | Seq-Length（输入样本序列长度） | 取值范围：不限制                                                              |
  | He           | Head-Size（隐藏层大小）        | A2、A3、A5取值固定为：1024、2048、3072、4096、5120、6144、7168、7680、8192  |
  | Hcq          | q 低秩矩阵维度                 | 取值固定为：1536                                                           |
  | N            | Head-Num（多头数）             | 取值范围：1、2、4、8、16、32、64、128                                       |
  | Hckv         | kv 低秩矩阵维度                | 取值固定为：512                                                             |
  | D            | qk 不含位置编码维度            | 取值固定为：128                                                             |
  | Dr           | qk 位置编码维度                | 取值固定为：64                                                              |
  | Nkv          | kv 的 head 数                  | 取值固定为：1                                                               |
  | BlockNum     | PagedAttention 场景下的块数    | 1. 当CacheMode="PA_BSND"/"PA_NZ"时，取值大于或等于 `(B*S)/BlockSize` 向上取整的结果。<br> 2. 当CacheMode="PA_BLK_BSND"/"PA_BLK_NZ"时，取值大于或等于`B` * `(S / BlockSize)`向上取整的结果（即`B * Ceil(S/BlockSize)`）。注：BS合轴场景，每个Batch中的S长度可以不同，因此BlockNum的取值需大于或等于各Batch中S长度除以BlockSize后的向上取整结果相加。举例：actualSeqLenOptional数值为[47，151，261，422]，blocksize=128，那么Batch中的长度分别为[47，104，110，161] ，此时BlockNum=Ceil（47/128）+ Ceil（104/128）+ Ceil（110/128）+ Ceil（161/128）=5|
  | BlockSize    | PagedAttention 场景下的块大小  | 取值范围：16~1024，且为16的倍数<br>                                              |
  | T            | BS 合轴后的大小                | 取值范围：不限制；注：若采用 BS 合轴，此时 tokenX、ropeSin、ropeCos 均为 2 维，cacheIndex 为 1 维，queryOut、queryRopeOut 为 3 维 |
  | Dtile        | kvCache的D维度的大小           | - Per-tile量化场景下，取值固定为656 <br> - 其他场景下，取值固定为Hckv（512）                                                       |

-   shape约束
    - 若tokenX的维度采用BS合轴，即(T, He)
        - ropeSin和ropeCos的shape为(T, Dr)
        - 当CacheMode为PA_BSND或PA_NZ时，cacheIndex的shape为(T)
        - 当CacheMode为PA_BLK_BSND或PA_BLK_NZ时，cacheIndex的shape为(Sum(Ceil(S_i/BlockSize)))，S_i为每个Batch中的S的长度
        - 当CacheMode为PA_BLK_BSND或PA_BLK_NZ时，actualSeqLenOptional需要传入，维度为(B)
        - int8全量化场景下，dequantScaleXOptional的shape为(T, 1)；mxfp8全量化场景下，dequantScaleXOptional的shape为(T, He/32)
        - queryOut的shape为(T, N, Hckv)
        - queryRopeOut的shape为(T, N, Dr)
        - int8全量化场景和mxfp8全量化场景下，dequantScaleQNopeOutOptional的shape为(T, N, 1)，其他场景下为(1)
    - 若tokenX的维度不采用BS合轴，即(B, S, He)
        - ropeSin和ropeCos的shape为(B, S, Dr)
        - 当CacheMode为PA_BSND或PA_NZ时，cacheIndex的shape为(B, S)
        - 当CacheMode为PA_BLK_BSND或PA_BLK_NZ时，cacheIndex的shape为(B,Ceil(S/BlockSize))
        - int8全量化场景下，dequantScaleXOptional的shape为(B\*S, 1)；mxfp8全量化场景下，dequantScaleXOptional的shape为(B*S, He/32)
        - queryOut的shape为(B, S, N, Hckv)
        - queryRopeOut的shape为(B, S, N, Dr)
        - int8全量化场景和mxfp8全量化场景下，dequantScaleQNopeOutOptional的shape为(B*S, N, 1)，其他场景下为(1)
    - B、S、T、Skv值允许一个或多个取0，即Shape与B、S、T、Skv值相关的入参允许传入空Tensor，其余入参不支持传入空Tensor。
        - 如果B、S、T取值为0，则queryOut、queryRopeOut输出空Tensor，kvCacheRef、krCacheRef不做更新。
        - 如果Skv取值为0，则queryOut、queryRopeOut、dequantScaleQNopeOutOptional正常计算，kvCacheRef、krCacheRef不做更新，即输出空Tensor。
    - 当CacheMode为BSND时
        - tokenX应不采用BS合轴，即维度为(B, S, He)
        - kvCache的维度为(B,S,Nkv,Dr)
    - 当CacheMode为TND时
        - tokenX应采用BS合轴，即维度为(T, He)
        - kvCache的维度为(T,Nkv,Dr)
    - 当ckvkrRepoMode=1时
        - krCache的维度应包含0，支持维度为(0)


- 特殊约束
  - per-tile量化模式下，ckvkrRepoMode和quantScaleRepoMode必须同时为1；其他量化模式以及非量化场景下，ckvkrRepoMode和quantScaleRepoMode必须同时为0。
  - per-tile量化模式下，CacheMode只支持PA_BSND, BSND和TND。
  - 当ckvkrRepoMode值为1时，krCache必须为空Tensor（即shape的乘积为0）。
- aclnnMlaPrologV3WeightNz接口支持场景：A2、A3当前不支持mxfp8全量化场景，A5当前仅支持非量化场景、部分量化kvCache非量化、部分量化kvCache per-channel量化和mxfp8全量化场景
  <table style="table-layout: auto;" border="1">
    <tr>
      <th colspan="2">场景</th>
      <th>含义</th>
    </tr>
    <tr>
      <td colspan="2">非量化</td>
      <td>
          入参：所有入参皆为非量化数据 <br> 
          出参：所有出参皆为非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">部分量化</td>
      <td>kvCache非量化 </td>
      <td>
          入参：weightUqQr传入per-token量化数据，其余入参皆为非量化数据。dequantScaleWUqQr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kvCache per-channel量化 </td>
      <td>
          入参：weightUqQr传入per-token量化数据，kvCacheRef、krCacheRef传入per-channel量化数据，其余入参皆为非量化数据。dequantScaleWUqQr、quantScaleCkv、quant_scale_ckr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：kvCacheRef、krCacheRef返回per-channel量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td>kvCache per-tile量化 </td>
      <td>
          入参：tokenX传入per-token量化数据，weightDq、weightUqQr、weightDkvKr传入per-channel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：kvCacheRef返回per-tile量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">int8全量化</td>
      <td> kvCache非量化</td>
      <td>
          入参：tokenX传入per-token量化数据，weightDq、weightUqQr、weightDkvKr传入per-channel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kvCache per-tensor量化 </td>
      <td>
          入参：tokenX传入per-token量化数据，weightDq、weightUqQr、weightDkvKr传入per-channel量化数据，kvCacheRef传入per-tensor量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr、quantScaleCkv字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：queryOut返回per-token-head量化数据，kvCacheRef出参返回per-tensor量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kvCache per-tile量化 </td>
      <td>
          入参：tokenX传入per-token量化数据，weightDq、weightUqQr、weightDkvKr传入per-channel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入，smoothScalesCq字段可选传入 <br>
          出参：kvCacheRef出参返回per-tile量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td rowspan="3">mxfp8全量化</td>
      <td> kvCache非量化</td>
      <td>
          入参：tokenX传入per-token量化数据，weightDq、weightUqQr、weightDkvKr传入per-channel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入 <br>
          出参：所有出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kvCache per-tensor量化 </td>
      <td>
          入参：tokenX传入per-token量化数据，weightDq、weightUqQr、weightDkvKr传入per-channel量化数据，kvCacheRef传入per-tensor量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr、quantScaleCkv字段必须传入 <br>
          出参：queryOut返回per-token-head量化数据，kvCacheRef出参返回per-tensor量化数据，其余出参返回非量化数据
      </td>
    </tr>
    <tr>
      <td> kvCache per-tile量化 </td>
      <td>
          入参：tokenX传入per-token量化数据，weightDq、weightUqQr、weightDkvKr传入per-channel量化数据，其余入参皆为非量化数据。dequantScaleX、dequantScaleWDq、dequantScaleWUqQr、dequantScaleWDkvKr字段必须传入 <br>
          出参：kvCacheRef出参返回per-tile量化数据，其余出参返回非量化数据
      </td>
    </tr>
  </table>

- 在不同量化场景下，参数的dtype组合需要满足如下条件：

  <div style="overflow-x: auto; width: 100%;">
  <table style="table-layout: auto;" border="1">
    <tr>
      <th rowspan="3">参数名</th>
      <th rowspan="2" colspan="1">非量化场景</th>
      <th colspan="3">部分量化场景</th>
      <th colspan="3">int8全量化场景</th>
      <th colspan="3">mxfp8全量化场景</th>
    </tr>
    <tr>
      <th colspan="1">kvCache非量化</th>
      <th colspan="1">kvCache per-channel量化</th>
      <th colspan="1">kvCache per-tile量化</th>
      <th colspan="1">kvCache非量化</th>
      <th colspan="1">kvCache per-tensor量化</th>
      <th colspan="1">kvCache per-tile量化</th>
      <th colspan="1">kvCache非量化</th>
      <th colspan="1">kvCache per-tensor量化</th>
      <th colspan="1">kvCache per-tile量化</th>
    </tr>
    <tr>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
      <th>dtype</th>
    </tr>
    <tr>
      <td>tokenX</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
    </tr>
    <tr>
      <td>weightDq</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
    </tr>
    <tr>
      <td>weightUqQr</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
    </tr>
    <tr>
      <td>weightUk</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td>weightDkvKr</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
    </tr>
    <tr>
      <td> rmsnormGammaCq </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> rmsnormGammaCkv </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> ropeSin </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> ropeCos </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> kvCacheRef </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>BFLOAT16</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
    </tr>
    <tr>
      <td> krCacheRef </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> cacheIndexOptional </td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
      <td>INT64</td>
    </tr>
    <tr>
      <td> dequantScaleXOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
    </tr>
    <tr>
      <td> dequantScaleWDqOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
    </tr>
    <tr>
      <td> dequantScaleWUqQrOptional </td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
    </tr>
    <tr>
      <td> dequantScaleWDkvKrOptional </td>
      <td> NULLPTR </td>
      <td> NULLPTR </td>
      <td> NULLPTR </td>
      <td> NULLPTR </td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
    </tr>
    <tr>
      <td> quantScaleCkvOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
    </tr>
    <tr>
      <td> quantScaleCkrOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
    </tr>
    <tr>
      <td> smoothScalesCqOptional </td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
    </tr>
    <tr>
      <td> kNopeClipAlphaOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
    </tr>
    <tr>
      <td> queryOut </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>FLOAT8_E4M3FN</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> queryRopeOut </td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
      <td>BFLOAT16</td>
    </tr>
    <tr>
      <td> dequantScaleQNopeOutOptional </td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>NULLPTR</td>
    </tr>
    <tr>
      <td> queryNormOutOptional </td>
      <td>BFLOAT16</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>INT8</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
      <td>FLOAT8_E4M3FN</td>
    </tr>
    <tr>
      <td> dequantScaleQNormOutOptional </td>
      <td>NULLPTR</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
      <td>FLOAT8_E8M0</td>
    </tr>
  </table>
  </div>
## 调用示例

A2、A3示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <cstdint>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_mla_prolog_v3_weight_nz.h"
  #include<unistd.h>

  #define CHECK_RET(cond, return_expr) \
    do {                               \
      if (!(cond)) {                   \
        return_expr;                   \
      }                                \
    } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
      int64_t shape_size = 1;
      for (auto i : shape) {
          shape_size *= i;
      }
      return shape_size;
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
  int CreateAclTensorND(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMalloc申请host侧内存
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensorNZ(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                      aclDataType dataType, aclTensor** tensor) {
      auto size = GetShapeSize(shape) * sizeof(T);
      // 调用aclrtMalloc申请device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMalloc申请host侧内存
      ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                                shape.data(), shape.size(), *deviceAddr);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  int TransToNZShape(std::vector<int64_t> &shapeND, size_t typeSize) {
      if (typeSize == static_cast<size_t>(0)) {
        return 0;
      }
      int64_t h = shapeND[0];
      int64_t w = shapeND[1];
      int64_t h0 = static_cast<int64_t>(16);
      int64_t w0 = static_cast<int64_t>(32) / static_cast<int64_t>(typeSize);
      int64_t h1 = h / h0;
      int64_t w1 = w / w0;
      shapeND[0] = w1;
      shapeND[1] = h1;
      shapeND.emplace_back(h0);
      shapeND.emplace_back(w0);
      return 0;
  }

  int main() {
      // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      // check根据自己的需要处理
      CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      // 2. 构造输入与输出，需要根据API的接口定义构造
      std::vector<int64_t> tokenXShape = {8, 1, 7168};            // B,S,He
      std::vector<int64_t> weightDqShape = {7168, 1536};          // He,Hcq
      std::vector<int64_t> weightUqQrShape = {1536, 6144};        // Hcq,N*(D+Dr)
      std::vector<int64_t> weightUkShape = {32, 128, 512};        // N,D,Hckv
      std::vector<int64_t> weightDkvKrShape = {7168, 576};        // He,Hckv+Dr
      std::vector<int64_t> rmsnormGammaCqShape = {1536};          // Hcq
      std::vector<int64_t> rmsnormGammaCkvShape = {512};          // Hckv
      std::vector<int64_t> ropeSinShape = {8, 1, 64};             // B,S,Dr
      std::vector<int64_t> ropeCosShape = {8, 1, 64};             // B,S,Dr
      std::vector<int64_t> cacheIndexShape = {8, 1};              // B,S
      std::vector<int64_t> kvCacheShape = {16, 128, 1, 512};      // BolckNum,BlockSize,Nkv,Hckv
      std::vector<int64_t> krCacheShape = {16, 128, 1, 64};       // BolckNum,BlockSize,Nkv,Dr
      std::vector<int64_t> dequantScaleXShape = {8, 1};           // B*S, 1
      std::vector<int64_t> dequantScaleWDqShape = {1, 1536};      // 1, Hcq
      std::vector<int64_t> dequantScaleWUqQrShape = {1, 6144};    // 1, N*(D+Dr)
      std::vector<int64_t> dequantScaleWDkvKrShape = {1, 576};    // 1, Hckv+Dr
      std::vector<int64_t> quantScaleCkvShape = {1};              // 1
      std::vector<int64_t> smoothScalesCqShape = {1, 1536};       // 1, Hcq
      std::vector<int64_t> queryShape = {8, 1, 32, 512};          // B,S,N,Hckv
      std::vector<int64_t> queryRopeShape = {8, 1, 32, 64};       // B,S,N,Dr
      std::vector<int64_t> dequantScaleQNopeShape = {8, 32, 1};   // B*S, N, 1
      double rmsnormEpsilonCq = 1e-5;
      double rmsnormEpsilonCkv = 1e-5;
      char cacheMode[] = "PA_BSND";

      void* tokenXDeviceAddr = nullptr;
      void* weightDqDeviceAddr = nullptr;
      void* weightUqQrDeviceAddr = nullptr;
      void* weightUkDeviceAddr = nullptr;
      void* weightDkvKrDeviceAddr = nullptr;
      void* rmsnormGammaCqDeviceAddr = nullptr;
      void* rmsnormGammaCkvDeviceAddr = nullptr;
      void* ropeSinDeviceAddr = nullptr;
      void* ropeCosDeviceAddr = nullptr;
      void* cacheIndexDeviceAddr = nullptr;
      void* kvCacheDeviceAddr = nullptr;
      void* krCacheDeviceAddr = nullptr;
      void* dequantScaleXDeviceAddr = nullptr;
      void* dequantScaleWDqDeviceAddr = nullptr;
      void* dequantScaleWUqQrDeviceAddr = nullptr;
      void* dequantScaleWDkvKrDeviceAddr = nullptr;
      void* quantScaleCkvDeviceAddr = nullptr;
      void* smoothScalesCqDeviceAddr = nullptr;
      void* queryDeviceAddr = nullptr;
      void* queryRopeDeviceAddr = nullptr;
      void* dequantScaleQNopeDeviceAddr = nullptr;

      void* tokenXHostAddr = nullptr;
      void* weightDqHostAddr = nullptr;
      void* weightUqQrHostAddr = nullptr;
      void* weightUkHostAddr = nullptr;
      void* weightDkvKrHostAddr = nullptr;
      void* rmsnormGammaCqHostAddr = nullptr;
      void* rmsnormGammaCkvHostAddr = nullptr;
      void* ropeSinHostAddr = nullptr;
      void* ropeCosHostAddr = nullptr;
      void* cacheIndexHostAddr = nullptr;
      void* kvCacheHostAddr = nullptr;
      void* krCacheHostAddr = nullptr;
      void* dequantScaleXHostAddr = nullptr;
      void* dequantScaleWDqHostAddr = nullptr;
      void* dequantScaleWUqQrHostAddr = nullptr;
      void* dequantScaleWDkvKrHostAddr = nullptr;
      void* quantScaleCkvHostAddr = nullptr;
      void* smoothScalesCqHostAddr = nullptr;
      void* queryHostAddr = nullptr;
      void* queryRopeHostAddr = nullptr;
      void* dequantScaleQNopeHostAddr = nullptr;

      aclTensor* tokenX = nullptr;
      aclTensor* weightDq = nullptr;
      aclTensor* weightUqQr = nullptr;
      aclTensor* weightUk = nullptr;
      aclTensor* weightDkvKr = nullptr;
      aclTensor* rmsnormGammaCq = nullptr;
      aclTensor* rmsnormGammaCkv = nullptr;
      aclTensor* ropeSin = nullptr;
      aclTensor* ropeCos = nullptr;
      aclTensor* cacheIndex = nullptr;
      aclTensor* kvCache = nullptr;
      aclTensor* krCache = nullptr;
      aclTensor* dequantScaleX = nullptr;
      aclTensor* dequantScaleWDq = nullptr;
      aclTensor* dequantScaleWUqQr = nullptr;
      aclTensor* dequantScaleWDkvKr = nullptr;
      aclTensor* quantScaleCkv = nullptr;
      aclTensor* smoothScalesCq = nullptr;
      bool queryNormFlag = false;
      int64_t weightQuantMode = 2;
      int64_t kvQuantMode = 1;
      int64_t queryQuantMode = 1;
      int64_t ckvkrRepoMode = 0;
      int64_t quantScaleRepoMode = 0;
      int64_t tileSize = 128;
      double kNopeClipAlpha = 1.0f;
      double qcQrScale = 1.0f;
      double kcScale = 1.0f;
      aclTensor* query = nullptr;
      aclTensor* queryRope = nullptr;
      aclTensor* dequantScaleQNope = nullptr;

      // 转换三个NZ格式变量的shape
      constexpr size_t EXAMPLE_INT8_SIZE = sizeof(int8_t);
      constexpr size_t EXAMPLE_BFLOAT16_SIZE = sizeof(int16_t);
      ret = TransToNZShape(weightDqShape, EXAMPLE_INT8_SIZE);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightUqQrShape, EXAMPLE_INT8_SIZE);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
      ret = TransToNZShape(weightDkvKrShape, EXAMPLE_INT8_SIZE);
      CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);

      // 创建tokenX aclTensor
      ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_INT8, &tokenX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightDq aclTensor
      ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_INT8, &weightDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightUqQr aclTensor
      ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_INT8, &weightUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightUk aclTensor
      ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建weightDkvKr aclTensor
      ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_INT8, &weightDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建rmsnormGammaCq aclTensor
      ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建rmsnormGammaCkv aclTensor
      ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建ropeSin aclTensor
      ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建ropeCos aclTensor
      ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建cacheIndex aclTensor
      ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建kvCache aclTensor
      ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_INT8, &kvCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建krCache aclTensor
      ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建dequantScaleX aclTensor
      ret = CreateAclTensorND(dequantScaleXShape, &dequantScaleXDeviceAddr, &dequantScaleXHostAddr, aclDataType::ACL_FLOAT, &dequantScaleX);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建dequantScaleWDq aclTensor
      ret = CreateAclTensorND(dequantScaleWDqShape, &dequantScaleWDqDeviceAddr, &dequantScaleWDqHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWDq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建dequantScaleWUqQr aclTensor
      ret = CreateAclTensorND(dequantScaleWUqQrShape, &dequantScaleWUqQrDeviceAddr, &dequantScaleWUqQrHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWUqQr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建dequantScaleWDkvKr aclTensor
      ret = CreateAclTensorND(dequantScaleWDkvKrShape, &dequantScaleWDkvKrDeviceAddr, &dequantScaleWDkvKrHostAddr, aclDataType::ACL_FLOAT, &dequantScaleWDkvKr);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建quantScaleCkv aclTensor
      ret = CreateAclTensorND(quantScaleCkvShape, &quantScaleCkvDeviceAddr, &quantScaleCkvHostAddr, aclDataType::ACL_FLOAT, &quantScaleCkv);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建smoothScalesCq aclTensor
      ret = CreateAclTensorND(smoothScalesCqShape, &smoothScalesCqDeviceAddr, &smoothScalesCqHostAddr, aclDataType::ACL_FLOAT, &smoothScalesCq);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建query aclTensor
      ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_INT8, &query);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建queryRope aclTensor
      ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建dequantScaleQNope aclTensor
      ret = CreateAclTensorND(dequantScaleQNopeShape, &dequantScaleQNopeDeviceAddr, &dequantScaleQNopeHostAddr, aclDataType::ACL_FLOAT, &dequantScaleQNope);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API，需要修改为具体的API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      // 调用aclnnMlaPrologV3WeightNz第一段接口
      ret = aclnnMlaPrologV3WeightNzGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, krCache, cacheIndex,
        dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, nullptr, smoothScalesCq, nullptr, nullptr,rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode,
        weightQuantMode, kvQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScale, kcScale,
        query, queryRope, dequantScaleQNope, nullptr, nullptr, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > static_cast<uint64_t>(0)) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
      }
      // 调用aclnnMlaPrologV3WeightNz第二段接口
      ret = aclnnMlaPrologV3WeightNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. 固定写法，同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(queryShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
      // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
      aclDestroyTensor(tokenX);
      aclDestroyTensor(weightDq);
      aclDestroyTensor(weightUqQr);
      aclDestroyTensor(weightUk);
      aclDestroyTensor(weightDkvKr);
      aclDestroyTensor(rmsnormGammaCq);
      aclDestroyTensor(rmsnormGammaCkv);
      aclDestroyTensor(ropeSin);
      aclDestroyTensor(ropeCos);
      aclDestroyTensor(cacheIndex);
      aclDestroyTensor(kvCache);
      aclDestroyTensor(krCache);
      aclDestroyTensor(dequantScaleX);
      aclDestroyTensor(dequantScaleWDq);
      aclDestroyTensor(dequantScaleWUqQr);
      aclDestroyTensor(dequantScaleWDkvKr);
      aclDestroyTensor(quantScaleCkv);
      aclDestroyTensor(smoothScalesCq);
      aclDestroyTensor(query);
      aclDestroyTensor(queryRope);
      aclDestroyTensor(dequantScaleQNope);

      // 7. 释放device 资源
      aclrtFree(tokenXDeviceAddr);
      aclrtFree(weightDqDeviceAddr);
      aclrtFree(weightUqQrDeviceAddr);
      aclrtFree(weightUkDeviceAddr);
      aclrtFree(weightDkvKrDeviceAddr);
      aclrtFree(rmsnormGammaCqDeviceAddr);
      aclrtFree(rmsnormGammaCkvDeviceAddr);
      aclrtFree(ropeSinDeviceAddr);
      aclrtFree(ropeCosDeviceAddr);
      aclrtFree(cacheIndexDeviceAddr);
      aclrtFree(kvCacheDeviceAddr);
      aclrtFree(krCacheDeviceAddr);
      aclrtFree(dequantScaleXDeviceAddr);
      aclrtFree(dequantScaleWDqDeviceAddr);
      aclrtFree(dequantScaleWUqQrDeviceAddr);
      aclrtFree(dequantScaleWDkvKrDeviceAddr);
      aclrtFree(quantScaleCkvDeviceAddr);
      aclrtFree(smoothScalesCqDeviceAddr);
      aclrtFree(queryDeviceAddr);
      aclrtFree(queryRopeDeviceAddr);
      aclrtFree(dequantScaleQNopeDeviceAddr);

      // 8. 释放host 资源
      aclrtFree(tokenXHostAddr);
      aclrtFree(weightDqHostAddr);
      aclrtFree(weightUqQrHostAddr);
      aclrtFree(weightUkHostAddr);
      aclrtFree(weightDkvKrHostAddr);
      aclrtFree(rmsnormGammaCqHostAddr);
      aclrtFree(rmsnormGammaCkvHostAddr);
      aclrtFree(ropeSinHostAddr);
      aclrtFree(ropeCosHostAddr);
      aclrtFree(cacheIndexHostAddr);
      aclrtFree(kvCacheHostAddr);
      aclrtFree(krCacheHostAddr);
      aclrtFree(dequantScaleXHostAddr);
      aclrtFree(dequantScaleWDqHostAddr);
      aclrtFree(dequantScaleWUqQrHostAddr);
      aclrtFree(dequantScaleWDkvKrHostAddr);
      aclrtFree(quantScaleCkvHostAddr);
      aclrtFree(smoothScalesCqHostAddr);
      aclrtFree(queryHostAddr);
      aclrtFree(queryRopeHostAddr);
      aclrtFree(dequantScaleQNopeHostAddr);

    if (workspaceSize > static_cast<uint64_t>(0)) {
      aclrtFree(workspaceAddr);
    }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();

      return 0;
  }
  ```

A5示例代码如下，仅供参考。

  ```Cpp
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_mla_prolog_v3_weight_nz.h"
#include <unistd.h>

#define CHECK_RET(cond, return_expr)    \
    do {                                \
        if (!(cond)) {                  \
            return_expr;                \
        }                               \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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
int CreateAclTensorND(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMalloc申请host侧内存
    ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensorNZ(const std::vector<T>& shape, void** deviceAddr, void** hostAddr,
                    aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMalloc申请host侧内存
    ret = aclrtMalloc(hostAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_FRACTAL_NZ,
                              shape.data(), shape.size(), *deviceAddr);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, *hostAddr, GetShapeSize(shape)*aclDataTypeSize(dataType), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    return 0;
}

int TransToNZShape(std::vector<int64_t> &shapeND, size_t typeSize) {
    if (typeSize == static_cast<size_t>(0)) {
        return 0;
    }
    int64_t h = shapeND[0];
    int64_t w = shapeND[1];
    int64_t h0 = static_cast<int64_t>(16);
    int64_t w0 = static_cast<int64_t>(32) / static_cast<int64_t>(typeSize);
    int64_t h1 = h / h0;
    int64_t w1 = w / w0;
    shapeND[0] = w1;
    shapeND[1] = h1;
    shapeND.emplace_back(h0);
    shapeND.emplace_back(w0);
    return 0;
}

int main() {
    // 1. 固定写法，device/stream初始化, 参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> tokenXShape = {8,1,7168};            // B,S,He
    std::vector<int64_t> weightDqShape = {7168,1536};          // He,Hcq
    std::vector<int64_t> weightUqQrShape = {1536,24576};        // Hcq,N*(D+Dr)
    std::vector<int64_t> weightUkShape = {128, 128, 512};        // N,D,Hckv
    std::vector<int64_t> weightDkvKrShape = {7168, 576};        // He,Hckv+Dr
    std::vector<int64_t> rmsnormGammaCqShape = {1536};          // Hcq
    std::vector<int64_t> rmsnormGammaCkvShape = {512};          // Hckv
    std::vector<int64_t> ropeSinShape = {8,1,64};             // B,S,Dr
    std::vector<int64_t> ropeCosShape = {8,1,64};             // B,S,Dr
    std::vector<int64_t> kvCacheShape = {1,16,1,512};      // BolckNum,BlockSize,Nkv,Hckv
    std::vector<int64_t> krCacheShape = {1,16,1,64};       // BolckNum,BlockSize,Nkv,Dr
    std::vector<int64_t> cacheIndexShape = {8,1};              // B,S
    std::vector<int64_t> dequantScaleXShape = {8,224};           // B*S, 1
    std::vector<int64_t> dequantScaleWDqShape = {1536,224};      // 1, Hcq
    std::vector<int64_t> dequantScaleWUqQrShape = {24576,48};    // 1, N*(D+Dr)
    std::vector<int64_t> dequantScaleWDkvKrShape = {576,224};    // 1, Hckv+Dr
    std::vector<int64_t> quantScaleCkvShape = {1};    // 1
    std::vector<int64_t> queryShape = {8,1,128,512};          // B,S,N,Hckv
    std::vector<int64_t> queryRopeShape = {8,1,128,64};       // B,S,N,Dr
    std::vector<int64_t> dequantScaleQNopeShape = {8,128,1};   // B*S, N, 1
    double rmsnormEpsilonCq = 1e-5;
    double rmsnormEpsilonCkv = 1e-5;
    char cacheMode[] = "PA_BSND";

    void* tokenXDeviceAddr = nullptr;
    void* weightDqDeviceAddr = nullptr;
    void* weightUqQrDeviceAddr = nullptr;
    void* weightUkDeviceAddr = nullptr;
    void* weightDkvKrDeviceAddr = nullptr;
    void* rmsnormGammaCqDeviceAddr = nullptr;
    void* rmsnormGammaCkvDeviceAddr = nullptr;
    void* ropeSinDeviceAddr = nullptr;
    void* ropeCosDeviceAddr = nullptr;
    void* cacheIndexDeviceAddr = nullptr;
    void* kvCacheDeviceAddr = nullptr;
    void* krCacheDeviceAddr = nullptr;
    void* dequantScaleXDeviceAddr = nullptr;
    void* dequantScaleWDqDeviceAddr = nullptr;
    void* dequantScaleWUqQrDeviceAddr = nullptr;
    void* dequantScaleWDkvKrDeviceAddr = nullptr;
    void* quantScaleCkvDeviceAddr = nullptr;
    void* queryDeviceAddr = nullptr;
    void* queryRopeDeviceAddr = nullptr;
    void* dequantScaleQNopeDeviceAddr = nullptr;

    void* tokenXHostAddr = nullptr;
    void* weightDqHostAddr = nullptr;
    void* weightUqQrHostAddr = nullptr;
    void* weightUkHostAddr = nullptr;
    void* weightDkvKrHostAddr = nullptr;
    void* rmsnormGammaCqHostAddr = nullptr;
    void* rmsnormGammaCkvHostAddr = nullptr;
    void* ropeSinHostAddr = nullptr;
    void* ropeCosHostAddr = nullptr;
    void* cacheIndexHostAddr = nullptr;
    void* kvCacheHostAddr = nullptr;
    void* krCacheHostAddr = nullptr;
    void* dequantScaleXHostAddr = nullptr;
    void* dequantScaleWDqHostAddr = nullptr;
    void* dequantScaleWUqQrHostAddr = nullptr;
    void* dequantScaleWDkvKrHostAddr = nullptr;
    void* quantScaleCkvHostAddr = nullptr;
    void* queryHostAddr = nullptr;
    void* queryRopeHostAddr = nullptr;
    void* dequantScaleQNopeHostAddr = nullptr;

    aclTensor* tokenX = nullptr;
    aclTensor* weightDq = nullptr;
    aclTensor* weightUqQr = nullptr;
    aclTensor* weightUk = nullptr;
    aclTensor* weightDkvKr = nullptr;
    aclTensor* rmsnormGammaCq = nullptr;
    aclTensor* rmsnormGammaCkv = nullptr;
    aclTensor* ropeSin = nullptr;
    aclTensor* ropeCos = nullptr;
    aclTensor* kvCache = nullptr;
    aclTensor* krCache = nullptr;
    aclTensor* cacheIndex = nullptr;
    aclTensor* dequantScaleX = nullptr;
    aclTensor* dequantScaleWDq = nullptr;
    aclTensor* dequantScaleWUqQr = nullptr;
    aclTensor* dequantScaleWDkvKr = nullptr;
    aclTensor* quantScaleCkv = nullptr;
    bool queryNormFlag = false;
    int64_t weightQuantMode = 3;
    int64_t kvQuantMode = 1;
    int64_t queryQuantMode = 1;
    int64_t ckvkrRepoMode = 0;
    int64_t quantScaleRepoMode = 0;
    int64_t tileSize = 128;
    double qcQrScale = 1.0f;
    double kcScale = 1.0f;
    aclTensor* query = nullptr;
    aclTensor* queryRope = nullptr;
    aclTensor* dequantScaleQNope = nullptr;

    // 转换三个NZ格式变量的shape
    constexpr size_t EXAMPLE_INT8_SIZE = sizeof(int8_t);
    constexpr size_t EXAMPLE_BFLOAT16_SIZE = sizeof(int16_t);
    ret = TransToNZShape(weightDqShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightUqQrShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);
    ret = TransToNZShape(weightDkvKrShape, EXAMPLE_INT8_SIZE);
    CHECK_RET(ret == 0, LOG_PRINT("trans NZ shape failed.\n"); return ret);

    // 创建tokenX aclTensor
    ret = CreateAclTensorND(tokenXShape, &tokenXDeviceAddr, &tokenXHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &tokenX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightDq aclTensor
    ret = CreateAclTensorNZ(weightDqShape, &weightDqDeviceAddr, &weightDqHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightDq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightUqQr aclTensor
    ret = CreateAclTensorNZ(weightUqQrShape, &weightUqQrDeviceAddr, &weightUqQrHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightUk aclTensor
    ret = CreateAclTensorND(weightUkShape, &weightUkDeviceAddr, &weightUkHostAddr, aclDataType::ACL_BF16, &weightUk);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weightDkvKr aclTensor
    ret = CreateAclTensorNZ(weightDkvKrShape, &weightDkvKrDeviceAddr, &weightDkvKrHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &weightDkvKr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rmsnormGammaCq aclTensor
    ret = CreateAclTensorND(rmsnormGammaCqShape, &rmsnormGammaCqDeviceAddr, &rmsnormGammaCqHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rmsnormGammaCkv aclTensor
    ret = CreateAclTensorND(rmsnormGammaCkvShape, &rmsnormGammaCkvDeviceAddr, &rmsnormGammaCkvHostAddr, aclDataType::ACL_BF16, &rmsnormGammaCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建ropeSin aclTensor
    ret = CreateAclTensorND(ropeSinShape, &ropeSinDeviceAddr, &ropeSinHostAddr, aclDataType::ACL_BF16, &ropeSin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建ropeCos aclTensor
    ret = CreateAclTensorND(ropeCosShape, &ropeCosDeviceAddr, &ropeCosHostAddr, aclDataType::ACL_BF16, &ropeCos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cacheIndex aclTensor
    ret = CreateAclTensorND(cacheIndexShape, &cacheIndexDeviceAddr, &cacheIndexHostAddr, aclDataType::ACL_INT64, &cacheIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建kvCache aclTensor
    ret = CreateAclTensorND(kvCacheShape, &kvCacheDeviceAddr, &kvCacheHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &kvCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建krCache aclTensor
    ret = CreateAclTensorND(krCacheShape, &krCacheDeviceAddr, &krCacheHostAddr, aclDataType::ACL_BF16, &krCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dequantScaleX aclTensor
    ret = CreateAclTensorND(dequantScaleXShape, &dequantScaleXDeviceAddr, &dequantScaleXHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dequantScaleWDq aclTensor
    ret = CreateAclTensorND(dequantScaleWDqShape, &dequantScaleWDqDeviceAddr, &dequantScaleWDqHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleWDq);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dequantScaleWUqQr aclTensor
    ret = CreateAclTensorND(dequantScaleWUqQrShape, &dequantScaleWUqQrDeviceAddr, &dequantScaleWUqQrHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleWUqQr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dequantScaleWDkvKr aclTensor
    ret = CreateAclTensorND(dequantScaleWDkvKrShape, &dequantScaleWDkvKrDeviceAddr, &dequantScaleWDkvKrHostAddr, aclDataType::ACL_FLOAT8_E8M0, &dequantScaleWDkvKr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建quantScaleCkv aclTensor
    ret = CreateAclTensorND(quantScaleCkvShape, &quantScaleCkvDeviceAddr, &quantScaleCkvHostAddr, aclDataType::ACL_FLOAT, &quantScaleCkv);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建query aclTensor
    ret = CreateAclTensorND(queryShape, &queryDeviceAddr, &queryHostAddr, aclDataType::ACL_FLOAT8_E4M3FN, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建queryRope aclTensor
    ret = CreateAclTensorND(queryRopeShape, &queryRopeDeviceAddr, &queryRopeHostAddr, aclDataType::ACL_BF16, &queryRope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dequantScaleQNope aclTensor
    ret = CreateAclTensorND(dequantScaleQNopeShape, &dequantScaleQNopeDeviceAddr, &dequantScaleQNopeHostAddr, aclDataType::ACL_FLOAT, &dequantScaleQNope);
    CHECK_RET(ret == ACL_SUCCESS, return ret);


    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 调用aclnnMlaPrologV3WeightNz第一段接口
    ret = aclnnMlaPrologV3WeightNzGetWorkspaceSize(tokenX, weightDq, weightUqQr, weightUk, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, kvCache, krCache, cacheIndex,
        dequantScaleX, dequantScaleWDq, dequantScaleWUqQr, dequantScaleWDkvKr, quantScaleCkv, nullptr, nullptr, nullptr, nullptr, rmsnormEpsilonCq, rmsnormEpsilonCkv, cacheMode,
        weightQuantMode, kvQuantMode, queryQuantMode, ckvkrRepoMode, quantScaleRepoMode, tileSize, qcQrScale, kcScale,
        query, queryRope, dequantScaleQNope, nullptr, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMlaPrologV3WeightNz第二段接口
    ret = aclnnMlaPrologV3WeightNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMlaPrologV3WeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(queryShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), queryDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(tokenX);
    aclDestroyTensor(weightDq);
    aclDestroyTensor(weightUqQr);
    aclDestroyTensor(weightUk);
    aclDestroyTensor(weightDkvKr);
    aclDestroyTensor(rmsnormGammaCq);
    aclDestroyTensor(rmsnormGammaCkv);
    aclDestroyTensor(ropeSin);
    aclDestroyTensor(ropeCos);
    aclDestroyTensor(cacheIndex);
    aclDestroyTensor(kvCache);
    aclDestroyTensor(krCache);
    aclDestroyTensor(dequantScaleX);
    aclDestroyTensor(dequantScaleWDq);
    aclDestroyTensor(dequantScaleWUqQr);
    aclDestroyTensor(dequantScaleWDkvKr);
    aclDestroyTensor(quantScaleCkv);
    aclDestroyTensor(query);
    aclDestroyTensor(queryRope);
    aclDestroyTensor(dequantScaleQNope);

    // 7. 释放device 资源
    aclrtFree(tokenXDeviceAddr);
    aclrtFree(weightDqDeviceAddr);
    aclrtFree(weightUqQrDeviceAddr);
    aclrtFree(weightUkDeviceAddr);
    aclrtFree(weightDkvKrDeviceAddr);
    aclrtFree(rmsnormGammaCqDeviceAddr);
    aclrtFree(rmsnormGammaCkvDeviceAddr);
    aclrtFree(ropeSinDeviceAddr);
    aclrtFree(ropeCosDeviceAddr);
    aclrtFree(cacheIndexDeviceAddr);
    aclrtFree(kvCacheDeviceAddr);
    aclrtFree(krCacheDeviceAddr);
    aclrtFree(dequantScaleXDeviceAddr);
    aclrtFree(dequantScaleWDqDeviceAddr);
    aclrtFree(dequantScaleWUqQrDeviceAddr);
    aclrtFree(dequantScaleWDkvKrDeviceAddr);
    aclrtFree(quantScaleCkvDeviceAddr);
    aclrtFree(queryDeviceAddr);
    aclrtFree(queryRopeDeviceAddr);
    aclrtFree(dequantScaleQNopeDeviceAddr);

    // 8. 释放host 资源
    aclrtFree(tokenXHostAddr);
    aclrtFree(weightDqHostAddr);
    aclrtFree(weightUqQrHostAddr);
    aclrtFree(weightUkHostAddr);
    aclrtFree(weightDkvKrHostAddr);
    aclrtFree(rmsnormGammaCqHostAddr);
    aclrtFree(rmsnormGammaCkvHostAddr);
    aclrtFree(ropeSinHostAddr);
    aclrtFree(ropeCosHostAddr);
    aclrtFree(cacheIndexHostAddr);
    aclrtFree(kvCacheHostAddr);
    aclrtFree(krCacheHostAddr);
    aclrtFree(dequantScaleXHostAddr);
    aclrtFree(dequantScaleWDqHostAddr);
    aclrtFree(dequantScaleWUqQrHostAddr);
    aclrtFree(dequantScaleWDkvKrHostAddr);
    aclrtFree(quantScaleCkvHostAddr);
    aclrtFree(queryHostAddr);
    aclrtFree(queryRopeHostAddr);
    aclrtFree(dequantScaleQNopeHostAddr);

    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    _exit(0);
}
  ```