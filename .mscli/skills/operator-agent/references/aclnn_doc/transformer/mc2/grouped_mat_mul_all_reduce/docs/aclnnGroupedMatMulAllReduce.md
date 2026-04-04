# aclnnGroupedMatMulAllReduce

**该接口后续版本会废弃，请不要使用该接口。**

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：在grouped_matmul的基础上实现多卡并行AllReduce功能，实现分组矩阵乘计算，每组矩阵乘的维度大小可以不同。根据x、weight、y的Tensor数量支持如下4种场景：
  - x、weight、y的Tensor数量等于组数，即每组的数据对应的Tensor是独立的。
  - x的Tensor数量为1，weight/y的Tensor数量等于组数，此时需要通过可选参数group_list说明x在行上的分组情况，如group_list[0]=10说明x的前10行参与第一组矩阵乘计算。
  - x、weight的Tensor数量等于组数，y的Tensor数量为1，此时每组矩阵乘的结果放在同一个Tensor中连续存放。
  - x、y的Tensor数量为1，weight数量等于组数，属于前两种情况的组合。

- 计算公式：

  非量化场景：

  $$
  y_i=x_i\times weight_i + bias_i
  $$

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnGroupedMatMulAllReduceGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnGroupedMatMulAllReduce”接口执行计算。

```cpp
aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(
    const aclTensorList* x,
    const aclTensorList* weight,
    const aclTensorList* bias,
    const aclIntArray* groupListOptional,
    int64_t splitItem,
    const char* group,
    const char* reduceOp,
    int64_t commTurn,
    int64_t streamMode,
    const aclTensorList* y,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnGroupedMatMulAllReduce(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
```

## aclnnGroupedMatMulAllReduceGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1013px"><colgroup>
  <col style="width: 160px">
  <col style="width: 111px">
  <col style="width: 429px">
  <col style="width: 188px">
  <col style="width: 125px">
  </colgroup>
  <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
    <td>x</td>
    <td>输入</td>
    <td>公式中的输入x，支持的最大长度为64个。</td>
    <td>FLOAT16、BFLOAT16（列表内张量数据类型）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>weight</td>
    <td>输入</td>
    <td>公式中的weight，支持的最大长度为64个。</td>
    <td>FLOAT16、BFLOAT16（列表内张量数据类型）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>bias</td>
    <td>可选输入</td>
    <td>公式中的bias，支持的最大长度为64个。</td>
    <td>FLOAT16、FLOAT32（列表内张量数据类型）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>groupListOptional</td>
    <td>可选输入</td>
    <td>Host侧的aclIntArray类型，代表输入和输出M方向的matmul大小分布，支持的最大长度为64个。</td>
    <td>INT64（数组元素类型）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>splitItem</td>
    <td>输入</td>
    <td>可选属性，代表输入和输出是否要做tensor切分：0（默认）=输入输出都不切分；1=输入切分、输出不切分；2=输入不切分、输出切分；3=输入输出都切分。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>Host侧标识列组的字符串，即通信域名称，通过Hccl接口HcclGetCommName获取commName作为该参数。</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>reduceOp</td>
    <td>输入</td>
    <td>reduce操作类型，当前版本仅支持输入"sum"。</td>
    <td>STRING</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>commTurn</td>
    <td>输入</td>
    <td>通信数据切分数（总数据量/单次通信量），当前版本仅支持输入0。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>streamMode</td>
    <td>输入</td>
    <td>acl流模式的枚举，当前只支持值1。</td>
    <td>INT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>公式中的输出y，支持的最大长度为64个。</td>
    <td>FLOAT16、BFLOAT16（列表内张量数据类型）</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>UINT64</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td>aclOpExecutor*</td>
    <td>ND</td>
    </tr>
  </tbody></table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 282px">
  <col style="width: 120px">
  <col style="width: 747px">
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
    <td>1. 传入参数要求是必选输入、输出或者必选属性，但实际传入了空指针。</td>
    </tr>
    <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>1. x、weight、bias、groupListOptional、splitItem、y的数据类型和数据格式不在支持的范围内；<br>2. x的长度不等于1且不等于weight的长度；<br>3. 如bias不为空，bias的长度不等于weight的长度；<br>4. splitItem为1的场景，x的长度不等于1；<br>5. splitItem为2的场景，y的长度不等于1；<br>6. splitItem为1、3的场景，x和y的长度不等于1；<br>7. reduceOp不等于“sum”；<br>8. commTurn不等于0；<br>9. streamMode不等于1。</td>
    </tr>
  </tbody></table>


## aclnnGroupedMatMulAllReduce

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatMulAllReduceGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见aclnn返回码。

## 约束说明

- 确定性计算：
  - aclnnGroupedMatMulAllReduce默认确定性实现。

- 数据类型组合约束：x、weight、bias支持的数据类型组合为：
  - “x-FLOAT16、weight-FLOAT16、bias-FLOAT16”
  - “x-BFLOAT16、weight-BFLOAT16、bias-FLOAT32”
- 维度约束：
  - 当splitItem为0时，x支持2-6维，y支持2-6维；
  - 当splitItem为1/2/3时，x支持2维，y支持2维；
  - 无论splitItem取值如何，weight均支持2维。
- 设备数量约束：支持2、4、8卡部署。
- 张量维度大小约束：
  - x和weight中每一组tensor的最后一维大小都应小于65536（x的最后一维：transpose_x为false时指K轴，为true时指M轴；weight的最后一维：transpose_weight为false时指N轴，为true时指K轴）。
  - x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32最大值（2147483647）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

```Cpp
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <memory>
#include <thread>
#include <iomanip>
#include <cassert>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include "hccl/hccl.h"
#include <hccl/hccl_types.h>

#include "acl/acl.h"
#include "acl/acl_base.h"
#include "aclnnop/aclnn_grouped_mat_mul_all_reduce.h"
using namespace std;

#ifdef __aarch64__
    #define gettid() syscall(SYS_gettid)
#endif
#define N 128

struct TensorInfo {
    std::string name;
    std::string dtype;
    std::string data_file;
    std::vector<int64_t> shape;
};

struct TensorsInfo {
    std::vector<TensorInfo> x_array;
    std::vector<TensorInfo> w_array;
    std::vector<TensorInfo> b_array;
    std::vector<TensorInfo> grouplist_array;
    std::vector<TensorInfo> y_array;
};

std::map<string, aclDataType> DtypeMap = {
    {"float16", ACL_FLOAT16},
    {"bfloat16", ACL_BF16},
    {"int64", ACL_INT64},
    {"uint64", ACL_UINT64}
};
thread_local int g_dev_id;
int32_t g_ndev = 0;
string input_dir = "./golden";
string output_dir = "./output";
string json_path;
vector<int> device_list;

int32_t loop = 1;

void set_device_id(int id)
{
    g_dev_id = id;
}
int get_device_id(void)
{
    return g_dev_id;
}

typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

#define ACL_CHECK(ret)                                                                                     \
    do {                                                                                                   \
        auto retcode = ret;                                                                                \
        if (retcode != ACL_SUCCESS) {                                                                      \
            printf("[ERROR] acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, retcode); \
            return retcode;                                                                                \
        }                                                                                                  \
    } while (0)

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)


struct DataSize {
    u64 min_bytes;
    u64 max_bytes;
    u64 step_bytes = 0;
    double step_factor;
    u64 count;
    u64 data_size;
    u64 type_size;
    int op_flag;
    void Print()
    {
        printf("acl min_bytes : %lld, max_bytes: %lld, step_bytes: %lld, step_factor: %f, count: %lld, data_size: "
               "%lld,  type_size: %lld\n",
            min_bytes,
            max_bytes,
            step_bytes,
            step_factor,
            count,
            data_size,
            type_size);
    }
};

struct Resource {
    aclrtStream rtStream;
    aclrtEvent startEvent;
    aclrtEvent endEvent;
    aclrtContext context;
  };

struct Args {
    int ndev;
    int rankId;
    uint32_t logicDeviceId;
    uint32_t rootRank;
    HcclComm hcclComm;
    Resource resources;
    DataSize dataPara;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    std::string dtype;
    std::string bin_path;
    std::string run_type;
    uint32_t loop_cnt;
    int bias_flag;
    std::map<std::string, int64_t> addrMap;
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtContext* context, aclrtStream* stream) {
  // 固定写法，资源初始化
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

int ReleaseAddr(vector<TensorInfo>& tensorList, map<string, int64_t> &addrMap) {
    int len = tensorList.size();
    if (len == 0 || (len == 1 && tensorList[0].shape.size() == 0)) {
        return 0;
    }
    for (size_t j=0; j < len; ++j) {
        string name = tensorList[j].name;
        if (addrMap.find(name) != addrMap.end()) {
            aclrtFree((void*)addrMap.at(name));
        }
    }
    return 0;
}

int DestroyTensor(void* items[], int item_type[], int num) {
  for (size_t i = 0; i < num; ++i) {
    if (items[i] != nullptr) {
      continue;
    }
    if (item_type[i] == 0) {
      aclDestroyTensor((aclTensor*)items[i]);
    } else if (item_type[i] == 1) {
      aclDestroyTensorList((aclTensorList*)items[i]);
    } else {
      aclDestroyIntArray((aclIntArray*)items[i]);
    }
  }
  return 0;
}

int CreateAclTensor(std::vector<int64_t>& shape,  std::string& name, std::string& dataTypeStr,
                    void*& deviceAddr, void*& tensor, bool int_array, int rankId=0) {
  uint32_t axis = 0;
  uint32_t byteBlock = 0;
  auto dataType = DtypeMap[dataTypeStr];
  uint32_t dataTypeSize = aclDataTypeSize(dataType);
  if (name.compare(0, 1, "x") == 0) {
    if (shape[1] % g_ndev != 0) {
      printf("[ERROR] X_k cannot be divided by ndev!\n");
      return -1;
    } 
    shape[1] /= g_ndev;
    axis = 1;
    byteBlock = shape[1] * dataTypeSize;
  } else if (name.compare(0, 6, "weight") == 0) {
    if (shape[0] % g_ndev != 0) {
      printf("[ERROR] Weight_k cannot be divided by ndev!\n");
      return -1;
    } 
    shape[0] /= g_ndev;
    axis = 0;
    byteBlock = shape[0] * dataTypeSize;
  } 
  auto size = GetShapeSize(shape) * dataTypeSize;
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(&deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
  uint8_t* hostData = new (std::nothrow) uint8_t[size];
  if (int_array) {
    tensor = (void*)aclCreateIntArray((int64_t*)hostData, size / sizeof(int64_t));
    delete[] hostData;
    return 0; 
  }
  ret = aclrtMemcpy(deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  tensor = (void*)aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,shape.data(), shape.size(), deviceAddr);
  return 0;
}

int CreateTensor_(int i, vector<TensorInfo>& tensorList, map<string, int64_t> &addrMap, void* items[], int item_type[], bool copy, int rankId) {
    int len = tensorList.size();
    if (len == 0 || (len == 1 && tensorList[0].shape.size() == 0)) {
        items[i] = nullptr;
        item_type[i] = 1;
        return 0;
    }
    void* tensors[len];
    for (int j = 0; j < len; ++j) {
        string name = tensorList[j].name;
        vector<int64_t> shape = tensorList[j].shape;
        string dtype = tensorList[j].dtype;
        void* deviceAddr = nullptr;
        auto ret = CreateAclTensor(shape, name, dtype, deviceAddr, tensors[j], false, rankId);
        addrMap[name] = (int64_t)deviceAddr;
        if (ret != 0) {return -1;}
    }
    items[i] = (void*)aclCreateTensorList((aclTensor**)tensors, len);
    item_type[i] = 1;
    return 0;
}


int CreateTensor(TensorsInfo& config, map<string, int64_t> &addrMap, void* items[], int item_type[], bool copy, int rankId) {
    int ret = 0;
    if (!copy) {
        ret = CreateTensor_(0, config.y_array, addrMap, items, item_type, copy, rankId);
        if (ret != 0) {return -1;}
    } else {
        ret = CreateTensor_(0, config.x_array, addrMap, items, item_type, copy, rankId);
        if (ret != 0) {return -1;}
        ret = CreateTensor_(1, config.w_array, addrMap, items, item_type, copy, rankId);
        if (ret != 0) {return -1;}
        ret = CreateTensor_(2, config.b_array, addrMap, items, item_type, copy, rankId);
        if (ret != 0) {return -1;}

        bool int_array = true;
        void* deviceAddr = nullptr;
        TensorInfo& grouplist = config.grouplist_array[0];
        ret = CreateAclTensor(grouplist.shape, grouplist.name, grouplist.dtype, deviceAddr, items[3], int_array, rankId);
        addrMap[grouplist.name] = (int64_t)deviceAddr;
        item_type[3] = 2;
        if (ret != 0) {return -1;}
    }
    return 0;
}

int PrepareDeviceMemAndData(TensorsInfo& config, std::map<std::string, int64_t>& addrMap, void** inputs, void** outputs, int* input_type, int* output_type, int rankId) {
  int ret = 0;
  ret = CreateTensor(config, addrMap, inputs, input_type, true, rankId);
  CHECK_RET(ret == ACL_SUCCESS, printf("CreateTensor Failed\n"); return -1);
  ret = CreateTensor(config, addrMap, outputs, output_type, false, rankId);
  CHECK_RET(ret == ACL_SUCCESS, printf("CreateTensor Failed\n"); return -1);
  return 0;
}

typedef int32_t rtError_t;
typedef void *rtNotify_t;
typedef void *rtStream_t;

// init hccl args end
extern "C" rtError_t rtNotifyWait(rtNotify_t notify, rtStream_t stm);
extern "C" rtError_t rtNotifyRecord(rtNotify_t notify, rtStream_t stm);
extern "C" rtError_t rtNotifyCreate(int32_t deviceId, rtNotify_t *notify);
extern "C" HcclResult HcclCreateComResource(char *commName, uint32_t streamMode, void **commContext);
extern "C" HcclResult HcclGetAicpuOpStreamNotify(char *commName, rtStream_t *Opstream, void **aicpuNotify);

int callMatmulAndAicpu(HcclComm hcclComm, void **inputs, void **outputs, int* input_type, int* output_type, DataSize &dataPara, aclrtStream rtStream, aclrtEvent startEvent, aclrtEvent endEvent, int checkErr, int ndev, int rankId, int deviceId, int rootRank, std::string dtype, std::map<std::string, int64_t>& addrMap)  // 主函数
{
    struct timespec tp1, tp2;
    long cost;
    clock_gettime(CLOCK_MONOTONIC, &tp1);
    set_device_id(deviceId);

    // **********Add Hcommname start********
    char hcom_name[N];
    auto hccl_name_ret = HcclGetCommName(hcclComm, hcom_name);
    CHECK_RET(
        hccl_name_ret == ACL_SUCCESS, printf("[ERROR] HcclGetCommName failed. hccl_name_ret = %d \n", hccl_name_ret);
        return -1);
    printf("rank %d hcom name is %s\n", rankId, hcom_name);
    // **********Add Hcommname end*********

    void *workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    int64_t split_item = 0;
    inputs[3] = nullptr;
    int64_t comm_turn = 0;
    
    auto aclnnRet = aclnnGroupedMatMulAllReduceGetWorkspaceSize(
        (aclTensorList*)inputs[0], (aclTensorList*)inputs[1], (aclTensorList*)inputs[2], (aclIntArray*)inputs[3], split_item,  hcom_name, "sum", comm_turn, 1, (aclTensorList*)outputs[0], &workspaceSize, &executor);
    CHECK_RET(aclnnRet == ACL_SUCCESS,
                printf("[ERROR] aclnnGroupedMatMulAllReduceGetWorkspaceSize failed. aclnnRet = %d \n", aclnnRet);
                return -1);
    printf("[INFO] gmm_allreduce workspaceSize = %lu\n", workspaceSize);
    if (workspaceSize > 0) {
        ACL_CHECK(aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    aclnnRet = aclnnGroupedMatMulAllReduce(workspaceAddr, workspaceSize, executor, rtStream);
    CHECK_RET(aclnnRet == ACL_SUCCESS, printf("[ERROR] aclnnGroupedMatMulAllReduce failed. aclnnRet = %d \n", aclnnRet);
                return -1);
    ACL_CHECK(aclrtSynchronizeStreamWithTimeout(rtStream, 10000));
    if (workspaceAddr != nullptr) {
      auto ret = aclrtFree(workspaceAddr);
      CHECK_RET(ret == ACL_SUCCESS, printf("[ERROR] aclnnFree workspaceAddr failed. ret = %d \n", ret);
                return -1);
    }

    clock_gettime(CLOCK_MONOTONIC, &tp2);
    if (tp2.tv_sec != tp1.tv_sec) {
        cost = tp2.tv_nsec - tp1.tv_nsec + 1000000000;
    } else {
        cost = tp2.tv_nsec - tp1.tv_nsec;
    }
    printf("[INFO] mc2 costtime = %lu, deviceid = %d\n", cost, rankId);

    return 0;
}

int launchOneThread(Args &args, TensorsInfo& config)
{
    std::string thread_name = "test" + std::to_string(args.logicDeviceId);
    prctl(PR_SET_NAME, thread_name.c_str());
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);

    int numa_node = args.logicDeviceId;
    int cpu_num = 8;
    int cpu_start = cpu_num * args.rankId;
    for (int i = cpu_start; i < cpu_start + cpu_num; i++) {
        CPU_SET(i, &cpu_set);
    }
    int ret = sched_setaffinity(gettid(), sizeof(cpu_set), &cpu_set);
    if (ret) {
        printf("failed to set cpu affinity, errno = %d\n", errno);
        return -1;
    }
    printf("0 rankId: %d, stream: %p, context : %p\n",
        args.logicDeviceId,
        args.resources.rtStream,
        args.resources.context);

    DataSize dataPara = args.dataPara;
    int checkErr = 0;
    ACL_CHECK(aclrtSetDevice(args.logicDeviceId));
    ACL_CHECK(aclrtSetCurrentContext(args.resources.context));
    
    constexpr int input_num = 4;
    void* inputs[input_num];
    int input_type[input_num];
    constexpr int output_num = 1;
    void* outputs[output_num];
    int output_type[output_num];
    args.addrMap.clear();
    ret = PrepareDeviceMemAndData(config, args.addrMap, inputs, outputs, input_type, output_type, args.rankId);
    CHECK_RET(ret == 0, return ret);

    for (int i = 0; i < args.loop_cnt; i++) {
        printf("======Startloop: %d / %d ============\n", i, args.loop_cnt);

        ret = callMatmulAndAicpu(args.hcclComm,
            inputs,
            outputs,
            input_type,
            output_type,
            dataPara,
            args.resources.rtStream,
            args.resources.startEvent,
            args.resources.endEvent,
            checkErr,
            args.ndev,
            args.logicDeviceId,
            args.rankId,
            args.rootRank,
            args.run_type,
            args.addrMap);
        if (ret != 0) {
            printf("TestCall execute AICPU failed, Detailed logs are stored in path: /root/ascend/log/");
            return ret;
        }
    }
    // save output
    // if (args.logicDeviceId == args.rootRank) {
    //   SaveTensor(config["outputs"], args.addrMap, output_dir);
    // }
    ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.resources.rtStream, 10000));
    DestroyTensor(inputs, input_type, input_num);
    DestroyTensor(outputs, output_type, output_num);
    ReleaseAddr(config.x_array, args.addrMap);
    ReleaseAddr(config.w_array, args.addrMap);
    ReleaseAddr(config.b_array, args.addrMap);
    ReleaseAddr(config.grouplist_array, args.addrMap);
    ReleaseAddr(config.y_array, args.addrMap);

    ACL_CHECK(aclrtSynchronizeStreamWithTimeout(args.resources.rtStream, 10000));
    ACL_CHECK(aclrtDestroyStreamForce(args.resources.rtStream));
    // 销毁集合通信域
    //  HCCLCHECK(HcclCommDestroy(args.hcclComm));
    HcclCommDestroy(args.hcclComm);
    // 重置设备
    ACL_CHECK(aclrtResetDevice(args.logicDeviceId));
    return 0;
}

int launchMultiThread(Args &input_args, int32_t *devices, HcclComm *comms, Resource *resources)
{
    DataSize dataPara;
    dataPara.data_size = 10485760;
    dataPara.type_size = sizeof(uint16_t);
    dataPara.count = (dataPara.data_size + sizeof(uint16_t) - 1) / sizeof(uint16_t);  // data->count;
    dataPara.step_factor = 1;                                                         // data->step_factor;
    dataPara.step_bytes = 1;                                                          // data->step_bytes;
    dataPara.max_bytes = 10485760;                                                    // data->max_bytes;
    dataPara.min_bytes = 10485760;                                                    // data->min_bytes;
    dataPara.op_flag = 0;                                                             // op_flag;
    input_args.bin_path = "";
    input_args.dtype = "float16";

    TensorsInfo config;
    TensorInfo x0_json;
    x0_json.name = "x0";
    x0_json.dtype = "float16";
    x0_json.data_file = "x_0.bin";
    x0_json.shape = vector<int64_t>{256,256};
    TensorInfo x1_json;
    x1_json.name = "x1";
    x1_json.dtype = "float16";
    x1_json.data_file = "x_1.bin";
    x1_json.shape = vector<int64_t>{1024,256};
    config.x_array.push_back(x0_json);
    config.x_array.push_back(x1_json);
    
    TensorInfo w0_json;
    w0_json.name = "weight0";
    w0_json.dtype = "float16";
    w0_json.data_file = "w_0.bin";
    w0_json.shape = vector<int64_t>{256,256};
    TensorInfo w1_json;
    w1_json.name = "weight1";
    w1_json.dtype = "float16";
    w1_json.data_file = "w_1.bin";
    w1_json.shape = vector<int64_t>{256, 1024};
    config.w_array.push_back(w0_json);
    config.w_array.push_back(w1_json);

    TensorInfo b0_json;
    b0_json.name = "b0";
    b0_json.dtype = "float16";
    b0_json.data_file = "b_0.bin";
    b0_json.shape = vector<int64_t>{256};
    TensorInfo b1_json;
    b1_json.name = "b1";
    b1_json.dtype = "float16";
    b1_json.data_file = "b_1.bin";
    b1_json.shape = vector<int64_t>{1024};
    config.b_array.push_back(b0_json);
    config.b_array.push_back(b1_json);
    
    
    TensorInfo group_list_json;
    group_list_json.name = "group_list";
    group_list_json.dtype = "int64";
    group_list_json.data_file = "";
    group_list_json.shape = vector<int64_t>{};
    config.grouplist_array.push_back(group_list_json);
    
    TensorInfo y0_json;
    y0_json.name = "y0";
    y0_json.dtype = "float16";
    y0_json.data_file = "y_0.bin";
    y0_json.shape = vector<int64_t>{256, 256};
    TensorInfo y1_json;
    y1_json.name = "y1";
    y1_json.dtype = "float16";
    y1_json.data_file = "y_1.bin";
    y1_json.shape = vector<int64_t>{1024, 1024};
    config.y_array.push_back(y0_json);
    config.y_array.push_back(y1_json);

    uint32_t ndev = input_args.ndev;
    Args args[ndev];
    aclrtStream rtStream[ndev];
    aclrtContext context[ndev];
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        ACL_CHECK(aclrtSetDevice(devices[rankId]));
        ACL_CHECK(aclrtGetCurrentContext(&context[rankId]));
        ACL_CHECK(aclrtCreateStream(&rtStream[rankId]));
        // printf("1 rankId: %d, stream: %p\n", rankId,  rtStream[rankId]);
    }

    std::vector<std::unique_ptr<std::thread>> threads(ndev);
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].rootRank = devices[0];
        args[rankId].ndev = ndev;
        args[rankId].logicDeviceId = devices[rankId];
        args[rankId].hcclComm = comms[rankId];
        args[rankId].resources = resources[rankId];
        args[rankId].dataPara = dataPara;
        args[rankId].resources.rtStream = rtStream[rankId];
        args[rankId].resources.context = context[rankId];
        args[rankId].m = input_args.m;
        args[rankId].k = input_args.k;
        args[rankId].n = input_args.n;
        args[rankId].dtype = input_args.dtype;
        args[rankId].bin_path = input_args.bin_path;
        args[rankId].run_type = input_args.run_type;
        args[rankId].loop_cnt = input_args.loop_cnt;
        args[rankId].bias_flag = input_args.bias_flag;
        threads[rankId].reset(new (std::nothrow) std::thread(&launchOneThread, std::ref(args[rankId]), std::ref(config)));
    }
    for (uint32_t rankId = 0; rankId < ndev; rankId++) {
        threads[rankId]->join();
    }
    return 0;
}

int InitParams(int argc, char** argv) {
  if (argc < 4) {
    LOG_PRINT("miss paramaters\n");
  }
  g_ndev = std::stoi(argv[1]);
  loop = std::stoi(argv[2]);
  std::stringstream str = std::stringstream(argv[3]);
  std::string seg;
  char c = ',';
  while(std::getline(str, seg, c)) {
    device_list.push_back(std::stoi(seg));
  }
  assert (g_ndev == device_list.size());
  return 0;
}


int main(int argc, char *argv[])
{
    // 本样例基于Atlas A3实现，必须在Atlas A3上运行
    // usage: ./main 2 1 0,1
    int ret = 0;
    CHECK_RET(InitParams(argc, argv) == 0, LOG_PRINT("init params failed.\n"); return 0);
    Args input_args;
    input_args.ndev = g_ndev;
    input_args.run_type = string("reduce");
    input_args.loop_cnt = loop;

    if (input_args.ndev != 1 && input_args.ndev != 2 && input_args.ndev != 4 && input_args.ndev != 8) {
        printf("device_num input error, only support 8,4,2,1.\n");
        return -1;
    }
    int32_t devices[input_args.ndev];
    for (int i = 0; i < input_args.ndev; i++) {
        devices[i] = device_list[i];
    }

    HcclComm comms[N];
    Resource resources[N];

    ACL_CHECK(aclInit(NULL));
    for (int i = 0; i < 1; i++) {
        // 初始化集合通信域
        for (int i = 0; i < input_args.ndev; i++) {
            ACL_CHECK(aclrtSetDevice(devices[i]));
        }
        ret = HcclCommInitAll(input_args.ndev, devices, comms);
        if (ret != 0) {
            printf("This is an error in init_hcclComm.\n");
            return -1;
        }

        // 启动测试
        ret = launchMultiThread(input_args, devices, comms, resources);
        if (ret != 0) {
            printf("This is an error in opbase_test_by_data_size.\n");
            // ACL_CHECK(aclFinalize());
            return -1;
        }
    }
    ACL_CHECK(aclFinalize());
    return ret;
}
```

