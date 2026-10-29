# PAG 接入 Knowhere

## 目的

在 Knowhere 中增加一个静态 fp32 索引，类型名是 `PAG`。它能 `Build`、`Search`，并能把整份索引放进 `BinarySet` 的一条记录。

PAG 算法库按 Faiss 的方式放进 `knowhere/thirdparty/pag`，编成静态库后由 `libknowhere` 链接。PAG 自己拥有内存读写器，不引用 Knowhere 的头文件。Knowhere 只在 `src/index/pag` 里写 `IndexNode` 适配层。

仓库根目录的 `PAG/` 保持现有的 AVX-512 实现，这次不拆内核，也不加 AVX2。Knowhere 拷贝的就是这一版。

## 不在范围内

- 在线插入，以及 `Train` / `Add`
- bitset 过滤、范围搜索、迭代器、按 ID 取原始向量
- fp16、bf16、int8、binary
- AVX2、标量、SSE、NEON、RVV 后端，以及按 CPU 选择内核
- 把根目录 `PAG/` 的 CLI、Python、examples、docs 拷进 Knowhere
- 让拷贝之后的两份 PAG 自动同步

## 与 IVF、HNSW 的对应

IVF 和 HNSW 都有 `Build`。它们没有自己重写这个函数，走的是 Knowhere 的默认实现：对同一批数据先 `Train`，再 `Add`。PAG 的 `Build` 对应的就是这一次完整的 `Build`，也就是 `Train` 加 `Add`，不能拆开，也不能之后再 `Add`。

| 功能 | IVF | HNSW | PAG | 对应关系 |
|---|---|---|---|---|
| `Build` | 有。默认实现，等于 `Train` 再 `Add` | 有。同样是 `Train` 再 `Add` | 有。自己实现，一次建完静态图 | PAG 的 `Build` 对应 IVF / HNSW 对全量数据调用一次 `Build` |
| `Train` | 有。学习聚类中心，倒排表仍是空的 | 有。Flat 基本只标记已训练，图还没建 | 未实现 | 没有单独对应。这部分工作已经包含在 PAG 的 `Build` 里 |
| `Add` | 有。把向量写入倒排表，可以多次追加 | 有。真正建图和后续插入都在这里 | 未实现 | 没有单独对应。全量插入包含在 PAG 的 `Build` 里，之后不能再追加 |
| `Search` | 有，支持 bitset | 有，支持 bitset | 有。只支持空 bitset 的 top-k | PAG 的 `Search` 对应 IVF / HNSW 传入空 bitset 的 `Search` |
| bitset 过滤 | 在 `Search` 里支持 | 在 `Search` 里支持 | 非空 bitset 返回未实现 | 没有对应实现 |
| `RangeSearch` | 有 | 有 | 未实现 | 没有对应实现 |
| `AnnIterator` | 有 | 有 | 未实现 | 没有对应实现 |
| `Serialize` / `Deserialize` | 有。写入 `BinarySet` | 有。写入 `BinarySet` | 有。一条名为 `"PAG"` 的记录 | 三者对应的是同一条内存序列化路径 |
| `DeserializeFromFile` | 有。从文件 mmap 加载 | 有。从文件 mmap 加载 | 未实现 | 没有对应实现。PAG 只能走 `Deserialize` |
| `GetVectorByIds` | IVF Flat 有；PQ、SQ 这类不保存原向量的类型没有 | 有。图里保存了可还原的原向量时可用 | 未实现 | 没有对应实现 |
| `HasRawData` | IVF Flat 为 true，压缩类型为 false | 可还原时为 true | 固定 false | 表示 PAG 不保留可按 ID 读回的原向量 |
| `GetIndexMeta` | IVF Flat 有 | 未实现 | 未实现 | 没有对应实现 |
| `CalcDistByIDs` | IVF Flat、IVF FlatCC 有 | 有 | 未实现 | 没有对应实现 |
| `GetQueryCodeSize` | 有 | 有 | 未实现，调用会抛异常 | 没有对应实现 |
| `Dim` / `Count` / `Size` / `Type` | 有 | 有 | 有 | 一一对应。PAG 的 `Size` 是打包后的 `index.bin` 字节数 |
| 数据类型 | fp32、fp16、bf16、int8、binary | fp32、fp16、bf16、int8、binary | 只有 fp32 | PAG 只对应两边的 fp32 版本 |
| 距离 | `L2`、`IP`、`COSINE` 等 | `L2`、`IP`、`COSINE` 等 | `L2`、`IP`、`COSINE` | 这三个距离一一对应 |

IVF 的 `Train` 学的是聚类中心，`Add` 才把向量放进倒排表。HNSW 的 `Train` 几乎不做建图，真正建图在 `Add`。PAG 没有这个两段式过程，调用方要对全量数据调用一次 `Build`。

## 指令集

初期只支持 AVX-512。`_mm512_*` 仍留在 `space_l2.h`、`space_ip.h`、`pag_search_primitives.h` 和 `pag_index_core.h` 里。静态库 `pag` 整体使用这些编译选项：

```text
-mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -mavx512vnni
```

这些选项只加在 `pag` 上，不加到整个 `libknowhere`。编译器不支持其中任一选项时，不创建 `pag` 目标，也不注册 `PAG`。`__X86_64` 为假时同样不编。

`PagIndexNode` 在 `Build` 和 `Deserialize` 进入 PAG 之前检查 CPU。需要同时具备 AVX512F、AVX512DQ、AVX512CD、AVX512BW、AVX512VL、AVX512VNNI。前五项用 Knowhere 已有的 `InstructionSet`。AVX512VNNI 读取 CPUID leaf 7 的 ECX bit 11，因为 `InstructionSet` 目前没有这一项。缺任何一项就返回 `Status::invalid_instruction_set`，不调用 `pag::Index`。这里只拒绝运行，不提供 AVX2 实现。

## 源码放置

拷进 `knowhere/thirdparty/pag/` 的文件：

- `pag.cpp`
- `pag_index.h`
- `pag_config.h`
- `paglib/` 下的全部头文件
- `cmake/PAGCompilerOptions.cmake`

不拷 `main.cpp`、`python/`、`examples/`、`docs/`。

`cmake/libs/libpag.cmake` 在 `__X86_64` 为真、且编译器支持上一节的 AVX-512 选项时，把这些源文件编成一个静态库 `pag`，并链接 OpenMP。`knowhere/CMakeLists.txt` 在链接 Faiss 的同一处把 `pag` 加入 `KNOWHERE_LINKER_LIBS`，并把 `thirdparty/pag` 加入 include 路径。

`src/index/*.cc` 会被全局搜集。`__X86_64` 为假，或编译器不支持这些 AVX-512 选项时，从 `KNOWHERE_SRCS` 去掉 `src/index/pag/*.cc`，不创建 `pag` 目标，也不注册索引类型。这与 DiskANN 源文件在未启用时被移出编译列表的做法相同。

## PAG 内存序列化

改动只发生在 `knowhere/thirdparty/pag`。目录接口保持拷贝来源的行为：

- `build`、`load(LoadOptions)`、`save()` 的签名不变
- `index_path` 仍必填，为空时仍抛出原来的异常
- 目录里仍写出 `index.bin`、`info.bin`、`permutation.bin`，这三段的字节与拷贝来源一致
- 现有 `save()` 对静态索引仍是空操作，只在在线索引收尾时写目录

新增类型和接口：

```cpp
namespace pag {

struct MemoryIOWriter {
    uint8_t* data();
    size_t tellg() const;
    std::pair<uint8_t*, size_t> release();
    size_t operator()(const void* ptr, size_t size, size_t nitems);
    template <typename T>
    size_t write(T* ptr, size_t size, size_t nitems = 1);
};

struct MemoryIOReader {
    MemoryIOReader(uint8_t* data, size_t size);
    size_t operator()(void* ptr, size_t size, size_t nitems);
    template <typename T>
    size_t read(T* ptr, size_t size, size_t nitems = 1);
    size_t tellg() const;
    void seekg(size_t pos);
};

void Index::save(MemoryIOWriter& writer) const;
void Index::load(MemoryIOReader& reader);

size_t Index::count() const;
size_t Index::size_bytes() const;
size_t Index::max_search_k() const;

}
```

`MemoryIOWriter` 按宿主字节序追加到一块可增长的 `uint8_t` 缓冲，不做字节交换。这样写进缓冲的三段与目录文件的字节相同。`release()` 交出指针和 `tellg()`，调用后 `data()` 为空，调用方获得所有权。Knowhere 用 `std::shared_ptr<uint8_t[]>` 接管该指针。没有调用 `release()` 就析构时，析构函数释放缓冲。`load` 不读取 `index_path`，metric 来自缓冲本身。

`save(MemoryIOWriter&)` 可以在静态索引已经载入内存后调用。它从内存写出，不重新打开目录。`size_bytes()` 返回 packed 图的字节数，与 `index.bin` 等长。`count()` 返回当前向量数。`max_search_k()` 返回构建时写入索引的 `max_query_top_k`。未加载时这三个访问器返回 0。

单块布局：

```text
char magic[4] = {'P', 'A', 'G', '1'}
uint32_t version = 1
uint32_t metric          // 0 = L2, 1 = Cosine, 2 = MaximumInnerProduct
uint64_t info_nbytes
uint8_t  info[info_nbytes]                 // 与 info.bin 相同
uint64_t index_nbytes
uint8_t  index[index_nbytes]               // 与 index.bin 相同
uint64_t permutation_nbytes
uint8_t  permutation[permutation_nbytes]   // 与 permutation.bin 相同
```

写这三段的函数只有一套。静态 `build()` 在打包后用这套函数写入三个文件，再从该目录重新加载。在线 `save()` 收尾时也用这套函数写三个文件。`save(writer)` 从已经在内存中的索引按上面的顺序写出单块，不调用现有的 `save()`。

`load(reader)` 在以下情况抛出异常：魔数不是 `PAG1`，版本不是 1，metric 不是 0、1、2，某一段长度超出剩余字节。成功后索引可直接搜索。

## Knowhere IndexNode

文件：

- `include/knowhere/comp/index_param.h` 增加 `INDEX_PAG = "PAG"`
- `src/index/pag/pag_config.h`
- `src/index/pag/pag.cc`

`PagIndexNode` 实现 `IndexNode`，只注册 fp32：

```cpp
KNOWHERE_SIMPLE_REGISTER_GLOBAL(PAG, PagIndexNode, fp32, knowhere::feature::FLOAT32)
```

`Build` 被覆盖，不再调用 `Train` 和 `Add`。`Train`、`Add`、`RangeSearch`、`AnnIterator`、`GetVectorByIds`、`GetIndexMeta`、`DeserializeFromFile` 返回 `Status::not_implemented`。`HasRawData` 返回 false。

`Build` 忽略 `use_knowhere_build_pool`。PAG 构建和批量搜索使用自己的 OpenMP。`Search` 对整批查询调用一次 `search_batch`，不把单条查询提交到 Knowhere 的搜索线程池。

`Build` 的数据规则：

- 张量是 fp32、行主序
- 行数必须能放进 `uint32`
- 数据集没有 ID 时，标签就是 `0 .. n-1`
- 数据集有 ID 时，第 `i` 个 ID 必须等于 `i`
- 违反上述规则返回 `Status::invalid_args`

因为现有 `pag::Index::build` 仍要求目录，包装层在系统临时目录下创建 `pag-build-<pid>-<64 位随机数>`，传给 `index_path`。并发的 `Build` 不共享目录。`build` 返回后 `pag::Index` 已在内存中可搜索。无论成功或失败，都删除该临时目录。再次 `Build` 替换节点里原来的 `pag::Index`。

`Dim()`、`Count()`、`Size()` 在未加载时返回 0。加载后分别返回 `dimension()`、`count()`、`size_bytes()`。`Type()` 返回 `"PAG"`。`CreateConfig()` 返回 `PagConfig`。

`Serialize` 在索引为空时返回 `Status::empty_index`。否则调用 `pag::Index::save(writer)`，用 `release()` 接管缓冲，以 `"PAG"` 为键放进 `BinarySet`。只放这一条。

`Deserialize` 读取键 `"PAG"`。缺失时返回 `Status::invalid_binary_set`。用 `pag::MemoryIOReader` 调用 `pag::Index::load`，替换节点里的旧索引。反序列化不使用配置里的 `metric_type`。之后的 `Search` 若配置 metric 与 `pag::Index::metric()` 不一致，返回 `Status::invalid_metric_type`。

`Build`、`Search`、`Serialize`、`Deserialize` 捕获 PAG 抛出的异常，返回 `Status::pag_inner_error`。临时目录创建或删除失败返回 `Status::disk_file_error`。指令集不足时按上一节返回 `Status::invalid_instruction_set`，此时还没有调用 PAG。

`Status` 增加 `pag_inner_error = 33`。`Status2String` 返回 `"pag inner error"`。

## 配置

`PagConfig` 继承 `BaseConfig`。默认值用 PAG 自己的，不用 HNSW 的。

| 字段 | 传给 PAG | 默认 | 约束 | 阶段 |
|---|---|---|---|---|
| `M` | `target_degree` | 16 | `[1, 2048]` | train |
| `efConstruction` | `ef_construction` | 200 | `[1, INT_MAX]` | train |
| `ef` | `ef_search` | 搜索时缺省为 `max(k, 16)` | 缺省允许为空；有值时 `>= k` | search |
| `projection_levels` | `projection_levels` | 64 | 正数且是 8 的倍数 | train |
| `max_search_k` | `max_search_k` | 100 | `[1, INT_MAX]` | train |

`projection_levels` 不满足约束时，配置检查返回 `Status::invalid_param_in_json`，不调用 PAG。

PAG 内部可以把 `ef_construction` 提高到至少 `2 * target_degree`。包装层原样传递配置值，不再做一次同样的调整。

`metric_type` 只接受 `L2`、`IP`、`COSINE`。其它值返回 `Status::invalid_metric_type`。

| Knowhere | PAG |
|---|---|
| `L2` | `Metric::L2` |
| `IP` | `Metric::MaximumInnerProduct` |
| `COSINE` | `Metric::Cosine` |

`ef < k` 时返回 `Status::out_of_range_in_json`，规则与 HNSW 相同。

`max_search_k` 在 `Build` 时进入 PAG，并保存在索引里。`Search` 使用加载后的 `pag::Index::max_search_k()`，不用搜索配置里的同名字段。`k` 大于该值时返回 `Status::out_of_range_in_json`。

## 搜索结果

`Search` 在 `bitset.count() > 0` 时返回 `Status::not_implemented`。空 bitset 允许搜索。

距离与 Faiss HNSW 同向：

- `L2`：写入 `SearchResult.distance`，越小越近
- `IP` 和 `COSINE`：写入 `SearchResult.inner_product`，越大越近

ID 从 PAG 的 `uint32` 转成 `int64`。结果行数等于查询数，每行 `k` 个结果。

## 测试

只把「fp32 的 `Build`、空 bitset 的 top-k `Search`、`BinarySet` 序列化」能覆盖到的用例加进去。依赖 bitset、`RangeSearch`、`AnnIterator`、binary、fp16、bf16、mmap 的用例先不加入。那些 benchmark 在搜索失败时直接取 `result.value()`，不能写成“期望 `not_implemented`”。对应接口实现之后，再按同一张表补上。

`knowhere_tests` 里的用例要打开 `with_ut` 才会编。当前 `build-linux-test.sh` 不打开它。CPU 没有 AVX-512（含 VNNI）时，`test_pag.cc` 和 `test_search.cc` 里的 PAG 搜索跳过，类型检查仍执行。`__X86_64` 为假或编译器不支持 AVX-512 选项时，不定义 `KNOWHERE_WITH_PAG`，这些 PAG 断言和用例不参与编译。

| 位置 | 用例 | 处理 |
|---|---|---|
| `benchmark/hdf5/benchmark_float_qps.cpp` | `TEST_PAG` | 已加入。fp32，按召回率二分 `ef`，再测 QPS |
| `benchmark/hdf5/benchmark_float.cpp` | `TEST_PAG` | 已加入。fp32，扫 `ef`、`nq`、`k`，打印延迟和召回率。`max_search_k` 取 `TOPKs_` 的最大值 |
| `tests/ut/test_pag.cc` | `PAG build and search` 等 | 已有。进程内小数据集，覆盖 `L2`、`IP`、`COSINE`、序列化、过大的 `k`、非空 bitset、`Train`、非法 `projection_levels` |
| `tests/ut/test_search.cc` | `Test Mem Index With Float Vector` / `Test Search` | 已加入。只走 fp32 的 `Build`、`Serialize`、`Deserialize`、`Search`，`max_search_k` 为 128。无 AVX-512 时跳过 |
| `tests/ut/test_index_check.cc` | 数据类型、mmap、`HasRawData` | 已加入。断言只接受 fp32，不支持 mmap，`HasRawData` 为 false |
| `benchmark_float_qps.cpp`、`benchmark_float.cpp` | 同一用例里的 fp16、bf16 | 不加。PAG 没有这两种类型 |
| `benchmark_float.cpp` | `TEST_HNSW_SQ`、`TEST_HNSW_PQ`、`TEST_HNSW_PRQ` | 不加。PAG 没有量化精排 |
| `benchmark_float_bitset.cpp` | `TEST_IVF_*`、`TEST_HNSW`、`TEST_DISKANN` | 不加。非空 bitset 返回 `not_implemented` |
| `benchmark_float_range.cpp` | `TEST_IDMAP`、`TEST_IVF_*`、`TEST_HNSW`、`TEST_DISKANN` | 不加。`RangeSearch` 未实现 |
| `benchmark_float_range_bitset.cpp` | `TEST_IVF_*`、`TEST_HNSW`、`TEST_DISKANN` | 不加。范围搜索和 bitset 都未实现 |
| `benchmark_binary.cpp`、`benchmark_binary_range.cpp` | `TEST_BINARY_*` | 不加。PAG 没有 binary |
| `benchmark_simd_qps.cpp`、`benchmark_sparse` | SIMD、稀疏距离 | 不加。不是 PAG 这种稠密索引的用例 |
| `tests/ut/test_search.cc` | `Test Range Search`、binary、取消范围搜索 | 不加。`RangeSearch` 或 binary 未实现 |
| `tests/ut/test_iterator.cc` | 各 `AnnIterator` 用例 | 不加。迭代器未实现 |
| `tests/ut/test_index_node.cc` | embedding-list bitset | 不加。PAG 没有 embedding list，也不做 bitset |

`test_pag.cc` 在 `__X86_64` 为假，或编译器不支持 AVX-512 选项时，按 `test_diskann.cc` 的方式从测试源文件列表移除。功能覆盖仍是：

- `L2`、`IP`、`COSINE` 各完成一次 `Build` 和 `Search`，返回 `k` 个 ID
- `Serialize` 之后用新索引 `Deserialize`，同一组和 `ef` 的搜索 ID 与距离逐项相等
- `k > max_search_k` 返回 `Status::out_of_range_in_json`
- 非空 bitset 返回 `Status::not_implemented`
- `Train` 返回 `Status::not_implemented`
- `projection_levels` 不是 8 的倍数时，`Build` 返回 `Status::invalid_param_in_json`

在 AVX-512（含 VNNI）的机器上运行：

```bash
./dist-amd64/bin/benchmark_float_qps --gtest_filter=Benchmark_float_qps.TEST_PAG
./dist-amd64/bin/benchmark_float --gtest_filter=Benchmark_float.TEST_PAG
```

## 数据流

```text
DataSet
  -> PagIndexNode::Build
  -> 临时目录 + pag::Index::build
  -> 内存中的 pag::Index
  -> 删除临时目录

Search(DataSet)
  -> pag::Index::search_batch
  -> DataSet(ids, distances)

Serialize
  -> pag::Index::save(MemoryIOWriter)
  -> BinarySet["PAG"]

Deserialize
  -> BinarySet["PAG"]
  -> pag::Index::load(MemoryIOReader)
  -> 内存中的 pag::Index
```
