# 高性能向量检索

**项目目的:** 给N个向量(每个长度(元素个数)为d),当输入一个query向量时，库返回距离它最近的K个向量，并且在大N是查询速度明显快于暴力,但结果肯呢个不是100%准确的

**项目结构:**

 - `include\vecsearch/`
 - `src/`
 - `bench/`
 - `tests/`
 - `README.md`

**项目目标:**

写一个C++高性能向量检索,然后可以变成一个可以被调用的库



## 快速开始 (Quick Start)

以下示例展示了如何初始化索引、并发插入数据、冻结图结构并进行快速查询：

```cpp
#include <iostream>
#include <vector>
#include "vecsearch/hnsw_index.h"
#include "vecsearch/types.h" // 假设包含 IndexConfig 等定义

int main() {
    // 1. 配置参数
    vecsearch::IndexConfig cfg;
    cfg.dim = 128;                     // 向量维度
    cfg.metric = vecsearch::Metric::L2; // 距离度量方式

    vecsearch::HNSWParams params;
    params.M = 32;                     // 每层最大邻居数
    params.ef_construction = 200;      // 建图时的候选集大小
    params.ef_search = 400;            // 搜索时的候选集大小

    // 2. 初始化 HNSW 索引
    vecsearch::HNSWIndex index(cfg, params);

    // 3. 准备测试数据 (ID 与 扁平化的向量数据)
    int N = 10000;
    std::vector<vecsearch::Id> ids(N);
    std::vector<float> data(N * cfg.dim);
    for (int i = 0; i < N; ++i) {
        ids[i] = i;
        // 填充假数据...
        for(int j=0; j<cfg.dim; ++j) data[i * cfg.dim + j] = (float)rand() / RAND_MAX;
    }

    // 4. 批量添加数据 (内部已实现 OpenMP 多线程并发建图)
    index.add_batch(ids, data);

    // 5. 冻结索引 (重要！建库完成后调用，使查询变为无锁极速模式)
    index.freeze();

    // 6. 搜索
    std::vector<float> query(cfg.dim, 0.5f);
    int topK = 10;
    
    // 返回距离 query 最近的 topK 个邻居
    std::vector<vecsearch::Neighbor> results = index.search_one(query.data(), topK);

    for (const auto& res : results) {
        std::cout << "ID: " << res.id << ", L2 Distance Sqr: " << res.dist << "\n";
    }

    return 0;
}
```



## 编译与依赖环境(Build & Dependencies)

**环境要求:**

- C++标准: C++20及以上
- 构建工具: Cmake 3.30及以上
- 依赖库: OpenMP(多线程加速)必备

**Linux(GCC)**

CMake 会自动探测本机 CPU 架构，开启 `-O3`, `-march=native`, `-ffast-math` 优化。

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./bench_runner
```





**Windows(MSVC)**

如果在 Windows 环境下使用 Visual Studio，CMake 会自动附加 `/O2` 与 `/arch:AVX2` 参数以显式开启 SIMD 加速。

```DOS
mkdir build
cd build
cmake .. 
cmake --build . --config Release
.\Release\bench_runner.exe
```



## 4. 核心参数调优指南 (Tuning Guide)

构建高性能的 HNSW 索引需要在速度、内存与精度之间寻找平衡，主要由以下三个参数控制：

- **`M` (默认: 32)**:

  决定了每个节点在图中最多保留的邻居数量（第 0 层上限为 $2M$，高层为 $M$）。

  - *建议*: 对于大多数应用，设为 16~32 是速度与召回率的最佳甜点。如果数据维度极高或对 Recall 要求极度苛刻，可以设置为 48 或 64，但这会增加建库时间与内存消耗。

- **`ef_construction` (默认: 200)**:

  建图时搜索候选集的大小。值越大，图的连通性越好，最终查询的 Recall 越高，但建库时间会线性增加。

  - *建议*: 设置在 100 ~ 400 之间。如果发现 Recall 卡在 0.8 左右上不去，可以尝试调高此值。

- **`ef_search` (默认: 400)**:

  查询时探索的候选集大小。必须 $\geq topK$。值越大，查询越准，但耗时越长。

  - *建议*: 作为运行时动态权衡速度与精度的旋钮，通常设置为 topK 的 5~10 倍。





## 算法基础

### 1) 单层HNSW

#### **1.1**

`M`: 每个点最多保留多少条边(邻居上限)

`efConstruction`:插入时候选集合大小(大 $\rightarrow$ 图更好 $\rightarrow$ 建库慢)

`efSearch`:查询时候选集合大小(大$\rightarrow $ 更准 $\rightarrow$ 更慢)



#### **1.2**

单层HNSW核心： **SearchLayer** (公用的在图里找邻居)

- 插入时用（efConstruction）
- 查询时用（efSearch）

返回对目标向量 `target` 来说，在图里找到的一堆“比较近”的点（数量约 ef）

##### **数据结构**

- `candidates` :小根堆，按距离从小到大
- `best`: 大顶堆(保留当前最好的ef个，堆顶是最差的那个)
- `visited`: bool数组或者hashset或者bitset，防止重复访问

#### **1.3**

##### **SearchLayer**流程:

- 取出`candidates`里当前候选最近的点,剪枝:如果`candidates`里面最好的已经比best里最差的还差就 **break**
- 对于每一个邻居
  - 如果已经访问过`continue`
  - 否则求出距离:
    - 如果`best`还没满 $ef$ 个 $or$  当前节点(nb)比当前最差的还要好,就 **加入**
    - 否则 不操作
- 返回最终结果 

##### **插入/建图** 流程:

对于每个新节点 $p$:

**1.**  用 SearchLayer 在当前图里找候选邻居（efConstruction）

**2.**  在候选里挑最近的M个 (SelectNeighbors)

**3.**  建边: p连接这M个点 (双向先’

**4.** 对被连接的点做删减，邻接表长度保持($\leq M\ or\ \leq 2M$)



**SelectNeighbors**:

查询返回的结果进行排序的取前M个



**add_batch**

上伪代码吧，写着方便还可以照着实现

```cpp
function AddBatch(ids, vectors):
    if graph empty:
        init graph size = N //因为总数是n
        init data size = N

    for i in 0..n-1:
        id = ids[i]
        data[id] = vectors[i]         // 存向量
        graph[id] = empty list        //清空

        if id is first inserted point:
            entry = id
            continue

        //找候选邻居
        candidates = SearchLayer(data[id], entry, efConstruction)

        //选 M 个最近邻居
        neigh = SelectNeighbors(candidates, M)   // 返回一堆 neighbor ids

        //建边（双向）
        for each v in neigh:
            graph[id].push_back(v)
            graph[v].push_back(id)

            // 裁剪 v 的邻接表（防止无限增长）
            if size(graph[v]) > M:
                // 对 v 的邻接表按 dist(v, nb) 排序，保留最近 M 个
                graph[v] = PruneNeighborsByDistance(v, graph[v], M)

        // 裁剪 id 的邻接表（保证 <= M）
        if size(graph[id]) > M:
            graph[id] = PruneNeighborsByDistance(id, graph[id], M)

        // entry 可以固定不变，也可以更新成最近插入的点
        entry = id   
```

**PruneNeighborsByDistance**

先不上一些启发式优化

```cpp
function PruneNeighborsByDistance(center_id, neighbor_list, M):
    // neighbor_list: [id1, id2, ...]
    sort neighbor_list by dist(data[center_id], data[nb]) asc
    remove duplicates
    resize to M
    return neighbor_list
```



**查询:search_one**

单层的话比较简单:

**1.** 用SearchLayer(query,entery,efSearch)得到best(约efSearch个)

**2.**把 best 排序，取前 topK，返回

```cpp
function SearchOne(query_vec, topK):
    if size == 0: return empty
    ef = max(efSearch, topK)   // ef 不能小于 K

    best = SearchLayer(query_vec, entry, ef)

    list = best as vector
    sort list by dist asc
    return first min(topK, list.size) as results (id, dist)
```



**attention：**

- efsearch必须>=k不然没有足够的候选
- visited需要存在
- 剪裁是必须的不然会爆掉
- 入口entry(图里搜索的起点)先简单一点在这里没必要





### 2) 多层 HNSW 与启发式导航

为了在千万级甚至更大规模的数据集上打破单层图的召回率（Recall）瓶颈，项目实现了完整的多层 HNSW 结构，并引入了相对邻近图（RNG）的启发式边修剪策略。

#### **2.1 核心概念与层级**

- **层级分配 (`random_level_`)**: 节点插入时被随机分配一个最高层级 $I$。概率呈指数衰减，绝大多数节点在第 0 层，极少数节点在高层充当“高速公路”的入口。
- **动态度数**: 第 0 层保留双倍的候选边（$2M$），高层保留 $M$ 个，以在底部保证绝对精度，在顶部保证跳转速度。

#### **2.2 贪心降落 (Greedy Descent)**

**目的**: 在不维护复杂优先队列的情况下，从图的最高层快速“降落”到目标层级（通常是节点待插入的最高层），找到一个足够好的局部最优起点。

**作用**: 极大减少搜索前期的遍历开销，是多层图快速锁定目标区域的核心。

**伪代码:**

C++

```
function GreedyDescent(target_vec, entry_id, from_level, to_level):
    curr = entry_id
    
    // 从高层向低层逐层降落
    for l in from_level down to (to_level + 1):
        changed = true
        while changed:
            changed = false
            curr_dist = dist(curr, target_vec)
            
            // 遍历当前节点在第 l 层的邻居
            for each nb in graph[curr][l]:
                d = dist(nb, target_vec)
                if d < curr_dist:
                    curr_dist = d
                    curr = nb
                    changed = true  // 找到更近的，继续在当前层贪心移动
                    
    return curr
```

#### **2.3 启发式邻居选择 (SelectNeighborsHeuristic)**

**目的**: 替代单纯的“按距离取前 M 个”。

**作用**: 在高维空间中，单纯取最近的邻居会导致图的连接极度“扎堆”（聚簇），形成相互孤立的孤岛。启发式策略引入了松弛系数 $\alpha$，在选择邻居时不仅看它离中心有多近，还要看它**与已经选中的邻居之间是否离得太近**。这保证了邻居的空间多样性。

**伪代码:**

C++

```
function SelectNeighborsHeuristic(center, candidates_heap, M, level):
    // 第0层适度放宽(1.20)保留局部连通，高层严格限制(1.05)增强长程跳跃
    alpha = (level == 0) ? 1.20 : 1.05
    alpha2 = alpha * alpha
    
    selected = empty list
    
    // 1. 启发式过滤（防止扎堆）
    for each cand in candidates_heap:
        if size(selected) >= M: break
        
        good = true
        for sid in selected:
            // 如果候选点与已选邻居的距离 d_cs 极小，说明方向重叠，剔除冗余
            d_cs = dist(cand, sid)
            if d_cs < (cand.dist / alpha2):
                good = false
                break
                
        if good: 
            selected.push_back(cand.id)
            
    // 2. 扫尾回填（保证最小连通度）
    // 如果启发式剔除得太狠导致边数不够 M，就把被剔除的节点补回来，防止断链
    if size(selected) < M:
        for each cand in candidates_heap:
            if size(selected) >= M: break
            if cand.id not in selected:
                selected.push_back(cand.id)
                
    return selected
```

#### **2.4 第0层对称性收尾 (FinalizeLevel0Symmetry)**

**目的**: 并发建库后的最终质量兜底。

**作用**: 在多线程（OpenMP）并发插入时，虽然加了锁，但由于插入顺序的交织，可能会导致部分反向边（有向图转无向图的过程）未能完美记录，或者节点的出边溢出。收尾阶段会强制修复第 0 层的所有双向关系，严格将度数裁剪至 $2M$，这是把 Recall 逼近极限的关键一步。

**伪代码:**

C++

```
function FinalizeLevel0Symmetry(M0):
    // 第一步：强制反向连接
    parallel for each u in all_nodes:
        neigh_copy = graph[u][0]
        for each nb in neigh_copy:
            if u not in graph[nb][0]:
                graph[nb][0].push_back(u)  // 补齐反向边
            
            // 如果补齐后超载了，触发一次启发式裁剪
            if size(graph[nb][0]) > M0 + prune_slack:
                graph[nb][0] = PruneNeighborsHeuristic(nb, graph[nb][0], M0, level=0)
                
    // 第二步：严格的最终裁剪
    parallel for each u in all_nodes:
        if size(graph[u][0]) > M0:
            graph[u][0] = PruneNeighborsHeuristic(u, graph[u][0], M0, level=0)
```

#### **2.5 多入口漏斗搜索 (Search Multi-entry)**

查询时 (`search_one`)，在最高层到第 1 层的遍历中，不再是单纯保留 1 个最好节点往下传，而是保留多个候选种子（`seeds`，例如 8 到 64 个）。这些种子作为下一层的多个入口同时开始 `search_layer_multi_`，从而大幅度降低陷入局部最优死胡同的概率。





## 优化



### 高级优化与并发设计 (Advanced Optimizations & Concurrency)

#### 1. 细粒度并发建图与无锁搜索 (Concurrent Build & Lock-Free Search)

- **细粒度读写锁 (Fine-grained Read-Write Locks)**: 建图阶段使用了节点级别的锁（`std::shared_mutex node_locks_`），支持多个线程同时对图的不同局部进行无冲突的邻居更新，大幅提升 OpenMP 并发建库时的吞吐量。
- **原子操作与 CAS 机制 (Atomic & CAS)**:
  - 全局入口点 `entry_point_` 和最高层级 `max_level_` 使用 `std::atomic` 进行管理。
  - 在更新全局入口点时，采用了**无锁的 CAS 操作** (`compare_exchange_weak`) 和严格的内存序（`std::memory_order_acq_rel`），彻底避免了多线程更新最高层级时的竞态条件。
- **无锁搜索模式 (Lock-Free Read-Only Mode)**:
  - 提供 `freeze()` 接口。索引构建完成后，通过 `std::memory_order_release` 写入 `frozen_` 标记。
  - 搜索时通过 `memory_order_acquire` 零成本检查冻结状态，一旦进入只读模式，完全绕过所有锁的加锁开销，实现极致的搜索吞吐。

#### 2. 高效的距离计算与 SIMD 极限压榨 (Distance Computation Speedup)

- **L2 距离公式展开**: 放弃了传统的遍历相减求平方和，采用数学展开式 $||a - b||^2 = ||a||^2 + ||b||^2 - 2(a \cdot b)$。在插入时提前预计算并缓存了所有向量的模长平方 (`norms_`)，将复杂的欧式距离计算降维打击为更适合硬件加速的**点积 (Dot Product)** 计算。
- **AVX2 / AVX512 与 FMA 融合乘加**:
  - 在 `distance.h` 中纯手写了底层的 SIMD 向量化运算。
  - 运用 `#if defined(__FMA__)` 自动检测并启用 `_mm256_fmadd_ps` 或 `_mm512_fmadd_ps`，在单条指令中同时完成乘法和加法。
  - **循环展开 (Loop Unrolling)**: 单次处理 16 个 float (双路 `__m256` 累加器 `sum0`, `sum1` 并行流水线)，打破数据依赖，榨干 CPU 发射端口性能。

#### 3. 内存与缓存微操 (Memory & Cache Micro-optimizations)

- **TLS 线程局部存储 (Thread Local Storage) 与 Tagging**:
  - `search_layer_` 中的 `visited` 集合、小根堆 `cand_heap` 和大顶堆 `top_heap` 均被声明为 `static thread_local`。消除了多线程并发搜索时极度昂贵的堆内存动态分配 (`new/delete`) 开销。
  - 彻底抛弃 `memset` 清空 `visited` 数组：引入了 `cur_tag` 机制。每次搜索只需 `++cur_tag`，将 $O(N)$ 的状态重置优化为 $O(1)$ 的整形自增。
- **L1 Cache 内存预取 (Software Prefetching)**:
  - 引入 `_mm_prefetch(..., _MM_HINT_T0)` 硬件预取指令。
  - 在搜索层遍历当前节点的邻居时，提前将邻居的向量数据地址告知 CPU 预取到 L1 缓存中，完美掩盖了内存墙带来的延迟 (Memory Latency Hiding)。

#### 4. 启发式图裁剪策略 (Heuristic Graph Pruning)

- **自适应松弛度 (Adaptive Alpha Relaxation)**: 实现了标准的 HNSW 相对邻近图启发式裁剪 (`select_neighbors_heuristic_`)。对于第 0 层采用 `alpha = 1.20` 适度放宽限制保留局部连通性；对于高层采用 `alpha = 1.05` 严格限制，确保长程导航跳跃的质量。
- **双遍扫描回填 (Two-pass Backfill)**: 优先防止邻居聚集（扎堆），然后再进行第二遍扫描将之前被剪枝的有效连接回填，保证图的最小出边度数，极大提高了极端情况下的召回率 (Recall)。

------





## 局限性与未来规划 (Limitations & TODO)

- **删除操作**: 当前 HNSW 图结构暂不支持动态删除（软删除/硬删除）已有节点。
- **持久化**: 亟需增加 `save(path)` 与 `load(path)` 接口，以支持索引构建后的磁盘序列化存储。
- **距离度量**: 目前仅硬编码支持了 L2（欧几里得距离），未来计划支持内积（IP）和余弦相似度（Cosine）。
- **量化压缩**: 计划引入基于 PQ (Product Quantization) 或 SQ 的标量量化机制，以大幅降低大规模数据下的内存占用。
