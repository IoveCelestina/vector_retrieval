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







## 优化



#### O3优化+AVX2+手写SIMD(-march=native) 让build时间快了很多

- **O3**:让小函数自动进行内联，同时把循环展开(Unroll)，甚至自动尝试使用SSE指令(1次算4个float)来向量化

- SIMD(-march=native):长出8只手(256位寄存器)，一次装配8个零件

  - `__m256`是一个256-bit向量寄存器，对于`float` 就是8 个 float打包在一起

  - `__m256 _mm256_loadu_ps(float const* mem_addr);`  

    - `mem_addr:` 指向float的指针
    - `loadu` 的`u` =unaligned，表示不要求32字节对齐

  - `__m256 _mm256_sub_ps(__m256 a, __m256 b);` 对8个float分别做减法

    - 返回每个lane: `a[j]-b[j]`

  - `__m256 _mm256_fmadd_ps(__m256 a, __m256 b, __m256 c);`

    - 计算 `a*b+c`

    - 老机器不支持FMA时用以下代替

      - ```
        __m256 _mm256_mul_ps(__m256 a, __m256 b); // a*b
        __m256 _mm256_add_ps(__m256 a, __m256 b); // a+b
        ```

  - `void _mm_prefetch(char const* p, int hint);`

    - `p` :要预取的地址，注意类型是`char const *`
    - `hint` 预取到那个缓存层/策略的提示

#### 内存预取

- 在search_layer_的过程中告诉CPU提前把neighbors拉到L1缓存

- ```
  _mm_prefetch(vec, _MM_HINT_T0);
  ```

  - _MM_HINT_T0表示预期稍后会频繁使用,拉到所有缓存层



#### 保留局部变量tag

`local_cur_tag` 是`static thread_local` 变量，存储在线程局部存储(TLS,Thread Local Storage) 区域，本质是一块特殊的内存,每次写`local_cur_tag`，cpu都要去计算TLS偏移量，然后去内存读这个值.

`tag` 是普通的局部栈变量,在开启`-O3`优化后，编译器会把它丢进寄存器里(Register)
