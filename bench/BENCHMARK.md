OS:Windows

Compiler：MinGW GCC 13.2.0

Build：Debug

dim = 128

sizes={10000,100000}

nq = 1000

topk = 10

seed = 20260121

low=-1,high - 1

warmup = 200



**Recall**:和结果的近似度，越大越好，最大1.0

**QPS**:1s可以完成多少次查询

**P99**:第99百分位延迟:把q次query的耗时从小到大排序，最慢的1%的门槛值

**build_qps**: 每秒插入多少向量





- **N = 10000**
- CPU:i9-13900kf

| Method                         | Params | Recall@10 | QPS     | P99(ms) |
| ------------------------------ | ------ | --------- | ------- | ------- |
| baseline (bruteforce)(debug)   | -      | 1.0       | 472.935 | 2.2105  |
| baseline (bruteforce)(release) | -      | 1.0       | 2445.66 | 0.432   |

- **N = 100000**
- CPU:i9-13900kf

| Method                         | Params | Recall@10 | QPS     | P99(ms) |
| ------------------------------ | ------ | --------- | ------- | ------- |
| baseline (bruteforce)(debug)   | -      | 1.0       | 47.8052 | 22.7271 |
| baseline (bruteforce)(release) | -      | 1.0       | 217.055 | 5.025   |







## 计时工具

`std::chrono::`

`hish_resolution_clock::now()` :返回当前时间点对象(time_point)

- 计时

- ```c++
  auto t0 = std::chrono::high_resolution_clock::now();
  // ... 一段要计时的代码 ...
  auto t1 = std::chrono::high_resolution_clock::now();
  auto dt = t1-t0
  ```

- `dt`是一个时间段对象`std::chrono::duration<Rep,Period>` ,`Rep`内部存储的数值类型,`Period` 表示单位



`duration<double>(t1-t0).count()`:计算总耗时(秒为默认单位)

- ```
  double total_s = std::chrono::duration<double>(t1 - t0).count();
  ```

- **毫秒计时**:

  - ```cpp
    auto qs = std::chrono::high_resolution_clock::now();
    auto qe = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(qe - qs).count();
    ```



`steady_clock::now()`:和`hish_resolution_clock::now()`一样更稳定





## 日志工具

每次运行bench_runner，在项目目录下生成/追加`benchmark_results/results.csv`

**内容示例:**

- ```shell
  build,N,dim,topk,nq,seed,method,qps,p99_ms,mean_recall
  Release,10000,128,10,1000,20260121,baseline_bruteforce,12345.6,0.31,1.0
  Release,100000,128,10,1000,20260121,baseline_bruteforce,1234.5,3.21,1.0
  ```



### 运行命令

### Debug

```
cmake -S . -B cmake-build-debug -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake --build cmake-build-debug
& ".\cmake-build-debug\bench_runner.exe"
```

### Release

```
cmake -S . -B cmake-build-release -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release
& ".\cmake-build-release\bench_runner.exe"
```



## 技术

引入工厂函数(本质是if/else创建不同类)这样就不用改标签

只需要修改cases里面的参数，就可以运行





