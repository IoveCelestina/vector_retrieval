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