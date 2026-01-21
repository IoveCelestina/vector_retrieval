#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using Id = uint32_t;

static float l2_sqr(const float* a, const float* b, int dim) {
  float sum = 0.0f;
  for (int i = 0; i < dim; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

// 最小暴力索引：只存向量，查询时全扫
struct BruteForceIndex {
  int dim = 0;
  std::vector<Id> ids;        // size = N
  std::vector<float> data;    // size = N * dim，row-major

  explicit BruteForceIndex(int d) : dim(d) {}

  size_t size() const { return ids.size(); }

  void add_batch(const std::vector<Id>& in_ids, const std::vector<float>& vectors) {
    const size_t n = in_ids.size();
    if (vectors.size() != n * (size_t)dim) {
      std::cerr << "add_batch: vectors size mismatch\n";
      std::exit(1);
    }
    ids = in_ids;
    data = vectors;
  }

  // 返回 topk 个 (id, dist)，按 dist 升序
  std::vector<std::pair<Id, float>> search_one(const float* q, int topk) const {
    const int n = (int)ids.size();
    if (n == 0 || topk <= 0) return {};
    if (topk > n) topk = n;

    // 先计算所有距离
    std::vector<std::pair<Id, float>> all;
    all.reserve(n);
    for (int i = 0; i < n; ++i) {
      const float* v = &data[(size_t)i * dim];
      float dist = l2_sqr(q, v, dim);
      all.emplace_back(ids[i], dist);
    }

    // 取前 topk（部分排序）
    std::nth_element(all.begin(), all.begin() + topk, all.end(),
                     [](auto& x, auto& y) { return x.second < y.second; });
    all.resize(topk);
    std::sort(all.begin(), all.end(),
              [](auto& x, auto& y) { return x.second < y.second; });
    return all;
  }
};

static std::vector<float> gen_vectors(int n, int dim, uint32_t seed, float low, float high) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(low, high);
  std::vector<float> v((size_t)n * dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

static std::vector<Id> gen_ids(int n) {
  std::vector<Id> ids(n);
  for (int i = 0; i < n; ++i) ids[i] = (Id)i;
  return ids;
}

static double percentile_ms(std::vector<double>& ms, double p) {
  if (ms.empty()) return 0.0;
  std::sort(ms.begin(), ms.end());
  // p=0.99 => index = ceil(p*N)-1
  size_t n = ms.size();
  size_t idx = (size_t)std::ceil(p * n) - 1;
  if (idx >= n) idx = n - 1;
  return ms[idx];
}

double_t check_Recall(std::vector<Id> &search_result_id,std::vector<Id>&search_correct_ans) {

}


int main() {
  // ====== 方案3：先把配置写死（对应你 YAML） ======
  const int dim = 128;
  const std::vector<int> sizes = {10000, 100000};
  const int num_queries = 1000;
  const int topk = 10;
  const uint32_t seed = 20260121;
  const float low = -1.0f;
  const float high = 1.0f;
  const std::vector<uint32_t> search_result_id;
  const std::vector<uint32_t> search_correct_ans;

  std::cout << "Benchmark (baseline bruteforce)\n";
  std::cout << "dim=" << dim << " sizes={10000,100000}"
            << " nq=" << num_queries << " topk=" << topk
            << " seed=" << seed << "\n";

  for (int N : sizes) {
    // 1) 生成 base vectors / ids
    auto base = gen_vectors(N, dim, seed, low, high);
    auto ids = gen_ids(N);

    // 2) 建索引（baseline 只是存起来）
    BruteForceIndex index(dim);
    index.add_batch(ids, base);

    // 3) 生成 queries（用 seed+1，确保与 base 不同但可复现）
    auto queries = gen_vectors(num_queries, dim, seed + 1, low, high);

    // 4) 预热（可选：少量 query）
    const int warmup = 200;
    for (int i = 0; i < std::min(warmup, num_queries); ++i) {
      index.search_one(&queries[(size_t)i * dim], topk);
    }

    // 5) 正式计时：统计每个 query latency + 总耗时
    std::vector<double> per_query_ms;
    per_query_ms.reserve(num_queries);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_queries; ++i) {
      auto qs = std::chrono::high_resolution_clock::now();
      auto res = index.search_one(&queries[(size_t)i * dim], topk);
      (void)res; // 先不打印结果


      auto qe = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration<double, std::milli>(qe - qs).count();
      per_query_ms.push_back(ms);
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double total_s = std::chrono::duration<double>(t1 - t0).count();
    double qps = num_queries / total_s;
    double p99 = percentile_ms(per_query_ms, 0.99);

    std::cout << "\nN=" << N << "\n";
    std::cout << "baseline Recall@10 = 1.0 (exact)\n";
    std::cout << "QPS = " << qps << "\n";
    std::cout << "P99(ms) = " << p99 << "\n";
  }

  return 0;
}
