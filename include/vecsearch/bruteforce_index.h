#pragma once
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "vecsearch/index.h"

namespace vecsearch {

static inline float l2_sqr(const float* a, const float* b, int dim) {
  float sum = 0.0f;
  for (int i = 0; i < dim; ++i) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

//  BruteForceIndex 继承 IIndex
class BruteForceIndex : public IIndex {
 public:
  explicit BruteForceIndex(IndexConfig cfg) : cfg_(cfg) {//构造函数,explicit禁用自动隐式转换
    if (cfg_.dim <= 0) {
      std::cerr << "BruteForceIndex: dim must be > 0\n";
      std::exit(1);
    }
    // 目前只支持 L2
    if (cfg_.metric != Metric::L2) {
      std::cerr << "BruteForceIndex: only L2 is supported for now\n";
      std::exit(1);
    }
  }

  IndexConfig config() const override { return cfg_; }//override重写父函数接口

  std::size_t size() const override { return ids_.size(); }

  void clear() override {
    ids_.clear();
    data_.clear();
  }


  void add_batch(const std::vector<Id>& in_ids,
                 const std::vector<float>& vectors) override {
    const std::size_t n = in_ids.size();
    if (vectors.size() != n * (std::size_t)cfg_.dim) {
      std::cerr << "add_batch: vectors size mismatch\n";
      std::exit(1);
    }
    ids_ = in_ids;
    data_ = vectors;
  }

  // 返回 topk 个 Neighbor（id+dist），按 dist 升序
  std::vector<Neighbor> search_one(const float* q, int topk) const override {
    const int n = (int)ids_.size();
    if (n == 0 || topk <= 0) return {};
    if (topk > n) topk = n;

    std::vector<Neighbor> all;
    all.reserve(n);

    for (int i = 0; i < n; ++i) {
      const float* v = &data_[(std::size_t)i * (std::size_t)cfg_.dim];
      float dist = l2_sqr(q, v, cfg_.dim);
      all.push_back(Neighbor{ids_[i], dist});
    }

    std::nth_element(all.begin(), all.begin() + topk, all.end(),
                     [](const Neighbor& x, const Neighbor& y) {
                       return x.dist < y.dist;
                     });
    all.resize(topk);
    std::sort(all.begin(), all.end(),
              [](const Neighbor& x, const Neighbor& y) {
                return x.dist < y.dist;
              });
    return all;
  }

  // 循环调用 search_one
  std::vector<std::vector<Neighbor>> search_batch(const float* queries,
                                                  int num_queries,
                                                  int topk) const override {
    std::vector<std::vector<Neighbor>> out;
    out.reserve(num_queries);
    for (int i = 0; i < num_queries; ++i) {
      const float* q = &queries[(std::size_t)i * (std::size_t)cfg_.dim];
      out.push_back(search_one(q, topk));
    }
    return out;
  }

  std::string name() const override { return "baseline_bruteforce"; }

  std::string params() const override { return "-"; }

 private:
  IndexConfig cfg_;
  std::vector<Id> ids_;     // size = N
  std::vector<float> data_; // size = N * dim
};

}
