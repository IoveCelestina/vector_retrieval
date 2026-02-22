#pragma once
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "vecsearch/index.h"
#include "vecsearch/distance.h"

namespace vecsearch {

class BruteForceIndex : public IIndex {
public:
    explicit BruteForceIndex(IndexConfig cfg) : cfg_(cfg) {
        if (cfg_.dim <= 0) {
            std::cerr << "BruteForceIndex: dim must be > 0\n";
            std::exit(1);
        }
        if (cfg_.metric != Metric::L2) {
            std::cerr << "BruteForceIndex: only L2 is supported for now\n";
            std::exit(1);
        }
    }

    IndexConfig config() const override { return cfg_; }
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

    // === 核心修正：使用标准 L2 计算，确保 Ground Truth 100% 正确 ===
    std::vector<Neighbor> search_one(const float* q, int topk) const override {
        const int n = (int)ids_.size();
        if (n == 0 || topk <= 0) return {};
        if (topk > n) topk = n;

        std::vector<Neighbor> all;
        all.reserve(n);

        for (int i = 0; i < n; ++i) {
            const float* v = &data_[(std::size_t)i * (std::size_t)cfg_.dim];

            // 直接调用 distance.h 中的 l2_sqr (SIMD 加速)
            // 它是精确的 sum((a-b)^2)
            float dist_sq = vecsearch::l2_sqr(q, v, cfg_.dim);

            all.push_back(Neighbor{ids_[i], dist_sq});
        }

        // TopK 排序
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

    std::vector<std::vector<Neighbor>> search_batch(const float* queries,
                                                    int num_queries,
                                                    int topk) const override {
        std::vector<std::vector<Neighbor>> out(num_queries);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_queries; ++i) {
            const float* q = &queries[(std::size_t)i * (std::size_t)cfg_.dim];
            out[i] = search_one(q, topk);
        }
        return out;
    }

    std::string name() const override { return "FlatL2(Standard)"; }
    std::string params() const override { return "AVX2"; }

private:
    IndexConfig cfg_;
    std::vector<Id> ids_;
    std::vector<float> data_;
    // 移除了 norms_，因为它只在错误公式里用到
};

} // namespace vecsearch