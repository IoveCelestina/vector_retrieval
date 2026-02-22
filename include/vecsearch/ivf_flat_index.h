#pragma once
#include <vector>
#include <mutex>
#include <random>
#include "vecsearch/index.h"

namespace vecsearch {
    struct IVFParams{
        int nlist = 256;//聚类中心数
        int nprobe = 8;//查询时探测的桶数
        int kmeans_iters = 20;//K-means 迭代次数
        int train_max_points = 0;//训练采样上限
    };

    class IVFFlatIndex:public IIndex{
    public:
        IVFFlatIndex(IndexConfig cfg, IVFParams params);

        IndexConfig config() const override { return cfg_; }
        std::size_t size() const override { return total_count_; }
        void clear() override;

        //核心逻辑:如果第一次调佣且为训练，会使用该批数据进行K-means训练
        //然后将数据分配到对应的倒排桶中
        void add_batch(const std::vector<Id> &ids,const std::vector<float> &vectors) override;
        std::vector<Neighbor> search_one(const float* q, int topk) const override;
        std::vector<std::vector<Neighbor>> search_batch(const float* queries,
                                                    int num_queries,
                                                    int topk) const override;
        std::string name() const override { return "IVF-Flat"; }
        std::string params() const override;
    private:
        //K-means训练(Lloyd's Alogrithm)
        void train_centroids_(const float* data,std::size_t n);

        //寻找最近的centroids，返回pairs of (centroid_index, distance_sq)
        std::vector<std::pair<int,float>> find_nearest_centroids_(const float* queries,int k)const ;

    private:
        IndexConfig cfg_;
        IVFParams p_;

        std::size_t total_count_ = 0;
        bool is_trained_ = false;

        // 聚类中心 [nlist * dim]
        std::vector<float> centroids_;
        // 倒排表
        // inverted_lists_ids_[i] 存储第 i 个桶内的 ID
        std::vector<std::vector<Id>> inverted_lists_ids_;
        // inverted_lists_vecs_[i] 存储第 i 个桶内的向量数据 (flattened)
        std::vector<std::vector<float>> inverted_lists_vecs_;
    };
}