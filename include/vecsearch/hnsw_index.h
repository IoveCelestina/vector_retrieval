#pragma once
#include<cstdint>
#include<string>
#include<vector>
#include<queue>


#include "vecsearch/index.h"

namespace vecsearch {
	struct HNSWParams {//参数,看README
          int M = 16;
          int ef_construction = 200;
          int ef_search = 50;
	};
    //貌似单层HNSW是单层ANN?
	//继承Index
	class HNSWIndex : public IIndex {
 		public:
  		HNSWIndex(IndexConfig cfg, HNSWParams p);

  		IndexConfig config() const override { return cfg_; }
  		std::size_t size() const override { return ids_.size(); }
  		void clear() override;

  		void add_batch(const std::vector<Id>& ids,
                 		const std::vector<float>& vectors) override;

  		std::vector<Neighbor> search_one(const float* q, int topk) const override;

  		std::vector<std::vector<Neighbor>> search_batch(const float* queries,
	                                                  int num_queries,
	                                                  int topk) const override;

  		std::string name() const override { return "hnsw"; }
  		std::string params() const override;

 		private:
  		//距离函数
 		 float dist_l2_sqr(const float* a, const float* b) const;


  		// 单层：搜索从 entry_ 开始
  		bool has_entry_ = false;
  		Id entry_ = 0;

  		// 返回 ef 个左右的候选,未排序
  		std::vector<Neighbor> search_layer_(const float* target,
	                                     Id entry,
	                                     int ef) const;

  		// 从 candidates 中挑最近的 M 个
  		std::vector<Id> select_neighbors_simple_(std::vector<Neighbor>& candidates,
	                                          int M) const;

  		// 把 neighbor_list 裁剪到最多 M 个：按 dist(center, nb) 排序，保留前 M
  		void prune_neighbors_by_distance_(Id center,
	                                   std::vector<Id>& neighbor_list,
	                                   int M);

  		// 建双向边 + 做裁剪
  		void connect_bidirectional_(Id u, const std::vector<Id>& neigh);

 		private:
  		IndexConfig cfg_;
  		HNSWParams p_;

  		// row-major存
  		std::vector<Id> ids_;        // size = N
  		std::vector<float> data_;    // size = N * dim

  		// 邻接表 graph_[i] 存邻居 id
  		// 默认 id = 0..N-1 这样
  		std::vector<std::vector<Id>> graph_;
		};

};

