#pragma once
#include<cstdint>
#include<string>
#include<vector>
#include<queue>
#include "vecsearch/index.h"
#include<mutex> //创建锁
#include<memory> //管理内存 std::unique_ptr
#include<random>
#include<cmath>

namespace vecsearch {
	struct HNSWParams {//参数,看README
          int M = 32;
          int ef_construction = 200;
          int ef_search = 400;
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
		//随机决定新节点的层数
		int get_random_level();


  		// 从 candidates 中挑最近的 M 个
  		std::vector<Id> select_neighbors_simple_(std::vector<Neighbor>& candidates,
	                                          int M) const;

  		// 把 neighbor_list 裁剪到最多 M 个：按 dist(center, nb) 排序，保留前 M
  		void prune_neighbors_by_distance_(Id center,
	                                   std::vector<Id>& neighbor_list,
	                                   int M);


		std::vector<Neighbor> search_layer_(const float* target, Id entry, int ef, int level) const;
		//建双向边
		void connect_bidirectional_(Id u, const std::vector<Id>& neigh, int level);

		//谈心降落，从entry开始在[from_level,to_level]区间内贪婪地找最近的一个点
		Id greedy_descent_(const float* target, Id entry, int from_level, int to_level) const;

		std::vector<Id> select_neighbors_heuristic_(Id center, std::vector<Neighbor>& candidates, int M) const;

		void prune_neighbors_heuristic_(Id center,
										  std::vector<Id>& neighbor_list,
										  int M);
 		private:
  		IndexConfig cfg_;
  		HNSWParams p_;

  		// row-major存数据
  		std::vector<Id> ids_;        // size = N
  		std::vector<float> data_;    // size = N * dim

  		//3D图结构 graph_[u] ->节点u的所有层信息
		//graph_[u][L] 节点u在第L层的邻居列表(vector<Id>)
  		std::vector<std::vector<std::vector<Id>>> graph_;

		//锁数组,每个节点有一把锁，用unique_ptr是因为锁不可以被复制，只能用指针管理
		std::vector<std::unique_ptr<std::mutex>> node_locks_;

		int current_max_level_ = -1; //当前图的最高层数初始为-1
		Id entry_point_ = 0; //全局入口点

		//随机层数生成器
		std::default_random_engine level_generator_;
		std::uniform_real_distribution<double> level_distribution_;
		double mult_; //归一化因子 1/ln(M)
	};

};

