#pragma once
#include <atomic>
#include<cstdint>
#include<string>
#include<vector>
#include<queue>
#include "vecsearch/index.h"
#include<mutex> //创建锁
#include<memory> //管理内存 std::unique_ptr
#include<random>
#include<cmath>
#include <shared_mutex>

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

  		void add_batch(const std::vector<Id>& ids,const std::vector<float>& vectors) override;

  		std::vector<Neighbor> search_one(const float* q, int topk) const override;
  		std::vector<std::vector<Neighbor>> search_batch(const float* queries,int num_queries,int topk) const override;

  		std::string name() const override { return "HNSW"; }
  		std::string params() const override;


		void freeze() {// 在构建（索引/图）完成后，调用 freeze() 使搜索变为无锁操作（进入只读模式）
			frozen_.store(true, std::memory_order_release);
			// 在只读模式下，不再需要锁（搜索变为无锁状态）
			std::deque<std::shared_mutex>().swap(node_locks_);
		}

	private:
  		//距离函数
		float dist_l2_sqr_id_query_(Id id, const float* q, float q_norm) const;
		float dist_l2_sqr_ids_(Id a, Id b) const;

		//随机决定新节点的层数
		int random_level_() noexcept;

		std::vector<Id> select_neighbors_heuristic_(Id center, std::vector<Neighbor>& candidates, int M,int level) const;

		void prune_neighbors_heuristic_(Id center,std::vector<Id>& neighbor_list,int M,int level);

		//谈心降落，从entry开始在[from_level,to_level]区间内贪婪地找最近的一个点
		Id greedy_descent_(const float* target, float target_norm,Id entry, int from_level, int to_level) const;

		std::vector<Neighbor> search_layer_(const float* target, float target_norm,Id entry, int ef, int level) const;

		std::vector<Neighbor> search_layer_multi_(const float *target,float target_norm,const std::vector<Id> &entries,int ef,int level) const;

		//建双向边
		void connect_bidirectional_(Id u, const std::vector<Id>& neigh, int level);

		//构建后的清理工作（比之前开销高昂的全局优化过程更快）,就是之前的recfine_level0
		void finalize_level0_symmetry_(int M0);
		void finalize_prune_all_levels_();

	private:
  		IndexConfig cfg_;
  		HNSWParams p_;

  		// row-major存数据
  		std::vector<Id> ids_;        // size = N
  		std::vector<float> data_;    // size = N * dim
		std::vector<float> norms_;

  		//3D图结构 graph_[u] ->节点u的所有层信息
		//graph_[u][L] 节点u在第L层的邻居列表(vector<Id>)
  		std::vector<std::vector<std::vector<Id>>> graph_;

		mutable std::deque<std::shared_mutex> node_locks_;

		//入口点以及当前最高层级（使用原子变量以支持并发构建）
		std::atomic<Id> entry_point_{0};
		std::atomic<int> max_level_{-1};

		// 用于分配层级的随机数生成器（RNG）
		std::mt19937 level_generator_;
		std::uniform_real_distribution<float> level_distribution_{0.0f, 1.0f};
		float level_mult_ = 1.0f; // 1 / log(M)

		// 当为 true 时，搜索变为无锁操作。
		std::atomic<bool> frozen_{false};

		//修剪余量（松弛量）：允许在执行修剪前出现临时的（边数）溢出（从而减少修剪操作的频率）。
		int prune_slack_ = 8;
	};
}

