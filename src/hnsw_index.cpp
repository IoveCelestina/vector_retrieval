#include "vecsearch/hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <mutex>
#include <utility>

#include "vecsearch/distance.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif


namespace vecsearch {

namespace {
	//预取（Prefetch）辅助函数
	static inline void prefetch_T0(const void *p) noexcept {
#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
		_mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0);
#elif defined(__GNUC__) || defined(__clang__)
		__builtin_prefetch(p, 0, 3);
#else
		(void)p;
#endif
	}

	static inline bool contains_id(const std::vector<Id>& v, Id x) noexcept {
		for (Id y : v) {
			if (y == x) return true;
		}
		return false;
	}

	static inline void dedup_inplace_small(std::vector<Id>& v) noexcept {

		std::size_t out = 0;
		for (std::size_t i = 0; i < v.size(); ++i) {
			const Id x = v[i];
			bool dup = false;
			for (std::size_t j = 0; j < out; ++j) {
				if (v[j] == x) { dup = true; break; }
			}
			if (!dup) v[out++] = x;
		}
		v.resize(out);
	}
}//namespace


	HNSWIndex::HNSWIndex(IndexConfig cfg,HNSWParams p):cfg_(cfg),p_(p) {
		if (cfg_.dim<=0) {
			std::cerr<<"HNSWIndex: dim must be >0\n";
			std::exit(1);
		}
		if (cfg_.metric!=Metric::L2) {//之后会改掉的
			std::cerr<<"Only L2 supported now!\n";
			std::exit(1);
		}
		if (p_.M <= 0 || p_.ef_construction <= 0 || p_.ef_search <= 0) {
			std::cerr << "HNSWIndex: params must be > 0\n";
			std::exit(1);
		}
		//固定种子以便复现
		level_generator_.seed(2026);
		// 标准的 HNSW 层级乘数：1 / ln(M)。
		level_mult_ = 1.0f / std::log((float)p_.M);
		//修剪余量（在执行修剪操作前允许的临时溢出量）
		prune_slack_ = std::max(8, p_.M / 2);
	}

	void HNSWIndex::clear() {
		ids_.clear();
		data_.clear();
		norms_.clear();
		graph_.clear();
		node_locks_.clear();
		entry_point_.store(0, std::memory_order_release);
		max_level_.store(-1, std::memory_order_release);
		frozen_.store(false, std::memory_order_release);
	}

	std::string HNSWIndex::params()const {
		return "M="+std::to_string(p_.M)+
				";efC="+std::to_string(p_.ef_construction)+
				";efS="+ std::to_string(p_.ef_search);
	}

	float HNSWIndex::dist_l2_sqr_id_query_(Id id, const float* q, float q_norm) const {
		const std::size_t dim = (std::size_t)cfg_.dim;
		const float* v = &data_[(std::size_t)id * dim];
		const float dot = vecsearch::inner_product(v, q, cfg_.dim);
		float d = norms_[(std::size_t)id] + q_norm - 2.0f * dot;
		//避免负数
		return (d > 0.0f) ? d : 0.0f;
	}

	float HNSWIndex::dist_l2_sqr_ids_(Id a, Id b) const {
		const std::size_t dim = (std::size_t)cfg_.dim;
		const float* va = &data_[(std::size_t)a * dim];
		const float* vb = &data_[(std::size_t)b * dim];
		const float dot = vecsearch::inner_product(va, vb, cfg_.dim);
		float d = norms_[(std::size_t)a] + norms_[(std::size_t)b] - 2.0f * dot;
		return (d > 0.0f) ? d : 0.0f;
	}

	int HNSWIndex::random_level_() noexcept {
		// level = floor(-ln(U) * (1/ln(M))), U ~ Uniform(0,1]
		float r = level_distribution_(level_generator_);
		if (r < 1e-9f) r = 1e-9f;
		const int level = (int)(-std::log(r) * level_mult_);
		//限制最大
		return std::min(level, 32);
	}

	std::vector<Id> HNSWIndex::select_neighbors_heuristic_(Id center,
                                                      std::vector<Neighbor>& candidates,
                                                      int M,int level ) const {
	    if (M <= 0 || candidates.empty()) return {};
		// 基于层级的松弛型 RNG（相对邻近图）启发式策略。
		// - 第 0 层：适度放宽限制，以保留局部的连通性。
		// - 较高层级：趋近于严格限制，以提供更好的空间多样性和导航（跳跃）能力。
		const float alpha = (level == 0) ? 1.20f : 1.05f;
		const float alpha2 = alpha * alpha;


	    std::vector<Id> selected;
	    selected.reserve((std::size_t)M);

	    // Heuristic:避免邻居扎堆
	    for (const auto& cand : candidates) {
	        if ((int)selected.size() >= M) break;

	        const Id cid = cand.id;
	        if (cid == center) continue;
	        if ((std::size_t)cid >= ids_.size()) continue;

	    	bool good = true;
	    	for (Id sid : selected) {
	    		const float d_cs = dist_l2_sqr_ids_(cid, sid);
	    		// 如果候选节点与某个已被选中的邻居“离得太近”，则拒绝（丢弃）该候选节点。
	    		if (d_cs < cand.dist / alpha2) {
	    			good = false;
	    			break;
	    		}
	    	}
	    	if (good) selected.emplace_back(cid);
	    }

		//第二遍扫描（回填）：保留之前被修剪掉的连接，以保证节点的出边度数（从而提升图的连通性）
		if ((int)selected.size() < M) {
			for (const auto& cand : candidates) {
				if ((int)selected.size() >= M) break;
				const Id cid = cand.id;
				if (cid == center) continue;
				if ((std::size_t)cid >= ids_.size()) continue;
				if (contains_id(selected, cid)) continue;
				selected.push_back(cid);
			}
		}
	    return selected;
	}



	void HNSWIndex::prune_neighbors_heuristic_(Id center, std::vector<Id> &neighbor_list, int M,int level) {
		if (M <= 0) {
			neighbor_list.clear();
			return;
		}
		if (neighbor_list.empty()) return;

		const std::size_t N = ids_.size();

		//过滤掉无效节点或指向自身的节点（防止自环）。
		neighbor_list.erase(
	   std::remove_if(neighbor_list.begin(), neighbor_list.end(),
					  [&](Id nb) { return nb == center || (std::size_t)nb >= N; }),
	   neighbor_list.end());

		if (neighbor_list.size() <= 1) return;

		//去重
		if (neighbor_list.size() <= 64) {
			dedup_inplace_small(neighbor_list);
		} else {
			std::sort(neighbor_list.begin(), neighbor_list.end());
			neighbor_list.erase(std::unique(neighbor_list.begin(), neighbor_list.end()), neighbor_list.end());
		}

		if ((int)neighbor_list.size() <= M) return;

		//构建候选集
		std::vector<Neighbor> cands;
		cands.reserve(neighbor_list.size());
		for (Id nb : neighbor_list) {
			cands.push_back(Neighbor{nb, dist_l2_sqr_ids_(center, nb)});
		}

		std::sort(cands.begin(), cands.end(),[](const Neighbor &a,const Neighbor &b) {
			return a.dist<b.dist;
		});

		// 限制候选池大小
		const int max_cand = std::min<int>((int)cands.size(), std::max(M * 4, M + 16));
		cands.resize((std::size_t)max_cand);

		neighbor_list = select_neighbors_heuristic_(center, cands, M, level);

	}


	Id HNSWIndex::greedy_descent_(const float *target, float target_norm,Id entry, int from_level, int to_level) const {
		Id curr = entry;

		for (int l = from_level; l > to_level; --l) {
			bool changed = true;
			while (changed) {
				changed = false;

				float curr_dist = dist_l2_sqr_id_query_(curr, target, target_norm);

				if (l >= (int)graph_[(std::size_t)curr].size()) break;

				if (frozen_.load(std::memory_order_acquire)) {
					const auto& neigh = graph_[(std::size_t)curr][l];
					for (Id nb : neigh) {
						prefetch_T0(&data_[(std::size_t)nb * (std::size_t)cfg_.dim]);
						const float d = dist_l2_sqr_id_query_(nb, target, target_norm);
						if (d < curr_dist) {
							curr_dist = d;
							curr = nb;
							changed = true;
						}
					}
				} else {
					std::shared_lock<std::shared_mutex> lk(node_locks_[(std::size_t)curr]);
					const auto& neigh = graph_[(std::size_t)curr][l];
					for (Id nb : neigh) {
						prefetch_T0(&data_[(std::size_t)nb * (std::size_t)cfg_.dim]);
						const float d = dist_l2_sqr_id_query_(nb, target, target_norm);
						if (d < curr_dist) {
							curr_dist = d;
							curr = nb;
							changed = true;
						}
					}
				}
			}
		}
		return curr;
	}

	std::vector<Neighbor> HNSWIndex::search_layer_(const float* target,float target_norm, Id entry, int ef, int level) const {
        // 基础检查
		if (ids_.empty()) return {};
		if (ef <= 0) return {};
		const std::size_t N = ids_.size();
		if ((std::size_t)entry >= N) return {};

		if (ef > (int)N) ef = (int)N;

        // 初始化 Visited Tag (线程安全 & 高性能)
		static thread_local std::vector<std::uint32_t> visited;
		static thread_local std::uint32_t cur_tag = 0;


        //Tag更新(避免 memset)
		if (visited.size() < N) visited.resize(N, 0);
		++cur_tag;
		if (cur_tag == 0) {
			std::fill(visited.begin(), visited.end(), 0);
			cur_tag = 1;
		}
		const std::uint32_t tag = cur_tag;

		//这里也上thread_local
		static thread_local std::vector<Neighbor> cand_heap; // min-heap
		static thread_local std::vector<Neighbor> top_heap;  // max-heap

		cand_heap.clear();
		top_heap.clear();
		cand_heap.reserve((std::size_t)ef * 2 + 32);
		top_heap.reserve((std::size_t)ef + 32);

		struct MinCmp {
			bool operator()(const Neighbor& a, const Neighbor& b) const noexcept { return a.dist > b.dist; }
		};
		struct MaxCmp {
			bool operator()(const Neighbor& a, const Neighbor& b) const noexcept { return a.dist < b.dist; }
		};
		const MinCmp mincmp{};
		const MaxCmp maxcmp{};



		const float d0 = dist_l2_sqr_id_query_(entry, target, target_norm);
		Neighbor e{entry, d0};

		cand_heap.push_back(e);
		std::push_heap(cand_heap.begin(), cand_heap.end(), mincmp);

		top_heap.push_back(e);
		std::push_heap(top_heap.begin(), top_heap.end(), maxcmp);

		visited[(std::size_t)entry] = tag;

        while (!cand_heap.empty()) {
	        std::pop_heap(cand_heap.begin(), cand_heap.end(), mincmp);
        	Neighbor cur = cand_heap.back();
        	cand_heap.pop_back();

        	const float worst = top_heap.front().dist;
        	if ((int)top_heap.size() >= ef && cur.dist > worst) break; // 剪枝:当前最近的候选比best里最差的还远且best够了,没必要找了

        	// 访问 graph_[cur.id]的第[level] 层
        	// 安全检查:防止该节点没有这一层 (虽然理论上entry保证了都在同一层)
        	if (level >= (int)graph_[(std::size_t)cur.id].size()) continue;

        	if (frozen_.load(std::memory_order_acquire)) {
        		const auto& neigh = graph_[(std::size_t)cur.id][level];

        		// Prefetch neighbor vectors.
        		for (Id nb : neigh) {
        			if ((std::size_t)nb < N && visited[(std::size_t)nb] != tag) {
        				prefetch_T0(&data_[(std::size_t)nb * (std::size_t)cfg_.dim]);
        			}
        		}

        		for (Id nb : neigh) {
        			const std::size_t nbi = (std::size_t)nb;
        			if (nbi >= N) continue;
        			if (visited[nbi] == tag) continue;
        			visited[nbi] = tag;

        			const float d = dist_l2_sqr_id_query_(nb, target, target_norm);
        			if ((int)top_heap.size() < ef || d < top_heap.front().dist) {
        				Neighbor cand{nb, d};

        				cand_heap.push_back(cand);
        				std::push_heap(cand_heap.begin(), cand_heap.end(), mincmp);

        				top_heap.push_back(cand);
        				std::push_heap(top_heap.begin(), top_heap.end(), maxcmp);

        				if ((int)top_heap.size() > ef) {
        					std::pop_heap(top_heap.begin(), top_heap.end(), maxcmp);
        					top_heap.pop_back();
        				}
        			}
        		}
        	}else {
        		std::shared_lock<std::shared_mutex> lk(node_locks_[(std::size_t)cur.id]);
        		const auto& neigh = graph_[(std::size_t)cur.id][level];

        		for (Id nb : neigh) {
        			if ((std::size_t)nb < N && visited[(std::size_t)nb] != tag) {
        				prefetch_T0(&data_[(std::size_t)nb * (std::size_t)cfg_.dim]);
        			}
        		}

        		for (Id nb : neigh) {
        			const std::size_t nbi = (std::size_t)nb;
        			if (nbi >= N) continue;
        			if (visited[nbi] == tag) continue;
        			visited[nbi] = tag;

        			const float d = dist_l2_sqr_id_query_(nb, target, target_norm);
        			if ((int)top_heap.size() < ef || d < top_heap.front().dist) {
        				Neighbor cand{nb, d};

        				cand_heap.push_back(cand);
        				std::push_heap(cand_heap.begin(), cand_heap.end(), mincmp);

        				top_heap.push_back(cand);
        				std::push_heap(top_heap.begin(), top_heap.end(), maxcmp);

        				if ((int)top_heap.size() > ef) {
        					std::pop_heap(top_heap.begin(), top_heap.end(), maxcmp);
        					top_heap.pop_back();
        				}
        			}
        		}
        	}
        }

		std::sort(top_heap.begin(), top_heap.end(), [](const Neighbor& a, const Neighbor& b) {
			return a.dist < b.dist;
		});
		return top_heap;
    }



	std::vector<Neighbor> HNSWIndex::search_layer_multi_(const float* target,
                                                    float target_norm,
                                                    const std::vector<Id>& entries,
                                                    int ef,
                                                    int level) const {
	    if (ids_.empty()) return {};
	    if (ef <= 0) return {};
	    const std::size_t N = ids_.size();

	    if (ef > (int)N) ef = (int)N;

	    static thread_local std::vector<std::uint32_t> visited;
	    static thread_local std::uint32_t cur_tag = 0;

	    if (visited.size() < N) visited.resize(N, 0);
	    ++cur_tag;
	    if (cur_tag == 0) {
	        std::fill(visited.begin(), visited.end(), 0);
	        cur_tag = 1;
	    }
	    const std::uint32_t tag = cur_tag;

	    static thread_local std::vector<Neighbor> cand_heap;
	    static thread_local std::vector<Neighbor> top_heap;

	    cand_heap.clear();
	    top_heap.clear();
	    cand_heap.reserve((std::size_t)ef * 2 + 64);
	    top_heap.reserve((std::size_t)ef + 64);

	    struct MinCmp {
	        bool operator()(const Neighbor& a, const Neighbor& b) const noexcept { return a.dist > b.dist; }
	    };
	    struct MaxCmp {
	        bool operator()(const Neighbor& a, const Neighbor& b) const noexcept { return a.dist < b.dist; }
	    };
	    const MinCmp mincmp{};
	    const MaxCmp maxcmp{};

	    // Initialize from multiple entry points (dedup via visited).
	    for (Id ep : entries) {
	        if ((std::size_t)ep >= N) continue;
	        if (visited[(std::size_t)ep] == tag) continue;
	        visited[(std::size_t)ep] = tag;

	        const float d0 = dist_l2_sqr_id_query_(ep, target, target_norm);
	        Neighbor e{ep, d0};

	        cand_heap.push_back(e);
	        std::push_heap(cand_heap.begin(), cand_heap.end(), mincmp);

	        top_heap.push_back(e);
	        std::push_heap(top_heap.begin(), top_heap.end(), maxcmp);
	    }

	    if (top_heap.empty()) return {};

	    while (!cand_heap.empty()) {
	        std::pop_heap(cand_heap.begin(), cand_heap.end(), mincmp);
	        Neighbor cur = cand_heap.back();
	        cand_heap.pop_back();

	        const float worst = top_heap.front().dist;
	        if ((int)top_heap.size() >= ef && cur.dist > worst) break;

	        if (level >= (int)graph_[(std::size_t)cur.id].size()) continue;

	        if (frozen_.load(std::memory_order_acquire)) {
	            const auto& neigh = graph_[(std::size_t)cur.id][level];

	            for (Id nb : neigh) {
	                if ((std::size_t)nb < N && visited[(std::size_t)nb] != tag) {
	                    prefetch_T0(&data_[(std::size_t)nb * (std::size_t)cfg_.dim]);
	                }
	            }

	            for (Id nb : neigh) {
	                const std::size_t nbi = (std::size_t)nb;
	                if (nbi >= N) continue;
	                if (visited[nbi] == tag) continue;
	                visited[nbi] = tag;

	                const float d = dist_l2_sqr_id_query_(nb, target, target_norm);
	                if ((int)top_heap.size() < ef || d < top_heap.front().dist) {
	                    Neighbor cand{nb, d};

	                    cand_heap.push_back(cand);
	                    std::push_heap(cand_heap.begin(), cand_heap.end(), mincmp);

	                    top_heap.push_back(cand);
	                    std::push_heap(top_heap.begin(), top_heap.end(), maxcmp);

	                    if ((int)top_heap.size() > ef) {
	                        std::pop_heap(top_heap.begin(), top_heap.end(), maxcmp);
	                        top_heap.pop_back();
	                    }
	                }
	            }
	        } else {
	            std::shared_lock<std::shared_mutex> lk(node_locks_[(std::size_t)cur.id]);
	            const auto& neigh = graph_[(std::size_t)cur.id][level];

	            for (Id nb : neigh) {
	                if ((std::size_t)nb < N && visited[(std::size_t)nb] != tag) {
	                    prefetch_T0(&data_[(std::size_t)nb * (std::size_t)cfg_.dim]);
	                }
	            }

	            for (Id nb : neigh) {
	                const std::size_t nbi = (std::size_t)nb;
	                if (nbi >= N) continue;
	                if (visited[nbi] == tag) continue;
	                visited[nbi] = tag;

	                const float d = dist_l2_sqr_id_query_(nb, target, target_norm);
	                if ((int)top_heap.size() < ef || d < top_heap.front().dist) {
	                    Neighbor cand{nb, d};

	                    cand_heap.push_back(cand);
	                    std::push_heap(cand_heap.begin(), cand_heap.end(), mincmp);

	                    top_heap.push_back(cand);
	                    std::push_heap(top_heap.begin(), top_heap.end(), maxcmp);

	                    if ((int)top_heap.size() > ef) {
	                        std::pop_heap(top_heap.begin(), top_heap.end(), maxcmp);
	                        top_heap.pop_back();
	                    }
	                }
	            }
	        }
	    }

	    std::sort(top_heap.begin(), top_heap.end(), [](const Neighbor& a, const Neighbor& b) {
	        return a.dist < b.dist;
	    });
	    return top_heap;
	}


	void HNSWIndex::connect_bidirectional_(Id u, const std::vector<Id>& neigh, int level) {
		if (neigh.empty()) return;
		const std::size_t N = ids_.size();
		if ((std::size_t)u >= N) return;
		if (level < 0) return;
		if (level >= (int)graph_[(std::size_t)u].size()) return;

        // 过滤与去重
        // 去掉 u 自己,去掉越界的 ID
		std::vector<Id> uniq;
		uniq.reserve(neigh.size());
		for (Id nb : neigh) {
			if (nb == u) continue;
			if ((std::size_t)nb >= N) continue;
			if (level >= (int)graph_[(std::size_t)nb].size()) continue;
			uniq.emplace_back(nb);
		}
		if (uniq.empty()) return;

		if (uniq.size() <= 64) {
			dedup_inplace_small(uniq);
		} else {
			std::sort(uniq.begin(), uniq.end());
			uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
		}


        //第0层允许两倍的M,其他层用 M
		const int M_max = (level == 0) ? (p_.M << 1) : p_.M;
		const int prune_trigger = M_max + prune_slack_;

        //更新节点 u (正向连接)
        {
			std::unique_lock<std::shared_mutex> lk(node_locks_[(std::size_t)u]);
			auto& list = graph_[(std::size_t)u][level];
			if ((int)list.capacity() < prune_trigger + 1) list.reserve((std::size_t)prune_trigger + 1);

			for (Id nb : uniq) {
				if (!contains_id(list, nb)) list.emplace_back(nb);
			}
			if ((int)list.size() > prune_trigger) {
				prune_neighbors_heuristic_(u, list, M_max, level);
			}
        }

        // 更新邻居节点 nb (反向连接)
        for (Id nb : uniq) {
        	std::unique_lock<std::shared_mutex> lk(node_locks_[(std::size_t)nb]);
        	auto& list = graph_[(std::size_t)nb][level];
        	if ((int)list.capacity() < prune_trigger + 1) list.reserve((std::size_t)prune_trigger + 1);

        	if (!contains_id(list, u)) list.push_back(u);
        	if ((int)list.size() > prune_trigger) {
        		prune_neighbors_heuristic_(nb, list, M_max, level);
        	}
        }
    }


	void HNSWIndex::finalize_prune_all_levels_() {
		const std::size_t N = ids_.size();
		if (N == 0) return;

	#pragma omp parallel for schedule(dynamic, 256)
		for (std::int64_t uu = 0; uu < (std::int64_t)N; ++uu) {
			const Id u = (Id)uu;
			std::unique_lock<std::shared_mutex> lk(node_locks_[(std::size_t)u]);

			const int maxL = (int)graph_[(std::size_t)u].size() - 1;
			for (int l = 0; l <= maxL; ++l) {
				const int M_max = (l == 0) ? (p_.M << 1) : p_.M;
				auto& list = graph_[(std::size_t)u][l];
				if ((int)list.size() > M_max) {
					prune_neighbors_heuristic_(u, list, M_max, l);
				} else {
					// Still clean invalid/self/dup.
					prune_neighbors_heuristic_(u, list, (int)list.size(), l);
				}
			}
		}
	}


	void HNSWIndex::finalize_level0_symmetry_(int M0) {
		const std::size_t N = ids_.size();
		if (N == 0) return;
		if (M0 <= 0) M0 = (p_.M << 1);

		const int prune_trigger = M0 + prune_slack_;

	#pragma omp parallel for schedule(dynamic, 256)
		for (std::int64_t uu = 0; uu < (std::int64_t)N; ++uu) {
			const Id u = (Id)uu;

			//在共享锁(读锁)的保护下拷贝节点u的邻居列表，以避免迭代器竞争和死锁风险。
			std::vector<Id> neigh_copy;
			{
				std::shared_lock<std::shared_mutex> lk(node_locks_[(std::size_t)u]);
				if (graph_[(std::size_t)u].empty()) continue;
				neigh_copy = graph_[(std::size_t)u][0];
			}

			for (Id nb : neigh_copy) {
				if (nb == u) continue;
				if ((std::size_t)nb >= N) continue;
				if (graph_[(std::size_t)nb].empty()) continue;

				std::unique_lock<std::shared_mutex> lk(node_locks_[(std::size_t)nb]);
				auto& list = graph_[(std::size_t)nb][0];
				if (!contains_id(list, u)) list.push_back(u);
				if ((int)list.size() > prune_trigger) {
					prune_neighbors_heuristic_(nb, list, M0, 0);
				}
			}
		}

		// 对第 0 层进行最终的裁剪，将邻居数量严格限制到 M0
	#pragma omp parallel for schedule(dynamic, 256)
		for (std::int64_t uu = 0; uu < (std::int64_t)N; ++uu) {
			const Id u = (Id)uu;
			std::unique_lock<std::shared_mutex> lk(node_locks_[(std::size_t)u]);
			if (graph_[(std::size_t)u].empty()) continue;
			auto& list = graph_[(std::size_t)u][0];
			if ((int)list.size() > M0) prune_neighbors_heuristic_(u, list, M0, 0);
		}
	}




	void HNSWIndex::add_batch(const std::vector<Id> &ids, const std::vector<float> &vectors) {
		const std::size_t n = ids.size();
		if (n==0) return;

		const std::size_t dim = (std::size_t)cfg_.dim;
		if (vectors.size()!=n*(std::size_t)cfg_.dim) {
			std::cerr << "HNSWIndex::add_batch: vectors size mismatch\n";
			std::exit(1);
		}

		if (frozen_.load(std::memory_order_acquire)) {//冻结了只能读
			std::cerr << "HNSWIndex::add_batch: index is frozen (read-only)\n";
			std::exit(1);
		}

		const std::size_t oldN = ids_.size();
		const std::size_t newN = oldN + n;

		std::vector<int> levels(n);
		int batch_max_level = -1;
		for (std::size_t i = 0; i < n; ++i) {
			const int lv = random_level_();
			levels[i] = lv;
			batch_max_level = std::max(batch_max_level, lv);
		}

	    //基于预期的节点总数 N 来限制最高层级（Cap levels），以避免在 N 较小时顶层过于稀疏，以及在 N 极大时出现病态的极端异常层级
		const int cap = std::min<int>(32, std::max<int>(0, (int)std::ceil(std::log((double)(newN + 1)) * level_mult_) + 1));
		for (int& lv : levels) lv = std::min(lv, cap);


		//按层级降序进行重排（有助于提升导航质量和入口点的稳定性）
		std::vector<std::size_t> perm(n);
		std::iota(perm.begin(), perm.end(), 0);
		std::sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b) {
			return levels[a] > levels[b];
		});
		//重新分配内存
		ids_.resize(newN);
		data_.resize(newN * dim);
		norms_.resize(newN);
		graph_.resize(newN);
		node_locks_.resize(newN);

		//按降序排列的顺序，将新节点插入到图结构中
		for (std::size_t i = 0; i < n; ++i) {
			const std::size_t real = perm[i];
			const std::size_t internal = oldN + i;

			ids_[internal] = ids[real];

			const float* src = &vectors[real * dim];
			float* dst = &data_[internal * dim];
			std::copy(src, src + dim, dst);

			norms_[internal] = vecsearch::l2_norm_sqr(dst, cfg_.dim);

			const int level = levels[real];
			graph_[internal].resize((std::size_t)level + 1);
			for (int l = 0; l <= level; ++l) {
				const int M_max = (l == 0) ? (p_.M << 1) : p_.M;
				graph_[internal][l].reserve((std::size_t)(M_max + prune_slack_ + 1));
			}
		}


		if (oldN == 0) {
			entry_point_.store(0, std::memory_order_release);
			max_level_.store((int)graph_[0].size() - 1, std::memory_order_release);
		}//初始化入口

		const Id start_u = (oldN == 0) ? 1 : (Id)oldN;

		//分块并发插入
		constexpr std::size_t BLOCK = 256;
	    for (std::size_t base = (std::size_t)start_u; base < newN; base += BLOCK) {
	        const std::size_t end = std::min(newN, base + BLOCK);

	#pragma omp parallel for schedule(dynamic, 1)
	        for (std::int64_t uu = (std::int64_t)base; uu < (std::int64_t)end; ++uu) {
	            const Id u = (Id)uu;
	            const int level = (int)graph_[(std::size_t)u].size() - 1;

	            const float* target = &data_[(std::size_t)u * dim];
	            const float target_norm = norms_[(std::size_t)u];

	            // 快照
	            Id curr_obj = entry_point_.load(std::memory_order_acquire);
	            int curr_max_level = max_level_.load(std::memory_order_acquire);
	            if (curr_max_level < 0) {
	                curr_obj = u;
	                curr_max_level = level;
	            }

	            // 贪心快速降落
	            if (curr_max_level > level) {
	                curr_obj = greedy_descent_(target, target_norm, curr_obj, curr_max_level, level);
	            }

	            const int connect_start_level = std::min(level, curr_max_level);

	            for (int l = connect_start_level; l >= 0; --l) {
	                auto best = search_layer_(target, target_norm, curr_obj, p_.ef_construction, l);
	                const int M_max = (l == 0) ? (p_.M << 1) : p_.M;

	                auto selected = select_neighbors_heuristic_(u, best, M_max, l);
	                connect_bidirectional_(u, selected, l);

	                if (!best.empty()) curr_obj = best[0].id;
	            }

	            // 如果当前节点的层级更高，则更新全局入口点
	            int observed = max_level_.load(std::memory_order_acquire);
	            while (level > observed &&
	                   !max_level_.compare_exchange_weak(observed, level,
	                                                     std::memory_order_acq_rel,
	                                                     std::memory_order_acquire)) {
	                //更新是通过 CAS（Compare-And-Swap，比较并交换）原子操作完成的
	            }
	            if (level > observed) {
	                entry_point_.store(u, std::memory_order_release);
	            }
	        }
	    }

		//快速的构建后清理工作（通过度数控制和第 0 层对称处理，来增强连通性和召回率）
		finalize_prune_all_levels_();
		finalize_level0_symmetry_(p_.M << 1);
	}



	std::vector<Neighbor> HNSWIndex::search_one(const float* q, int topk) const {
		if (ids_.empty() || topk <= 0) return {};
		const int maxL = max_level_.load(std::memory_order_acquire);
		if (maxL < 0) return {};

		const std::size_t dim = (std::size_t)cfg_.dim;
		const float q_norm = vecsearch::l2_norm_sqr(q, cfg_.dim);

		int ef = p_.ef_search;
		if (ef < topk) ef = topk;

		std::vector<Id> seeds;
		seeds.reserve(16);
		const Id ep = entry_point_.load(std::memory_order_acquire);
		seeds.push_back(ep);

		// Multi-entry narrowing on upper layers.
		for (int l = maxL; l > 0; --l) {
			const int ef_l = std::min(64, ef);
			auto top = search_layer_multi_(q, q_norm, seeds, ef_l, l);

			seeds.clear();
			int K = 8;
			if ((int)top.size() < K) K = (int)top.size();
			for (int i = 0; i < K; ++i) seeds.push_back(top[i].id);
			if (seeds.empty()) seeds.push_back(ep);
		}

		auto best = search_layer_multi_(q, q_norm, seeds, ef, 0);
		if ((int)best.size() > topk) best.resize((std::size_t)topk);

		// Map internal ids to external ids.
		for (auto& nb : best) nb.id = ids_[(std::size_t)nb.id];
		return best;
	}

	std::vector<std::vector<Neighbor>> HNSWIndex::search_batch(const float* queries,
																   int num_queries,
																   int topk) const {
		std::vector<std::vector<Neighbor>> out((std::size_t)num_queries);

	#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < num_queries; ++i) {
			const float* q = &queries[(std::size_t)i * (std::size_t)cfg_.dim];
			out[(std::size_t)i] = search_one(q, topk);
		}
		return out;
	}

}
