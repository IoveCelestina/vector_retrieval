#include "vecsearch/hnsw_index.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include<array>
#include<immintrin.h>

namespace vecsearch {
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
	}

	void HNSWIndex::clear() {
		ids_.clear();
		data_.clear();
		graph_.clear();
		has_entry_=false;
		entry_= 0;
		locks_.clear();
	}

	std::string HNSWIndex::params()const {
		return "M="+std::to_string(p_.M)+
				";efC="+std::to_string(p_.ef_construction)+
				";efS="+ std::to_string(p_.ef_search);
	}
	// float HNSWIndex::dist_l2_sqr(const float *a, const float *b) const {
	// 	float res=0;
	// 	for (int i = 0;i<cfg_.dim;++i) {
	// 		float d=a[i]-b[i];
	// 		res+=d*d;
	// 	}
	// 	return res;
	// }

	float HNSWIndex::dist_l2_sqr(const float *a, const float *b) const {
		const std::size_t d = (std::size_t)cfg_.dim;
		std::size_t i = 0;
		float res = 0;

#if defined(__AVX2__)
		__m256 sum = _mm256_setzero_ps();

		// 主循环：每次处理 8 个 float
		for (; i + 8 <= d; i += 8) {
			__m256 v1 = _mm256_loadu_ps(a + i);
			__m256 v2 = _mm256_loadu_ps(b + i);
			__m256 diff = _mm256_sub_ps(v1, v2);
			sum = _mm256_fmadd_ps(diff, diff, sum);
		}

		// 寄存器内水平求和
		// 将 8 个 float逐步折叠相加,避免存入内存
		__m128 sum_high = _mm256_extractf128_ps(sum, 1); // 取出高 128 位
		__m128 sum_low  = _mm256_castps256_ps128(sum);   // 取出低 128 位
		__m128 vres     = _mm_add_ps(sum_low, sum_high); // 8 -> 4
		vres = _mm_hadd_ps(vres, vres);                  // 4 -> 2
		vres = _mm_hadd_ps(vres, vres);                  // 2 -> 1

		float temp;
		_mm_store_ss(&temp, vres); //只存最后1个float
		res = temp;

#endif

		// 处理剩余的维度
		for (; i < d; ++i) {
			float diff = a[i] - b[i];
			res += diff * diff;
		}
		return res;
	}

	//SearchLayer，重要的？
	std::vector<Neighbor> HNSWIndex::search_layer_(const float *target, Id entry, int ef) const {
		if (ids_.empty()) return {};
		if (ef<=0) return{};
		if ((std::size_t)entry>=ids_.size()) return {};

		//使用static thread_local定义线程局部变量
		//每个线程第一次调用时会创建一个独立的local_visited_tag,后续调用会复用没有malloc开销且线程安全
		static thread_local std::vector<std::uint32_t> local_visited_tag;
		//还是需要保留tag去实现O(1)清空===========
		static thread_local std::uint32_t local_cur_tag=0;

		//因为是线程局部的，当主ids_扩容时，这里的局部副本也需要跟随扩容
		if(local_visited_tag.size()<ids_.size())  local_visited_tag.resize(ids_.size());

		//local tag++,处理溢出
		++local_cur_tag;
		if (local_cur_tag==0) {
			std::fill(local_visited_tag.begin(),local_visited_tag.end(),0);
			local_cur_tag = 1;
		}
		std::uint32_t tag = local_cur_tag;//兼容之前写的

		auto cmp_MinHeap=[&](const Neighbor &a,const Neighbor &b){return a.dist>b.dist;};
		auto cmp_MAXHeap=[&](const Neighbor &a,const Neighbor &b){return a.dist<b.dist;};
		std::priority_queue<Neighbor, std::vector<Neighbor>,decltype(cmp_MinHeap)> candidates(cmp_MinHeap);
		std::priority_queue<Neighbor,std::vector<Neighbor>,decltype(cmp_MAXHeap)> best(cmp_MAXHeap);
		//先写id连续的版本

		//初始化,把entry这个入口发进去
		const float* entry_vsc = &data_[(size_t)entry*(size_t)cfg_.dim];
		float d0 = dist_l2_sqr(entry_vsc,target);

		Neighbor e{entry,d0};
		candidates.push(e);
		best.push(e);
		//visited.insert(entry);
		local_visited_tag[entry]=tag;

		while (!candidates.empty()) {
			Neighbor cur = candidates.top();
			candidates.pop();
			//剪枝:当前候选比best最差还差，stop

			float worst = best.top().dist;
			if (cur.dist>worst) {
				break;
			}

			//扩展邻居
			auto &neighbors = graph_[cur.id];

			//=================内存预取开始=================
			//提前告诉CPU把邻居的向量数据从内存拉到L1缓存
			for (const auto& nb : neighbors) {
				if ((std::size_t)nb>=local_visited_tag.size()||local_visited_tag[nb]==tag) continue;
				// 计算该邻居向量在data_中的内存地址
				const char* vec= reinterpret_cast<const char*>(&data_[(std::size_t)nb * (std::size_t)cfg_.dim]);
				//_MM_HINT_T0表示预期稍后会频繁使用,拉到所有缓存层
				_mm_prefetch(vec, _MM_HINT_T0);
			}
			// ============内存预取结束====================


			for (Id nb: neighbors) {
				// if (visited.find(nb)!=visited.end()) continue;
				// visited.insert(nb);
				if ((std::size_t)nb>=local_visited_tag.size()||local_visited_tag[nb]==tag) continue;
				local_visited_tag[nb]=tag;
				const float *nb_vec = &data_[(size_t)nb*(size_t)cfg_.dim];
				float d = dist_l2_sqr(nb_vec,target);

				//如果best<ef,或者nb比最差的好
				if ((int) best.size()<ef || d<best.top().dist) {
					Neighbor cand{nb,d};
					candidates.push(cand);
					best.push(cand);

					//超过ef弹出
					if ((int) best.size()>ef) best.pop();
				}
			}
		}

		std::vector<Neighbor> res;
		res.reserve(best.size());
		while (!best.empty()) {
			res.emplace_back(best.top());
			best.pop();
		}
		std::reverse(res.begin(),res.end());//这样就是从小到到大了
		return res;
	}

	// 简单的选邻居:按 dist 升序取前 M
	std::vector<Id> HNSWIndex::select_neighbors_simple_(std::vector<Neighbor>& candidates,
	                                                   int M) const {
		if (M <= 0 || candidates.empty()) return {};
		auto by_dist=[](const Neighbor &a,const Neighbor &b){return a.dist<b.dist;};
		if (candidates.size () > M) {
			std::nth_element(candidates.begin(),candidates.begin()+M,candidates.end(),by_dist);
			candidates.resize(M);
		}
		std::sort(candidates.begin(), candidates.end(),by_dist);

		std::vector<Id> out;
		out.reserve(candidates.size());
		for (const auto& x : candidates) out.push_back(x.id);
		return out;
	}


	std::vector<Id> HNSWIndex::select_neighbors_heuristic_(Id center,
                                                      std::vector<Neighbor>& candidates,
                                                      int M) const {
    if (M <= 0 || candidates.empty()) return {};

    const std::size_t dim = (std::size_t)cfg_.dim;

    // 限制候选池大小,减少 cand-vs-selected 的距离计算
    const int L = std::min<int>((int)candidates.size(), std::max((3 * M) / 2, M));


    // search_layer_返回的candidates 已经是按dist 升序
    // 所以这里直接截断前L个即可，不再排序/不再nth_element
    if ((int)candidates.size() > L) {
        candidates.resize(L);
    }

    std::vector<Id> selected;
    selected.reserve((std::size_t)M);

    // Heuristic:避免邻居扎堆
    for (const auto& cand : candidates) {
        if ((int)selected.size() >= M) break;

        const Id cid = cand.id;
        if (cid == center) continue;
        if ((std::size_t)cid >= ids_.size()) continue;

        const float* cvec = &data_[(std::size_t)cid * dim];

        bool good = true;
        for (Id sid : selected) {
            const float* svec = &data_[(std::size_t)sid * dim];

            // 若 dist(cand, selected_neighbor) < dist(center, cand),说明 cand 与已有邻居太近
            const float d_cs = dist_l2_sqr(cvec, svec);
            if (d_cs < cand.dist) {
                good = false;
                break;
            }
        }

        if (good) selected.push_back(cid);
    }

    // 不足 M：按中心距离从近到远补齐(在前 L 个里补就够了)
    if ((int)selected.size() < M) {
        for (const auto& cand : candidates) {
            if ((int)selected.size() >= M) break;

            const Id cid = cand.id;
            if (cid == center) continue;
            if ((std::size_t)cid >= ids_.size()) continue;

            bool dup = false;
            for (Id sid : selected) {
                if (sid == cid) {
                    dup = true;
                    break;
                }
            }
            if (!dup) selected.push_back(cid);
        }
    }

    return selected;
}


	// 裁剪邻居表:按 dist(center, nb) 排序，保留前 M
	void HNSWIndex::prune_neighbors_by_distance_(Id center,
	                                            std::vector<Id>& neighbor_list,
	                                            int M) {
		if ((int)neighbor_list.size()<=1) return;
		if (M<=0) {
			neighbor_list.clear();
			return;
		}

		if ((int)neighbor_list.size() > 2 * M) {
			std::sort(neighbor_list.begin(), neighbor_list.end());
			neighbor_list.erase(std::unique(neighbor_list.begin(), neighbor_list.end()), neighbor_list.end());
		}//只在数组过大时才做sort+unique，避免小数组sort开销


		std::vector<Neighbor> res;
		res.reserve(neighbor_list.size());

		const float* center_vsc = &data_[(size_t)center*(size_t)cfg_.dim];

		for (const Id &nb : neighbor_list) {
			if (nb==center||(size_t)nb>=ids_.size()) continue;//防止越界
			res.emplace_back(nb,dist_l2_sqr(&data_[(size_t)nb*(size_t)cfg_.dim],center_vsc));
			//一样的算距离放进去
		}

		auto by_dist = [](const Neighbor &a,const Neighbor &b) {
			return a.dist<b.dist;
		};


		if ((int)res.size() > 2 * M) {
			std::nth_element(res.begin(), res.begin() + M, res.end(), by_dist);
			res.resize(M);
			std::sort(res.begin(), res.end(), by_dist);
		} else {
			std::sort(res.begin(), res.end(), by_dist);
			if ((int)res.size() > M) res.resize(M);
		}//有些时候线性去重比sort+unique慢


		neighbor_list.clear();
		neighbor_list.reserve(res.size());
		for (const auto &x : res) neighbor_list.push_back(x.id);
	}

	void HNSWIndex::connect_bidirectional_(Id u, const std::vector<Id>& neigh) {
		if ((std::size_t)u >= graph_.size()) return;
		if (neigh.empty()) return;

		// 先过滤：去掉 self / 越界（不去重）
		std::vector<Id> buf;
		buf.reserve(neigh.size());
		for (Id nb : neigh) {
			if (nb == u) continue;
			if ((std::size_t)nb >= graph_.size()) continue;
			buf.push_back(nb);
		}
		if (buf.empty()) return;

		// 混合去重：
		// - 很小：线性去重（分支少、无 sort）
		// - 大一点：sort+unique（避免 O(t^2)）
		constexpr int LINEAR_UNIQ_THRESHOLD = 16;

		std::vector<Id> uniq;
		if ((int)buf.size() <= LINEAR_UNIQ_THRESHOLD) {
			uniq.reserve(buf.size());
			for (Id nb : buf) {
				bool dup = false;
				for (Id x : uniq) {
					if (x == nb) { dup = true; break; }
				}
				if (!dup) uniq.push_back(nb);
			}
		} else {
			uniq = std::move(buf);
			std::sort(uniq.begin(), uniq.end());
			uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
		}

		//更新当前节点u,必须加大括号限制锁的范围
		{
			std::lock_guard<std::mutex> lock(*locks_[u]);
			for (Id nb : uniq) {
				graph_[u].emplace_back(nb);
			}
			if ((int)graph_[u].size() > p_.M) {
				prune_neighbors_by_distance_(u, graph_[u], p_.M);
			}
		}//出这个括号锁自动释放

		// prune nb
		for (Id nb : uniq) {
			std::lock_guard<std::mutex> lock(*locks_[nb]);
			graph_[nb].emplace_back(u);
			if ((int)graph_[nb].size() > p_.M) {
				prune_neighbors_by_distance_(nb, graph_[nb], p_.M);
			}
		}
	}



	// ===== add_batch：插入 + 建图 =====
	void HNSWIndex::add_batch(const std::vector<Id>& ids,
	                          const std::vector<float>& vectors) {
		const std::size_t n = ids.size();
		if (n==0) return;
		if (vectors.size() != n * (std::size_t)cfg_.dim) {
			std::cerr << "HNSWIndex::add_batch: vectors size mismatch\n";
			std::exit(1);
		}
		//这里假设Id是连续的，先把数据append仅需然后在search_layer才不会越界
		std::size_t oldN = ids_.size();
		std::size_t dim = (size_t)cfg_.dim;
		ids_.reserve(oldN+n);
		data_.reserve((oldN+n) * dim);
		graph_.reserve(oldN+n);
		//初始化locks_
		if (locks_.size() < oldN+n) {
			locks_.reserve(oldN+n);
			for (std::size_t i = locks_.size();i<oldN+n;i++) {
				locks_.emplace_back(std::make_unique<std::mutex>());
			}
		}

		for (std::size_t  i = 0;i<n;++i) {
			const Id expected_id = (Id)(oldN + i);
			if (ids[i] != expected_id) {
				std::cerr << "Only continuous ids supported: expect "<<expected_id<< ", got " << ids[i]<<"\n";
				std::exit(1);
			}
			ids_.emplace_back(ids[i]);
			const float* vec = &vectors[i*dim];
			data_.insert(data_.end(), vec, vec+dim);
			graph_.emplace_back();//空邻接表
			graph_.back().reserve(p_.M<<1);
		}

		//初始化入口，第一次插入的时候
		if (!has_entry_) {
			has_entry_ = true;
			entry_=0;
		}
		//对新插入的点建图
		//如果oldN=0，第0个点作为入口不需要连边，从u=1开始
		std::size_t start_u = (oldN==0)?1:oldN;
		//循环开始前,把当前的entry拿出来,存在局部变量里,这样循环里的所有线程都只读这个局部变量,互不干扰
		//循环开始前,把当前的entry拿出来,存在局部变量里,这样循环里的所有线程都只读这个局部变量,互不干扰
		Id current_entry = entry_;
		#pragma omp parallel for schedule(dynamic)
		for (std::size_t u = start_u;u<oldN+n;++u) {
			const float *target = &data_[u*dim];
			//使用局部变量进行搜索，都是读操作，线程安全
			auto candidates = search_layer_(target,current_entry,p_.ef_construction);

			auto neigh = select_neighbors_heuristic_((Id)u, candidates, p_.M);
			//去掉自己保险一点
			neigh.erase(std::remove(neigh.begin(), neigh.end(),(Id)u),neigh.end());

			//连线,内部有锁,安全
			connect_bidirectional_((Id)u, neigh);
			//不在这里写entry_= u,优化性能
		}

		//等所有线程都干完活统一更新一次
		//这里(oldN+n-1)就是这一批最后一个点的ID
		if (oldN+n>0) {
			entry_ = (Id)(oldN+n-1);
		}
		//局部读+统一写是高性能并行标准写法
	}

	// ===== search_one:查询 =====
	std::vector<Neighbor> HNSWIndex::search_one(const float* q, int topk) const {
		if (!has_entry_ || ids_.empty() || topk <= 0) return {};
		if (topk > (int)ids_.size()) topk = (int)ids_.size();

		int ef = p_.ef_search;
		if (ef < topk) ef = topk;
		std::vector<Neighbor> best = search_layer_(q,entry_,ef);//找谁，入口，找几个(最多几个)
		//ef_search是候选集合大小上限，需要确保大于等于topk，返回的已经是升序，所以下一步不用排序
		if (best.size()>topk) best.resize(topk);
		return best;
	}

	std::vector<std::vector<Neighbor>> HNSWIndex::search_batch(const float* queries,
	                                                           int num_queries,
	                                                           int topk) const {
	  std::vector<std::vector<Neighbor>> out;
	  out.reserve(num_queries);
	  for (int i = 0; i < num_queries; ++i) {
	    const float* q = &queries[(std::size_t)i * (std::size_t)cfg_.dim];
	    out.push_back(search_one(q, topk));
	  }
	  return out;
	}

}