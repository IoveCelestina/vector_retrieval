#include "vecsearch/hnsw_index.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <unordered_set>

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
	}

	std::string HNSWIndex::params()const {
		return "M="+std::to_string(p_.M)+
				";efC="+std::to_string(p_.ef_construction)+
				";efS="+ std::to_string(p_.ef_search);
	}
	float HNSWIndex::dist_l2_sqr(const float *a, const float *b) const {
		float res=0;
		for (int i = 0;i<cfg_.dim;++i) {
			float d=a[i]-b[i];
			res+=d*d;
		}
		return res;
	}

	//SearchLayer，重要的？
	std::vector<Neighbor> HNSWIndex::search_layer_(const float *target, Id entry, int ef) const {
		if (ids_.empty()) return {};
		if (ef<=0) return{};
		if ((std::size_t)entry>=ids_.size()) return {};

		auto cmp_MinHeap=[&](const Neighbor &a,const Neighbor &b){return a.dist>b.dist;};
		auto cmp_MAXHeap=[&](const Neighbor &a,const Neighbor &b){return a.dist<b.dist;};
		std::priority_queue<Neighbor, std::vector<Neighbor>,decltype(cmp_MinHeap)> candidates(cmp_MinHeap);
		std::priority_queue<Neighbor,std::vector<Neighbor>,decltype(cmp_MAXHeap)> best(cmp_MAXHeap);
		//先写id连续的版本
		// std::unordered_set<Id> visited;
		// visited.reserve((std::size_t)ef*4);
		std::vector<std::uint8_t> visited(ids_.size(), 0);

		//初始化,把entry这个入口发进去
		const float* entry_vsc = &data_[(size_t)entry*(size_t)cfg_.dim];
		float d0 = dist_l2_sqr(entry_vsc,target);

		Neighbor e{entry,d0};
		candidates.push(e);
		best.push(e);
		//visited.insert(entry);
		visited[entry]=1;

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
			for (Id nb: neighbors) {
				// if (visited.find(nb)!=visited.end()) continue;
				// visited.insert(nb);
				if ((size_t)nb>=visited.size()||visited[nb]) continue;
				visited[nb]=1;
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
		return res;
	}

	// 简单的选邻居:按 dist 升序取前 M
	std::vector<Id> HNSWIndex::select_neighbors_simple_(std::vector<Neighbor>& candidates,
	                                                   int M) const {
	  std::sort(candidates.begin(), candidates.end(),
	            [&](const Neighbor& a, const Neighbor& b) { return a.dist < b.dist; });
	  if ((int)candidates.size() > M) candidates.resize(M);

	  std::vector<Id> out;
	  out.reserve(candidates.size());
	  for (const auto& x : candidates) out.push_back(x.id);
	  return out;
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

		std::sort(neighbor_list.begin(), neighbor_list.end());
		neighbor_list.erase(std::unique(neighbor_list.begin(), neighbor_list.end()),
							neighbor_list.end());//id去个重先

		std::vector<Neighbor> res;
		res.reserve(neighbor_list.size());

		const float* center_vsc = &data_[(size_t)center*(size_t)cfg_.dim];

		for (const Id &nb : neighbor_list) {
			if (nb==center||(size_t)nb>=ids_.size()) continue;//防止越界
			res.emplace_back(nb,dist_l2_sqr(&data_[(size_t)nb*(size_t)cfg_.dim],center_vsc));
			//一样的算距离放进去
		}
		sort(res.begin(), res.end(),[&](const auto &a,const auto &b) {
			return a.dist<b.dist;
		});
		if ((int) res.size()>M) res.resize(M);

		neighbor_list.clear();
		neighbor_list.reserve(res.size());
		for (const auto &x : res) neighbor_list.push_back(x.id);
	}

	void HNSWIndex::connect_bidirectional_(Id u, const std::vector<Id>& neigh) {
		if ((std::size_t)u >= graph_.size()) return;

		std::vector<Id> uniq = neigh;
		std::sort(uniq.begin(), uniq.end());
		uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());

		for (Id nb : uniq) {
			if (nb == u) continue;
			if ((std::size_t)nb >= graph_.size()) continue;
			graph_[u].push_back(nb);
			graph_[nb].push_back(u);
		}

		if ((int)graph_[u].size() > p_.M) {
			prune_neighbors_by_distance_(u, graph_[u], p_.M);
		}

		for (Id nb : uniq) {
			if (nb == u) continue;
			if ((std::size_t)nb >= graph_.size()) continue;
			if ((int)graph_[nb].size() > p_.M) {
				prune_neighbors_by_distance_(nb, graph_[nb], p_.M);
			}
		}
	}


	// ===== add_batch：插入 + 建图 =====
	void HNSWIndex::add_batch(const std::vector<Id>& ids,
	                          const std::vector<float>& vectors) {
	  const std::size_t n = ids.size();
	  if (vectors.size() != n * (std::size_t)cfg_.dim) {
	    std::cerr << "HNSWIndex::add_batch: vectors size mismatch\n";
	    std::exit(1);
	  }

	  // TODO(简化假设)：
	  // - 先假设 ids = 0..N-1 且连续（你现在 gen_ids 就是这样）
	  // - 直接 ids_ = ids; data_ = vectors
	  // - graph_.resize(N)
	  //
	  // 然后逐点插入建图：
	  // for each id:
	  //   if first: entry_=id; continue
	  //   candidates = search_layer_(vec_of_id, entry_, p_.ef_construction)
	  //   neigh = select_neighbors_simple_(candidates, p_.M)
	  //   connect_bidirectional_(id, neigh)
	  //   entry_ = id (可选)
	  (void)ids;
	  (void)vectors;
	}

	// ===== search_one:查询 =====
	std::vector<Neighbor> HNSWIndex::search_one(const float* q, int topk) const {
	  if (!has_entry_ || ids_.empty() || topk <= 0) return {};
	  if (topk > (int)ids_.size()) topk = (int)ids_.size();

	  int ef = p_.ef_search;
	  if (ef < topk) ef = topk;

	  // TODO：
	  // - best = search_layer_(q, entry_, ef)
	  // - sort best by dist asc
	  // - resize topk
	  (void)q;
	  return {};
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