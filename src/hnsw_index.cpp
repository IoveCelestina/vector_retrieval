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
		//固定种子以便复现
		level_generator_.seed(2026);
		//均匀分布[0.0,1.0]
		level_distribution_ = std::uniform_real_distribution<double>(0.0,1.0);
		//计算层数因子 mult = 1/ln(M)
		mult_ = 1.0/std::log(1.0*p_.M);
	}

	void HNSWIndex::clear() {
		ids_.clear();
		data_.clear();
		graph_.clear();
		node_locks_.clear();
		current_max_level_ = -1;
		entry_point_ = 0;
		//随机生成器状态无需重置
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
#if defined(__FMA__)
			sum = _mm256_fmadd_ps(diff, diff, sum);
#else
			__m256 mul = _mm256_mul_ps(diff, diff);
			sum = _mm256_add_ps(sum, mul);
#endif
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

	int HNSWIndex::get_random_level(){
		//HNSW论文公式 level = floor(-ln(uniform(0,1)*mult),这会让大部分节点停在Level 0,极少数升到高层
		double r = level_distribution_(level_generator_);
		if (r<1e-9) r = 1e-9;//防止log 0
		int level = (int)(-std::log(r)*mult_);
		return level;
	}


	std::vector<Id> HNSWIndex::select_neighbors_heuristic_(Id center,
                                                      std::vector<Neighbor>& candidates,
                                                      int M) const {
	    if (M <= 0 || candidates.empty()) return {};

	    const std::size_t dim = (std::size_t)cfg_.dim;


	    std::vector<Id> selected;
	    selected.reserve((std::size_t)M);

		//宽松因子,默认(1.0)严格RNG,可以稍微变大用于提升recall
		//意思是及时neighbor离cand更近,只要没近太多,还是保留cand
		const float alpha =1.05f;
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
	            if (d_cs * alpha < cand.dist) {
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


	void HNSWIndex::prune_neighbors_heuristic_(Id center, std::vector<Id> &neighbor_list, int M) {
		if ((int)neighbor_list.size()<=M) return;
		std::vector<Neighbor> cands;
		cands.reserve(neighbor_list.size());
		const float* center_vsc = &data_[(size_t)center*(size_t)cfg_.dim];

		for (Id nb:neighbor_list) {
			if (nb==center||(std::size_t)nb>=ids_.size()) continue;
				float d = dist_l2_sqr(&data_[(size_t)nb*(size_t)cfg_.dim],center_vsc);
				cands.emplace_back(nb,d);
		}
		std::sort(cands.begin(), cands.end(),[](const Neighbor &a,const Neighbor &b) {
			return a.dist<b.dist;
		});

		auto selected = select_neighbors_heuristic_(center, cands, M);
		neighbor_list = std::move(selected);

	}


	std::vector<Neighbor> HNSWIndex::search_layer_(const float* target, Id entry, int ef, int level) const {
        // 基础检查
        if (ids_.empty()) return {};
        if (ef <= 0) return {};
        if ((std::size_t)entry >= ids_.size()) return {};

        // 初始化 Visited Tag (线程安全 & 高性能)
        static thread_local std::vector<std::uint32_t> local_visited_tag;
        static thread_local std::uint32_t local_cur_tag = 0;

        // 扩容检查
        if (local_visited_tag.size() < ids_.size()+100) {//加个buffer
            local_visited_tag.resize(ids_.size()+100, 0);
        }

        //Tag更新(避免 memset)
        local_cur_tag++;
        if (local_cur_tag == 0) {
            std::fill(local_visited_tag.begin(), local_visited_tag.end(), 0);
            local_cur_tag = 1;
        }
        std::uint32_t tag = local_cur_tag;

        // 初始化候选堆
        auto cmp_MinHeap = [&](const Neighbor &a, const Neighbor &b) { return a.dist > b.dist; };
        auto cmp_MAXHeap = [&](const Neighbor &a, const Neighbor &b) { return a.dist < b.dist; };

        // candidates:小根堆,存待探索的节点(离target最近的在最上面)
        std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp_MinHeap)> candidates(cmp_MinHeap);
        // best: 大根堆,存目前找到的Top-EF(离 target 最远的在顶)
        std::priority_queue<Neighbor, std::vector<Neighbor>, decltype(cmp_MAXHeap)> best(cmp_MAXHeap);

        const float* entry_vsc = &data_[(size_t)entry * (size_t)cfg_.dim];
        float d0 = dist_l2_sqr(entry_vsc, target);

        Neighbor e{entry, d0};
        candidates.push(e);
        best.push(e);
        local_visited_tag[entry] = tag;

        while (!candidates.empty()) {
            Neighbor cur = candidates.top();
            candidates.pop();

            float worst_dist = best.top().dist;
            if ((int)best.size() >= ef && cur.dist > worst_dist) break; // 剪枝:当前最近的候选比best里最差的还远且best够了,没必要找了

            // 访问 graph_[cur.id]的第[level] 层
            // 安全检查:防止该节点没有这一层 (虽然理论上entry保证了都在同一层)
            if (level >= (int)graph_[cur.id].size()) continue;

        	static thread_local std::vector<Id> neighbors_buffer;
            {
            	// 锁住 cur.id，只为了安全地拷贝一份邻居列表
            	std::lock_guard<std::mutex> lock(*node_locks_[cur.id]);
            	neighbors_buffer = graph_[cur.id][level];
            }

            //内存预取 (Prefetch)
            for (const auto& nb : neighbors_buffer) {
                if ((std::size_t)nb < local_visited_tag.size() && local_visited_tag[nb] != tag) {
                    const char* vec = reinterpret_cast<const char*>(&data_[(std::size_t)nb * (std::size_t)cfg_.dim]);
                    _mm_prefetch(vec, _MM_HINT_T0);
                }
            }
            for (Id nb : neighbors_buffer) {
                if ((std::size_t)nb >= local_visited_tag.size() || local_visited_tag[nb] == tag) continue;
                local_visited_tag[nb] = tag;

                const float *nb_vec = &data_[(size_t)nb * (size_t)cfg_.dim];
                float d = dist_l2_sqr(nb_vec, target);

                if ((int)best.size() < ef || d < best.top().dist) {
                    Neighbor cand{nb, d};
                    candidates.push(cand);
                    best.push(cand);
                    if ((int)best.size() > ef) best.pop(); // 保持 best大小不超过 ef
                }
            }
        }

		//结果
        std::vector<Neighbor> res;
        res.reserve(best.size());
        while (!best.empty()) {
            res.emplace_back(best.top());
            best.pop();
        }
        std::reverse(res.begin(), res.end()); // 从近到远排序
        return res;
    }

	void HNSWIndex::connect_bidirectional_(Id u, const std::vector<Id>& neigh, int level) {
        if ((std::size_t)u >= graph_.size()) return;
        if (neigh.empty()) return;

        // 过滤与去重
        // 去掉 u 自己,去掉越界的 ID
        std::vector<Id> buf;
        buf.reserve(neigh.size());
        for (Id nb : neigh) {
            if (nb == u) continue;
            // 简单检查 graph 大小，防止访问不存在的节点
            if ((std::size_t)nb >= graph_.size()) continue;
            buf.push_back(nb);
        }
        if (buf.empty()) return;

        // 去重逻辑:小规模用线性查，大规模用 sort+unique
        constexpr int LINEAR_UNIQ_THRESHOLD = 16;
        std::vector<Id> uniq;
        if ((int)buf.size() <= LINEAR_UNIQ_THRESHOLD) {
            uniq.reserve(buf.size());
            for (Id nb : buf) {
                bool dup = false;
                for (Id x : uniq) { if (x == nb) { dup = true; break; } }
                if (!dup) uniq.push_back(nb);
            }
        } else {
            uniq = std::move(buf);
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        }

        //第0层允许两倍的M,其他层用 M
        int M_max = (level == 0) ? (p_.M<<1) : p_.M;

        //更新节点 u (正向连接)
        {
            std::lock_guard<std::mutex> lock(*node_locks_[u]);
            // 确保 u 有这一层 (add_batch 初始化时应该保证了，但多加检查无害)
            if (level < (int)graph_[u].size()) {
                for (Id nb : uniq) {
                    graph_[u][level].emplace_back(nb);
                }
                // 如果边太多,裁剪
                if ((int)graph_[u][level].size() > M_max) {
                    prune_neighbors_heuristic_(u, graph_[u][level], M_max);
                }
            }
        }

        // 更新邻居节点 nb (反向连接)
        for (Id nb : uniq) {
            // 邻居 nb 必须也拥有第 level 层，否则不能连
            // 在 HNSW 插入逻辑中，我们是在现有图上走的，遇到的 nb 理论上肯定有这层
            if (level >= (int)graph_[nb].size()) continue;

            std::lock_guard<std::mutex> lock(*node_locks_[nb]);

            // 把 u 加到 nb 的邻居表里
            graph_[nb][level].emplace_back(u);

            // 裁剪 nb 的边
            if ((int)graph_[nb][level].size() > M_max) {
                prune_neighbors_heuristic_(nb, graph_[nb][level], M_max);
            }
        }
    }


	void HNSWIndex::add_batch(const std::vector<Id> &ids, const std::vector<float> &vectors) {
		const std::size_t n = ids.size();
		if (n==0) return;
		if (vectors.size()!=n*(std::size_t)cfg_.dim) {
			std::cerr << "HNSWIndex::add_batch: vectors size mismatch\n";
			std::exit(1);
		}

		std::size_t oldN = ids_.size();
		std::size_t dim = (size_t)cfg_.dim;

		//预分配主数据
		ids_.reserve(oldN+n);
		data_.reserve((oldN+n)*dim);
		graph_.reserve(oldN+n);
		//初始化锁
		if (node_locks_.size()<oldN+n) {
			node_locks_.reserve(oldN+n);
			for (std::size_t i = node_locks_.size(); i<oldN+n; i++) {
				node_locks_.emplace_back(std::make_unique<std::mutex>());
			}
		}


		std::vector<int> new_levels(n);
		int batch_max_level = -1;
		Id batch_max_level_id = 0;
		std::size_t batch_max_level_local_id = 0;//内部最好的索引,不加oldN
		for (std::size_t i = 0;i<n;++i) {
			new_levels[i] = get_random_level();
			if (new_levels[i] > batch_max_level) {
				batch_max_level = new_levels[i];
				batch_max_level_local_id = i;
			}
		}

		//提升recall的做法，如果是第一批数据，强制吧最高点换到第0点
		//这样entry_point 一上来就是最高层,我们需要维护一个映射数组,避免物理移动vectors数据太慢
		std::vector<std::size_t> permuitation(n);
		for (std::size_t i = 0;i<n;++i) permuitation[i] = i;

		if (oldN==0) {
			std::swap(permuitation[0],permuitation[batch_max_level_local_id]);
			//注意new_levels也要跟着换，方便后面直接查
			// std::swap(new_levels[0],new_levels[batch_max_level_local_id]);
		}//batch_max_level_id 赋值的是全局id = oldN+i,当oldN==0时恰好等于i,一旦不等于0,batch_max_level_id会大于等于n，会越界


		std::vector<int> levels_internal(n);
		for (std::size_t i = 0; i < n; ++i) {
			std::size_t real_idx = permuitation[i];
			levels_internal[i] = new_levels[real_idx];
		}

		//开始真正插入,所有访问都要经过permutation
		for (std::size_t i = 0;i<n;++i) {
			std::size_t real_idx = permuitation[i];
			ids_.emplace_back(ids[real_idx]);
			const float *vec = &vectors[real_idx*dim];
			data_.insert(data_.end(), vec, vec+dim);

			int level = levels_internal[i];


			graph_.emplace_back(level+1);
			for (int l = 0;l<=level;++l) {
				int M_max = (l==0)?(p_.M<<1):p_.M;
				graph_.back()[l].reserve(M_max+1);
			}
		}
		//seeding 串行插入骨架
		std::size_t seed_len = (oldN==0)?std::min(n,(std::size_t)1000):0;

		if (oldN==0) {
			entry_point_= 0;
			current_max_level_ = levels_internal[0];

			//注意数据已经是按permutation顺序存入ids_和data_了，在后面直接访问data_即可

			for (std::size_t u = 1;u<seed_len;++u) {
				int max_level = levels_internal[u];
				const float *target = &data_[u*dim];//vector是原始乱序的，而我们在seeding过程是经过交换的，写成&vectors是错误的

				//降落
				Id curr_obj = entry_point_;
				if (current_max_level_ > max_level) {
					curr_obj = greedy_descent_(target,entry_point_,current_max_level_, max_level);
				}
				//连边&搜索
				for (int l = std::min(max_level, current_max_level_); l >= 0; --l) {
					auto best = search_layer_(target, curr_obj, p_.ef_construction, l);
					int M = (l == 0) ? (p_.M << 1) : p_.M;
					auto selected = select_neighbors_heuristic_((Id)u, best, M);
					connect_bidirectional_((Id)u, selected, l);
					if (!best.empty()) curr_obj = best[0].id;
				}


				//更新入口
				if (max_level > current_max_level_) {
					current_max_level_ = max_level;
					entry_point_ = (Id)u;
				}
			}
		}



		Id curr_global_entry = entry_point_;
		int curr_global_max_level = current_max_level_;
		//这里的curr_global_max_level现在已经是最高层了

		//分块并行
		std::size_t start_u = (oldN==0)?seed_len:oldN;
		std::size_t end_u = oldN+n;


		const std::size_t chunk_size = 2000;

		for (std::size_t chunk_start = start_u; chunk_start<end_u; chunk_start += chunk_size) {
			std::size_t chunk_end = std::min(end_u,chunk_start+chunk_size);


			//获取当前最新的全局入口快照(由主线程读取,是安全的)
			Id snapshot_entry = entry_point_;
			int snapshot_max_level = current_max_level_;

			#pragma omp parallel for schedule(dynamic)
			for (std::size_t u = chunk_start; u < chunk_end; ++u) {
				int max_level = levels_internal[u - oldN];
				const float* target = &data_[u * dim];

				Id curr_obj = snapshot_entry; // 必须从已构建好的快照入口开始

				// 先从全局最高层贪婪降落到 max_level+1（停在 max_level 层的入口附近）
				if (snapshot_max_level > max_level) {
					curr_obj = greedy_descent_(target, curr_obj, snapshot_max_level, max_level);
				}

				// 从 min(max_level, snapshot_max_level) 一路到 0 层：search + connect
				for (int l = std::min(max_level, snapshot_max_level); l >= 0; --l) {
					auto best = search_layer_(target, curr_obj, p_.ef_construction, l);

					int M = (l == 0) ? (p_.M << 1) : p_.M;
					auto selected = select_neighbors_heuristic_((Id)u, best, M);
					connect_bidirectional_((Id)u, selected, l);

					// 下一层入口：本层找到的最近点（best 已经是从近到远）
					if (!best.empty()) curr_obj = best[0].id;
				}
			}


			//并行结束,先找出当前 Chunk 内的"局部最强"
			int chunk_max_level = -1;
			Id chunk_best_id = 0;
			for (std::size_t u = chunk_start; u < chunk_end; ++u) {
				// new_levels 是相对 batch 的,u 是全局的，所以要减 oldN
				int level = levels_internal[u - oldN];

				// 这里使用 > 而不是 >=，意味着如果有多个同为最高层的点，
				// 我们优先选择 ID 较小的（先遇到的）。
				// 这样做有利于 Cache（旧数据可能更热），且保证确定性。
				if (level > chunk_max_level) {
					chunk_max_level = level;
					chunk_best_id = (Id)u;
				}
			}

			// 与全局比较,做一次性更新
			if (chunk_max_level > current_max_level_) {
				current_max_level_ = chunk_max_level;
				entry_point_ = chunk_best_id;

				// Debug Log
				// std::cout << "Entry point updated to " << entry_point_
				//           << " at level " << current_max_level_ << "\n";
			}
		}
	}


	Id HNSWIndex::greedy_descent_(const float *target, Id entry, int from_level, int to_level) const {
		Id curr_obj = entry;
		static thread_local std::vector<Id> neighbors_copy;

		for (int l = from_level; l > to_level; --l) {
			bool changed = true;
			while (changed) {
				changed = false;
				float min_dist = dist_l2_sqr(&data_[curr_obj * (size_t)cfg_.dim], target);
				// 安全检查
				if (l >= (int)graph_[curr_obj].size()) break;
				// 贪婪搜索
				{
					std::lock_guard<std::mutex> lock(*node_locks_[curr_obj]);
					neighbors_copy = graph_[curr_obj][l];
				}
				for (Id nb : neighbors_copy) {
					// 让 CPU 提前拉取数据
					_mm_prefetch((const char*)&data_[nb * (size_t)cfg_.dim], _MM_HINT_T0);
				}
				for (Id nb : neighbors_copy) {
					float d = dist_l2_sqr(&data_[nb * (size_t)cfg_.dim], target);
					if (d < min_dist) {
						min_dist = d;
						curr_obj = nb;
						changed = true;
					}
				}
			}
		}
		return curr_obj;
	}




	// ===== search_one:查询 =====
	std::vector<Neighbor> HNSWIndex::search_one(const float* q, int topk) const {
		if (ids_.empty() || topk <= 0) return {};
		if (current_max_level_==-1) return {};

		Id curr_obj = entry_point_;
		for (int l = current_max_level_;l > 0;--l) {
			// 在层 l 做纯贪心:一直找更近的邻居直到不能改进
			curr_obj = greedy_descent_(q, curr_obj, l, l-1); // 或者写个 greedy_one_level(curr,l)
		}

		int ef = p_.ef_search;
		if (ef < topk) ef = topk;
		std::vector<Neighbor> best = search_layer_(q,curr_obj,ef,0);//找谁，入口，找几个(最多几个),第几层
		//ef_search是候选集合大小上限，需要确保大于等于topk，返回的已经是升序，所以下一步不用排序
		if (best.size()>topk) best.resize(topk);
		for (auto &nb:best) {
			nb.id = ids_[nb.id];
			//关键，需要内部id变回外部id,因为我的bench_runner传了外部id,这是导致recall降低的关键原因
		}
		return best;
	}

	std::vector<std::vector<Neighbor>> HNSWIndex::search_batch(const float* queries,
	                                                           int num_queries,
	                                                           int topk) const {
	    std::vector<std::vector<Neighbor>> out(num_queries);

		//开启并行
		#pragma omp parallel for schedule(dynamic)
	    for (int i = 0; i < num_queries; ++i) {
	      const float* q = &queries[(std::size_t)i * (std::size_t)cfg_.dim];
	      out[i]  = search_one(q,topk);
	    }
	    return out;
	}

}
