#include "vecsearch/ivf_flat_index.h"
#include "vecsearch/distance.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace vecsearch {
	IVFFlatIndex::IVFFlatIndex(IndexConfig cfg,IVFParams params):cfg_(cfg),p_(params){
		if (cfg_.dim <= 0) {
			std::cerr << "IVFFlatIndex: dim must be > 0\n";
			std::exit(1);
		}
		if (p_.nlist <= 0) p_.nlist = 100;
		if (p_.nprobe <= 0) p_.nprobe = 1;
		if (p_.nprobe > p_.nlist) p_.nprobe = p_.nlist;

		inverted_lists_ids_.resize(p_.nlist);
		inverted_lists_vecs_.resize(p_.nlist);
	}


    void IVFFlatIndex::clear() {
        centroids_.clear();
        for(auto &list:inverted_lists_ids_) list.clear();
        for(auto &list:inverted_lists_vecs_) list.clear();
    	total_count_ =0;
        is_trained_ = false;
    }

	std::string IVFFlatIndex::params() const {
		return "nlist=" + std::to_string(p_.nlist) +
			   ";nprobe=" + std::to_string(p_.nprobe) +
			   ";iter=" + std::to_string(p_.kmeans_iters);
	}

    //简单的K-means实现
    void IVFFlatIndex::train_centroids_(const float* data, std::size_t n) {
		int d = cfg_.dim;
        int k = p_.nlist;

		if (n < (std::size_t)k) {
			std::cerr << "IVF Warning: Training points (" << n << ") < nlist (" << k << "). Reducing nlist.\n";
			k = (int)n; // 强制降级
			// 实际上这时应该报错或抛异常，这里做简单处理
		}
		centroids_.resize(k * d);
        //随机选择k个点作为中心,为简化直接选前k个
        for(int i = 0;i<k;++i){
          const float* src = data + (std::size_t)i * d;
          float* dst = &centroids_[i*d];
          std::copy(src,src+d,dst);
        }

        std::vector<int> assign(n);
		std::vector<float> new_centroids(k * d, 0.0f);
		std::vector<int> counts(k, 0);

        //迭代
        for(int iter = 0;iter<p_.kmeans_iters;++iter){
            //E-step：分配
            #pragma omp parallel for schedule(static)
            for(std::int64_t i = 0;i<(std::int64_t)n;++i){
            	const float *vec = data + (std::size_t)i * d;//换种写法熟悉一下
                float min_dist = std::numeric_limits<float>::max();
                int best_c = 0;

                //暴力扫描所有中心
                for(int c = 0;c<k;++c){
                	float dist = vecsearch::l2_sqr(vec,&centroids_[c*d],d);
                    if(dist < min_dist){
                      	min_dist = dist;
                      	best_c = c;
                    }
                }
				assign[i] = best_c;
            }


            //M-step:更新中心
            std::fill(new_centroids.begin(),new_centroids.end(),0.0f);
            std::fill(counts.begin(),counts.end(),0);
			//累加,这里未做并行规约，简单串行累加
            for(std::size_t i = 0;i<n;++i){
            	int c = assign[i];
                counts[c]++;
                const float *vec = data + (std::size_t)i * d;
                float *center = &new_centroids[c*d];
                for(int j = 0;j<d;++j){
                	center[j] += vec[j];
                }
            }

            //平均
            for(int c = 0;c<k;++c){
                if(counts[c] > 0){
                	float scale = 1.0f/(float)counts[c];
                    for(int j = 0;j<d;++j){
                    	centroids_[c*d + j] = new_centroids[c*d + j] * scale;
                    }
                }else{
                 	//空簇处理：保持旧中心不变，或者随机重置。这里保持不变。
                }
            }
        }
		is_trained_ = true;
		std::cout << "IVF trained with " << n << " vectors. nlist=" << k << "\n";
	}

	void IVFFlatIndex::add_batch(const std::vector<Id>& ids, const std::vector<float>& vectors){
		const std::size_t n = ids.size();
		if (n == 0) return;
		// 如果还没训练，用第一批数据训练
		if (!is_trained_) {
			// 采样逻辑 (如果数据量太大，只取一部分训练)
			if (p_.train_max_points > 0 && n > (std::size_t)p_.train_max_points) {
				// 简单取前 train_max_points 个
				train_centroids_(vectors.data(), p_.train_max_points);
			} else {
				train_centroids_(vectors.data(), n);
			}
		}

        const int d = cfg_.dim;
		const int k = (int)inverted_lists_ids_.size(); // 实际 nlist

        //分配桶
        std::vector<int>assignment(n);
    	#pragma omp parallel for schedule(static)
        for(std::int64_t i = 0;i<(std::int64_t)n;++i){
          	const float *vec = &vectors[(std::size_t)i * d];
            //找最近中心
        	float min_dist = std::numeric_limits<float>::max();
        	int best_c = 0;
        	for (int c = 0; c < k; ++c) {
        		float dist = vecsearch::l2_sqr(vec, &centroids_[c * d], d);
        		if (dist < min_dist) {
        			min_dist = dist;
        			best_c = c;
        		}
        	}
        	assignment[i] = best_c;
        }
		//写入倒排表(串行写入或者分桶加锁)
		for (std::size_t i = 0; i < n; ++i) {
			int c = assignment[i];
			inverted_lists_ids_[c].push_back(ids[i]);

			const float* vec = &vectors[i * d];
			inverted_lists_vecs_[c].insert(inverted_lists_vecs_[c].end(), vec, vec + d);
		}
		total_count_ += n;
    }

	//找query最近的k个中心
    std::vector<std::pair<int,float> > IVFFlatIndex::find_nearest_centroids_(const float* q, int k) const {
		int nlist = (int) inverted_lists_ids_.size();
        if(k>nlist) k = nlist;

        std::vector<std::pair<int,float> >dists;
        dists.reserve(nlist);

        for(int c = 0;c<nlist;++c){
        	float d = vecsearch::l2_sqr(q,&centroids_[c*cfg_.dim],cfg_.dim);
            dists.emplace_back(c,d);
        }

		std::nth_element(dists.begin(), dists.begin() + k, dists.end(),
			[](const auto& a, const auto& b){ return a.second < b.second; });
        dists.resize(k);
        return dists;
    }


    std::vector<Neighbor> IVFFlatIndex::search_one(const float *q,int topk) const{
    	if(total_count_==0||topk<=0) return{};

        //粗搜,找最近的 nprobe个桶
        auto nearest_buckets = find_nearest_centroids_(q,p_.nprobe);

        //细搜:扫描这些桶
		std::vector<Neighbor> candidates;
        for(const auto &bucket: nearest_buckets){
          	int list_idx = bucket.first;
            const auto &ids = inverted_lists_ids_[list_idx];
            const auto &vecs = inverted_lists_vecs_[list_idx];
            std::size_t sz = ids.size();
        	for(std::size_t i = 0;i<sz;++i){
            	const float *vec = &vecs[(std::size_t)i * cfg_.dim];
                float dist = vecsearch::l2_sqr(q,vec,cfg_.dim);
        		candidates.emplace_back(ids[i],dist);
            }
        }

        //topk排序
		if (candidates.empty()) return {};
		if (topk > (int)candidates.size()) topk = (int)candidates.size();
		std::nth_element(candidates.begin(), candidates.begin() + topk, candidates.end(),
					 [](const Neighbor& a, const Neighbor& b) {
						 return a.dist < b.dist;
					 });
		candidates.resize(topk);
		std::sort(candidates.begin(), candidates.end(),
				  [](const Neighbor& a, const Neighbor& b) {
					  return a.dist < b.dist;
				  });
		return candidates;
    }


	std::vector<std::vector<Neighbor>> IVFFlatIndex::search_batch(const float* queries,
															  int num_queries,
															  int topk) const {
		std::vector<std::vector<Neighbor>> out(num_queries);
        #pragma omp parallel for schedule(dynamic)
		for(int i = 0;i<num_queries;++i){
			out[i] = search_one(&queries[i * cfg_.dim], topk);
		}
        return out;
    }

}//namespace