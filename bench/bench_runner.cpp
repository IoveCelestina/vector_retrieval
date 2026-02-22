#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <utility>
#include <vector>
#include<filesystem>
#include<fstream>
#include <cmath>       // std::ceil, double_t
#include <stdexcept>   // std::runtime_error
#include <sstream>
#include <cctype>

#include "../include/vecsearch/bruteforce_index.h"
#include "../include/vecsearch/hnsw_index.h"
#include "../include/vecsearch/ivf_flat_index.h"
#ifdef _OPENMP
#include <omp.h>
#endif


// 通用字符串 trim
static std::string trim_str(std::string s) {
    const char* ws = " \t\r\n";
    s.erase(0, s.find_first_not_of(ws));
    if (s.empty()) return s;
    s.erase(s.find_last_not_of(ws) + 1);
    return s;
}

//解析函数
static vecsearch::HNSWParams ParseHNSWParamsOrDie(const std::string& para_str) {
  vecsearch::HNSWParams p;//默认值
  auto trim=[](std::string s) {
    const char* ws = "\t\r\n";
    s.erase(0,s.find_first_not_of(ws));
    s.erase(s.find_last_not_of(ws)+1);
    return s;
  };
  //支持;和,
  std::string s = para_str;
  for (char &c:s) {
    if (c==',') c=';';
  }
  std::stringstream ss(s);
  std::string item;
  //小模拟
  while (std::getline(ss, item, ';')) {
    item = trim(item);
    if (item.empty()) continue;

    auto pos = item.find('=');
    if (pos == std::string::npos) {
      throw std::runtime_error("Bad hnsw param (missing '='): " + item);
    }

    std::string key = trim(item.substr(0, pos));
    std::string val = trim(item.substr(pos + 1));

    // key 统一转小写
    for (auto& ch : key) ch = (char)std::tolower((unsigned char)ch);

    int v = 0;
    try {
      v = std::stoi(val);
    } catch (...) {
      throw std::runtime_error("Bad hnsw param value: " + item);
    }

    if (key == "m") {
      p.M = v;
    } else if (key == "efc" || key == "ef_construction" || key == "efconstruction") {
      p.ef_construction = v;
    } else if (key == "efs" || key == "ef_search" || key == "efsearch") {
      p.ef_search = v;
    } else {
      throw std::runtime_error("Unknown hnsw param key: " + key);
    }
  }

  // 参数校验
  if (p.M <= 0 || p.ef_construction <= 0 || p.ef_search <= 0) {
    throw std::runtime_error("HNSW params must be > 0: " + para_str);
  }
  return p;
}


static vecsearch::IVFParams ParseIVFParamsOrDie(const std::string& para_str) {
    vecsearch::IVFParams p;
    std::string s = para_str;
    for (char &c:s) {if (c==',') c=';';}
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss,item,';')) {
        item = trim_str(item);
        if (item.empty()) continue;
        auto pos = item.find('=');
        if (pos==std::string::npos) throw std::runtime_error("Bad ivf param: " + item);
        std::string key = trim_str(item.substr(0, pos));
        std::string val = trim_str(item.substr(pos + 1));
        for (auto& ch : key) ch = (char)std::tolower((unsigned char)ch);
        int v = 0;
        try {
            v = std::stoi(val);
        }catch (...) {
            throw std::runtime_error("Bad ivf param value: " + item);
        }

        if (key == "nlist") {
          p.nlist = v;
        } else if (key == "nprobe") {
          p.nprobe = v;
        } else if (key == "iter" || key == "kmeans_iters") {
          p.kmeans_iters = v;
        } else if (key == "max_points" || key == "train_max") {
          p.train_max_points = v;
        }
    }
    return p;
}



//=========工厂函数===
static std::unique_ptr<vecsearch::IIndex> CreateIndexOrDie(
                                          const std::string& type,
                                          const vecsearch::IndexConfig& cfg,
                                          const std::string&para_str="") {
    //用下划线太多了
    if (type=="baseline_bruteforce"||type=="bruteforce"||type=="flat") {
        return std::make_unique<vecsearch::BruteForceIndex>(cfg);//创建unique_ptr指针
    }
    if (type == "hnsw") {
        // vecsearch::HNSWParams p; // 先用默认参数（M=16, ef_construction=200, ef_search=50）
        //        // return std::make_unique<vecsearch::HNSWIndex>(cfg, p);
        vecsearch::HNSWParams p;
        if (!para_str.empty()) {
            p = ParseHNSWParamsOrDie(para_str);
        }
        return std::make_unique<vecsearch::HNSWIndex>(cfg, p);
    }
    if (type=="ivf"||type=="ivf_flat") {
        vecsearch::IVFParams p;
        if (!para_str.empty()) {
          p = ParseIVFParamsOrDie(para_str);
        }
        return std::make_unique<vecsearch::IVFFlatIndex>(cfg, p);
    }
    throw std::runtime_error("Unknown index type: "+type);

}


//=======csv小工具==========太少了就放这里

static void append_csv_row(const std::string& path,
                           const std::string& header,
                           const std::string& row) {
  namespace fs = std::filesystem;
  fs::path p(path);

  // 目录存在吗?
  if (p.has_parent_path()) {
    fs::create_directories(p.parent_path());
  }

  const bool need_header = !fs::exists(p);

  std::ofstream out(path, std::ios::app);
  if (!out) {
    throw std::runtime_error("cannot open csv file: " + path);
  }

  if (need_header) {
    out << header << "\n";
  }
  out << row << "\n";
}
//===========小工具结束=========



using Id = uint32_t;


static std::vector<float> gen_vectors(int n, int dim, uint32_t seed, float low, float high) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(low, high);
    std::vector<float> v((size_t)n * dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

static std::vector<Id> gen_ids(int n) {
    std::vector<Id> ids(n);
    for (int i = 0; i < n; ++i) ids[i] = (Id)i;
    return ids;
}

static double percentile_ms(std::vector<double>& ms, double p) {
    if (ms.empty()) return 0.0;
    std::sort(ms.begin(), ms.end());
    // p=0.99 => index = ceil(p*N)-1
    size_t n = ms.size();
    size_t idx = (size_t)std::ceil(p * n) - 1;
    if (idx >= n) idx = n - 1;
    return ms[idx];
}

double_t check_Recall(const std::vector<Id>& search_result_id,
                      const std::vector<Id>& search_correct_ans,
                      uint32_t topk) {
  // 真实K：以 gt 为准（gt 的长度就是实际 topk，可能被 N 截断）
  const uint32_t K = std::min<uint32_t>(topk, static_cast<uint32_t>(search_correct_ans.size()));
  if (K == 0) return 0.0;

  // pred 不应该比 gt 长（你的判断保留）
  if (search_result_id.size() > search_correct_ans.size()) {
    throw std::runtime_error("check_Recall: pred size > gt size");
  }

  // 拷贝一份再排序：不破坏调用方的数据
  std::vector<Id> pred = search_result_id;
  std::vector<Id> gt   = search_correct_ans;
  std::sort(pred.begin(), pred.end());
  std::sort(gt.begin(), gt.end());

  // 去重：避免 pred 里重复 id 被重复计数
  pred.erase(std::unique(pred.begin(), pred.end()), pred.end());

  uint32_t hit = 0;
  for (const auto& x : pred) {
    auto it = std::lower_bound(gt.begin(), gt.end(), x);
    if (it != gt.end() && *it == x) {
      ++hit;
    }
  }

  // Recall@K = hit / K（用真实 K）
  return static_cast<double_t>(hit) / static_cast<double_t>(K);
}

static std::vector<Id> extract_ids(const std::vector<vecsearch::Neighbor>& res) {//提取查询到的<Id,floay>的id列表
    std::vector<Id> ids;
    ids.reserve(res.size());
    for (const auto& kv : res) ids.push_back(kv.id);
    return ids;
}

struct Case {
    std::string type;
    std::string params;
};


struct BenchmarkResult {//抽象成结果结构体
    double recall_mean = 0.0;
    double qps = 0.0;
    double p99_ms = 0.0;
};

static BenchmarkResult RunCase(const vecsearch::IIndex& pred,
                               const vecsearch::BruteForceIndex& gt,
                               const std::vector<float>& queries,
                               int dim,
                               int num_queries,
                               int topk) {//原来的东西抽成函数
    //  预热
    const int warmup = 200;
    for (int i = 0; i < std::min(warmup, num_queries); ++i) {
        pred.search_one(&queries[(size_t)i * dim], topk);
    }

    // 计时+recall,统计每个 query latency + 总耗时
    std::vector<double> per_query_ms;
    per_query_ms.reserve(num_queries);

    auto t0 = std::chrono::high_resolution_clock::now();
    double recall_sum = 0.0;

    for (int i = 0; i < num_queries; ++i) {
      const float* q = &queries[(size_t)i * dim];

      // 真实值
      auto gt_res = gt.search_one(q, topk);
      auto gt_ids = extract_ids(gt_res);

      // 计算被测索引,先算pred(被测索引),先baseline自己
      auto qs = std::chrono::high_resolution_clock::now();
      auto pred_res = pred.search_one(q, topk);
      auto qe = std::chrono::high_resolution_clock::now();

      auto pred_ids = extract_ids(pred_res);
      recall_sum += check_Recall(pred_ids, gt_ids, topk);

      double ms = std::chrono::duration<double, std::milli>(qe - qs).count();
      per_query_ms.push_back(ms);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(t1 - t0).count();

    BenchmarkResult r;
    r.recall_mean = recall_sum / num_queries;


    double total_pred_time_ms = 0;
    for(double ms : per_query_ms) total_pred_time_ms += ms;
    // 原来计算 QPS 只用 total_s。
    // 但 total_s 是 t1-t0，中间夹杂了 gt.search_one。
    // 所以 QPS 会被 GT 拖慢
    r.qps = num_queries / (total_pred_time_ms / 1000.0);
    r.p99_ms = percentile_ms(per_query_ms, 0.99);
    return r;
}




int main() {
#ifdef _OPENMP
    std::cerr << "[OMP] max_threads=" << omp_get_max_threads()
              << " num_procs=" << omp_get_num_procs() << "\n";
#endif


    const std::vector<Case> cases = {
      {"baseline_bruteforce", ""},
          // 2. IVF-Flat 对比 (N=100W时, nlist=1024, nprobe=32 是常见配置)
      {"ivf", "nlist=1024;nprobe=16"},
      {"ivf", "nlist=1024;nprobe=32"},
      {"ivf", "nlist=1024;nprobe=64"},

      // 3. HNSW 对比
    {"hnsw", "M=16;efC=200;efS=50"},
    {"hnsw", "M=32;efC=200;efS=100"},
    };


    //先把配置写死
    const int dim = 128;
    const std::vector<int> sizes = {1000000};
    const int num_queries = 1000;
    const int topk = 10;
    const uint32_t seed = 20260121;
	std::cout<<"Preparing Data...\n";
    const float low = -1.0f;
    const float high = 1.0f;
	int N = sizes[0];
	auto base = gen_vectors(N, dim, seed, low, high);
	auto ids = gen_ids(N);
	auto queries = gen_vectors(num_queries, dim, seed+1, low, high);

	vecsearch::IndexConfig cfg;
	cfg.dim = dim;
	cfg.metric = vecsearch::Metric::L2;

	// 建立 Ground Truth (使用优化后的 BruteForce)
	std::cout << "Building Ground Truth (Optimized Flat)...\n";
	vecsearch::BruteForceIndex gt(cfg);
	gt.add_batch(ids, base);

	std::cout << "\n====================\n";
	std::cout << "N=" << N << " Dim=" << dim << " TopK=" << topk << "\n";

	const std::string csv_path = "benchmark_results/results.csv";
	const std::string csv_header = "type,params,build_ms,qps,recall,p99";
	for (auto &c : cases) {
		std::cout << "\n------------------------------------------------\n";
		std::cout << "Running Case: " << c.type << " [" << c.params << "]\n";

		auto index = CreateIndexOrDie(c.type, cfg, c.params);

		// Build
		auto t0 = std::chrono::high_resolution_clock::now();
		index->add_batch(ids, base);
		auto t1 = std::chrono::high_resolution_clock::now();

		// === Phase 5 核心: 激活 HNSW Freeze ===
		if (auto hnsw = dynamic_cast<vecsearch::HNSWIndex*>(index.get())) {
			hnsw->freeze();
			std::cout << "[HNSW] Index frozen (Lock-free mode enabled).\n";
		}

		double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
		std::cout << "Build Time: " << build_ms << " ms\n";

		// Run
		auto res = RunCase(*index, gt, queries, dim, num_queries, topk);

		std::cout << "Recall: " << res.recall_mean << "\n";
		std::cout << "QPS:    " << res.qps << "\n";
		std::cout << "P99:    " << res.p99_ms << " ms\n";

		// Log to CSV
		std::string row = c.type + "," + c.params + "," +
						  std::to_string(build_ms) + "," +
						  std::to_string(res.qps) + "," +
						  std::to_string(res.recall_mean) + "," +
						  std::to_string(res.p99_ms);
		append_csv_row(csv_path, csv_header, row);
	}

    return 0;
}
