//运行 ./bench_runner --config bench/configs/v1_random_l2_d128.yaml
//输出示例 dim=128 seed=20260121 sizes=10000 100000 num_queries=1000 topk=10

#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

int main(int argc, char** argv) {
    std::string config_path = "bench/configs/v1_random_l2_d128.yaml";
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--config") {
            config_path = argv[i + 1];
        }
    }

    YAML::Node cfg = YAML::LoadFile(config_path);

    auto dataset = cfg["dataset"];
    int dim = dataset["dim"].as<int>();
    int seed = dataset["seed"].as<int>();
    int num_queries = dataset["num_queries"].as<int>();
    int topk = dataset["topk"].as<int>();

    std::vector<int> sizes;
    for (auto s : dataset["sizes"]) sizes.push_back(s.as<int>());

    std::cout << "Config loaded: " << config_path << "\n";
    std::cout << "dim=" << dim << ", seed=" << seed
              << ", num_queries=" << num_queries << ", topk=" << topk << "\n";
    std::cout << "sizes: ";
    for (auto s : sizes) std::cout << s << " ";
    std::cout << "\n";

    return 0;
}
