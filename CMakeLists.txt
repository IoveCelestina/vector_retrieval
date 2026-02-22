cmake_minimum_required(VERSION 3.30)
project(vector_retrieval)

set(CMAKE_CXX_STANDARD 20)

add_executable(bench_runner
        bench/bench_runner.cpp
        include/vecsearch/hnsw_index.h
        src/hnsw_index.cpp
        include/vecsearch/distance.h
        include/vecsearch/ivf_flat_index.h
        src/ivf_flat_index.cpp
        src/distance.cpp
)

# 针对 MinGW/GCC/Clang 的核心优化
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # -O3: 最高级别通用优化
    # -march=native: 自动探测本机CPU架构，开启 AVX2/FMA 等指令集
    # -ffast-math: 允许浮点运算在极小精度损失下大幅加速
    target_compile_options(bench_runner PRIVATE -O3 -march=native -ffast-math)
endif()

# 针对MSVC(如果在Windows VS环境跑,备用)
if(MSVC)
    target_compile_options(bench_runner PRIVATE /O2 /arch:AVX2)
endif()

target_compile_features(bench_runner PRIVATE cxx_std_20)

# 把 include 目录加进去
target_include_directories(bench_runner PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
#寻找 OpenwrtMP(多线程支持)
find_package(OpenMP REQUIRED)
#把OpenMP库链接到bench_runner
target_link_libraries(bench_runner PRIVATE OpenMP::OpenMP_CXX  )