cmake_minimum_required(VERSION 3.30)
project(vector_retrieval)


set(CMAKE_CXX_STANDARD 20)

add_executable(bench_runner bench/bench_runner.cpp)
target_compile_features(bench_runner PRIVATE cxx_std_20)
