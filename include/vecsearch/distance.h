#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace vecsearch {

    // 说明：
    // - 函数在头文件中被定义为内联（inline），以支持跨编译单元的内联优化
    //   （这对于 HNSW 的构建和搜索非常重要，因为在这些过程中距离函数的调用极其频繁）。
    // - 具体实现提供了从 AVX512F -> AVX2 -> 标量（scalar）的降级回退机制。

#if defined(__AVX2__)
static inline float hsum256_ps(__m256 v) noexcept {
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 sum   = _mm_add_ps(vlow, vhigh);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float out;
    _mm_store_ss(&out, sum);
    return out;
}
#endif

#if defined(__AVX512F__)
static inline float hsum512_ps(__m512 v) noexcept {
    __m256 vlow  = _mm512_castps512_ps256(v);
    __m256 vhigh = _mm512_extractf32x8_ps(v, 1);
    return hsum256_ps(_mm256_add_ps(vlow, vhigh));
}
#endif

// Squared L2 distance (no sqrt).
static inline float l2_sqr(const float* a, const float* b, int dim) noexcept {
    float res = 0.0f;
    int i = 0;

#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (; i + 16 <= dim; i += 16) {
        __m512 v1   = _mm512_loadu_ps(a + i);
        __m512 v2   = _mm512_loadu_ps(b + i);
        __m512 diff = _mm512_sub_ps(v1, v2);
    #if defined(__FMA__)
        sum = _mm512_fmadd_ps(diff, diff, sum);
    #else
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    #endif
    }
    res = hsum512_ps(sum);

#elif defined(__AVX2__)
    // Unroll x2 (16 floats) to reduce loop overhead.
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (; i + 16 <= dim; i += 16) {
        __m256 v1a  = _mm256_loadu_ps(a + i);
        __m256 v2a  = _mm256_loadu_ps(b + i);
        __m256 difa = _mm256_sub_ps(v1a, v2a);

        __m256 v1b  = _mm256_loadu_ps(a + i + 8);
        __m256 v2b  = _mm256_loadu_ps(b + i + 8);
        __m256 difb = _mm256_sub_ps(v1b, v2b);

    #if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(difa, difa, sum0);
        sum1 = _mm256_fmadd_ps(difb, difb, sum1);
    #else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(difa, difa));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(difb, difb));
    #endif
    }
    // Handle the remaining 8-float block if present.
    if (i + 8 <= dim) {
        __m256 v1  = _mm256_loadu_ps(a + i);
        __m256 v2  = _mm256_loadu_ps(b + i);
        __m256 dif = _mm256_sub_ps(v1, v2);
    #if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(dif, dif, sum0);
    #else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(dif, dif));
    #endif
        i += 8;
    }
    res = hsum256_ps(_mm256_add_ps(sum0, sum1));
#endif

    for (; i < dim; ++i) {
        const float d = a[i] - b[i];
        res += d * d;
    }
    return res;
}

// Inner product (dot product).
static inline float inner_product(const float* a, const float* b, int dim) noexcept {
    float res = 0.0f;
    int i = 0;

#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();
    for (; i + 16 <= dim; i += 16) {
        __m512 v1 = _mm512_loadu_ps(a + i);
        __m512 v2 = _mm512_loadu_ps(b + i);
    #if defined(__FMA__)
        sum = _mm512_fmadd_ps(v1, v2, sum);
    #else
        sum = _mm512_add_ps(sum, _mm512_mul_ps(v1, v2));
    #endif
    }
    res = hsum512_ps(sum);

#elif defined(__AVX2__)
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    for (; i + 16 <= dim; i += 16) {
        __m256 v1a = _mm256_loadu_ps(a + i);
        __m256 v2a = _mm256_loadu_ps(b + i);
        __m256 v1b = _mm256_loadu_ps(a + i + 8);
        __m256 v2b = _mm256_loadu_ps(b + i + 8);

    #if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(v1a, v2a, sum0);
        sum1 = _mm256_fmadd_ps(v1b, v2b, sum1);
    #else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(v1a, v2a));
        sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(v1b, v2b));
    #endif
    }
    if (i + 8 <= dim) {
        __m256 v1 = _mm256_loadu_ps(a + i);
        __m256 v2 = _mm256_loadu_ps(b + i);
    #if defined(__FMA__)
        sum0 = _mm256_fmadd_ps(v1, v2, sum0);
    #else
        sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(v1, v2));
    #endif
        i += 8;
    }
    res = hsum256_ps(_mm256_add_ps(sum0, sum1));
#endif

    for (; i < dim; ++i) res += a[i] * b[i];
    return res;
}

// Squared L2 norm.
static inline float l2_norm_sqr(const float* a, int dim) noexcept {
    return inner_product(a, a, dim);
}

} // namespace vecsearch
