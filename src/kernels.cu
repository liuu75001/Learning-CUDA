#include <vector>
#include <iostream>
#include <limits>
#include <cmath>

#include "../tester/utils.h"

/**
 * @brief Find the k-th largest element in a vector using CUDA.
 * 
 * @tparam T Type of elements in the input vector (should support `int` and `float`).
 * @param h_input Host-side input vector.
 * @param k 1-based index of the element to find (e.g., `k=1` returns the largest element).
 * @return T The k-th largest element in `h_input`.

 * @note Must use CUDA kernels for all compute-intensive steps; no significant CPU allowed.
 * @note Library functions that can directly complete a significant part of the work are NOT allowed. 
 * @note For invalid cases, return T(-100).
 * @note Handles device memory management (allocate/copy/free) internally. Errors should be thrown.
 */
template <typename T>
__global__ void bitonicSortKernel(T* data, int n, int j, int k) {
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x; // 线程全局索引
    unsigned int ixj = i ^ j;  // 位运算 XOR，得到要比较交换的“对手”元素索引

    // 确保索引合法，避免越界
    if (ixj > i && ixj < n) {
        // 决定升序还是降序：bitonic sort 的关键逻辑
        if ((i & k) == 0) { 
            // 升序比较（保持较大值在前面，方便找 Top-K）
            if (data[i] < data[ixj]) {
                T tmp = data[i];
                data[i] = data[ixj];
                data[ixj] = tmp;
            }
        } else {
            // 降序比较
            if (data[i] > data[ixj]) {
                T tmp = data[i];
                data[i] = data[ixj];
                data[ixj] = tmp;
            }
        }
    }
}


inline size_t nextPow2(size_t n) {
    if (n == 0) return 1;
    size_t pow2 = 1;
    while (pow2 < n) pow2 <<= 1;  // 左移直到 >= n
    return pow2;
}


template <typename T>
T kthLargest(const std::vector<T>& h_input, size_t k) {
    // 参数检查：空数组 / k 越界直接返回错误值
    if (h_input.empty() || k == 0 || k > h_input.size()) {
        return T(-100);
    }

    size_t n = h_input.size();            // 原始长度
    size_t n_padded = nextPow2(n);        // 补齐后的长度（2 的幂）

    
    std::vector<T> h_data(n_padded);

    // 拷贝原始数据
    for (size_t i = 0; i < n; i++) h_data[i] = h_input[i];

    // 补齐部分填充“负无穷” (-∞)，保证它们排在最后，不影响前 k 大
    for (size_t i = n; i < n_padded; i++) 
        h_data[i] = std::numeric_limits<T>::lowest();


    T* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n_padded * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), n_padded * sizeof(T), cudaMemcpyHostToDevice));

  
    int threads = 256;                               // 每个 block 的线程数
    int blocks = (n_padded + threads - 1) / threads; // block 数量，覆盖所有元素

    // 双层循环控制 Bitonic sort 的阶段
    // 外层循环：子序列长度 K 从 2 开始，每次翻倍
    // 内层循环：步长 J，从 K/2 开始，每次除以 2
    for (int K = 2; K <= (int)n_padded; K <<= 1) {
        for (int J = K >> 1; J > 0; J >>= 1) {
            bitonicSortKernel<<<blocks, threads>>>(d_data, n_padded, J, K);
            CUDA_CHECK(cudaDeviceSynchronize()); // 等待当前阶段完成
        }
    }

    // ------------------- 结果拷贝回 Host -------------------
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, n_padded * sizeof(T), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data)); // 释放显存

    // ------------------- 返回结果 -------------------
    // Bitonic Sort 结果：降序排列
    return h_data[k-1];
}




/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

// checkCuda: 如果 CUDA API 返回错误，则打印出错信息并抛异常
inline void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " - " << cudaGetErrorString(err) << "\n";
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
#define CHECK_CUDA(call) checkCuda((call), __FILE__, __LINE__)

// ------------------------ 索引辅助函数（row-major） ------------------------
// 这些函数把多维下标 [b, t, h, d] 转换成扁平数组的偏移。
// 保证 host/device 两侧的内存布局一致（行主序）。
__device__ __forceinline__
size_t idx_Q(int b, int t, int h, int d, int B, int Tt, int Hq, int D) {
    // (((b * Tt) + t) * Hq + h) * D + d
    return (((size_t)b * Tt + t) * Hq + h) * D + d;
}
__device__ __forceinline__
size_t idx_KV(int b, int t, int h, int d, int B, int Ts, int H, int D) {
    // (((b * Ts) + t) * H + h) * D + d
    return (((size_t)b * Ts + t) * H + h) * D + d;
}
__device__ __forceinline__
size_t idx_O(int b, int t, int h, int d, int B, int Tt, int Hq, int D) {
    // 输出与 Q 的布局相同
    return (((size_t)b * Tt + t) * Hq + h) * D + d;
}

// ------------------------ Kernel 实现（three-pass, correct & stable） ------------------------
/*
 * 设计要点（高层）：
 * - 每个 CUDA block 负责处理一个 query 行：(batch b, query_head h_q, target position t_q)
 * - 使用三遍策略避免数值和并发错误：
 *    Pass 1: 计算每个 j 的 score s_j = (q · k_j)/sqrt(D) 并求 m = max_j s_j
 *    Pass 2: 计算 l = sum_j exp(s_j - m)
 *    Pass 3: 用 alpha_j = exp(s_j - m) / l 累加输出 O += alpha_j * v_j
 * - 这样等价于标准 softmax，但更稳定，并且无需大内存保存 scores
 *
 * 共享内存布局（动态 shared memory）：
 * - q_sh[0..D-1]    : 存放当前 (b,t_q,h_q) 的 Q 向量（head_dim 个 float）
 * - red[0..THREADS-1]: reduction scratch（用于 dot-product 的块内规约）
 *
 * 注意：
 * - 该实现多次计算 dot(q,k_j)（每个 j 在不同 pass 里重新计算），因此正确但有重复计算；
 *   可优化为把 dot 的中间结果缓存到 shared/global（不能太大），或使用一次流式算法。
 * - THREADS 必须为 2 的幂（以简化 tree-reduction）。
 */
template <int THREADS>
__global__ void flash_attention_row_kernel_tp(
    const float* __restrict__ Q, // [B, Tt, Hq, D]
    const float* __restrict__ K, // [B, Ts, Hkv, D]
    const float* __restrict__ V, // [B, Ts, Hkv, D]
    float* __restrict__ O,       // [B, Tt, Hq, D]
    int B, int Tt, int Ts, int Hq, int Hkv, int D,
    bool is_causal
) {
    // -------------------- Block -> (t_q, h_q, b) 映射 --------------------
    // grid.x = Tt, grid.y = Hq, grid.z = B
    int t_q = blockIdx.x; // target position index
    int h_q = blockIdx.y; // query head index
    int b   = blockIdx.z; // batch index

    // -------------------- GQA（grouped-query-attention）映射 --------------------
    // 当 kv_heads < query_heads 时，每个 kv head 对应若干 query heads
    // 这里简单使用整除映射：group_size = Hq / Hkv
    int group_size = Hq / Hkv; // assume divisible (checked on host)
    int h_kv = h_q / group_size; // 对应的 kv head 的索引

    // -------------------- Shared memory 分配 --------------------
    // extern __shared__ float shmem[];   // 动态分配
    // q_sh -> shmem[0 .. D-1]
    // red  -> shmem[D .. D + THREADS - 1]
    extern __shared__ float shmem[];
    float* q_sh = shmem;           // 存放 Q 的 head_dim 元素
    float* red  = shmem + D;       // 规约 scratch，大小 = THREADS

    const int tid = threadIdx.x;

    // -------------------- 加载 Q 到 shared memory --------------------
    // 每个线程负责 stride 为 THREADS 的元素（coalesced-friendly）
    for (int d = tid; d < D; d += THREADS) {
        q_sh[d] = Q[idx_Q(b, t_q, h_q, d, B, Tt, Hq, D)];
    }
    __syncthreads(); // 确保 Q 已加载好供后续使用

    // 归一化系数 1/sqrt(D)（用 rsqrtf 在 device 上更快）
    const float inv_sqrt_d = rsqrtf((float)D);

    // =========== PASS 1: 计算 m = max_j s_j = max_j (q·k_j)/sqrt(D) ===========
    // 初始化 m 为 -inf（INFINITY 宏在 device 上可用）
    float m = -INFINITY;
    // 对每个 source 位置 j
    for (int j = 0; j < Ts; ++j) {
        // 每个线程计算其负责维度的部分点积 partial
        float partial = 0.0f;
        for (int d = tid; d < D; d += THREADS) {
            float qd = q_sh[d];
            float kd = K[idx_KV(b, j, h_kv, d, B, Ts, Hkv, D)];
            partial += qd * kd;
        }
        // 把每个线程的 partial 写入 red[tid]，准备共享内存规约
        red[tid] = partial;
        __syncthreads();

        // 使用简单的 tree-reduction（shared memory）把 red[] 规约到 red[0]
        for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        // 只有线程 0 拿到完整的 dot(q,k_j)
        if (tid == 0) {
            float s = red[0] * inv_sqrt_d; // scale
            if (is_causal && j > t_q) s = -INFINITY; // causal mask：屏蔽未来位置
            if (s > m) m = s; // 更新最大值
        }
        // 把 m 广播到所有线程（通过 red[0]）
        if (tid == 0) red[0] = m;
        __syncthreads();
        m = red[0];
        __syncthreads();
    }

    // =========== PASS 2: 计算 l = sum_j exp(s_j - m) ===========
    // 这一步用来得到 softmax 的分母；用 s - m 提高数值稳定性
    float l = 0.0f;
    for (int j = 0; j < Ts; ++j) {
        // 计算 dot(q, k_j) 同上（每次重新计算）
        float partial = 0.0f;
        for (int d = tid; d < D; d += THREADS) {
            float qd = q_sh[d];
            float kd = K[idx_KV(b, j, h_kv, d, B, Ts, Hkv, D)];
            partial += qd * kd;
        }
        red[tid] = partial;
        __syncthreads();

        // tree-reduction -> red[0] contains dot
        for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            float s = red[0] * inv_sqrt_d;
            if (is_causal && j > t_q) s = -INFINITY;
            // exp(s - m)；若 s = -inf 则 e = 0
            float e = (s == -INFINITY) ? 0.0f : expf(s - m);
            red[0] = e; // 暂存在 red[0]
        }
        __syncthreads();

        // 将 red[0]（有意义）与其他线程的 red[tid]（无意义或垃圾）一致化为规约输入
        if (tid != 0) red[tid] = 0.0f;
        __syncthreads();

        // 规约（本步其实大部分线程为 0，只是重复使用 reduction 代码）
        for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        // 汇总到 l（仅线程 0 做累加）
        if (tid == 0) l += red[0];
        __syncthreads();
    }

    // 广播 m 和 l 给所有线程：放到 red[0]/red[1]
    if (tid == 0) {
        red[0] = m;
        // 使用 red[1] 需要保证 red 数组长度 >= 2; 我们这里 red 大小为 THREADS，因此安全
        red[1] = l;
    }
    __syncthreads();
    m = red[0];
    l = red[1];

    // 特殊情况处理：如果所有位置都被 mask（l == 0），输出置 0
    if (l == 0.0f) {
        for (int d0 = tid; d0 < D; d0 += THREADS) {
            O[idx_O(b, t_q, h_q, d0, B, Tt, Hq, D)] = 0.0f;
        }
        return;
    }

    // =========== 按照 alpha_j 累加输出 O = sum_j alpha_j * v_j ===========
    // alpha_j = exp(s_j - m) / l
    // 每个线程负责输出向量中索引为 d0 的子集（stride = THREADS）
    for (int d0 = tid; d0 < D; d0 += THREADS) {
        float acc = 0.0f;

        // 对每个 j 再次计算 s_j（重复计算，保证简明与正确）
        for (int j = 0; j < Ts; ++j) {
            // 计算 dot(q,k_j) -> reuse reduction pattern
            float my_partial = 0.0f;
            for (int d = tid; d < D; d += THREADS) {
                my_partial += q_sh[d] * K[idx_KV(b, j, h_kv, d, B, Ts, Hkv, D)];
            }
            red[tid] = my_partial;
            __syncthreads();
            for (int stride = THREADS >> 1; stride > 0; stride >>= 1) {
                if (tid < stride) red[tid] += red[tid + stride];
                __syncthreads();
            }
            // 线程 0 拿到完整的 dot
            float s = red[0] * inv_sqrt_d;
            if (is_causal && j > t_q) s = -INFINITY;
            float alpha = (s == -INFINITY) ? 0.0f : expf(s - m) / l;

            // 读取 V 的该维度并累加
            float vjd0 = V[idx_KV(b, j, h_kv, d0, B, Ts, Hkv, D)];
            acc += alpha * vjd0;

            __syncthreads(); // 确保 red[] 可以安全复用到下一轮 j
        }
        // 把该维的累加写回输出
        O[idx_O(b, t_q, h_q, d0, B, Tt, Hq, D)] = acc;
    }

    // kernel 结束，block 内所有线程完成该 query 写回
}


template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {      
    static_assert(std::is_same<T, float>::value, "Only float supported in this ref impl.");

    // 计算各 buffer 的期望大小并校验
    size_t nQ = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t nK = (size_t)batch_size * src_seq_len    * kv_heads   * head_dim;
    size_t nV = nK;
    size_t nO = nQ;

    if (h_q.size() != nQ || h_k.size() != nK || h_v.size() != nV) {
        throw std::runtime_error("Invalid Q/K/V sizes");
    }
    if (h_o.size() != nO) {
        throw std::runtime_error("Invalid output buffer size");
    }
    if (query_heads % kv_heads != 0) {
        throw std::runtime_error("query_heads must be divisible by kv_heads");
    }

    // -------------------- 分配 device 缓冲区并拷贝 --------------------
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    CHECK_CUDA(cudaMalloc(&dQ, nQ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dK, nK * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dV, nV * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dO, nO * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dQ, h_q.data(), nQ * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dK, h_k.data(), nK * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dV, h_v.data(), nV * sizeof(float), cudaMemcpyHostToDevice));
    // 将输出初始化为 0
    CHECK_CUDA(cudaMemset(dO, 0, nO * sizeof(float)));


    // grid: (target_seq_len, query_heads, batch_size)
    // 每个 block 负责 (t_q, h_q, b)
    dim3 grid((unsigned)target_seq_len, (unsigned)query_heads, (unsigned)batch_size);

    // 线程数选择：选择 2 的幂以便简单 reductions；128 或 256 常用
    constexpr int THREADS = 128;
    dim3 block(THREADS, 1, 1);

    // 动态 shared memory 大小：head_dim (float) + THREADS (float) for reductions
    size_t shmem_bytes = (size_t)head_dim * sizeof(float) + (size_t)THREADS * sizeof(float);

    // 启动 kernel（模板实例化）
    flash_attention_row_kernel_tp<THREADS><<<grid, block, shmem_bytes>>>(
        dQ, dK, dV, dO,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal
    );

    // 同步并检查错误
    CHECK_CUDA(cudaDeviceSynchronize());

    // -------------------- 结果拷回并释放 --------------------
    CHECK_CUDA(cudaMemcpy(h_o.data(), dO, nO * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dQ));
    CHECK_CUDA(cudaFree(dK));
    CHECK_CUDA(cudaFree(dV));
    CHECK_CUDA(cudaFree(dO));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int kthLargest<int>(const std::vector<int>&, size_t);
template float kthLargest<float>(const std::vector<float>&, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
