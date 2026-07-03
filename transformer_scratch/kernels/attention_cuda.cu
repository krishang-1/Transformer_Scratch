#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// ============================================================
// FORWARD KERNEL
// One block per (batch, head, query_position i).
// Computes: scores -> causal mask -> softmax -> weighted sum over V
//
// NOTE (honest limitation): max/sum reductions below are done by a
// single thread (tid==0) per block for simplicity/correctness-first.
// This is a real bottleneck at large seq_len - a proper parallel
// reduction is the natural first optimization for Phase 4 stretch work.
// ============================================================
__global__ void attention_forward_kernel(
    const float* __restrict__ Q,   // (B,H,S,D)
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,         // (B,H,S,D)
    float* __restrict__ P,         // (B,H,S,S) saved attention weights for backward
    int B, int H, int S, int D,
    float scale
) {
    int idx = blockIdx.x;          // 0 .. B*H*S - 1
    int i = idx % S;
    int bh = idx / S;
    int h = bh % H;
    int b = bh / H;

    const float* Qrow  = Q + (((long)(b*H + h)*S + i) * D);
    const float* Kbase = K + ((long)(b*H + h)*S) * D;
    const float* Vbase = V + ((long)(b*H + h)*S) * D;
    float* Prow = P + (((long)(b*H + h)*S + i) * S);
    float* Orow = O + (((long)(b*H + h)*S + i) * D);

    extern __shared__ float shmem[];
    float* scores   = shmem;       // size S
    float* outAccum = shmem + S;   // size D

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Step 1: raw scores, causal masked
    for (int j = tid; j < S; j += nthreads) {
        if (j <= i) {
            float dot = 0.0f;
            const float* Krow = Kbase + (long)j * D;
            for (int d = 0; d < D; d++) dot += Qrow[d] * Krow[d];
            scores[j] = dot * scale;
        } else {
            scores[j] = -1e30f;
        }
    }
    __syncthreads();

    // Step 2: row max (single-thread reduction - see note above)
    __shared__ float rowmax;
    __shared__ float rowsum;
    if (tid == 0) {
        float m = -1e30f;
        for (int j = 0; j <= i; j++) if (scores[j] > m) m = scores[j];
        rowmax = m;
    }
    __syncthreads();

    // Step 3: exponentiate
    for (int j = tid; j < S; j += nthreads) {
        scores[j] = (j <= i) ? expf(scores[j] - rowmax) : 0.0f;
    }
    __syncthreads();

    // Step 4: row sum
    if (tid == 0) {
        float s = 0.0f;
        for (int j = 0; j <= i; j++) s += scores[j];
        rowsum = s + 1e-10f;
    }
    __syncthreads();

    // Step 5: normalize -> P
    for (int j = tid; j < S; j += nthreads) {
        scores[j] = scores[j] / rowsum;
        Prow[j] = scores[j];
    }
    __syncthreads();

    // Step 6: init output accumulator
    for (int d = tid; d < D; d += nthreads) outAccum[d] = 0.0f;
    __syncthreads();

    // Step 7: weighted sum over V (atomics into shared mem)
    for (int j = tid; j <= i; j += nthreads) {
        float p = scores[j];
        const float* Vrow = Vbase + (long)j * D;
        for (int d = 0; d < D; d++) {
            atomicAdd(&outAccum[d], p * Vrow[d]);
        }
    }
    __syncthreads();

    for (int d = tid; d < D; d += nthreads) Orow[d] = outAccum[d];
}

// ============================================================
// BACKWARD KERNEL
// One block per (batch, head, query_position i) - same layout as forward.
// dQ is written directly (each row owned exclusively by one block).
// dK, dV require atomics (multiple i-blocks touch the same key index j).
// ============================================================
__global__ void attention_backward_kernel(
    const float* __restrict__ dO,  // (B,H,S,D)
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ P,   // saved from forward (B,H,S,S)
    float* __restrict__ dQ,        // (B,H,S,D) direct write
    float* __restrict__ dK,        // (B,H,S,D) MUST be zero-initialized before launch
    float* __restrict__ dV,        // (B,H,S,D) MUST be zero-initialized before launch
    int B, int H, int S, int D,
    float scale
) {
    int idx = blockIdx.x;
    int i = idx % S;
    int bh = idx / S;
    int h = bh % H;
    int b = bh / H;

    const float* dOrow = dO + (((long)(b*H+h)*S+i) * D);
    const float* Prow  = P  + (((long)(b*H+h)*S+i) * S);
    const float* Qrow  = Q  + (((long)(b*H+h)*S+i) * D);
    const float* Kbase = K  + ((long)(b*H+h)*S) * D;
    const float* Vbase = V  + ((long)(b*H+h)*S) * D;
    float* dQrow  = dQ + (((long)(b*H+h)*S+i) * D);
    float* dKbase = dK + ((long)(b*H+h)*S) * D;
    float* dVbase = dV + ((long)(b*H+h)*S) * D;

    extern __shared__ float shmem[];
    float* dP = shmem;         // size S
    float* dS = shmem + S;     // size S

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Step 1: dP[j] = dot(dO[i], V[j]) for j <= i
    for (int j = tid; j < S; j += nthreads) {
        if (j <= i) {
            float dot = 0.0f;
            const float* Vrow = Vbase + (long)j * D;
            for (int d = 0; d < D; d++) dot += dOrow[d] * Vrow[d];
            dP[j] = dot;
        } else {
            dP[j] = 0.0f;
        }
    }
    __syncthreads();

    // Step 2: softmax backward correction term: sum_k P[i,k] * dP[i,k]
    __shared__ float dot_P_dP;
    if (tid == 0) {
        float s = 0.0f;
        for (int j = 0; j <= i; j++) s += Prow[j] * dP[j];
        dot_P_dP = s;
    }
    __syncthreads();

    // Step 3: dS[j] = P[j] * (dP[j] - dot_P_dP)
    for (int j = tid; j < S; j += nthreads) {
        dS[j] = (j <= i) ? Prow[j] * (dP[j] - dot_P_dP) : 0.0f;
    }
    __syncthreads();

    // Step 4: dQ[i,d] = scale * sum_j dS[j] * K[j,d]  (exclusive to this block)
    for (int d = tid; d < D; d += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j <= i; j++) acc += dS[j] * Kbase[(long)j*D + d];
        dQrow[d] = scale * acc;
    }

    // Step 5/6: dK[j,d] += scale*dS[j]*Q[i,d] ; dV[j,d] += P[j]*dO[i,d]  (atomics)
    for (int j = tid; j <= i; j += nthreads) {
        float dsj = dS[j];
        float pj  = Prow[j];
        for (int d = 0; d < D; d++) {
            atomicAdd(&dKbase[(long)j*D + d], scale * dsj * Qrow[d]);
            atomicAdd(&dVbase[(long)j*D + d], pj * dOrow[d]);
        }
    }
}

// ============================================================
// HOST-SIDE WRAPPERS (called from Python via pybind11)
// ============================================================
std::vector<torch::Tensor> attention_forward_cuda(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, double scale) {

    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Kernel currently supports float32 only");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous");

    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);

    auto O = torch::zeros_like(Q);
    auto P = torch::zeros({B, H, S, S}, Q.options());

    int threads = std::min((int64_t)256, S);
    int64_t blocks = B * H * S;
    size_t shmem_size = (size_t)(S + D) * sizeof(float);

    attention_forward_kernel<<<blocks, threads, shmem_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), P.data_ptr<float>(),
        B, H, S, D, (float)scale
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA forward kernel launch failed: ", cudaGetErrorString(err));

    return {O, P};
}

std::vector<torch::Tensor> attention_backward_cuda(
    torch::Tensor dO, torch::Tensor Q, torch::Tensor K, torch::Tensor V,
    torch::Tensor P, double scale) {

    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);

    auto dQ = torch::zeros_like(Q);
    auto dK = torch::zeros_like(K);   // zero-init required: backward accumulates via atomicAdd
    auto dV = torch::zeros_like(V);   // zero-init required: same reason

    int threads = std::min((int64_t)256, S);
    int64_t blocks = B * H * S;
    size_t shmem_size = (size_t)(2 * S) * sizeof(float);

    attention_backward_kernel<<<blocks, threads, shmem_size>>>(
        dO.contiguous().data_ptr<float>(),
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        P.data_ptr<float>(),
        dQ.data_ptr<float>(), dK.data_ptr<float>(), dV.data_ptr<float>(),
        B, H, S, D, (float)scale
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA backward kernel launch failed: ", cudaGetErrorString(err));

    return {dQ, dK, dV};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_forward_cuda, "Custom causal attention forward (CUDA)");
    m.def("backward", &attention_backward_cuda, "Custom causal attention backward (CUDA)");
}