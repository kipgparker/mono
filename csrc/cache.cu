#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

template <typename scalar_t, typename cache_t>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, block_size, num_heads,
                                         // head_size]
    cache_t* __restrict__ value_cache,   // [num_blocks, block_size, num_heads,
                                         // head_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;
    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int64_t tgt_key_value_idx = block_idx * block_stride +
                                      block_offset * page_stride +
                                      head_idx * head_stride + head_offset;
    key_cache[tgt_key_value_idx] = key[src_key_idx];
    value_cache[tgt_key_value_idx] = value[src_value_idx];
  }
}

#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T) \
  reshape_and_cache_flash_kernel<KV_T, CACHE_T> \
      <<<grid, block, 0, stream>>>( \
          reinterpret_cast<KV_T*>(key.data_ptr()), \
          reinterpret_cast<KV_T*>(value.data_ptr()), \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()), \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()), \
          slot_mapping.data_ptr<int64_t>(), block_stride, page_stride, \
          head_stride, key_stride, value_stride, num_heads, head_size, block_size);

#define DISPATCH_FLOAT_HALF(SRC_DTYPE, FN) \
  if (SRC_DTYPE == at::ScalarType::Float) { \
    FN(float, float); \
  } else if (SRC_DTYPE == at::ScalarType::Half) { \
    FN(at::Half, at::Half); \
  } else { \
    TORCH_CHECK(false, "Only float32 and float16 are supported."); \
  }

void reshape_and_cache_flash(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping) {  // [num_tokens] or [num_actual_tokens]
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_FLOAT_HALF(key.dtype(), CALL_RESHAPE_AND_CACHE_FLASH);
}