# import torch
# import mono_cpp  # This is the name defined in your setup.py
# from flash_attn import flash_attn_varlen_func

# num_tokens = 1
# num_heads = 1
# head_size = 128
# num_blocks = 2
# block_size = 256

# query_tensor = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16)
# key_tensor = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16)
# value_tensor = torch.randn(num_tokens, num_heads, head_size, device="cuda", dtype=torch.float16)

# key_cache_tensor = torch.zeros(num_blocks, block_size, num_heads, head_size, device="cuda", dtype=torch.float16)
# value_cache_tensor = torch.zeros(num_blocks, block_size, num_heads, head_size, device="cuda", dtype=torch.float16)
# slot_mapping_tensor = torch.full((num_tokens,), 4, dtype=torch.int64, device="cuda")  # Using int64 for slot mapping
# block_table = torch.tensor(torch.zeros((num_tokens, num_blocks)), dtype=torch.int32, device="cuda")


# # print sum of key_tensor and value_tensor
# print(key_cache_tensor.sum())
# print(value_cache_tensor.sum())

# # sleep 10 seconds
# # import time
# # time.sleep(3)

# # Call the cache kernel
# mono_cpp.reshape_and_cache_flash(
#     key_tensor,
#     value_tensor,
#     key_cache_tensor,
#     value_cache_tensor,
#     slot_mapping_tensor
# )

# cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
# cu_seqlens_k = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
# max_seqlen_q = 1
# max_seqlen_k = 1

# print(block_table.shape)

# # block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.

# out = flash_attn_varlen_func(
#     q = query_tensor,
#     k = key_cache_tensor,
#     v = value_cache_tensor,
#     cu_seqlens_q = cu_seqlens_q,
#     cu_seqlens_k = cu_seqlens_k,
#     max_seqlen_q = max_seqlen_q,
#     max_seqlen_k = max_seqlen_k,
#     dropout_p = 0.0,
#     block_table = block_table,
# )

# print(out)
import torch
from einops import rearrange

num_blocks = 2
# block_size = 256
batch_size = 1
device = "cuda"

block_table = rearrange(
    torch.randperm(num_blocks, dtype=torch.int32, device=device),
    "(b nblocks) -> b nblocks",
    b=batch_size,
)
print(block_table)