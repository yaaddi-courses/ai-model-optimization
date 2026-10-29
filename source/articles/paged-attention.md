# PagedAttention: managing the KV cache in blocks

**Covers:** PagedAttention, KV block, Block table, Memory fragmentation

## The memory waste problem {#overview}
Every conversation needs its own KV cache, and no one knows in advance how long a conversation will get. The simple solution is to reserve space for the longest possible reply the moment a request arrives. Most requests never use it all, so much of the reserved memory sits empty, and yet it cannot be given to anyone else. On a busy server, this waste decides how many users fit on one GPU.

## Memory fragmentation {#memory-fragmentation}
**Memory fragmentation** is wasted GPU memory that exists but cannot be used. It has two forms. Reserved but unused space belongs to a request that has not grown into it. Scattered gaps are the holes left between allocations when requests finish at different times. In both cases free memory is there, but no free piece is large enough for a new request, so the server turns people away or slows down.

## KV block {#kv-block}
A **KV block** is a small, fixed-size chunk of KV cache memory that holds the keys and values for a set number of tokens, for example 16. Instead of one big reservation per request, the cache is built from many blocks of the same small size, handed out only when needed.

## Block table {#block-table}
The blocks belonging to one request need not sit next to each other in memory. A **block table** is a lookup that maps a request's logical positions (token 0 to 15, 16 to 31, and so on) to the physical blocks where they are stored. The table is how the attention code finds the right data even though the blocks are scattered, much as a page table does in an operating system.

## PagedAttention {#pagedattention}
**PagedAttention** is the technique that puts these pieces together: it stores the KV cache in small fixed-size blocks allocated on demand, found through a block table. A new block is added only when a request has filled its last one. When a request ends, its blocks go back to a shared pool at once. Almost nothing is reserved ahead of need, so far less memory is wasted and more requests fit.

## A worked example {#example}
Four requests with 300, 700, 120 and 1,500 tokens so far. The simple scheme reserves 2,048 positions for each. Compare with blocks of 16.

```python
import math

used = [300, 700, 120, 1500]
max_length = 2048
block_size = 16

reserved = len(used) * max_length
waste_reserved = 1 - sum(used) / reserved
print(reserved, sum(used), round(waste_reserved, 2))    # 8192 2620 0.68

blocks_needed = [math.ceil(tokens / block_size) for tokens in used]
paged = sum(blocks_needed) * block_size
waste_paged = 1 - sum(used) / paged
print(blocks_needed, paged, round(waste_paged, 3))      # [19, 44, 8, 94] 2640 0.008

block_table = {"request_a": [7, 2, 9]}                  # logical blocks 0, 1, 2 live in blocks 7, 2, 9
print(block_table["request_a"][1])                      # 2
```

Reading it:

1. Reserving the maximum for four requests sets aside 8,192 positions, but only 2,620 are in use.
2. About 68% of the reserved memory is wasted.
3. With 16-token blocks the same requests need 19, 44, 8 and 94 blocks, 2,640 positions in all.
4. Waste falls to about 0.8%, which is just the unfilled end of each request's last block.

The block table at the end shows how a request's three logical blocks can sit in physical blocks 7, 2 and 9. Nothing needs to be contiguous.

## A common mistake {#mistake}
Making blocks very small to cut waste further. Smaller blocks mean larger tables and more lookups, and the attention code reads more scattered pieces. Pick a block size that balances the two.

## Key points {#key-points}
- Reserving space for the longest reply wastes memory and limits how many users fit.
- Memory fragmentation is free memory that cannot be used because of reserved space and gaps.
- PagedAttention stores the cache in small fixed blocks allocated on demand.
- A block table maps each request's positions to wherever its blocks actually are.
