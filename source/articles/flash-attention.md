# FlashAttention: attention without the memory traffic

**Covers:** FlashAttention, SRAM, HBM

## Attention is mostly about memory {#overview}
Attention compares every token with every other token. For a text of n tokens, that produces a table of n × n scores, and the table grows with the square of the length. A text twice as long makes a table four times as big. The arithmetic is not the main trouble. The trouble is that a naive implementation writes this huge table out to the GPU's main memory and reads it back more than once. FlashAttention is a way to avoid that.

## Two kinds of GPU memory {#memory-kinds}
A GPU has two very different memories, and the whole technique is about using the right one.

## HBM {#hbm}
**HBM**, or high-bandwidth memory, is the large main memory of a GPU. The model's weights live there, and it holds tens of gigabytes. It is fast compared with ordinary computer memory, but it is noticeably slower than the memory on the chip itself, and reading from it is what usually limits attention.

## SRAM {#sram}
**SRAM** is very fast but tiny memory built onto the GPU chip. A whole GPU has only a few tens of megabytes of it. It is great for data that is used right now, and far too small to hold a model or a big score table. Moving data between HBM and SRAM is the expensive step.

## FlashAttention {#flashattention}
**FlashAttention** is an exact attention algorithm that cuts the traffic between HBM and SRAM. Instead of building the whole n × n table, it cuts the work into small tiles that fit in SRAM. For each tile it loads the pieces it needs, computes the scores, updates a running result and moves on, never writing the full table to HBM.

It gives the same answer as the standard method, within normal number rounding. It is faster because it moves less data, and it uses far less memory because the big table never exists. Memory use grows with the text length, not its square.

## A worked example {#example}
The size of the score table for one layer, if it were written out in full, for a text of 4,096 tokens.

```python
tokens = 4096
bytes_per_score = 2
heads = 32

scores_per_head = tokens * tokens
per_head_mb = scores_per_head * bytes_per_score / 1e6
all_heads_gb = per_head_mb * heads / 1000

print(scores_per_head)             # 16777216
print(round(per_head_mb, 1))       # 33.6
print(round(all_heads_gb, 2))      # 1.07

doubled = 8192
print((doubled * doubled) / scores_per_head)    # 4.0
```

What it means:

1. One head's table has 4,096 × 4,096 = about 16.8 million scores.
2. At 2 bytes each, that is 33.6 MB for one head, a size that does not fit in SRAM.
3. All 32 heads together, for a single layer and a single text, would be about 1.07 GB.
4. Doubling the length to 8,192 makes every table four times bigger.

FlashAttention never stores these tables. It keeps only a small tile in SRAM at a time, so the large number above never shows up in memory.

## A common mistake {#mistake}
Thinking FlashAttention is an approximation that trades accuracy for speed. It is not. It computes the same attention exactly, with a smarter order of operations. Approximate methods exist, but this one is not among them.

## Key points {#key-points}
- Standard attention builds an n × n table, so memory and traffic grow with the square of the length.
- HBM is the GPU's large, slower main memory; SRAM is tiny, very fast memory on the chip.
- FlashAttention works in tiles that fit in SRAM, so the full table is never written to HBM.
- It gives the same results as normal attention, faster and with less memory.
