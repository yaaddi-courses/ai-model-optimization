# Grouped-query and sliding-window attention

**Covers:** Grouped-query attention, GQA, Sliding-window attention

## Two ways to make attention cheaper {#overview}
Standard attention is generous. Every attention head keeps its own keys and values, and every token looks at every earlier token. Both habits cost memory: the first makes the KV cache large, and the second makes the work grow with the length of the text. Two variants change one habit each. Grouped-query attention shares keys and values between heads, and sliding-window attention limits how far back a token looks.

## Grouped-query attention {#grouped-query-attention}
**Grouped-query attention** lets several attention heads share one set of keys and values. Each head still has its own query, so each can still look for something different, but they read the same stored keys and values. Instead of 32 sets to store for every token, a model might keep only 8, each shared by a group of 4 heads.

The benefit is a much smaller KV cache, which means longer conversations or more users per GPU. The accuracy loss is usually small. Models trained this way from the start do better than models converted afterwards.

## GQA {#gqa}
**GQA** is the common abbreviation for grouped-query attention. You will see it in model descriptions, for example "32 query heads and 8 KV heads". At one extreme, every head has its own keys and values, which is the standard design. At the other, all heads share a single set. GQA sits between them.

## Sliding-window attention {#sliding-window-attention}
**Sliding-window attention** is a variant where each token attends only to a fixed number of recent tokens, the window, instead of to everything before it. The work and the memory no longer grow with the full length of the text. They grow with the window size.

The price is reach: a token cannot directly see anything older than the window. Information can still pass further back indirectly, since each layer builds on the one before it, but long-range recall is weaker.

## A worked example {#example}
First the KV cache saving of GQA, then how many token pairs sliding-window attention compares.

```python
layers, head_dim, bytes_per_number = 32, 128, 2

def cache_bytes_per_token(kv_heads):
    return 2 * layers * kv_heads * head_dim * bytes_per_number

full = cache_bytes_per_token(32)
grouped = cache_bytes_per_token(8)
print(full, grouped, full / grouped)            # 524288 131072 4.0
print(round(grouped * 4000 / 1e6))              # 524 MB for 4,000 tokens, not 2,097 MB

def pairs_compared(n, window=None):
    return sum(i if window is None else min(i, window) for i in range(1, n + 1))

print(pairs_compared(8))                        # 36
print(pairs_compared(8, window=3))              # 21
print(pairs_compared(1000), pairs_compared(1000, window=128))   # 500500 119872
```

How to read it:

1. With 32 sets of keys and values, one token's cache is 524,288 bytes. With 8 sets, it is 131,072, four times smaller.
2. For a 4,000-token conversation that is 524 MB instead of about 2.1 GB.
3. For 8 tokens, full attention compares 36 pairs, while a window of 3 compares 21.
4. At 1,000 tokens the gap is large: about 500,000 pairs against about 120,000 with a window of 128. The longer the text, the more the window saves.

## A common mistake {#mistake}
Combining the two ideas without checking quality. A model's design fixes which variant it uses. You cannot switch a trained model to GQA or a window without retraining or fine-tuning, so choose a model that already has it.

## Key points {#key-points}
- Grouped-query attention shares keys and values among groups of heads, shrinking the KV cache.
- GQA describes designs between one set per head and one set for all heads.
- Sliding-window attention limits each token to recent tokens, so cost follows the window, not the full length.
- Both are decided when a model is designed and trained.
