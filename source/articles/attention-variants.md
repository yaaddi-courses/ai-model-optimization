# Grouped-query and sliding-window attention

**Covers:** Grouped-query attention, GQA, Sliding-window attention

## Two ways to make attention cheaper {#overview}
You already know that standard attention is generous. Every attention head keeps its own keys and values, and every token looks back at every earlier token. Each habit costs something. The first makes the KV cache large. The second makes the work grow with the length of the text. Two variants each change one habit: grouped-query attention shares keys and values between heads, and sliding-window attention limits how far back a token looks. Let's take them one at a time.

## Grouped-query attention {#grouped-query-attention}
Why does every head need its own keys and values? Often it does not. In **grouped-query attention**, several heads share one set of keys and values. Each head still has its own query, so each can still look for something different, but they all read the same stored keys and values. Instead of storing 32 sets for every token, a model might keep only 8, each shared by a group of 4 heads.

The benefit is a much smaller KV cache, which means longer conversations or more users on each GPU. The accuracy loss is usually small. Models trained this way from the start do better than models converted afterwards.

## GQA {#gqa}
**GQA** is the short name for grouped-query attention, and you will meet it in model descriptions such as "32 query heads and 8 KV heads". Picture a dial. At one end, every head has its own keys and values, which is the standard design. At the other end, all heads share a single set. GQA is any setting between the two ends.

## Sliding-window attention {#sliding-window-attention}
The second habit is looking at everything. In **sliding-window attention**, each token looks only at a fixed number of recent tokens, called the window, instead of at the whole text before it. The work and the memory no longer grow with the full length of the text. They grow with the size of the window.

What do you give up? Reach. A token cannot directly see anything older than the window. Information can still travel further back indirectly, because each layer builds on the one before it, but recall over long distances is weaker.

## A worked example {#example}
We will do two things. First we measure the KV cache saving from GQA, then we count how many token pairs sliding-window attention compares.

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

Here is what the numbers say:

1. With 32 sets of keys and values, one token's cache is 524,288 bytes. With 8 sets, it is 131,072 bytes, which is four times smaller.
2. For a 4,000-token conversation, that is 524 MB instead of about 2.1 GB.
3. For 8 tokens, full attention compares 36 pairs, while a window of 3 compares 21.
4. At 1,000 tokens the gap is wide: about 500,000 pairs against about 120,000 with a window of 128. The longer the text, the more the window saves.

## A common mistake {#mistake}
A common slip is thinking you can switch these on like a setting. A model's design fixes which variant it uses. You cannot turn a trained model into a GQA model or give it a window without retraining or fine-tuning it. So choose a model that already has the feature you need.

## Key points {#key-points}
- Grouped-query attention shares keys and values among groups of heads, which shrinks the KV cache.
- GQA covers any design between one set per head and one set for all heads.
- Sliding-window attention limits each token to recent tokens, so the cost follows the window, not the full length.
- Both are decided when a model is designed and trained.
