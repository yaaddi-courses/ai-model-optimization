# Attention and the KV cache: how a model remembers what it has read

**Covers:** Attention, Attention head, KV, KV cache, Autoregressive decoding

## How a language model writes {#overview}
A language model does not write a whole reply at once. It writes one piece at a time, and each new piece depends on everything before it. That simple fact explains most of the memory and speed problems of serving language models, and it is why a special cache exists. To follow it you need three ideas: attention, the step-by-step writing loop, and the cache that stops the loop from repeating work.

## Attention {#attention}
**Attention** is the step where each token in the text is compared with the other tokens to decide which ones matter most for predicting what comes next. In the sentence "The trophy did not fit in the suitcase because it was too big", the word "it" has to look back to decide whether it means the trophy or the suitcase. Attention is how the model makes that look-back.

For each token the model builds three lists of numbers: a **query** that says what it is looking for, and a **key** and a **value** that describe what it offers. A token compares its query with every key, which produces scores. The scores say how much to listen to each earlier token, and the model mixes the values in those proportions.

## Attention head {#attention-head}
One round of comparison would capture only one kind of relationship. An **attention head** is one of several parallel attention computations in a layer. Each head can specialise, for example one following grammar, another tracking which noun a pronoun refers to, a third following topic. The results of all heads are combined. A typical large model has 32 or more heads in each layer.

## KV {#kv}
The keys and values together are often shortened to **KV**. They are the per-token lists of numbers that later tokens look at. Once a token's KV has been computed, it never changes while the reply is being written, which sets up the next idea.

## Autoregressive decoding {#autoregressive-decoding}
**Autoregressive decoding** is how language models write: they produce one token at a time, and each new token is predicted from all the tokens before it. The new token is added to the text, and the loop runs again. A reply of 100 tokens needs 100 passes through the model, one after another. They cannot be run all at the same time, because each depends on the one before.

## KV cache {#kv-cache}
Without any trick, every pass would recompute the keys and values of the entire text again, even though the old ones have not changed. The **KV cache** stores the keys and values of earlier tokens, so each pass computes them only for the newest token and reads the rest from memory. This saves a great deal of arithmetic.

The cost is memory. The cache grows with every token and with every user being served at once, and it can become as big as the model's weights.

## A worked example {#example}
How large is the KV cache of a model with 32 layers, 32 heads per layer, 128 numbers per head, and 2 bytes per number?

```python
layers = 32
heads = 32
head_dim = 128
bytes_per_number = 2

per_token = 2 * layers * heads * head_dim * bytes_per_number   # 2 = keys and values
print(per_token)                               # 524288 bytes
print(per_token / 1e6)                         # 0.524288 MB for one token

tokens = 4000
print(round(per_token * tokens / 1e9, 2))      # 2.1 GB for a 4,000-token text
```

How to read it:

1. Each token stores a key and a value, hence the factor 2.
2. Each is kept for every head in every layer: 32 × 32 × 128 numbers.
3. At 2 bytes each, one token costs 524,288 bytes, about half a megabyte.
4. A 4,000-token conversation therefore holds about 2.1 GB, for one user.

Ten users with such conversations would need 21 GB of cache, on top of the weights.

## A common mistake {#mistake}
Counting only the weights when planning memory. For long texts or many users the KV cache can be larger than the weights, and it grows as the text grows.

## Key points {#key-points}
- Attention lets each token weigh the earlier tokens by relevance.
- A head is one parallel attention computation; layers have many.
- Models write autoregressively: one token per pass, each based on all earlier ones.
- The KV cache keeps earlier keys and values so they are not recomputed, but it takes memory that grows with text length and with the number of users.
