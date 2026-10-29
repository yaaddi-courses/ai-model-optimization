# Attention and the KV cache: how a model remembers what it has read

**Covers:** Attention, Attention head, KV, KV cache, Autoregressive decoding

## How a language model writes {#overview}
When you type a message on your phone, you pick one word at a time, and each word depends on the ones before it. A language model writes the same way. It does not produce a whole reply at once; it produces one piece at a time. This one fact explains most of the memory and speed problems of serving language models, and it is the reason a special cache exists. To follow the story, we need three ideas: attention, the step-by-step writing loop, and the cache that keeps the loop from repeating work.

## Attention {#attention}
Take the sentence "The trophy did not fit in the suitcase because it was too big". What does "it" mean? You know it is the trophy, because you looked back at the earlier words. The model needs to do the same look-back, and **attention** is the step where it does. Each token is compared with the other tokens in the text, so that the model can decide which ones matter most for predicting what comes next.

How does the comparison work? For each token the model builds three lists of numbers. A **query** says what the token is looking for. A **key** and a **value** describe what the token has to offer. A token compares its query with every key, and this gives a score for each earlier token. The scores say how much to listen to each one, and the model mixes the values in those proportions.

## Attention head {#attention-head}
One round of comparison can capture only one kind of relationship. So a layer runs several rounds side by side, and each is called an **attention head**. Each head is free to specialise: one might follow grammar, another might track which noun a pronoun points to, a third might follow the topic. The results of all heads are combined afterwards. A typical large model has 32 or more heads in every layer.

## KV {#kv}
You will often see the keys and values spoken of together as **KV**. They are the per-token lists of numbers that later tokens look at. Notice something important: once a token's KV has been computed, it does not change while the reply is being written. That detail sets up the cache below.

## Autoregressive decoding {#autoregressive-decoding}
Now the writing loop. In **autoregressive decoding**, the model predicts one token from all the tokens before it, adds that token to the text, and then runs again. A reply of 100 tokens needs 100 passes through the model, one after another. They cannot be run at the same time, because each pass needs the result of the one before.

## KV cache {#kv-cache}
Here is the waste. Without any trick, each pass would recompute the keys and values for the entire text, even though the old ones have not changed. The **KV cache** fixes this by storing the keys and values of earlier tokens. Each pass then computes them only for the newest token and reads the rest from memory. This saves a great deal of arithmetic.

There is a price, and it is memory. The cache grows with every token, and it grows again with every user being served at once. It can become as big as the model's weights.

## A worked example {#example}
Let's measure it. How large is the KV cache of a model with 32 layers, 32 heads per layer, 128 numbers per head, and 2 bytes per number?

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

We read it step by step:

1. Each token stores a key and a value, hence the factor of 2.
2. Each of those is kept for every head in every layer: 32 × 32 × 128 numbers.
3. At 2 bytes each, one token costs 524,288 bytes, about half a megabyte.
4. A 4,000-token conversation therefore holds about 2.1 GB, for one user.

Ten users with conversations like that would need 21 GB of cache, and that is on top of the weights.

## A common mistake {#mistake}
A common slip is counting only the weights when planning memory. The weights are fixed, so they are easy to count, and the cache is easy to forget. But for long texts or many users the KV cache can be larger than the weights, and it keeps growing as the text grows.

## Key points {#key-points}
- Attention lets each token weigh the earlier tokens by how relevant they are.
- A head is one parallel attention computation, and layers have many.
- Models write autoregressively: one token per pass, each based on all earlier ones.
- The KV cache keeps earlier keys and values so they are not recomputed, but it takes memory that grows with text length and with the number of users.
