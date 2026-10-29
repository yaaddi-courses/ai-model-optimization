# Semantic and prefix caching, and embeddings

**Covers:** Semantic caching, Embedding, Similarity threshold, Prefix caching, Tied embeddings

## Repeats that look different {#overview}
A plain response cache helps only when the request is identical, word for word. Real users rarely repeat themselves exactly. "How do I reset my password?" and "I forgot my password, what now?" ask the same thing, yet a plain cache sees two different strings. Two techniques widen the net. One reuses answers for requests that mean the same thing. The other reuses the work done on the beginning of a prompt. Both depend on a way to compare meanings, which is the embedding.

## Embedding {#embedding}
An **embedding** is a list of numbers that captures the meaning of a piece of text. Texts with similar meanings get lists that sit close together, and unrelated texts get lists that are far apart. Closeness is measured with a short formula, such as cosine similarity, which gives a score near 1 for matching directions and near 0 for unrelated ones. A small, cheap model produces embeddings.

## Tied embeddings {#tied-embeddings}
Inside a language model there is also an embedding table: one list of numbers for every word piece in its vocabulary. The model reads from such a table on the way in, and it uses another one to turn its final numbers back into scores on the way out. **Tied embeddings** use one shared table for both sides, which saves the parameters of a whole table. With 32,000 word pieces and 4,096 numbers each, a table holds 131 million numbers, so tying removes that many from the file. This is weight sharing in one specific place. It is a different use of the word from the meaning embeddings above, although both are lists of numbers.

## Semantic caching {#semantic-caching}
Back to the cache. **Semantic caching** reuses a stored answer when a new request means about the same thing as an earlier one. The service turns each incoming request into an embedding and searches the stored requests for the closest one. If it is close enough, the service returns that request's saved answer. Differently worded repeats can then skip the model.

## Similarity threshold {#similarity-threshold}
How close is close enough? The **similarity threshold** is the cutoff for how close two embeddings must be before a saved answer is reused. A high threshold is safe but finds fewer matches. A low threshold gets more hits but risks giving an answer to a question that only looked similar. Tune it using real examples, and set it higher where a wrong answer is costly.

## Prefix caching {#prefix-caching}
The second technique looks at the start of a prompt. **Prefix caching** stores the KV cache computed for the shared beginning of prompts, such as a long system prompt that every request starts with. A later request with the same start skips reprocessing that part and begins from the saved state. It responds faster and costs less compute.

## A worked example {#example}
Let's compute the cosine similarity for three toy embeddings, and then see what prefix caching saves.

```python
import math

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

question = [0.9, 0.1, 0.3]
reworded = [0.85, 0.15, 0.35]
unrelated = [0.1, 0.9, 0.2]
threshold = 0.9

print(round(cosine(question, reworded), 3))      # 0.996
print(round(cosine(question, unrelated), 3))     # 0.271
print(cosine(question, reworded) >= threshold)   # True

system_prompt_tokens = 2000
requests = 100
print(system_prompt_tokens * requests)           # 200000 tokens processed without prefix caching
print(system_prompt_tokens)                      # 2000 with it: processed once
```

Following the output:

1. The reworded question scores 0.996 against the original. That is above the 0.9 threshold, so its cached answer is reused.
2. The unrelated one scores 0.271 and is a miss.
3. Without prefix caching, 100 requests with a 2,000-token system prompt process 200,000 tokens of the same text.
4. With it, the shared start is processed once, and later requests reuse its saved state.

## A common mistake {#mistake}
A common slip is setting the similarity threshold from a few examples. A request that is nearly identical in words can mean the opposite ("cancel my order" and "do not cancel my order"). Test the threshold on tricky pairs like these.

## Key points {#key-points}
- An embedding is a list of numbers that represents meaning, and closeness means similar meaning.
- Semantic caching reuses an answer when a new request's embedding is close enough.
- The similarity threshold trades more hits against more wrong answers.
- Prefix caching reuses the saved KV cache of a shared prompt beginning.
- Tied embeddings are a separate idea: one vocabulary table shared by a model's input and output.
