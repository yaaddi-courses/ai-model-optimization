# Response caching: not running the model at all

**Covers:** Response cache, Cache hit, Cache miss, Cache misses, TTL

## The cheapest request is the one you skip {#overview}
Every optimization so far makes the model run faster or cheaper. A cache goes further: for some requests, the model does not run at all. Many services get the same questions again and again, such as "what are your opening hours?" or the same document summarised for many people. If the answer was good once, it can be returned from storage in microseconds at almost no cost.

## Response cache {#response-cache}
A **response cache** stores previous model answers, keyed by the request. When the same request arrives again, the service looks it up and returns the saved answer instead of calling the model. A cache can be as small as a dictionary in memory or as large as a shared database across many servers.

## Cache hit {#cache-hit}
A **cache hit** happens when the lookup finds a stored answer. The model does not run, the reply is almost instant, and the cost is close to zero. The share of requests that are hits is the hit rate, and it decides how much a cache is worth.

## Cache miss {#cache-miss}
A **cache miss** is the opposite: the lookup finds nothing. The model runs as usual, and the fresh answer is saved so that future repeats can be hits. A miss is slightly slower than having no cache at all, because the lookup came first, but the difference is tiny.

## TTL {#ttl}
Think about a saved answer from this morning. Is it still right this afternoon? **TTL**, short for time to live, is how long a cache entry stays valid before it expires. A stock price from this morning is wrong by afternoon, and a policy answer is wrong after the policy changes. A short TTL keeps answers fresh but lowers the hit rate. A long TTL raises the hit rate at the risk of serving old information. The right value depends on how quickly the right answer changes.

## A worked example {#example}
Here is a small cache with a time limit. It uses a fake clock, so the result is the same every time.

```python
cache = {}
ttl = 60
model_calls = 0

def answer(question, now):
    global model_calls
    entry = cache.get(question)
    if entry and now - entry["time"] < ttl:
        return entry["text"], "hit"
    model_calls += 1
    text = f"answer {model_calls}"
    cache[question] = {"text": text, "time": now}
    return text, "miss"

print(answer("opening hours?", 0))       # ('answer 1', 'miss')
print(answer("opening hours?", 30))      # ('answer 1', 'hit')
print(answer("opening hours?", 90))      # ('answer 2', 'miss')

requests = 10_000
hit_rate = 0.3
cost = 0.002
print(round(requests * cost, 2))                        # 20.0
print(round(requests * (1 - hit_rate) * cost, 2))       # 14.0
```

Reading the output:

1. The first request finds nothing, so it is a miss and calls the model.
2. Thirty seconds later the saved answer is still valid, so it is a hit and the model is not called.
3. At 90 seconds the entry is older than the 60-second TTL, so it counts as a miss, and a new answer replaces it.
4. With 10,000 requests at 0.2 cents each, a 30% hit rate cuts the model bill from 20 to 14 dollars.

## A common mistake {#mistake}
A common slip is caching answers that depend on the user or on the moment. If a reply uses personal data or today's date, the same words can need a different answer for a different person. Include everything that affects the answer in the cache key, or do not cache it.

## Key points {#key-points}
- A response cache returns a saved answer instead of running the model.
- A hit skips the model, and a miss runs it and saves the result.
- TTL limits how long an answer may be reused, trading freshness against hit rate.
- Anything that changes the right answer must be part of the cache key.
