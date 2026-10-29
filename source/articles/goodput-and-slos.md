# Goodput, service level objectives and chunked prefill

**Covers:** Chunked prefill, Goodput, Service level objective, SLO

## Fast on average is not good enough {#overview}
Imagine a restaurant that serves a record number of meals each evening, yet half of the guests waited an hour. The kitchen counts that as a great night. The guests do not. A serving system can behave the same way: an impressive number of requests per second, while many users are unhappy because their answers came too late. Raw throughput counts everything the system finished, but what you really want to know is how much of it was finished in time. In this article we look at the measure for that, the promise it is measured against, and one scheduling trick that helps keep the promise.

## Service level objective {#service-level-objective}
Before you can say "in time", you must say what in time means. A **service level objective** is a measurable target that a service promises to meet. For example: "the first token appears within 1 second for 95% of requests." Notice the three parts: a metric, a threshold, and the share of requests that must meet it. It is written SLO for short. An SLO turns a vague wish for "fast" into something you can test, alert on and design for.

## SLO {#slo}
**SLO** is the standard abbreviation, and in practice you will have several, one for each metric, such as a time-to-first-token SLO and a time-per-token SLO. A request counts as good only if it meets all of them.

## Goodput {#goodput}
Now the measure. **Goodput** is the rate of requests a system completes while still meeting its SLOs. Throughput counts a request that took ten seconds the same as one that took half a second. Goodput counts only the ones that were on time. So a system pushed past its limits can show higher throughput and lower goodput at the same time: it finishes more requests, but makes most of them late.

## Chunked prefill {#chunked-prefill}
How do you protect the promise when one user sends a very long prompt? That prompt has a long prefill. While the GPU works through it, the other users' decode steps are stalled, and their text freezes. **Chunked prefill** splits the long prompt's prefill into smaller chunks that are scheduled together with other requests' decode steps. Each step does a little prefill and a little decoding. Long prompts no longer stall everyone, and tail latency improves.

## A worked example {#example}
Let's use ten requests with their time to first token, in seconds, over a ten-second window. The SLO says the first token must appear within 1.0 second.

```python
ttft_seconds = [0.5, 0.7, 0.9, 1.2, 0.8, 2.5, 0.6, 0.95, 3.0, 0.7]
slo = 1.0
window = 10

throughput = len(ttft_seconds) / window
on_time = [t for t in ttft_seconds if t <= slo]
goodput = len(on_time) / window

print(throughput)                          # 1.0
print(len(on_time), goodput)               # 7 0.7

prefill_speed = 10_000                     # prompt tokens processed per second
prompt = 8000
chunk = 1000
print(prompt / prefill_speed)              # 0.8 seconds of stall without chunking
print(chunk / prefill_speed)               # 0.1 seconds at most with chunking
```

We read it like this:

1. The system finished 10 requests in 10 seconds, so throughput is 1.0 request per second.
2. Only 7 of them met the 1-second target, so goodput is 0.7 requests per second.
3. A single 8,000-token prompt at 10,000 tokens per second would block everyone else for 0.8 seconds.
4. Cut into 1,000-token chunks, each step blocks the others for at most 0.1 seconds, and decode steps continue in between.

Chunking leaves the total work the same. It changes who has to wait.

## A common mistake {#mistake}
A common slip is writing an SLO nobody can verify, or one that mentions only the average. An average hides the slow requests that people complain about. Say the percentage of requests that must meet the target, and then measure that percentage.

## Key points {#key-points}
- An SLO is a measurable promise: a metric, a threshold and a share of requests.
- Goodput counts only requests that met their SLOs, while throughput counts all of them.
- Pushing throughput too far can lower goodput.
- Chunked prefill interleaves pieces of a long prompt with other requests' decode steps.
