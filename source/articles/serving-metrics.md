# Serving metrics for language models

**Covers:** Prefill, Decode phase, Time to first token, TTFT, Time per output token, TPOT, Tokens per second

## One request, two phases {#overview}
For an ordinary model, one number, latency, is enough. A language model is different because a reply arrives piece by piece. A user cares how long they wait before anything appears, and also how quickly the text then flows. These are separate experiences with separate causes, so they have separate measures. To understand them, split a request into its two phases.

## Prefill {#prefill}
**Prefill** is the first step of a request. The model reads the entire prompt in one pass, builds the KV cache for it and produces the first token. Because the whole prompt is processed together, prefill does a lot of arithmetic at once and is usually compute-bound. A longer prompt means a longer prefill.

## Decode phase {#decode-phase}
The **decode phase** is everything after the first token: the model generates the reply one token at a time. Each step re-reads the weights and the KV cache to produce a single token, so this phase is usually limited by memory bandwidth, not by arithmetic.

## Time to first token {#time-to-first-token}
**Time to first token** is how long a user waits after sending a prompt until the first piece of the answer appears. It is mostly the cost of prefill, plus any time spent waiting in a queue. It is often written TTFT. For chat, it sets how responsive the system feels.

## Time per output token {#time-per-output-token}
**Time per output token** is the average time between tokens after the first. It is mostly decode speed, and it is written TPOT. If it is long, text appears slowly; if it is short, text streams fast, faster than most people read.

## Tokens per second {#tokens-per-second}
**Tokens per second** is the number of output tokens produced each second. Measured for one user it is about 1 divided by TPOT. Measured for a server it is the sum over all users at that moment, which is a throughput figure. Always say which of the two you mean.

## A worked example {#example}
A request with a TTFT of 0.4 seconds and a TPOT of 0.03 seconds, producing a 200-token reply.

```python
ttft = 0.4
tpot = 0.03
reply_tokens = 200

total_time = ttft + (reply_tokens - 1) * tpot
print(round(total_time, 2))                  # 6.37 seconds in all
print(round(1 / tpot, 1))                    # 33.3 tokens per second for this user

users = 8
print(round(users / tpot))                   # 267 tokens per second across 8 users

long_prompt_ttft = 1.6
print(round(long_prompt_ttft + (reply_tokens - 1) * tpot, 2))   # 7.57
```

How to read it:

1. The first token arrives at 0.4 s. The other 199 tokens take 0.03 s each, so the whole reply takes 6.37 s.
2. A TPOT of 0.03 s means about 33 tokens per second for one reader.
3. Eight users decoding together, if each keeps that speed, give the server about 267 tokens per second.
4. A longer prompt raises TTFT and so lengthens the total, even though TPOT is the same.

## A common mistake {#mistake}
Reporting one average latency for a chat service. It blends the wait for the first token with the speed of the rest and hides which one is the problem. Track TTFT and TPOT separately.

## Key points {#key-points}
- Prefill reads the prompt and makes the first token; decode produces the rest one token at a time.
- TTFT is the wait for the first token; it is mostly prefill and queueing.
- TPOT is the time between later tokens; it is mostly decode speed.
- Tokens per second means 1 divided by TPOT for one user, or a total for a whole server.
