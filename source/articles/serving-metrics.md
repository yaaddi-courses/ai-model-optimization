# Serving metrics for language models

**Covers:** Prefill, Decode phase, Time to first token, TTFT, Time per output token, TPOT, Tokens per second

## One request, two phases {#overview}
For an ordinary model, one number, latency, is enough. A language model is different, because its reply arrives piece by piece. Think about what you feel when you use a chat window. You care how long you wait before anything appears, and then you care how quickly the text flows. These are separate experiences with separate causes, so they have separate measures. To understand them, we split a request into its two phases.

## Prefill {#prefill}
**Prefill** is the first step of a request. The model reads the entire prompt in one pass, builds the KV cache for it and produces the first token. Because the whole prompt is processed together, prefill does a lot of arithmetic at once and is usually compute-bound. A longer prompt means a longer prefill.

## Decode phase {#decode-phase}
The **decode phase** is everything after the first token. The model generates the reply one token at a time. Each step re-reads the weights and the KV cache to produce a single token, so this phase is usually limited by memory bandwidth, not by arithmetic.

## Time to first token {#time-to-first-token}
**Time to first token**, often written TTFT, is how long a user waits after sending a prompt until the first piece of the answer appears. It is mostly the cost of prefill, plus any time spent waiting in a queue. For chat, it sets how responsive the system feels.

## Time per output token {#time-per-output-token}
**Time per output token**, written TPOT, is the average time between tokens after the first. It is mostly decode speed. If it is long, text appears slowly. If it is short, text streams fast, even faster than most people read.

## Tokens per second {#tokens-per-second}
**Tokens per second** is the number of output tokens produced each second. Measured for one user, it is about 1 divided by TPOT. Measured for a server, it is the sum over all users at that moment, which makes it a throughput figure. Always say which of the two you mean.

## A worked example {#example}
Let's take a request with a TTFT of 0.4 seconds and a TPOT of 0.03 seconds, producing a 200-token reply.

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

We read it step by step:

1. The first token arrives at 0.4 s. The other 199 tokens take 0.03 s each, so the whole reply takes 6.37 s.
2. A TPOT of 0.03 s means about 33 tokens per second for one reader.
3. Eight users decoding together, if each keeps that speed, give the server about 267 tokens per second.
4. A longer prompt raises TTFT and so lengthens the total, even though TPOT is the same.

## A common mistake {#mistake}
A common slip is reporting one average latency for a chat service. It blends the wait for the first token with the speed of the rest, and it hides which one is the problem. Track TTFT and TPOT separately.

## Key points {#key-points}
- Prefill reads the prompt and makes the first token, and decode produces the rest one token at a time.
- TTFT is the wait for the first token, and it is mostly prefill and queueing.
- TPOT is the time between later tokens, and it is mostly decode speed.
- Tokens per second means 1 divided by TPOT for one user, or a total for a whole server.
