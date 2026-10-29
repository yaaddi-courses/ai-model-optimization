# Model routing and cascades: the right model for each request

**Covers:** Model routing, Cascade

## One model for everything is wasteful {#overview}
Not every request needs the biggest model. A question like "what is 2 + 2?" or "summarise this one-line note" can be answered by a small, cheap model as well as by a large one. A hard question may need the big one. If you send everything to the largest model, you pay for power that most requests never use. Routing fixes that by matching each request to a model that suits it.

## Model routing {#model-routing}
**Model routing** sends each request to the most suitable model: a cheap small one for easy tasks, a large one for hard tasks. Something has to decide, and there are a few common ways. You can use simple rules based on the task type, a small classifier that predicts how hard the request is, or the model's own confidence. Done well, routing saves money without hurting quality on the requests that matter.

## Choosing how to route {#how-to-route}
Three signals are in common use. The first is the kind of task: a request to translate a sentence or to extract a date can go to a small model, while a request for a long analysis goes to a large one. The second is the length or difficulty of the input, estimated by a tiny classifier or even by simple rules. The third is the cheap model's own answer, which is the idea behind a cascade.

Where should you begin? Start with the simplest signal that works. Rules based on task type are easy to explain and to debug. Add a learned classifier only when the rules leave too much money on the table, and keep a sample of routed requests so you can check the quality afterwards.

## Cascade {#cascade}
A **cascade** is a routing setup where a small, cheap model answers first and passes the request to a larger model only when it is unsure. There is no separate judge up front. The small model is the first step, and its confidence is the signal. If it is confident, the answer is returned right away. If not, the request moves up a level, and so on, until a model answers or the largest one handles it.

There is a cost to notice. A request that escalates pays for two models. A cascade only pays off when most requests stay at the cheap level.

## A worked example {#example}
Let's put numbers on it. A small model costs 1 unit per request, and a large one costs 10. In a cascade, every request goes to the small model first, and 20% escalate to the large one.

```python
small_cost = 1
large_cost = 10
escalation_rate = 0.2

cascade_cost = small_cost + escalation_rate * large_cost
always_large = large_cost

print(cascade_cost)                                # 3.0
print(always_large)                                # 10
print(round(1 - cascade_cost / always_large, 2))   # 0.7

# What if more requests escalate?
for rate in (0.2, 0.5, 0.9):
    print(rate, small_cost + rate * large_cost)
# 0.2 3.0
# 0.5 6.0
# 0.9 10.0
```

Here is how to read it:

1. Each request pays 1 for the small model, plus 10 for the 20% that go on, so the average cost is 3.
2. Always using the large model costs 10 per request, so the cascade saves 70%.
3. At a 50% escalation rate, the cost rises to 6.
4. At 90% it reaches 10, no cheaper than using the large model alone, and slower too, since most requests wait for two models.

The cascade only wins when the small model really handles most of the traffic.

## A common mistake {#mistake}
A common slip is trusting the small model's confidence without checking it. Some models are confidently wrong. Before you rely on the signal, measure how often a "confident" small answer is actually correct on a labelled set.

## Key points {#key-points}
- Routing matches each request to a model of the right size and cost.
- A cascade tries a small model first and escalates to a larger one only when unsure.
- The saving depends on the share of requests the small model can handle.
- Check that the small model's confidence is reliable before using it as the gate.
