# Mixture of experts: many specialists, few at a time

**Covers:** Mixture of experts, Mixture-of-experts, MoE, Expert, Active parameters

## Big models without the full cost {#overview}
A dense model uses every weight for every token it processes. Doubling its size doubles the arithmetic per token. A mixture-of-experts design breaks that link. It builds a very large model out of many specialist parts and, for each token, uses only a few of them. The model stores a lot of knowledge, yet each token costs far less than the stored size suggests.

## Mixture of experts {#mixture-of-experts}
A **mixture of experts**, shortened to **MoE**, is a model design made of many expert sub-networks, with a small router that decides which ones to use. In a transformer, it usually replaces the large feed-forward block of a layer with a set of parallel blocks plus the router. The same idea is also written mixture-of-experts with hyphens, for example when used as a describing phrase: a mixture-of-experts model.

## Expert {#expert}
An **expert** is one of the sub-networks inside a mixture-of-experts layer, chosen by the router for some tokens. Experts do not specialise in neat human topics such as "law" or "cooking". They specialise in whatever pattern helps the model's training goal, and the pattern is often subtle.

## The router {#router}
For each token, the router scores all experts, picks the top few (often two) and combines their outputs, weighted by the scores. The other experts do no work for this token. Different tokens, even in the same sentence, can go to different experts.

## Active parameters {#active-parameters}
**Active parameters** count the weights that really take part when one token is processed. In a mixture-of-experts model the figure is far below the total parameter count, because only the chosen experts and the shared parts are used. Active parameters set the arithmetic per token, so they influence speed. The total sets the memory needed.

## A worked example {#example}
A toy model with 2 billion shared parameters and 8 experts of 5 billion each, where the router uses 2 experts per token.

```python
shared = 2          # billions of parameters used for every token
expert_size = 5
experts = 8
used_per_token = 2

total = shared + experts * expert_size
active = shared + used_per_token * expert_size

print(total)                            # 42
print(active)                           # 12
print(round(active / total, 2))         # 0.29

# But memory must hold everything:
bytes_per_parameter = 2
print(total * bytes_per_parameter)      # 84 GB
print(active * bytes_per_parameter)     # 24 GB of weights actually read per token
```

How to read it:

1. The model stores 42 billion parameters in total.
2. For any one token, only 12 billion take part: the shared 2 billion and the two chosen experts.
3. That is 29% of the total, so the arithmetic per token is roughly that of a 12-billion-parameter model.
4. At 2 bytes each, all 42 billion need 84 GB of memory, even though a token touches only 24 GB.

So a mixture-of-experts model is cheaper to run per token than its size suggests, but it is not cheaper to hold in memory.

## A common mistake {#mistake}
Judging the memory need by the active parameters. All experts must be available, in memory, because the router can pick any of them for the next token.

## Key points {#key-points}
- A mixture of experts has many expert sub-networks and a router that picks a few per token.
- Active parameters, the ones used for one token, are far fewer than the total.
- Compute per token follows the active count; memory follows the total.
- Experts specialise in learned patterns, not human subjects.
