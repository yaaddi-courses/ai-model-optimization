# Speculative decoding: guess ahead, then check

**Covers:** Draft model, Target model, Speculative decoding, Speculative-decoding, Acceptance rate

## Why writing is slow {#overview}
A large model writes one token per pass, and each pass has to read all of its weights from memory. That makes text generation slow, and the cost is mostly reading, not arithmetic. A surprising fact helps here: checking several tokens at once costs barely more than producing one. If we could guess the next few tokens cheaply, the big model could verify the whole guess in a single pass. Speculative decoding does exactly that.

## Target model {#target-model}
The **target model** is the large, high-quality model whose output you actually want. It has the final say. Whatever it would have written on its own is what the user must receive, so quality is the same as running it alone.

## Draft model {#draft-model}
The **draft model** is a small, fast model that quickly guesses several upcoming tokens. It is far less accurate than the target, but it is cheap, and it often guesses right on easy stretches of text such as common phrases or repeated words.

## Speculative decoding {#speculative-decoding}
**Speculative decoding** puts the two together in a loop:

1. The draft model writes a short run of tokens, for example four.
2. The target model checks all four in one pass, comparing what it would have written at each position.
3. It accepts the draft tokens up to the first one it disagrees with, and replaces that one with its own choice.
4. The loop repeats from there.

Each round gives at least one token from the target, and often several. The method is written Speculative-decoding in some places, but it is the same thing. Because the target verifies everything, the final text is the one the target would have produced alone.

## Acceptance rate {#acceptance-rate}
The **acceptance rate** is the share of drafted tokens that the target model agrees with. A high rate means few wasted guesses and a big speed-up. A low rate means most drafts are thrown away and the extra work is not worth it. The rate depends on how closely the draft model resembles the target and on how predictable the text is.

## A worked example {#example}
The expected number of tokens per round, and the speed-up, when the draft model writes 4 tokens per round and each is accepted with probability 0.8. Suppose one draft token costs a tenth of a target pass.

```python
acceptance = 0.8
draft_length = 4
draft_cost = 0.1          # one drafted token costs 0.1 of a target pass

tokens_per_round = (1 - acceptance ** (draft_length + 1)) / (1 - acceptance)
round_cost = 1 + draft_length * draft_cost

print(round(tokens_per_round, 2))                 # 3.36
print(round(round_cost, 2))                       # 1.4
print(round(tokens_per_round / round_cost, 2))    # 2.4 times faster

for rate in (0.5, 0.8, 0.95):
    n = (1 - rate ** (draft_length + 1)) / (1 - rate)
    print(rate, round(n / round_cost, 2))
# 0.5 1.38
# 0.8 2.4
# 0.95 3.23
```

Reading it:

1. With a rate of 0.8, each round produces about 3.36 tokens on average, counting the one the target always contributes.
2. A round costs one target pass plus four cheap draft steps: 1.4 passes in total.
3. Dividing gives a speed-up of 2.4 times over writing one token per target pass.
4. At a rate of 0.5 the gain falls to 1.38 times, and at 0.95 it reaches 3.23. The rate is the lever.

## A common mistake {#mistake}
Choosing a draft model that is too large, or too different from the target. A large draft costs too much, and a different one is wrong too often. The draft should be tiny, and it should be trained on similar text.

## Key points {#key-points}
- Speculative decoding drafts several tokens with a small model and checks them in one pass with a large one.
- The target model decides, so the output matches what it would write alone.
- The draft model must be cheap and reasonably close to the target.
- The acceptance rate drives the speed-up: more accepted tokens means fewer slow passes.
