# The accuracy budget: how much quality are you willing to spend?

**Covers:** Accuracy-speed trade-off, Accuracy budget, Baseline, Held-out test set, Slice

## Faster usually costs something {#overview}
Think about the last time you made something cheaper, such as a smaller phone plan or a lighter suitcase. You almost always gave something up. Models work the same way: nearly every method that makes one smaller or faster costs a little accuracy. Sometimes the loss is too small to notice, and sometimes it is large. So the sensible plan is to decide in advance how much you are willing to pay, and then to measure honestly whether you stayed inside that limit. In this article we build the vocabulary for doing that.

## The accuracy-speed trade-off {#accuracy-speed-trade-off}
Suppose you shrink a model so that it runs twice as fast, and it becomes one percentage point less accurate. Is that a good deal? It depends on who uses it. For a spam filter, one point is a bargain. For a medical tool, it could be unacceptable. This is the **accuracy-speed trade-off**: the price of optimization, paid in correct answers. No answer is right in general, because it depends on what a mistake costs and what a delay costs in your situation.

## Baseline {#baseline}
To say that a model got worse, you need something to compare it with. That something is the **baseline**: the measured accuracy and speed of the original model, before you changed anything. Every optimized version is compared against it. Without it, "91% accuracy" tells you nothing, because the original might have scored 91% or 99%.

Measure the baseline with care, using the same test data and the same hardware you will use for the new versions. If the baseline is sloppy, every comparison after it inherits the sloppiness.

## Accuracy budget {#accuracy-budget}
Now you can set a limit. An **accuracy budget** is the largest drop in accuracy you agree to accept, and you write it down before you start. For example: "We can lose up to one percentage point, and no more."

Why decide first? Because once you see the results, your judgement bends. A speed-up that looks exciting makes a bigger loss feel acceptable, and nervousness can make you stop too early. A budget fixed in advance removes the argument. If a result lands inside it, you ship it. If it lands outside, you try something else.

## Held-out test set {#held-out-test-set}
Where do the accuracy numbers come from? They must come from data the model has never seen, otherwise the result flatters it, much like a student who has seen the exam questions in advance. A **held-out test set** is a collection of examples you keep aside and never use for training or for tuning. Scoring on it shows how the model handles new inputs, which is exactly what real users will send.

Here is the part that trips people up. If you keep adjusting your optimization until it scores well on one test set, you are slowly training on that set, even though no training step ever touched it. Keep a second set untouched for the final check.

## Slice {#slice}
One accuracy number for the whole test set is an average, and an average can hide a local problem. A **slice** is a subgroup of the test data: short inputs versus long ones, one language versus another, one customer type versus another. When you score each slice on its own, you see damage that the overall figure covers up.

## A worked example {#example}
Let's work through one. A baseline scores 920 correct out of 1,000. After optimizing, a candidate scores 914 out of 1,000. The budget is one percentage point. The data has two slices, short and long inputs, and we will check both.

```python
baseline = {"short": (590, 600), "long": (330, 400)}
candidate = {"short": (598, 600), "long": (316, 400)}

def accuracy(groups):
    correct = sum(c for c, n in groups.values())
    total = sum(n for c, n in groups.values())
    return 100 * correct / total

budget = 1.0
drop = accuracy(baseline) - accuracy(candidate)
print(accuracy(baseline), accuracy(candidate))   # 92.0 91.4
print(round(drop, 1), drop <= budget)            # 0.6 True

for name in baseline:
    a = 100 * baseline[name][0] / baseline[name][1]
    b = 100 * candidate[name][0] / candidate[name][1]
    print(name, round(a - b, 1))                 # short -1.3, long 3.5
```

Here is how to read the output:

1. Overall accuracy fell from 92.0% to 91.4%. That is a drop of 0.6 points, inside the budget of 1.0.
2. On short inputs the candidate is 1.3 points better, so the drop there is negative.
3. On long inputs it is 3.5 points worse: 82.5% down to 79.0%.
4. The overall number passed, yet anyone who sends long inputs will notice a real drop.

So a passing average did not mean everyone was safe. The fix is to give each important slice its own budget, as well as the whole.

## A common mistake {#mistake}
A common slip is using the same data to tune and to judge. It feels natural, because that data is already at hand. But the score then says how well you fitted that data, not how the model will behave tomorrow. Choose your settings on one set, then report accuracy on a held-out test set that you touched only once.

## Key points {#key-points}
- Optimizing usually trades some accuracy for speed or size.
- A baseline is the original model's measured score, and every comparison uses it.
- An accuracy budget is the drop you accept, fixed before you start.
- Measure on a held-out test set, and check slices as well as the overall number.
