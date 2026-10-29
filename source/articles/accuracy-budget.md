# The accuracy budget: how much quality are you willing to spend?

**Covers:** Accuracy-speed trade-off, Accuracy budget, Baseline, Held-out test set, Slice

## Faster usually costs something {#overview}
Almost every way to make a model smaller or faster gives up a little accuracy. Sometimes the loss is too small to notice, and sometimes it is large. Because it is a trade, the sensible way to optimize is to decide in advance how much you will pay, and then to measure honestly whether you stayed inside that limit. This article covers the vocabulary for that.

## The accuracy-speed trade-off {#accuracy-speed-trade-off}
The **accuracy-speed trade-off** is the price of optimization. A model that has been shrunk to run twice as fast may be one percentage point less accurate. That could be a bargain for a spam filter and a disaster for a medical tool. Neither answer is correct in general, because it depends on what a mistake costs and what a delay costs.

## Baseline {#baseline}
A **baseline** is the measured accuracy and speed of the original, unoptimized model. Every optimized version is compared against it. Without a baseline, "91% accuracy" means nothing, because you do not know whether the original scored 91% or 99%.

Measure the baseline carefully, on the same test data and the same hardware you will use for the new versions. If the baseline is sloppy, every comparison inherits the sloppiness.

## Accuracy budget {#accuracy-budget}
An **accuracy budget** is the largest accuracy drop you will accept from an optimization, decided before you start. For example: "We can lose up to one percentage point, no more."

Setting the budget first protects you from two mistakes. One is stopping too early because you are nervous. The other is accepting a bigger loss because the speed-up looks exciting. When a result lands inside the budget you ship it, and when it lands outside you try something else.

## Held-out test set {#held-out-test-set}
Accuracy has to be measured on data the model has never seen, otherwise the result flatters it. A **held-out test set** is data kept aside and never used for training or tuning. Measuring on it shows how the model handles new inputs, which is what real users send.

If you tune your optimization until it scores well on the same test data, you are slowly training on it. Keep a second set untouched for the final check.

## Slice {#slice}
One accuracy number for the whole test set can hide a local problem. A **slice** is a subgroup of the test data, such as short inputs versus long inputs, one language versus another, or one customer type. Evaluating each slice on its own shows damage that the overall average hides.

## A worked example {#example}
A baseline scores 920 correct out of 1,000. After optimizing, a candidate scores 914. The budget is one percentage point.

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

The reading:

1. Overall accuracy fell from 92.0% to 91.4%, a drop of 0.6 points, inside the budget of 1.0.
2. On short inputs the candidate is actually 1.3 points better.
3. On long inputs it is 3.5 points worse (82.5% down to 79.0%).
4. The overall number passed, but anyone who sends long inputs will notice a clear drop.

The fix is to give each important slice its own budget, not only the whole.

## A common mistake {#mistake}
Using the same data to tune and to judge. Choose settings on one set, then report accuracy on a held-out test set that you touched only once.

## Key points {#key-points}
- Optimizing usually trades some accuracy for speed or size.
- A baseline is the original model's measured score; every comparison uses it.
- An accuracy budget is the drop you accept, fixed before you start.
- Measure on a held-out test set, and check slices as well as the overall number.
