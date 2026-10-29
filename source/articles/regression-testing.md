# Regression testing after optimizing

**Covers:** Regression, Golden set, Canary release, Shadow testing

## Faster is only good if it is still right {#overview}
Every optimization changes the model or the way it runs, and any change can break something that used to work. A smaller model may get a rare kind of question wrong. A new runtime may round a number differently. You find out only if you look. So testing after an optimization is not optional extra work. It is what turns "it is faster" into "it is faster and still correct".

## Regression {#regression}
A **regression** is a change that makes something that used to work correctly get worse. After an optimization, a regression could be a drop in accuracy, a new kind of wrong answer or a broken edge case. The word covers speed too: a change that makes a fast thing slow is a regression as well. The point of testing is to catch these before your users do.

## Golden set {#golden-set}
How do you notice a regression? You compare against something fixed. A **golden set** is a fixed collection of test inputs with known good outputs. You reuse it every time you change something, running the old and the new model on the same inputs and comparing the results. Because the inputs never change, any difference is caused by your change. A good golden set includes ordinary cases, unusual cases, and past failures that you fixed, so they cannot come back unnoticed.

## Shadow testing {#shadow-testing}
Some problems only show up on real traffic. **Shadow testing** runs the new model on copies of real requests without showing its answers to users. The current model still serves everyone. You collect the new model's answers on the side and compare them with the current one. There is no risk to users, but it doubles the compute for the period of the test.

## Canary release {#canary-release}
A **canary release** takes a more careful step toward real users. It sends a small share of live traffic to the new model first. If something is wrong, only a few users are affected, and you can roll back before everyone is. If all looks good, the share grows in steps until the new model takes over. The name comes from the canary that miners kept in coal mines to warn them of bad air.

## A worked example {#example}
Let's compare an old and an optimized model on a golden set of five cases, and then look at what a canary would expose.

```python
golden = [
    {"input": "refund policy", "expected": "30 days"},
    {"input": "shipping time", "expected": "3-5 days"},
    {"input": "opening hours", "expected": "9 to 5"},
    {"input": "contact", "expected": "support@example.com"},
    {"input": "warranty", "expected": "1 year"},
]
old_answers = ["30 days", "3-5 days", "9 to 5", "support@example.com", "1 year"]
new_answers = ["30 days", "3-5 days", "9 to 5", "support@example.com", "2 years"]

def score(answers):
    return sum(a == case["expected"] for a, case in zip(answers, golden)) / len(golden)

print(score(old_answers), score(new_answers))      # 1.0 0.8

regressions = [case["input"] for a, case in zip(new_answers, golden) if a != case["expected"]]
print(regressions)                                  # ['warranty']

canary_share = 0.05
print(canary_share * 100_000)                       # 5000.0 requests out of 100,000 see the new model
```

What happens here:

1. The old model gets all five right, and the optimized one gets four of five.
2. Listing the failures names the regression: the warranty answer changed from 1 year to 2 years.
3. This is a real fault and not noise, because the golden set never changes.
4. If the model passed, a 5% canary on 100,000 requests would expose it to only 5,000 of them while you watch.

## A common mistake {#mistake}
A common slip is building the golden set from the same examples you used to tune the optimization. It will look fine while real traffic suffers. Include cases from production logs, and add every bug you ever find.

## Key points {#key-points}
- A regression is a change that makes something worse, and optimization can cause them.
- A golden set is a fixed test set with known answers that you rerun after every change.
- Shadow testing compares the new model on real requests without showing users its answers.
- A canary release exposes a small share of users first, so a problem stays small.
