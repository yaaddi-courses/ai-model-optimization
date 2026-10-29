# Knowledge distillation: a small model learns from a big one

**Covers:** Knowledge distillation, Teacher model, Student model, Soft labels, Temperature, DistilBERT, BERT, ResNet

## The idea {#overview}
Suppose you want a small model that behaves like a big one. You could cut the big model down, but there is another way: train a different, smaller model to imitate it. This is **knowledge distillation**. A large, accurate model guides a small one during training. The small one ends up much cheaper to run, and it keeps a good part of the large one's quality. The shrinking happens through learning, not through deleting or rounding.

## Teacher model {#teacher-model}
Every lesson needs a teacher. The **teacher model** is the large, accurate model whose behaviour you want to copy. You use it only for its answers. Its own weights are not changed during distillation, and you do not need to deploy it afterwards.

## Student model {#student-model}
The **student model** is the small model that learns to imitate the teacher. It is the one you actually ship. You are free to choose its size and design, which makes distillation more flexible than other methods, because the student does not need the same shape as the teacher.

## Soft labels {#soft-labels}
Why can a student learn well from a teacher? Think about an ordinary training label. It is hard: this photo is a cat, full stop. A teacher tells you more. Its **soft labels** are its full probability outputs, such as 70% cat, 25% fox and 5% truck. They reveal which wrong answers are nearly right. The student learns that a cat looks a bit like a fox and almost nothing like a truck, which a plain label cannot teach. That extra information is the main reason a student can learn well from fewer examples.

## Temperature {#temperature}
There is a difficulty. Often the teacher is too confident: it gives 99.9% to its top answer and almost nothing to the rest, which hides exactly what we want to pass on. **Temperature** is a knob that divides the model's raw scores before they are turned into probabilities. A higher temperature flattens the distribution and brings the near-misses back into view. A lower temperature sharpens it. Distillation commonly uses a temperature above 1, so the student sees the finer detail.

## BERT, DistilBERT and ResNet {#distilbert}
**BERT** is a language model that Google released in 2018, and it is a common teacher in examples. **DistilBERT** is a smaller version of BERT created by distillation. Its authors reported a model about 40% smaller and 60% faster that kept about 97% of BERT's score on a language-understanding benchmark. Results always depend on the task, so measure on yours.

**ResNet** is something else: a neural network for images. It is not a distilled language model, and it appears in questions as a name that does not fit.

## A worked example {#example}
Let's take raw scores from a teacher for three classes and turn them into probabilities at three temperatures.

```python
import math

def softmax(scores, temperature):
    exps = [math.exp(s / temperature) for s in scores]
    total = sum(exps)
    return [round(e / total, 3) for e in exps]

scores = [4.0, 2.0, 0.5]
print(softmax(scores, 1))   # [0.858, 0.116, 0.026]
print(softmax(scores, 2))   # [0.649, 0.239, 0.113]
print(softmax(scores, 4))   # [0.494, 0.3, 0.206]
```

Here is what we see:

1. At temperature 1, the top class gets 85.8% and the others look unimportant.
2. At temperature 2, the top class drops to 64.9% and the second gets 23.9%.
3. At temperature 4, the picture is much flatter: 49.4%, 30.0% and 20.6%.
4. The order never changes, but the flatter version shows how close the second choice is. That is the signal the student learns from.

The student is trained to match these softened probabilities, usually mixed with the ordinary loss on the true labels.

## A common mistake {#mistake}
A common slip is expecting the student to match the teacher everywhere. It copies the teacher's behaviour on the kinds of inputs it was trained on, and it can fall short elsewhere. Test it on the cases that matter to you.

## Key points {#key-points}
- Distillation trains a small student to copy a large teacher's outputs.
- Soft labels carry how close the wrong answers are, not only which one is right.
- Temperature softens the teacher's probabilities so the student can learn from them.
- DistilBERT is a distilled BERT, and ResNet is an image network that is not part of that family.
