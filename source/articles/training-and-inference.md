# Training and inference: learning versus answering

**Covers:** Training, Inference, Forward pass, Backward pass, Gradients, Fine-tuning, Pretrained model

## Two phases of a model's life {#overview}
Think of a student who studies for months and then sits an exam. A model's life has the same two parts. First it learns, which happens once or rarely and takes enormous effort. Then it answers, which happens every time someone uses it. Almost everything in this course is about the second job, because that is where the daily cost and the waiting time live. To see why, you need to know exactly what each phase does.

## Training {#training}
**Training** is the process of adjusting a model's weights using many examples until its predictions are good. It starts from random numbers. The model sees an example, makes a guess, is told how wrong it was, and nudges its weights to be a little less wrong next time. Repeat that millions of times, and random numbers turn into something useful.

Training is expensive. It can need many GPUs for days or weeks. The good news is that it happens rarely: a company may train a model once and then use it for a year.

## Forward pass {#forward-pass}
Both phases begin the same way. The **forward pass** sends an input through the model's layers one after another, each layer multiplying by its weights, until an output comes out. If the model reads an email and says "spam", that answer is the result of one forward pass.

## Gradients {#gradients}
During training, the model also needs to know how to change. **Gradients** answer that. For each weight, the gradient says which direction to move it, and by how much, to reduce the error. A large gradient means a change to that weight would matter a lot. A gradient near zero means the weight barely affects the mistake.

## Backward pass {#backward-pass}
The **backward pass** is the step that computes the gradients. It starts at the error and works backwards through the layers, working out each weight's share of the blame. Then the weights are updated using those gradients, and the next example begins.

The backward pass is extra work on top of the forward pass, and it needs the intermediate results from the forward pass kept in memory. That is a large part of why training costs so much more than answering.

## Inference {#inference}
**Inference** means running an already-trained model on new input to get a prediction. It uses only forward passes. The weights are frozen, no gradients are computed, and nothing is learned.

Inference happens on every user request, so a model that is trained once may serve billions of answers. Saving a few milliseconds or a few cents per answer adds up quickly, which is why this course focuses here.

## A worked example {#example}
Let's walk through one training step on a model with a single weight. The model predicts `weight × input`. The input is 2, the target is 3, and the weight starts at 0.5.

```python
weight = 0.5
x, target = 2.0, 3.0
learning_rate = 0.1

prediction = weight * x                # forward pass
error = prediction - target
gradient = 2 * error * x               # backward pass for squared error
weight = weight - learning_rate * gradient

print(prediction, error, gradient)     # 1.0 -2.0 -8.0
print(round(weight, 2))                # 1.3
print(round(weight * x, 2))            # 2.6
```

The steps are:

1. Forward pass: 0.5 × 2 = 1.0, but the target is 3.0, so the error is −2.0.
2. Backward pass: the gradient of the squared error is 2 × (−2.0) × 2 = −8.0. A negative gradient means the weight should go up.
3. Update: 0.5 − 0.1 × (−8.0) = 1.3.
4. The new prediction is 1.3 × 2 = 2.6, much closer to 3.0.

Repeat the step a few times and the prediction reaches the target. Inference would stop after step 1.

## Pretrained model {#pretrained-model}
Training from scratch is so costly that most people never do it. A **pretrained model** is one that someone has already trained on a large amount of data. You download it and use it as it is, or you adapt it.

## Fine-tuning {#fine-tuning}
How do you adapt it? With **fine-tuning**: a short extra round of training on top of a pretrained model. You give it examples from your own task, such as your support tickets, and its weights shift to fit them. It costs far less than training from scratch, because the model already knows the basics. Later articles show cheaper ways to fine-tune, and also that fine-tuning is often used to win back accuracy after shrinking a model.

A common slip is to think that using a model keeps teaching it. It does not. During inference the weights are frozen, so a model that answered a million questions yesterday is exactly the same model today.

## Key points {#key-points}
- Training adjusts weights from examples, and it is costly but rare.
- The forward pass turns an input into an output, and the backward pass computes gradients.
- Gradients say which way and how far to move each weight.
- Inference is forward passes only, with frozen weights, and it runs on every request.
- A pretrained model saves you from training from scratch, and fine-tuning adapts it cheaply.
