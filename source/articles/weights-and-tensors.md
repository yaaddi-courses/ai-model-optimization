# Weights and tensors: what a model is made of

**Covers:** Weights, Tensor, Scalar, Vector, Matrix, Parameter count, AI, ML

## The big picture {#overview}
Machine learning (ML) is a way of getting a computer to learn a pattern from examples instead of following rules a person wrote by hand. AI is the wider name for software that does tasks which normally need human judgment, and most AI you meet today is built with ML. A trained ML model is not magic. Open the file and you find one thing: a very long list of numbers. Everything the model "knows" is stored in those numbers, and everything it does is arithmetic on them.

This article names the three ideas that make that list easy to talk about: weights, tensors and the parameter count. Later chapters shrink and speed up models, and every one of those tricks works by changing these numbers or the way they are stored.

## Weights {#weights}
A weight is one learned number. Training starts with random weights and adjusts them, example after example, until the model's answers are good. After training the weights are fixed, and they alone decide what the model predicts.

Think of a recipe card where each ingredient has an amount. The amounts are the weights. Change one amount slightly and the dish changes slightly. Change many of them and you get a different dish. A model file is mostly a huge recipe card, with millions or billions of amounts.

## Scalars, vectors and matrices {#scalar}
Models store numbers in grids, and each shape has a name:

- A **scalar** is a single number, like `3.5`. It has no dimensions.
- A **vector** is a list of numbers, like `[1.0, 2.0, 3.0]`. It has one dimension: a length.
- A **matrix** is a table of numbers with rows and columns. It has two dimensions.

Most layers of a model are matrices of weights. A layer that turns three input numbers into two output numbers needs a matrix with two rows and three columns. That is six weights.

## Tensor {#tensor}
A tensor is the general name for all of these. A scalar is a zero-dimensional tensor, a vector is a one-dimensional tensor and a matrix is a two-dimensional tensor. A colour photo is a three-dimensional tensor (height, width and three colour channels). A stack of photos is a four-dimensional tensor.

The word matters because every library for building models, such as PyTorch or TensorFlow, is built around tensors. Weights, inputs and outputs are all tensors, so once you can read tensor shapes you can read most model code.

## A worked example {#example}
Here is a tiny layer with three inputs and two outputs. Its weights form a matrix with two rows and three columns.

```python
inputs = [1.0, 2.0, 3.0]
weights = [
    [0.5, -1.0, 0.0],
    [2.0, 0.0, 1.0],
]

outputs = [sum(w * x for w, x in zip(row, inputs)) for row in weights]
print(outputs)  # [-1.5, 5.0]

count = sum(len(row) for row in weights)
print(count)  # 6
```

By hand, the steps are:

1. Take the first row of weights: `0.5, -1.0, 0.0`.
2. Multiply each weight by the matching input and add the results: 0.5 × 1 + (−1.0) × 2 + 0.0 × 3 = −1.5.
3. Repeat with the second row: 2.0 × 1 + 0.0 × 2 + 1.0 × 3 = 5.0.
4. Count the weights: two rows of three is 6.

A real layer does exactly this, only with thousands of rows and columns.

## Parameter count {#parameter-count}
The parameter count is the number of individual weights in a model. Our toy layer has 6. A small language model has about 100 million. The models behind modern chatbots have tens or hundreds of billions.

People write it with a short suffix. `7B` means seven billion, and `70B` means seventy billion. It is the quickest guide to how large and costly a model will be. A model with ten times the parameters needs roughly ten times the storage for its weights, and it usually needs more computing for every answer.

## A common mistake {#mistake}
Do not treat a bigger parameter count as a better model. A well-trained model with fewer parameters can beat a poorly trained larger one. Parameter count tells you the size of the model, not its quality.

## Key points {#key-points}
- Weights are the learned numbers inside a model; they decide its predictions.
- Scalars, vectors and matrices are grids of numbers with zero, one and two dimensions; a tensor is the general name for all of them.
- Most of a model is matrices of weights, and a layer's output is a sum of weights times inputs.
- The parameter count is how many weights there are, written like `7B`; it measures size, not quality.
