# Pruning: removing the weights that do little

**Covers:** Pruning, Magnitude pruning, Structured pruning, Unstructured pruning, Sparsity

## The idea {#overview}
Not every weight in a trained model matters equally. Many contribute almost nothing, so removing them barely changes the output. Pruning takes advantage of that by deleting the weights that matter least. A gardener prunes a tree the same way: cut the branches that bear no fruit, and the tree keeps producing.

## Pruning {#pruning}
**Pruning** removes weights, or larger groups of them, that contribute little to a model's output. The result is a smaller or cheaper model. Most pruning is followed by a short round of fine-tuning, because the cuts cause a small accuracy drop and the remaining weights can adapt to recover it.

Pruning is not free to apply: you decide what to remove, how much, and whether the hardware can use the result. The next sections show those choices.

## Sparsity {#sparsity}
**Sparsity** is the fraction of a model's weights that are exactly zero. A model that is 90% sparse has nine zeros for every non-zero weight. A pruned weight is set to zero, so sparsity is the usual way to say how heavily a model was pruned.

The catch is that a zero still sits in memory unless the software stores the model in a special sparse format. A file full of zeros is not automatically smaller.

## Magnitude pruning {#magnitude-pruning}
**Magnitude pruning** uses the simplest rule for choosing: the weights with the smallest absolute values are assumed to matter least, so they are set to zero first. It is easy to implement and a common baseline, though a small weight is not always an unimportant one.

## Unstructured pruning {#unstructured-pruning}
**Unstructured pruning** zeroes out individual weights wherever they happen to be. It can reach high sparsity with little accuracy loss, because it picks the best weights across the whole model. The zeros are scattered, however, and ordinary hardware gains nothing from scattered zeros. A real speed-up needs software or chips that are built to skip them.

## Structured pruning {#structured-pruning}
**Structured pruning** removes whole units at once, such as an entire neuron, a channel or an attention head. What remains is a smaller dense model that regular hardware runs faster with no special support. The cost is that cutting whole units is a blunter tool, so it usually loses more accuracy at the same amount removed.

## A worked example {#example}
Eight weights, with the smallest half removed by magnitude.

```python
weights = [0.8, -0.05, 0.3, 0.01, -0.6, 0.02, 0.1, -0.04]

cutoff = sorted(abs(w) for w in weights)[len(weights) // 2 - 1]
pruned = [w if abs(w) > cutoff else 0.0 for w in weights]

print(cutoff)                              # 0.05
print(pruned)                              # [0.8, 0.0, 0.3, 0.0, -0.6, 0.0, 0.1, 0.0]
print(pruned.count(0.0) / len(pruned))     # 0.5
```

Step by step:

1. Sort the absolute values: 0.01, 0.02, 0.04, 0.05, 0.1, 0.3, 0.6, 0.8.
2. The fourth value, 0.05, is the cutoff for removing half.
3. Every weight at or below 0.05 in size becomes zero.
4. Four of the eight weights are now zero, so the sparsity is 0.5.

This is unstructured pruning. The zeros are scattered through the list, so storing the list as ordinary numbers uses the same memory as before.

## A common mistake {#mistake}
Reporting a high sparsity as if it were a speed-up. Sparsity counts zeros. Whether those zeros save time depends on the structure of the cut and on hardware support, so measure the actual speed.

## Key points {#key-points}
- Pruning removes weights that contribute little and is usually followed by fine-tuning.
- Sparsity is the share of weights that are exactly zero.
- Magnitude pruning removes the smallest-valued weights first.
- Unstructured pruning scatters zeros and needs sparse support; structured pruning removes whole units and speeds up ordinary hardware.
