# Low-rank factorization and weight sharing: storing fewer numbers

**Covers:** Low-rank factorization, Rank, Weight sharing, ALBERT

## Two ways to store less {#overview}
So far you have seen two ways to shrink a model: pruning removes weights, and quantization stores each weight in fewer bits. A third family goes after the layout itself. It finds a way to describe the same weights using fewer numbers in total. We will look at its two main versions: approximating a big matrix by smaller ones, and reusing the same weights in several places.

## Low-rank factorization {#low-rank-factorization}
Imagine a table with 1,000 rows and 1,000 columns. Storing it takes a million numbers. Now suppose the table has a simple underlying pattern. Then you may be able to write it as a tall thin table (1,000 rows, a few columns) multiplied by a short wide one (a few rows, 1,000 columns). **Low-rank factorization** does exactly that: it approximates one big weight matrix as the product of two thinner matrices. You store only a small fraction of the numbers.

## Rank {#rank}
What does "a few columns" mean exactly? That number is the **rank**: the shared inner width of the two thin matrices. A lower rank compresses more but approximates the original less faithfully. A higher rank stays closer to the original and saves less. Choosing the rank is the central decision. The word rank also has an everyday sense, a position in an ordered list, but in this article it always means this inner width.

## Weight sharing {#weight-sharing}
The second idea is different. **Weight sharing** reuses the same set of weights in several places in a model. Instead of every layer having its own copy, several layers (or several places inside one layer) point at one copy. The model still computes with the shared weights wherever they are used, but it stores them only once, so the parameter count falls.

## ALBERT {#albert}
**ALBERT** is a variant of BERT built around this idea. It shares weights across all its layers, so a stack of 12 layers stores the parameters of roughly one. ALBERT-base has about 12 million parameters, compared with about 110 million for BERT-base. It is smaller to store, but it does not run any faster, because the computing is still done once per layer.

## A worked example {#example}
Let's count the saving from low-rank factorization.

```python
# Low-rank: a 1000 x 1000 matrix replaced by (1000 x r) times (r x 1000)
n, r = 1000, 10
full = n * n
factored = 2 * n * r
print(full, factored, factored / full)       # 1000000 20000 0.02
print(2 * n * 100 / full)                    # 0.2
```

Reading the output:

1. The full matrix has 1,000,000 numbers.
2. The two thin matrices have 2 × 1,000 × 10 = 20,000 numbers.
3. That is 2% of the storage, if a rank of 10 is good enough to approximate the matrix.
4. Try rank 100, and the factored form has 200,000 numbers, which is 20% of the original. A lower rank saves more.

Whether the low-rank copy is accurate is a question you must test. The saving is certain, but the quality is not.

## A common mistake {#mistake}
A common slip is assuming that fewer parameters means faster. Weight sharing cuts storage, but the model still does the same number of calculations. Low-rank factorization can cut both, because multiplying by two thin matrices costs less than multiplying by one large one.

## Key points {#key-points}
- Low-rank factorization replaces a large matrix with a product of two thin ones.
- Rank is the inner width, and a lower rank saves more but approximates less.
- Weight sharing stores one set of weights and reuses it, and ALBERT shares across layers.
- Sharing saves storage but not computation, because the model still runs every layer, while low-rank factorization can reduce both.
