# Tokens, prompts and the context window

**Covers:** Token, Tokenizer, Prompt, Context window, LLM

## What a language model reads {#overview}
A large language model, or LLM, is a neural network trained on huge amounts of text to predict and generate language. It does not read letters or whole words. It reads small pieces called tokens, and everything about its cost and its limits is counted in them. Knowing how tokens work lets you estimate cost, plan prompts and understand why a long conversation eventually breaks.

## Token {#token}
A **token** is a small chunk of text: a whole common word, part of a rarer word, or a symbol. The word "the" is usually one token. A rare word such as "unbreakable" might be three: "un", "break" and "able". In English, one token averages roughly three quarters of a word, so 1,000 tokens is about 750 words. Other languages often need more tokens for the same meaning.

## Tokenizer {#tokenizer}
The **tokenizer** is the program that splits raw text into tokens using a fixed vocabulary, and turns tokens back into readable text at the end. Each token has an ID number, and those numbers are what the model actually receives. Every model has its own tokenizer, so the same sentence can produce a different count in different models.

## Prompt {#prompt}
The **prompt** is the input text you give the model: a question, an instruction, or a long document with a question at the end. The model reads the whole prompt first and then continues it with a reply. Everything in the prompt counts toward the cost and toward the limit below, including instructions you reuse every time.

## Context window {#context-window}
The **context window** is the maximum number of tokens a model can consider at once, counting both the prompt and the reply. If the window is 8,000 tokens and your prompt uses 6,000, the reply can use at most 2,000. A conversation that grows past the window must be trimmed or summarised, or the model will not see the earliest parts.

A larger window is not free. The memory kept for earlier tokens grows with every token, so long windows make serving more expensive.

## A worked example {#example}
A toy tokenizer that always takes the longest piece it knows, and then a budget check.

```python
vocab = {"un", "break", "able", "cat", "s", "the", " "}

def tokenize(text):
    pieces = []
    i = 0
    while i < len(text):
        for j in range(len(text), i, -1):
            if text[i:j] in vocab:
                pieces.append(text[i:j])
                i = j
                break
        else:
            pieces.append(text[i])
            i += 1
    return pieces

print(tokenize("unbreakable"))   # ['un', 'break', 'able']
print(tokenize("the cats"))      # ['the', ' ', 'cat', 's']
print(tokenize("xyz"))           # ['x', 'y', 'z']

window = 8000
prompt_tokens = 6000
print(window - prompt_tokens)    # 2000 tokens left for the reply
```

What it shows:

1. A known word is cut into the largest pieces in the vocabulary: "unbreakable" becomes three tokens.
2. Spaces can be tokens too, and a plural can split off its "s".
3. A word the vocabulary does not know falls back to single characters, so "xyz" costs three tokens. Unusual text is more expensive than common text of the same length.
4. With an 8,000-token window and a 6,000-token prompt, only 2,000 tokens are left for the reply.

Real tokenizers learn their vocabulary from data, but the idea is the same.

## A common mistake {#mistake}
Estimating cost from the number of words or characters. Providers bill and limit by tokens, so count with the model's own tokenizer before you plan a budget.

## Key points {#key-points}
- An LLM reads and writes tokens, small chunks of text, not letters or whole words.
- The tokenizer splits text into tokens and turns them back; each model has its own.
- The prompt is the input, and every prompt token costs time and money.
- The context window caps prompt plus reply together, and longer windows need more memory.
