# Extra notes — attention by hand

A tiny worked example I did to convince myself self-attention isn't magic. Three words, two dimensions, made-up numbers, no framework. Just to watch the formula actually run.

Sentence: **"the cat sat"**. Pretend each word is a 2-D vector after embedding:

```
the = [1, 0]
cat = [0, 1]
sat = [1, 1]
```

To keep it readable I'll cheat and let Q = K = V = the embeddings themselves (in the real model these come from learned weight matrices Wq, Wk, Wv).

### Step 1 — scores for the word "cat" (its query vs every key)

`score = Q · Kᵀ`  with Q = cat = [0, 1]:

```
cat · the = (0×1)+(1×0) = 0
cat · cat = (0×0)+(1×1) = 1
cat · sat = (0×1)+(1×1) = 1
```

So before normalising, "cat" finds itself and "sat" equally relevant, and "the" irrelevant. Makes sense.

### Step 2 — scale by √dₖ

dₖ = 2, so divide each by √2 ≈ 1.41:

```
[0, 1, 1] / 1.41 ≈ [0, 0.71, 0.71]
```

### Step 3 — softmax → attention weights

```
softmax([0, 0.71, 0.71]) ≈ [0.21, 0.39, 0.39]
```

"cat" will build its new representation from **21% of 'the', 39% of 'cat', 39% of 'sat'**.

### Step 4 — weighted sum of the values

```
out(cat) = 0.21·[1,0] + 0.39·[0,1] + 0.39·[1,1]
         = [0.21+0+0.39 , 0+0.39+0.39]
         = [0.60, 0.78]
```

That `[0.60, 0.78]` is the new, context-aware vector for "cat" — it has now mixed in a bit of "the" and a lot of "sat." Do this for every word, in parallel, and you've run one attention layer.

The only thing the real Transformer adds on top of this is: learned Q/K/V projections, several heads doing it at once, the √dₖ scaling (which I used), residual connections, and a feed-forward layer after. The heart of it is exactly the four steps above.
