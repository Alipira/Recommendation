# Architecture

## Data contract

Every source (Instacart, ERP, fake data) is turned into the same three tables
(`hybridrec/data/schema.py`):

| Table | Columns |
|-------|---------|
| interactions | `user_id`, `item_id`, `basket_id`, `ts` (days), `qty` |
| items | `item_id`, `title`, plus text columns used in prompts (aisle, group, flavour, ...) |
| users (optional) | `user_id`, plus numeric feature columns |

`Dataset.validate()` checks it: no unknown items, one user and one time per
basket, numeric user features. After that point, nothing knows where the data
came from.

## Splits

Last-basket split: each customer's last basket is test, the one before is
validation, the rest is training. Customers with fewer than 3 baskets stay in
training only. A time-cutoff split is also there for a single global "now".

What each stage may see:

| Stage | Learns from | Checked on |
|-------|-------------|------------|
| VAE, LightGCN | train | validation (graph early stopping) |
| Reranker | validation baskets (the graph never saw them) | 10% of validation customers |
| Final report | — | test baskets, with train + validation as history |

## Customer VAE (`models/vae.py`)

Input: one standardised feature row per customer. With no customer table,
features come from training history: number of baskets, basket size, days
between orders, repeat share, distinct items, and share of purchases per top
category.

Encoder (2 layers, 64 wide) → mean and log-variance (16 dims) → decoder.
Loss: reconstruction error + β × KL, with β growing from 0 to 0.5 over the
first 10 epochs so the model learns to reconstruct before it is pushed toward
a smooth latent space. The mean vector is the customer's output.

Ready-made CustomerVAE vectors can replace this stage (`data.user_vectors`).

## LightGCN (`models/lightgcn.py`)

Graph: customers + products as nodes, one edge per (customer, product)
bought in training. The adjacency is normalised as D^-1/2 A D^-1/2.

Starting vector of a customer:

```
e0(u) = id_embedding(u) + Linear(vae_vector(u))
```

Each layer replaces every node's vector with the weighted average of its
neighbours. The final vector is the mean of the starting vector and the 3
layer outputs. There are no weights inside the layers, which is why LightGCN
is fast and hard to overfit.

**ID dropout:** in each training step, 10% of nodes lose their ID part and
must be placed from their features alone. This teaches the feature
projection to carry real information, which helps new customers.

**Loss:** BPR. For a bought pair (u, i) and a random item j the customer did not
buy, push `score(u,i) − score(u,j)` up. Plus a small L2 penalty on the starting
vectors. Early stopping on validation recall@20.

## Candidates (`candidates.py`)

Up to 8 of the customer's own most bought items, then the graph's top items,
no duplicates, 20 in total. Repeat items are added on purpose: in basket data
a large share of every next order is reorders, and a "just repeat" baseline is
very strong (see `references/`). The graph adds the new items.

The share of held-out items that make it into the 20 candidates is the
ceiling for the reranker. It is printed as "candidate coverage".

## LLM reranker (`llm/`)

**Prompt** (`llm/prompt.py`):

```
Customer: <|user_vec|>
Recent purchases, newest first: Milk 1L; Bread; Tea
Candidates:
A. Milk 1L <|item_vec|>
B. Cheddar 200g <|item_vec|>
...
Which candidate is the customer most likely to buy next?
Answer:
```

**Vector tokens:** `<|user_vec|>` and `<|item_vec|>` are new special tokens.
After the prompt is turned into embeddings, those positions are overwritten
with `MLP(graph vector)`. One MLP for customers and one for products, each
`64 → 512 → hidden size of the LLM` (1024 for Qwen3-0.6B). This is the
LLaRA idea: the LLM gets the text and the collaborative signal side by side.

**Scoring:** take the LLM's next-token logits at the last position and keep
only the tokens for "A", "B", "C", ... That gives one score per candidate in a
single forward pass, with no text generation (the LlamaRec idea). Because
nothing is generated, the model cannot name a product that is not in the
list.

**Loss:** a held-out basket can contain several candidates, so the loss is
`−log(sum of probabilities of all bought candidates)`. Customers with no
bought item among the candidates are skipped during training.

**What trains:** LoRA (rank 16) on the q, k, v, o projections, and the two
MLPs. The base Qwen weights stay frozen.

## Evaluation (`eval/`)

- Full catalogue, no sampled negatives.
- Recall@K, NDCG@K, hit@K, plus recall on repeat items and on new items.
- Baselines: popular items, and each customer's own most bought items.
- `paired_bootstrap()` to check that a gain between two models is real.

The report compares: popular, repeat, graph only, merged candidates in their
original order, and graph + LLM reranker. The reranker is only worth its cost
if it beats the merged-candidate order and the repeat baseline.
