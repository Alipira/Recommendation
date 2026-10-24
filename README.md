# Recommendation

A recommendation system for B2B ordering on ERP data. Given a retailer, a
product and the retailer's current shopping cart, the model predicts how much
of that product the retailer will order. Scores are then used to rank products
for each retailer: what they usually reorder, and new products they are likely
to want.

This repo holds the first working model (a PyTorch deep model trained on
order history). The longer plan, which adds sequence models, a ranker and a
small language model, is in [docs/roadmap.md](docs/roadmap.md).

## What is in the repo

```
src/
  recsys_model.py     the model (embeddings, cart attention, interaction nets, head)
  Interaction_net.py  pairwise field-interaction block used twice in the model
  train.py            training loop, weighted Huber loss, MLflow logging
  consumer.py         reads ready-made batches from Redis Streams
  loss.py             extra loss functions (not used by train.py right now)
  training_viz.py     loss curves, predicted-vs-true plots, ROC plot
  embedding_viz.py    t-SNE / UMAP plots of the learned embeddings
utils/
  read_env.py         reads Redis settings from .env or the environment
docs/
  architecture.md     how the model works, step by step
  roadmap.md          where the project is going next
references/           papers and the research report the design is based on
```

## How it works, in short

1. **Inputs.** Each sample is one (retailer, product, date) row with
   17 categorical features (product groups, customer, location, sales visitor, ...),
   91 numeric features (price, cheque and payment history, recent order volumes
   for the retailer, its path and its zone, date and holidays), and the
   products already in the cart.
2. **Cart summary.** Every cart item is combined with the row's features,
   passed through an interaction network, and scored by a small attention
   network. The weighted sum gives one vector that describes the cart.
3. **Main path.** The row's features are grouped into 14 fields
   (product, price, customer, geography, payment, RFM, date, ...). The cart
   vector is added as a 15th field. Every pair of fields gets its own small
   network, so the model learns how fields work together.
4. **Head.** A 3-layer MLP turns the result into one score: the predicted
   `log1p(quantity)`.
5. **Loss.** Huber loss, weighted so that rare retailer-product pairs and
   larger orders count more. Plain loss would let frequent, small orders
   dominate.

More detail in [docs/architecture.md](docs/architecture.md).

## Data flow

Training data is not read from files. A separate producer service builds
batches from the ERP database and writes them to Redis Streams (one stream each
for train, validation and test). `consumer.py` reads one message per batch,
acknowledges and deletes it, and hands the arrays to PyTorch.

Each message contains these arrays, each with `_dtype` and `_shape` entries:

| Key                      | Shape       | Meaning                              |
|--------------------------|-------------|--------------------------------------|
| `batch_row_embed`        | (B, 17)     | categorical indices (0 = unknown)    |
| `batch_row_float`        | (B, 91)     | numeric features                     |
| `batch_sc_embed`         | (B, T, 5)   | cart item product-group indices      |
| `batch_sc_float`         | (B, T, 5)   | cart item weight, prices, quantity   |
| `mask`                   | (B, T, 1)   | 1 = real cart item, 0 = padding      |
| `batch_labels`           | (B, 1)      | target, `log1p(quantity)`            |
| `batch_person_frequency` | (B,)        | weight for rare retailer-product pairs |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
cp .env.example .env        # then fill in your Redis settings
```

Start an MLflow server (training logs to `http://127.0.0.1:5000`):

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Make sure the producer is writing to the three streams, then:

```bash
python -m src.train
```

Run it from the repo root so the `src.` and `utils.` imports work.

## Debugging bad indices

If training stops with a CUDA "device-side assert", an embedding index is
usually out of range. Build the model with `check_indices=True`:

```python
model = get_recsys_model(..., check_indices=True)
```

It then checks every batch and tells you which feature has a bad value.
Turn it off again for normal runs.

## Known issues

These are open questions, not yet changed in the code:

- **Output layer ends in ReLU.** If the last layer's input goes negative, the
  prediction is 0 and no gradient flows back. Softplus would keep the output
  positive without this problem.
- **BatchNorm sees cart padding.** In the cart path, padded slots are passed
  through BatchNorm together with real items, so padding shifts the batch
  statistics.
- **Learning-rate step never fires.** `LR_DECAY_STEPS = 100_000`, but a run is
  10 × 300 = 3,000 steps.
- **Plateau scheduler and early stopping use the same patience (5),** so early
  stopping ends the run before the learning rate is ever reduced.
- **Only regression metrics are tracked** (loss, MAE, MSE). Ranking metrics
  (Recall@K, NDCG@K on the full catalogue, split by repeat and new items) are
  the next step; see the roadmap.
- **Shape dtype.** Array shapes are sent as `int16`, which breaks for any
  dimension above 32,767. The consumer now accepts a `<field>_shape_dtype`
  entry, so the producer can switch to `int32` safely.

## License

MIT, see [LICENSE](LICENSE).
