# Recommendation

A recommender that turns customers into vectors with a graph network and a
VAE, and feeds those vectors into a small LLM (Qwen) that picks the best
products from a short list.

It is built for B2B ordering on ERP data (shops reordering from a
distributor) and is tested first on the public Instacart dataset.

```
 orders ──► customer–product graph ──► LightGCN ──► customer & product vectors
                                          ▲                    │
 customer features ──► VAE ──► node features                   │
                                                               ▼
                      repeat items + graph top items ──► 20 candidates
                                                               │
          prompt: customer token, recent purchases,            ▼
          each candidate's title + its vector token ──► Qwen3 + LoRA
                                                               │
                                                               ▼
                                                     ranked recommendations
```

## How it works

1. **Customer VAE.** Each customer's feature row (order habits, category mix,
   or your own ERP profile columns) is squeezed into a 16-number vector.
   Customers with similar habits get similar vectors, and a brand-new customer
   still gets one from their profile alone.
2. **LightGCN.** Customers and products are nodes; each purchase is an edge.
   Every customer starts as `ID vector + projected VAE vector`. Three rounds of
   neighbour averaging spread "who buys what" through the graph. Trained with
   BPR loss: bought products should score above random ones.
3. **Candidates.** For each customer: up to 8 of their own most bought items
   (shops reorder a lot), filled up to 20 with the graph's top items.
4. **LLM reranker.** One prompt per customer lists their recent purchases and
   the 20 candidates by name. The prompt also holds special tokens whose
   embeddings are replaced by the graph vectors (passed through a small MLP).
   So Qwen reads both the product names and where each product and customer
   sits in the purchase graph. The next-token scores for the letters A, B, C,
   ... rank all candidates in one pass. Only LoRA adapters and the two MLPs
   are trained; Qwen itself stays frozen.
5. **Evaluation.** Recall, NDCG and hit rate over the full catalogue, also
   split into repeat items and new items, against "popular" and "repeat"
   baselines.

More detail in [docs/architecture.md](docs/architecture.md).

## Repo layout

```
hybridrec/
  data/         data contract, Instacart and ERP loaders, splits, customer features, fake data
  models/       customer VAE, LightGCN, graph training loop
  llm/          prompt, reranker (Qwen + LoRA + vector tokens), reranker training
  eval/         metrics and baselines
  candidates.py merge repeat and graph candidates
  retrieval.py  top-K search over graph vectors
  config.py     YAML config
  pipeline.py   runs every stage
configs/        smoke.yaml (fake data), instacart.yaml, erp.yaml
tests/          pytest suite
docs/           architecture and roadmap
references/     papers and the research report
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA
pip install -r requirements.txt
```

Check that everything works (fake data, tiny random LLM, CPU, about a minute):

```bash
pytest
python -m hybridrec.pipeline --config configs/smoke.yaml
```

The smoke run's reranker numbers mean nothing; it only checks the code paths.

## Run on Instacart

1. Download "Instacart Market Basket Analysis" from Kaggle and unzip the CSV
   files into `data/instacart/`.
2. Run:

```bash
python -m hybridrec.pipeline --config configs/instacart.yaml
```

The config starts with 20,000 customers so a first run is quick. Remove
`max_users` for the full dataset. Each stage saves its output in `runs/instacart/`,
so you can rerun one part:

```bash
python -m hybridrec.pipeline --config configs/instacart.yaml --stages rerank evaluate
```

Results are printed and saved to `runs/instacart/metrics.json`.

## Run on ERP data

1. Edit the SQL in `hybridrec/data/erp.py` (`ErpConfig`) to match your tables.
2. `export ERP_DB_URL="mssql+pyodbc://..."`
3. To use your CustomerVAE vectors instead of the built-in VAE, export them to
   parquet (`user_id` + one column per dimension) and set
   `data.user_vectors` in `configs/erp.yaml`.
4. `python -m hybridrec.pipeline --config configs/erp.yaml`

## Hardware

Set for 2× RTX 4080 (16 GB). Training runs on `cuda:1` (`train_device` in the
config), leaving `cuda:0` free for serving. Qwen3-0.6B with LoRA, batch 8 and
20 candidates fits in 16 GB in bfloat16. For Qwen3-1.7B, turn on
`gradient_checkpointing`.

## What has and has not been checked

- The full pipeline and all tests run on fake data, with a tiny random Qwen3
  model built offline.
- Not yet run with the real Qwen3 weights or on the real Instacart files.
  The first Instacart run is the real test; see the roadmap for what to look at.

## Old code

The previous quantity-prediction model is kept on the
[`legacy-quantity-model`](../../tree/legacy-quantity-model) branch.

## License

MIT, see [LICENSE](LICENSE).
