# Roadmap

The current model predicts order quantity for one retailer-product pair at a
time. The plan is to grow it into a full recommender that works for three
cases with one codebase:

1. **B2B ordering on ERP data:** retailers × products from order history,
   covering both reorders and new-product discovery.
2. **B2C e-commerce:** end customers browsing a large catalogue.
3. **Public benchmarks:** Amazon Reviews 2023 and Instacart, so results can be
   compared with published work.

## Hardware

Two RTX 4080 GPUs (16 GB each). GPU 0 serves requests; GPU 1 trains and runs
batch jobs with the language model. No model is split across both cards.

## Target design

**Online path (p99 under 150 ms)**

1. Candidate retrieval: several sources (sequence model, co-purchase rules,
   popular items, content similarity) each return a short list.
2. Ranker: LightGBM LambdaRank scores the merged candidates.
3. Business rules: stock, price lists, banned items, diversity.
4. Optional: a small Qwen reranker (Qwen3-Reranker-0.6B + LoRA) re-scores the
   top 20. It has a 60 ms time limit; if it is late, the ranker's order is used.

**Near-line path:** new products get a content vector as soon as they are
created, so they can be retrieved within minutes.

**Batch path:** B2B order suggestions are computed ahead of time with the
full stack, including the slower parts.

## Cold start

- Every product gets a text embedding from day one (comparing
  Qwen3-Embedding-0.6B and BGE-M3).
- Item IDs are randomly dropped during training, so the model learns to use
  content when the ID is missing.
- No per-user ID embeddings; users are described by what they did.
- A few exploration slots per list, with the chance of showing each item logged,
  so results can be evaluated without bias later.

## Where the language model helps

- Offline enrichment: Qwen3.5-4B fills in missing product attributes with
  JSON-schema-constrained output.
- KAR-style knowledge features: the LLM describes user preferences and item
  facts, and the encoded text becomes extra ranker features.
- The optional reranker above.
- Short explanations for why an item is recommended, generated offline.

## Not now

Ideas from the research that are left out of the first version because their
cost is high or the gain is unproven at this scale: per-request knowledge-graph
agents, GraphLoRA (kept as a research branch), Q-Former projectors, and
semantic-ID generative retrieval.

## Evaluation

- Time-based splits (or last-basket splits for basket data), never random.
- Metrics over the full catalogue, not sampled negatives (sampled metrics can
  change which model looks best).
- Results broken down by slice: warm users, cold users, cold items, repeat
  purchases, new-item exploration. Repeat and explore are reported apart,
  because a model can look strong only by repeating past orders.
- Paired bootstrap to check that a gain is real.
- A model moves to production only after passing each check in order:
  offline metrics, slice checks, latency, shadow traffic, small A/B test,
  wider A/B test, full rollout.

## Steps

| # | Step | Status |
|---|------|--------|
| 0 | Quantity model on ERP data (this repo) | working, being cleaned up |
| 1 | Data contract, adapters for each data source, time-based splits | spec and tests written |
| 2 | Evaluation harness and simple baselines | next |
| 3 | Content vectors for items | |
| 4 | SASRec with hybrid (ID + content) items | |
| 5 | Candidate sources and LightGBM ranker | |
| 6 | Serving and orchestration | |
| 7 | Qwen enrichment | |
| 8 | Qwen reranker | |
| 9 | Explanations and rollout | |

The current quantity model can stay useful as a ranker feature or as a
candidate source for B2B reorders once step 5 is in place.
