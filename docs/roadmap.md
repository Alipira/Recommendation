# Roadmap

## Done

- Data contract, Instacart and ERP loaders, last-basket and time-cutoff splits
- Customer VAE, LightGCN with VAE node features and ID dropout
- Candidate merge (repeat + graph)
- Qwen + LoRA reranker with graph-vector tokens and letter scoring
- Full-catalogue metrics, repeat/explore split, baselines, paired bootstrap
- End-to-end pipeline and tests on fake data

## Next

1. **First Instacart run (20k customers).** Look at:
   - graph vs. repeat baseline on recall of new items (the graph's job)
   - candidate coverage at 20 (the reranker's ceiling)
   - reranker NDCG@10 vs. the merged-candidate order on the hold-out set.
     If it does not beat it, the reranker is not earning its cost yet.
2. **Ablations:** graph with and without VAE features; reranker with and
   without vector tokens (text only). Use `paired_bootstrap` for each.
3. **Curriculum (LLaRA):** first train the reranker on text-only prompts,
   then switch on the vector tokens. Helps if the vector tokens make training
   unstable.
4. **Product vectors from text** (Qwen3-Embedding-0.6B) as item node features,
   so new products can be scored before anyone buys them.
5. **ERP run** with CustomerVAE vectors and the real customer and product
   tables. Shops reorder much more than Instacart users, so check if
   `n_repeat` needs to grow.
6. **Serving:** precompute graph vectors daily; at request time build
   candidates, run the reranker on GPU 0 with a time limit, and fall back to
   the merged-candidate order if it is late.

## Later

Ideas from the research report kept for later because their cost is high or
the gain is unproven at this scale: GraphLoRA (graph signal inside the LoRA
weights), Q-Former projectors, knowledge-graph reasoning per request, and
semantic-ID generative retrieval.
