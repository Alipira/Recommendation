# References

Papers and notes the design of this project draws on. PDFs are in
[`papers/`](papers/). They are kept here for private study only; most are
under the publisher's or arXiv's licence, so remove them before making this
repo public and keep the links below instead.

## Papers

| File | Paper | Why it matters here |
|------|-------|---------------------|
| [2023_LlamaRec.pdf](papers/2023_LlamaRec.pdf) | Yue, Rabhi, et al. *LlamaRec: Two-Stage Recommendation using Large Language Models for Ranking.* 2023. [arXiv:2311.02089](https://arxiv.org/abs/2311.02089) | A fast sequence model finds candidates, then an LLM reranks only the short list. Basis for the optional Qwen reranker on the top 20. |
| [2024_LLaRA.pdf](papers/2024_LLaRA.pdf) | Liao, Li, Yang, et al. *LLaRA: Large Language-Recommendation Assistant.* SIGIR 2024. [arXiv:2312.02445](https://arxiv.org/abs/2312.02445) | Feeds SASRec item vectors into an LLM next to the item text, with a curriculum from text-only to hybrid prompts. Idea behind the hybrid (ID + content) item encoder. |
| [2023_KAR.pdf](papers/2023_KAR.pdf) | Xi, Liu, Lin, et al. *Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models.* 2023. [arXiv:2306.10933](https://arxiv.org/abs/2306.10933) | The LLM writes user-preference and item-fact text offline; the encoded text becomes extra features for a normal ranker. Basis for the LLM enrichment step. |
| [2026_GraphLoRA.pdf](papers/2026_GraphLoRA.pdf) | Mu, Wang, Ni, et al. *GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation.* 2026. [arXiv:2606.07526](https://arxiv.org/abs/2606.07526) | Puts graph (GNN) signals inside the LoRA adapters instead of the prompt. Kept as a research branch, not in the first version. |
| [2020_Sampled_Metrics_Item_Recommendation.pdf](papers/2020_Sampled_Metrics_Item_Recommendation.pdf) | Krichene, Rendle. *On Sampled Metrics for Item Recommendation.* KDD 2020. | Metrics computed on a few sampled negatives can rank models in the wrong order. Reason the evaluation uses the full catalogue. |
| [2023_Next_Basket_Reality_Check.pdf](papers/2023_Next_Basket_Reality_Check.pdf) | Li, Jullien, Ariannezhad, de Rijke. *A Next Basket Recommendation Reality Check.* ACM TOIS 2023. [arXiv:2109.14233](https://arxiv.org/abs/2109.14233) | Many basket models win mostly by repeating past purchases. Reason results are split into repeat and explore slices, and why simple repeat baselines are needed. |

## Notes

- [research-report.md](research-report.md): a survey of ways to combine
  classic recommenders (association rules, collaborative filtering, GNNs,
  knowledge graphs, sequence models) with LLMs, ending in a proposed Qwen-based
  design. The roadmap in [../docs/roadmap.md](../docs/roadmap.md) keeps the
  parts that fit our latency and hardware limits and leaves out the rest.
