# Next-Generation Recommendation Architectures: Synthesizing Traditional Recommenders with Large Language Models

> Research report that the design in this repo is based on. Kept here as background reading. Source titles are listed at the end; some are cut short.

## Introduction to the Shifting Recommender Paradigm

The architecture of recommender systems (RS) has undergone a profound transformation over the last two decades, evolving from heuristic rule-based engines to deep representation learning, and currently entering the era of large language models (LLMs). Historically, recommender systems have relied heavily on capturing the collaborative signals between users and items based on explicit and implicit feedback, predominantly utilizing unique identifiers (IDs) mapped into continuous, dense latent spaces. While these ID-based methodologies excel at capturing behavioral patterns and user-item interactions, they intrinsically lack the semantic understanding necessary to interpret the rich textual, visual, and contextual data that surrounds modern digital interactions.

The synthesis of large language models and recommender systems—often encapsulated under the umbrella term LLM4Rec—has emerged as a disruptive paradigm, fundamentally altering how personalization is approached across domains such as e-commerce, media, healthcare, and education. Large language models, driven by vast pre-training corpora and advanced Transformer architectures, bring unprecedented zero-shot and few-shot inference capabilities, deep semantic comprehension, and step-by-step logical synthesis to the recommendation pipeline. However, the integration of LLMs into recommendation architectures introduces acute, systemic challenges regarding computational latency, structural context retention, and the handling of highly sparse behavioral matrices. LLMs operate within a discrete semantic token space, whereas traditional recommenders operate within a continuous collaborative embedding space, creating an inherent and challenging modality gap that must be bridged for optimal performance.

This report provides an exhaustive investigation into the methodologies for integrating traditional recommendation paradigms—specifically 

### Association Rules

 (AR), 

### Collaborative Filtering

 (CF), 

### Graph Neural Networks

 (GNNs), 

### Knowledge Graphs

 (KGs), and Sequential Recommender models—with LLMs. By analyzing these integrations through the critical lenses of scalability, latency, data sparsity handling, and dynamic behavior capture, this analysis synthesizes the empirical findings to determine the optimal architectural approach for concatenating these foundational models with the Qwen LLM series. The resulting framework aims to establish a robust, highly scalable architecture for next-generation, semantic-aware recommendation systems.

## Foundational Recommendation Paradigms in the LLM Era

To construct a hybrid LLM-based architecture, it is logically imperative to first deconstruct the functional mechanics, mathematical foundations, and inherent limitations of the foundational recommendation models that serve as the structural backbone for these advanced systems.

Association RulesAssociation Rule (AR) mining represents one of the earliest and most transparent forms of recommendation logic. It operates by identifying frequent itemsets and inferring conditional probabilities between co-occurring events within transactional databases. Systems utilizing AR generate explicit, human-readable rules (e.g., "If User views Item A, then User is 80% likely to view Item B") based on historical support and confidence metrics. While AR provides unparalleled transparency and explainability, it is fundamentally constrained by its strict reliance on exact pattern matching. In the context of vast, sparse datasets typical of modern platforms, AR fails entirely to generalize to the long-tail distribution of unobserved item combinations. Consequently, in contemporary LLM-augmented architectures, AR is rarely deployed as a standalone predictive engine. Instead, it serves as a deterministic baseline or a feature engineering mechanism to explicitly constrain LLM generative outputs, preventing hallucinations by grounding the model's logic in verified historical correlations.

Collaborative FilteringCollaborative Filtering (CF) operates on the foundational premise that users with historically similar interaction behaviors will exhibit similar preferences in the future. Traditional CF models, particularly latent factor models and matrix factorization techniques, explicitly construct user and item representations as learnable dense embeddings based on user-item interaction bipartite graphs. CF models are highly optimized for mapping the collaborative signals of warm-start users and items, utilizing highly efficient inner-product calculations to generate recommendations. However, CF suffers significantly from the cold-start problem—the systemic inability to generate accurate recommendations for new users or items lacking sufficient interaction history. Furthermore, CF embeddings are inherently devoid of semantic context; they are mere mathematical abstractions of behavior, making them notoriously difficult to interface directly with the semantic token spaces of language models without sophisticated alignment techniques.

Graph Neural NetworksGraph Neural Networks (GNNs) extend the foundational principles of CF by explicitly modeling the higher-order structural topology of user-item interactions. By iteratively aggregating embeddings from neighboring nodes in a defined graph, GNNs capture multi-hop transitive relationships that shallow CF models inherently miss. Architectures such as LightGCN demonstrate high efficacy in smoothing representations over graphs, propagating collaborative signals deeply through the network. When coupled with LLMs in frameworks like GraphLLM or OpenGraph, GNNs serve as powerful structural encoders. They provide the language model with an aggregated, mathematical representation of the user's localized network neighborhood, theoretically allowing the LLM to "see" beyond immediate interactions to infer broader community preferences.

Knowledge GraphsWhile GNNs primarily model user-item behavioral graphs, Knowledge Graphs (KGs) represent external, factual, and semantic relationships between entities in the form of structured triples $(h, r, t)$ representing head, relation, and tail. KGs introduce domain-specific ontological structures (e.g., "Movie X - Directed By - Director Y") into the recommendation process. The integration of KGs addresses one of the most critical flaws of LLMs: the intrinsic hallucination problem. Frameworks leveraging KGs execute multi-hop path traversals, providing the LLM with a highly interpretable and explicit trail of logic. By forcing the LLM to ground its generative process in verified factual pathways derived from the KG, the system ensures that the resulting recommendations are not only personalized but factually coherent and highly explainable.

### Sequential Recommender Models

Sequential models capture the temporal dynamics of user preferences by treating historical interactions as ordered chains, rather than static sets. Architectures like SASRec (Self-Attentive Sequential Recommendation), GRU4Rec, and Caser utilize unidirectional Transformer blocks or recurrent architectures to assign contextual weights to historical items. This allows the system to effectively model both long-term preferences and short-term shifts in user intent. In the context of LLM integration, sequential models are critical for tracking evolving behavioral states. They provide the necessary temporal trajectory that static CF or AR models lack, acting as a highly compressed, dynamic contextual window that informs the LLM of the user's immediate cognitive trajectory.

## Methodologies for LLM Integration and Modality Alignment

Integrating the aforementioned deterministic, structural, and temporal models with an autoregressive LLM requires crossing a fundamental semantic divide between continuous dense vectors and discrete linguistic tokens. Current research establishes three primary methodologies for this concatenation: Input-Space Alignment, Modality Bridging via Projectors, and 

### Parameter-Space Alignment

.

### Input-Space Alignment (Prompting Strategies)

Input-space alignment treats the LLM largely as a frozen black box, encoding structural or collaborative data entirely into natural language prompts. Foundational frameworks like P5 and TALLRec convert sequential interaction histories and collaborative signals into linear textual instructions (e.g., "User bought Item A and Item B. Will they buy Item C? Answer Yes or No"). The system then applies Low-Rank Adaptation (LoRA) to instruction-tune the LLM for binary classification or direct item generation.

While computationally straightforward to implement, this approach suffers from significant information degradation. Flattening complex topological structures—such as GNN neighborhoods or multi-hop KG subgraphs—into linear text strings effectively destroys the multi-dimensional relational context. Furthermore, processing long textual representations of graphs significantly increases the token sequence length. This exacerbates the LLM's inference latency, dramatically increases API or compute costs, and increases the model's susceptibility to the "lost in the middle" phenomenon, where the LLM forgets context presented in the center of a long prompt.

### Modality Bridging via Projectors (MLPs and Q-Formers)

To circumvent the limitations of text-only prompting, intermediate mapping networks are deployed to project collaborative, sequential, or structural embeddings directly into the LLM's input embedding space, aligning the dimensions. This process constructs a continuous "soft prompt" that seamlessly bypasses the discrete tokenizer, injecting behavioral mathematics directly alongside linguistic tokens.

The integration typically relies on Multi-Layer Perceptrons (MLPs). A standard MLP acts as a linear or non-linear adapter to align the dimensionality of the external encoder (e.g., SASRec) with the hidden dimension of the LLM, denoted as $d_{LLM}$. The mathematical equation for this alignment is formalized as $E_X = \text{ReLU}(X W_1 + b_1)W_2 + b_2$, where $X$ represents the structural embedding from the traditional model.

For highly complex external data, such as variable-length sequences or large subgraphs, Querying Transformers (Q-Formers) are deployed. A Q-Former utilizes a set of learnable query vectors to extract the most salient features from the external modality via cross-attention. This compresses the representation before it is passed through the MLP projector. This acts as a critical information bottleneck, ensuring the LLM is not overwhelmed by noisy, irrelevant, or highly sparse structural data. Models like ILM demonstrate that training the Q-Former through item-item and item-text contrastive learning significantly closes the modality gap.

Parameter-Space AlignmentThe most advanced integration paradigm moves beyond input manipulation entirely and alters the LLM's internal parameter update mechanisms. Parameter-space alignment explicitly injects collaborative or structural signals directly into the model's adaptation matrices. Methods like CoRA and GraphLoRA embed trainable message-passing mechanisms or collaborative features directly within the LoRA bottleneck.

In a standard LLM attention layer, the weight update is parameterized via low-rank matrices as $\Delta W = B A$. Under the GraphLoRA paradigm, the update for a given layer incorporates the structural node embedding $z_n$ derived from the external GNN. The forward pass is modified such that the structural signal explicitly guides the gradient updates, ensuring the collaborative topology dynamically shapes the LLM's internal semantic representations without inflating the input context window. This achieves a profound, synergistic coupling of mathematical structure and linguistic semantics with minimal computational overhead during inference.

## Architectures Enhancing Semantic Inference and Reasoning

To elevate recommendation systems from mere pattern matchers to cognitive systems capable of true semantic inference, architectures must synergize the broad world knowledge inherent in LLMs with the precise, structured data of KGs and CF matrices.

### Collaborative-LLM Synergy for Deep Semantic Matching

The CoLLM architecture exemplifies the enhancement of semantic understanding by explicitly mapping CF embeddings into the LLM's latent space. By allowing the LLM to access the pre-trained collaborative embeddings of a traditional matrix factorization model, the system bridges the gap between raw collaborative patterns and natural language reasoning. The traditional CF model isolates complex co-occurrence signals, while the LLM interprets the item descriptions and user profiles.

This hybrid approach demonstrates remarkable robustness across data distributions. It allows the LLM to utilize its zero-shot reasoning capabilities to maintain high accuracy in cold-start scenarios (where CF models traditionally fail), while leveraging the CF embeddings to achieve state-of-the-art metrics, such as NDCG@10, on warm-start users. Empirical benchmarks reveal that CoLLM significantly outperforms standalone CF and standalone text-prompted LLMs by capitalizing on this dual-knowledge representation.

### Knowledge-Augmented Reasoning Frameworks

To achieve interpretable reasoning, frameworks such as Reasoning on Graphs (RoG) and Knowledge Augmentation for Recommendation (KAR) implement sophisticated planning-retrieval-reasoning pipelines. In RoG, the LLM first generates a relation path plan based on the user's implicit or explicit query. This plan is used to traverse the external Knowledge Graph, retrieving grounded, factual paths (e.g., $e_0 \xrightarrow{r_1} e_1 \xrightarrow{r_2} e_2$). The retrieved reasoning paths, denoted as $Z$, are then fed back into the LLM as contextual constraints.

The optimization objective for RoG maximizes the probability of generating the correct recommendation $a$ given the query $q$ and the graph $G$, formulated mathematically as $\mathcal{L} = \log P_\theta(a\vert{}q, Z^*, G)$. This objective function forces the LLM to ground its generative output strictly in the topological reality of the KG, effectively neutralizing the hallucination problem. Models like XoG (Exploration on Graphs) and FoG (Foresight on Graphs) take this paradigm further by incorporating iterative recovery mechanisms and memory subgraphs to handle incomplete KGs. These architectures allow the LLM to act as a dynamic agent that directs graph traversal based on evolving contextual certainty, preventing the irreversible loss of answer-critical triples during beam pruning.

### Reinforcement Learning from AI Feedback (RecLAIF)

Enhancing semantic reasoning also requires rigorous evaluation of the LLM's outputs. The RecLAIF framework introduces a dual-role architecture leveraging Reinforcement Learning from AI Feedback. It uses one LLM instance as a recommender to generate candidate responses based on user interaction histories and item attributes, and a second LLM instance as a judge to evaluate the quality of those recommendations against metrics like relevance, diversity, and explainability. By generating preference pairs (chosen versus rejected outputs) and fine-tuning the recommender via Direct Preference Optimization (DPO), the system aligns the LLM's semantic reasoning explicitly with high-quality recommendation heuristics.

## Capturing Dynamic User Behavior

Static representations fail to account for the temporal drift in user interests. Concatenating Sequential Recommender models with LLMs addresses this limitation by injecting explicit temporal awareness directly into the language model's cognitive process.

### Sequential Backbones and the LLaRA Framework

The LLaRA (Large Language-Recommendation Assistant) architecture exemplifies the state-of-the-art in integrating sequential models with LLMs. LLaRA utilizes SASRec to encode the user's chronological interaction sequence into a dense behavioral token. This sequence token is transformed via a continuous hybrid projection—typically an MLP adapter—and injected into the LLM's input sequence alongside standard text tokens.LLaRA employs a highly sophisticated curriculum learning strategy to prevent capability degradation. Initially, the LLM is fine-tuned solely on text-based prompt representations of the user's history, forcing it to rely on its inherent linguistic reasoning. Gradually, the learning focus shifts to the hybrid prompts containing the SASRec continuous embeddings. This phased approach prevents the LLM from experiencing catastrophic forgetting of its broad world knowledge while it learns to interpret the dense sequential modality. The outcome is a robust system that intuitively understands the sequential cadence of user behavior while retaining the semantic capacity to explain precisely why a shift in preference occurred.

### Dynamic Routing and Contextual Adaptation

Real-time dynamic behavior capture requires adaptive retrieval mechanisms that evolve with the user. The AdaptRec framework introduces a self-adaptive user-contextualized sequential recommendation system. Instead of passively receiving sequential embeddings, the LLM in AdaptRec acts as an active agent in the loop. It evaluates the similarity between the target user's sequence and collaborative historical sequences, actively refining the pool of similar users to extract highly relevant demonstrations for in-context learning. Empirical results demonstrate that AdaptRec achieves significant gains, outperforming baseline LLM architectures by +7.13% on MovieLens and +18.16% on LastFM in HR@1 metrics, highlighting the power of LLM-driven collaborative data selection.

Similarly, the LlamaRec architecture implements a two-stage routing mechanism to handle dynamic intent at scale. Recognizing that invoking a massive LLM for every item in a global catalog is computationally prohibitive, LlamaRec utilizes the lightweight SASRec model as a first-stage retriever to generate a small, highly relevant candidate set. The LLM then acts as a sophisticated reranker, applying deep semantic analysis to the sequential trajectory and the candidate items to yield the final recommendation, effectively blending speed with semantic depth.

## Comparative Evaluation of LLM Integration Paradigms

A rigorous comparative analysis of AR, CF, GNN, KG, and Sequential models in the context of LLM integration reveals distinct architectural trade-offs across four critical dimensions: scalability, latency, data sparsity, and structural context retention.

| Feature | Association Rules | Collaborative Filtering | GNN | Knowledge Graph | Sequential (SASRec) |
|---|---|---|---|---|---|
| Primary output | Explicit deterministic rules | Dense latent user/item vectors | Topology-aware embeddings | Factual relational paths | Time-aware behavioral tokens |
| LLM integration | Input text constraints | Soft prompt projection / LoRA | Parameter space (GraphLoRA) | Agent-based path traversal | Hybrid prompts (LLaRA) |
| Cold-start resilience | Very low | Low | Moderate | High (ontological links) | Low (needs history) |
| Structural context | None | Low | Very high | Very high | Moderate (temporal only) |
| Inference overhead | Low | Low (if decoupled) | High | Very high (iterative LLM calls) | Moderate (two-stage routing) |

### Scalability and Real-Time Inference Latency

Scalability remains the primary bottleneck for deploying LLM-based recommender systems in production environments.

Association Rules & CF: These represent the most inherently scalable traditional methods. CF operations rely on highly optimized inner-product vector operations, enabling sub-millisecond latency for candidate generation. However, when integrated via input-space textual prompting (e.g., early iterations of TALLRec), the latency scales linearly with the sequence length due to the autoregressive nature of LLMs, making real-time deployment of text-only CF unfeasible.GNN & KG: Graph traversals introduce significant latency. If a multi-hop KG path is flattened into text, the LLM prompt becomes prohibitively long. Agent-based KG models (where the LLM iteratively queries the graph, such as in ToG or XoG) suffer from high API call overhead and compounding latency, limiting their use in strictly real-time environments.

Sequential (Two-Stage): The two-stage paradigm (e.g., SASRec retrieval followed by LLM reranking) offers the optimal balance for production scalability. By restricting the LLM's autoregressive inference strictly to a top-K reranking phase, the system maintains the low latency of traditional models for the heavy lifting while benefiting from the LLM's semantic depth for final selection.

### Handling of Data Sparsity and Cold-Start Scenarios

CF & AR: Traditional CF and AR are notoriously brittle in cold-start environments. Without historical co-occurrence data, the latent representations collapse, and no rules can be mined.LLMs + CF (CoLLM): The integration of LLMs completely reshapes cold-start performance. LLMs leverage their extensive pre-trained world knowledge. If a new item is added with a rich textual description but zero interaction history, the LLM can infer its utility immediately via zero-shot reasoning, entirely circumventing the CF matrix's sparsity.KGs: KGs fundamentally mitigate sparsity by connecting cold items to known entities via explicit ontological links (e.g., linking a new, unrated shoe to a known brand node), providing the LLM with an immediate structural anchor to ground its reasoning.

### Structural Context Retention

Sequential Models: Retain strict temporal ordering but fundamentally fail to capture non-linear, multi-dimensional collaborative effects across user communities.

Input-Space GNN/KG: Flattening a graph into a one-dimensional token sequence fundamentally destroys the spatial and multi-hop relationships. The LLM's self-attention mechanism treats the data as a linear sequence, losing the explicit edge connections of the graph, leading to degraded structural awareness.

Parameter-Space GNN (GraphLoRA): This represents the superior method for structural context retention. By passing the GNN message-passing output directly into the LLM's LoRA adapters, the high-order structural topology is mathematically preserved and fused with the semantic weights, preventing any structural degradation during inference.

## Synthesizing the Optimal Architectural Approach for Qwen

The ultimate objective of this synthesis is to architect the optimal framework concatenating these traditional paradigms with the Qwen Large Language Model. The Qwen series—particularly the recent Qwen2.5 iterations—presents distinct architectural advantages uniquely suited for this synthesis. Built upon an advanced, modified Transformer architecture, Qwen2.5 features Grouped Query Attention (GQA) for efficient inference, SwiGLU activation functions, and a massive byte-pair encoding (BPE) vocabulary of 151,936 tokens (with 151,643 usable tokens), which allows it to encode diverse linguistic and code-based data highly efficiently. Its variations span from highly efficient 0.5B and 1.5B parameter models up to massive 72B parameter dense models, alongside 8-Omni multimodal and Sparse Mixture of Experts (MoE) configurations, offering unprecedented flexibility for tiered deployment in recommendation ecosystems.

Based on the rigorous comparative synthesis of scalability, modality alignment, and context retention, the optimal architecture is the Qwen-Graph Sequential Adapter (QGSA) Framework. This is a three-tiered architecture deploying a frozen Qwen2.5 backbone enhanced by parameter-space structural injection, sequential soft-prompting, and agentic graph verification.

### 1. Stage One: The Retrieval and Modality Alignment Tier

Given Qwen's autoregressive latency (even with GQA optimizations), evaluating an entire item catalog natively is computationally invalid for real-time systems. Therefore, the first tier utilizes traditional, highly optimized models acting in parallel.A SASRec module generates a dense sequential trajectory vector based on the user's real-time, chronological interaction history.

Simultaneously, a LightGCN (Graph CF) module generates a structural collaborative vector, capturing deep community interaction patterns.Q-Former and MLP Projection: These dual vectors are passed into a trainable Q-Former, which compresses the signals into a fixed number of latent query tokens. An MLP then performs the exact dimensional alignment, transforming these query tokens to perfectly match Qwen2.5's specific hidden dimension ($d_{LLM}$). This generates a continuous "soft prompt" that is prepended to the user's textual input, bypassing the 151,643-token discrete vocabulary bottleneck entirely.

### 2. Stage Two: Parameter-Space Structural Injection

To preserve the complex multi-hop topology of the user-item graph without expanding Qwen's token context window (which would dilute semantic focus and increase compute costs), the architecture employs the GraphLoRA paradigm.

Trainable Low-Rank Adaptation (LoRA) matrices are attached to Qwen2.5's query, key, value, and output projection layers ($W_q, W_k, W_v, W_o$).

Instead of standard textual LoRA fine-tuning, the $z_n$ structural embeddings derived from an external GNN are fed directly into the intermediate bottleneck of the LoRA modules. As Qwen processes the text and soft prompts, the GraphLoRA matrices dynamically adjust the attention weights based on the user's explicit topological position in the graph. The massive Qwen backbone remains strictly frozen, preserving its pre-trained semantic integrity, zero-shot reasoning, and broad world knowledge.

### 3. Stage Three: Knowledge-Grounded Semantic Reasoning

With the sequential data loaded via soft prompts and the structural data injected directly via GraphLoRA parameters, Qwen2.5 is deployed strictly as a top-K reranker and reasoning engine.

To strictly mitigate hallucinations and enforce logical continuity, the architecture integrates the Reasoning on Graphs (RoG) and Foresight on Graphs (FoG) methodologies.

When Qwen requires explicit attribute verification (e.g., verifying if a candidate movie matches a user's highly specific genre preference extracted from reviews), it queries an external Knowledge Graph. The retrieved factual triples $(h, r, t)$ are injected into the standard textual prompt as explicit constraints.

Qwen then synthesizes the sequential trajectory (from the soft prompt), the structural community preferences (from GraphLoRA), and the explicit item facts (from the KG) to generate the final personalized recommendation along with a natural language justification.

### Strategic Justification of the QGSA Architecture

This proposed concatenation maximizes the profound strengths of all constituent models while explicitly neutralizing their systemic weaknesses. It solves the LLM latency bottleneck by relegating brute-force item search to the SASRec/GCN tier, utilizing Qwen solely for nuanced reranking. It decisively bridges the modality gap and solves the context-flattening problem by bypassing textual graphs entirely, instead using Q-Former MLPs and GraphLoRA parameter-space injections to retain pure topological mathematics. Finally, by utilizing Qwen2.5's massive parameter base, SwiGLU-activated logic pathways, and 151k+ vocabulary size in conjunction with KG grounding, the system gracefully handles complete cold-start data sparsity. If collaborative matrices fail due to a lack of data, the system seamlessly defaults to Qwen's pre-trained semantic reasoning, extracting intent from raw text. This unified, multi-tiered architectural approach represents the optimal, highly scalable future of large language model-driven recommendation systems.

## Sources

- LLM4Rec: A Comprehensive Survey on the Integration of Large (mdpi.com)
- CoLLM: Integrating Collaborative Embeddings into Large Language (arxiv.org)
- LLM4Rec: A Comprehensive Survey on the Integration of Large (researchgate.net)
- A Survey on Large Language Models for Recommendation - arXiv (arxiv.org)
- Review-driven Personalized Preference Reasoning with Large (arxiv.org)
- Large Language Models for Recommender Systems: A Problem (sciopen.com)
- Enhancing LLM-based Recommendation with Preference Hint (arxiv.org)
- Item-Language Model for Conversational Recommendation - arXiv (arxiv.org)
- Recent Developments in Recommender Systems: A Survey - arXiv (arxiv.org)
- A comprehensive survey of modern recommendation systems (techrxiv.org)
- A Comprehensive Analysis Based on Data, Algorithms, and Evaluation (arxiv.org)
- LLM Collaborative Filtering: User-Item Graph as New Language (www4.comp.polyu.edu.hk)
- Enhancing LLM-based Recommendation through Semantic-Aligned (arxiv.org)
- (PDF) A review of large language models and the recommendation (researchgate.net)
- Multimodal graph neural network with large language models for (frontiersin.org)
- A Survey of Graph Meets Large Language Model - IJCAI (ijcai.org)
- Boosting Graph Reasoning Ability of Large Language Model (openreview.net)
- Trustworthy GNNs with LLMs: A Systematic Review and Taxonomy (arxiv.org)
- Explore-on-Graph: Hybrid Embedding–LLM Reasoning for ... - arXiv (arxiv.org)
- (PDF) Knowledge graph-based recommendation system enhanced (researchgate.net)
- Faithful and Interpretable Large Language Model Reasoning - arXiv (arxiv.org)
- arXiv:2310.01061v2 [cs.CL] 24 Feb 2024 (arxiv.org)
- Foresight-over-Graph: Reasoning Beyond Local Horizons for ... - arXiv (arxiv.org)
- KG-TRACES: Enhancing Large Language Models with Knowledge (arxiv.org)
- LSF-SR: Latent Semantic Fusion for Sequential Recommendation (arxiv.org)
- HORIZON: A Benchmark for in-the-wild User Behavior Modelling (arxiv.org)
- Prompting Large Language Models for Recommender Systems - arXiv (arxiv.org)
- SBT-Rec: A Structured Behavioral Tokenization Framework for LLM (mdpi.com)
- Bringing Collaborative Filtering to LLMs with AdaptRec - Shaped.ai (shaped.ai)
- arXiv:2501.13344v1 [cs.IR] 23 Jan 2025 (arxiv.org)
- GraphLoRA: Structure-Aware Low-Rank Adaptation for Large ... - arXiv (arxiv.org)
- Review-driven Personalized Preference Reasoning with Large (arxiv.org)
- A Survey of Item Identifiers in Generative Recommendation - TechRxiv (techrxiv.org)
- Parameter-Efficient Fine-Tuning of Large Language Models for Unit (arxiv.org)
- LLM Monitoring at Speculative-Decoding Cost - arXiv (arxiv.org)
- Making LLMs Better Many-to-Many Speech-to-Text Translators with (arxiv.org)
- arXiv:2411.15453v1 [cs.CV] 23 Nov 2024 (arxiv.org)
- MLaGA: Multimodal Large Language and Graph Assistant - arXiv (arxiv.org)
- Hierarchical Frame Q-Former for Effective Speech Modality Adaptation (arxiv.org)
- GraphLoRA: Structure-Aware Low-Rank Adaptation for Large ... - arXiv (arxiv.org)
- arXiv:2408.10645v3 [cs.IR] 25 Oct 2024 (arxiv.org)
- Text-like Encoding of Collaborative Information in Large Language (alphaxiv.org)
- Towards Open-World Recommendation with Knowledge ... - alphaXiv (alphaxiv.org)
- RecLAIF: Reinforcement Learning from AI Feedback for (oars-workshop.github.io)
- C2KD: Cross-layer and Cross-head Knowledge Distillation for Small (aclanthology.org)
- ULMRec: User-centric Large Language Model for Sequential ... - arXiv (arxiv.org)
- Predicting and Explaining Tomorrow's Hits with a Fine-Tuned LLM (arxiv.org)
- Faithful Reasoning on Knowledge Graphs with Large Language (arxiv.org)
- [2412.15115] Qwen2.5 Technical Report - arXiv (arxiv.org)
- Qwen Technical Report - arXiv (arxiv.org)
- Upload folder using huggingface_hub · tjake/Qwen2.5-0.5B-Instruct (huggingface.co)
- Upload folder using huggingface_hub · CalamitousFelicitousness (huggingface.co)
- Qwen3.8-Omni: Towards Native Omni-Modal Agents - arXiv (arxiv.org)
- A Systematic Benchmark of Fine-Tuning Strategies - arXiv (arxiv.org)
- Lightweight Fairness for LLM-Based Recommendations via ... - arXiv (arxiv.org)
- Repurposing Pre-trained LLMs as High Fidelity Continuous Text (arxiv.org)
- CodeForge-MA: Execution-Verified Multi-Agent Learning with ... - arXiv (arxiv.org)
- KazByte: Adapting Qwen models to Kazakh via Byte-level ... - arXiv (arxiv.org)
- Make Large Language Model a Better Ranker - arXiv (arxiv.org)
- AutoData: A Multi-Agent System for Open Web Data Collection (neurips.cc)
- Evaluating the Capability of Prompted LLMs to Recommend NFR (sol.sbc.org.br)
- LLM-Enhanced Recommendations via Graphs | PDF - Scribd (scribd.com)
- Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets (drops.dagstuhl.de)
- CHIANGEL/Awesome-LLM-for-RecSys: Survey - GitHub (github.com)
- Retrieval-Augmented Recommender System for Data (d-nb.info)
- Large Language Models Are Universal Recommendation Learners (arxiv.org)
- Large Language Models, Architectures, Protocols, and Applications (arxiv.org)
- On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency (arxiv.org)
- LLM-Based Generative Recommendation via Structured Term (arxiv.org)
- Decision-Oriented Recommendation Reranking: An Empirical Study (arxiv.org)
- Qwen3 Embedding: Advancing Text Embedding and Reranking (arxiv.org)
- Qwen3 Technical Report - arXiv (arxiv.org)
- Injecting Ordinal Semantics into LLM-Based Recommenders - arXiv (arxiv.org)
- RRCM: Ranking-Driven Retrieval over Collaborative and Meta (arxiv.org)
- Large Language Models Enhanced Collaborative Filtering | alphaXiv (alphaxiv.org)
- Local Large Language Models for Recommendation (joonseok.net)
- Towards Open-World Recommendation with Knowledge ... - arXiv (arxiv.org)
- RLRF4Rec: Reinforcement Learning from Recsys Feedback ... - arXiv (arxiv.org)
- RLRF4Rec: Reinforcement Learning from Recsys ... - OpenReview (openreview.net)
- Towards Open-World Recommendation with ... - OpenReview (openreview.net)
- Reasoning over Knowledge Graph Paths to Improve Large ... - arXiv (arxiv.org)
- Research Trends for the Interplay between Large Language Models (arxiv.org)
- Literature Review: from RAG to LLM Reasoning - GitHub (github.com)
- G-Refer: Graph Retrieval-Augmented Large Language Model for (arxiv.org)
- LLM AS GNN: GRAPH VOCABULARY LEARNING ... - OpenReview (openreview.net)
- Large Language Models Meet Graph Neural Networks - arXiv (arxiv.org)
- Text Is All You Need: Learning Language Representations for (alphaxiv.org)
- 111 Towards Next-Generation LLM-based ... - OpenReview (openreview.net)
- UCGRec: User-Centric Graph Learning for LLM-based Sequential (aclanthology.org)
- Bidirectional Knowledge Distillation for Enhancing Sequential (alphaxiv.org)
- Qwen3.5: Towards Native Multimodal Agents (qwen.ai)
- Membership Inference Attack Against Large Language Model (researchgate.net)
- (PDF) Laser: Parameter-Efficient LLM Bi-Tuning for Sequential (researchgate.net)
- Efficient and Transferable Text Embedding via Dual-LLM Soft ... - arXiv (arxiv.org)
- arXiv:2502.12134v2 [cs.CL] 27 May 2025 (arxiv.org)
- Neurosymbolic LoRA: Why and When to Tune Weights vs. Rewrite (arxiv.org)
- Rethinking Parameter Sharing for LLM Fine-Tuning with Multiple (aclanthology.org)
- cot-faithfulness-feasibility/PRELIM-PLAN.md at main - GitHub (github.com)
- Add Qwen 2.5 1.5B instruct with compressed-tensors configuration (huggingface.co)
- arXiv:2502.19672v1 [cs.CV] 27 Feb 2025 (arxiv.org)
- Vision-aligned Latent Reasoning for Multi-modal Large ... - arXiv (arxiv.org)
- MiniGPT-3D: Efficiently Aligning 3D Point Clouds with Large ... - arXiv (arxiv.org)
- Visual Representation Alignment for Multimodal Large Language (arxiv.org)
- A Survey on MLLMs Aligned with Multi-modalities - arXiv (arxiv.org)
- Efficient Prompting Methods for Large Language Models: A Survey (arxiv.org)
- System Prompt Optimization with Meta-Learning - arXiv (arxiv.org)
