Do not use bias before Batch normalization layer, it is redundant
Use Kaiman initilaization when use ReLU activation
for scoring to findout which cart is important used layer normalization
To handle imbalance dataset used weighted Huber-loss

# FIX 2 — CRITICAL: apply mask BEFORE softmax.

Old code (WRONG):
weights = softmax(scores, dim=1)
includes padding positions
out = (weights * digest * mask).sum(dim=1)
zeros padding after the fact
Problem: softmax distributes probability over ALL T slots, including padding.
Zeroing after means the surviving weights sum to < 1 (e.g. 0.6 for a 3-of-5
cart), silently shrinking the digest vector for every padded sequence.
Fix: set padding logits to -inf so softmax assigns them exactly 0 probability.
The remaining weights sum to 1 over real items only — no post-hoc mask needed.

# FIX 3 - CRTIICAL: Weighting Strategy Has a Conceptual Mismatch
* Combine both weights — frequency AND value
