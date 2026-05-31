"""
recsys_model.py
───────────────
Full PyTorch recommendation system model.

Architecture
============
Input tensors
  i1  (B, 17)        int64   — categorical indices for 17 normal features
  i2  (B, 91)        float32 — scalar features
  i3  (B, T, 5)      int64   — shopping-cart embedding indices (variable T)
  i4  (B, T, 5)      float32 — shopping-cart scalar features
  i5  (B, T, 1)      float32 — padding mask  (1 = valid item, 0 = padding)

Forward flow
  1. Embed all categorical features.  Shopping-cart product fields (Group*,
     FlavorID) share the same embedding tables as the normal path.

  2. SCC path  (per cart-item, shape B×T×D):
     - Assemble per-field tensors: normal features are broadcast over T,
       cart features are taken directly from i3/i4.
     - InteractionNet (3-D mode) → scc_digest_tensor (B, T, scc_digest_dim)
     - SCCAttentionNet scores each item → softmax attention weights
     - Masked weighted sum → scc_digest_vector (B, scc_digest_dim)

  3. Main path  (per-transaction, shape B×D):
     - Assemble per-field tensors from i1/i2 + scc_digest_vector.
     - InteractionNet (2-D mode) → neck_vector
     - Optional residual skip-connections from raw field embeddings

  4. HeadNet  (3-layer MLP) → output score (B, 1)
"""


from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


from src.Interaction_net import InteractionNet


# ─────────────────────────────────────────────────────────────────────────────
# Feature schemas  (single source of truth — used everywhere in this file)
# ─────────────────────────────────────────────────────────────────────────────

# 17 categorical features for the normal (per-transaction) path
# input_dim = max_category_index + 1.
# Data is 1-indexed (categories run 1 → N), so input_dim must be N+1
# to keep index N inside the valid range [0, input_dim).
# Index 0 is reserved as the padding/unknown slot (padding_idx=0).
EMBEDDING_FEATURES: list[dict] = [
    {'name': 'Group4Code',          'input_dim': 247,    'embed_dim': 6},
    {'name': 'FlavorID',            'input_dim': 29,     'embed_dim': 4},
    {'name': 'Group3Code',          'input_dim': 60,     'embed_dim': 5},
    {'name': 'Group2Code',          'input_dim': 35,     'embed_dim': 4},
    {'name': 'Group1Code',          'input_dim': 16,     'embed_dim': 3},
    {'name': 'PersonID',            'input_dim': 56_428, 'embed_dim': 16},
    {'name': 'CustomerSubGroupID',  'input_dim': 42,     'embed_dim': 5},
    {'name': 'CustomerMainGroupID', 'input_dim': 6,      'embed_dim': 2},
    {'name': 'OwnerTypeID',         'input_dim': 4,      'embed_dim': 2},
    {'name': 'CustomerTypeID',      'input_dim': 4,      'embed_dim': 2},
    {'name': 'DefaultPaymentTypeID','input_dim': 7,      'embed_dim': 3},
    {'name': 'PathCode',            'input_dim': 1_336,  'embed_dim': 10},
    {'name': 'ZoneCode',            'input_dim': 301,    'embed_dim': 7},
    {'name': 'CityCode',            'input_dim': 153,    'embed_dim': 6},
    {'name': 'StateCode',           'input_dim': 19,     'embed_dim': 3},
    {'name': 'BranchCode',          'input_dim': 8,      'embed_dim': 3},
    {'name': 'VisitorEmployeeID',   'input_dim': 364,    'embed_dim': 8},
]

# 5 product-group embedding features shared with the shopping-cart path
# (same nn.Embedding tables — indices 0-4 of EMBEDDING_FEATURES)
SCC_EMBEDDING_FEATURES: list[dict] = EMBEDDING_FEATURES[:5]

# 91 scalar features for the normal path (order = column order in i2)
SCALAR_FEATURES: list[str] = [
    'NetWeight', '_UnitSalePrice', '_UnitEndCustomerPrice',
    '_CustomerProfitMargin', 'CustomerAge', 'DefaultChequeDuration',
    'RunningSumCurrentlyRejectedChequeAmount', 'RunningSumTotalChequeAmount',
    'RunningSumRejectedChequeAmount', 'RejectedChequeAmountRatio',
    'CurrentlyRejectedChequeAmountRatio', 'RejectedChequeCountRatio',
    'CurrentlyRejectedChequeCountRatio',
    'RunningSumPaymentCountRatio_1', 'RunningSumPaymentCountRatio_2',
    'RunningSumPaymentCountRatio_3', 'RunningSumPaymentCountRatio_4',
    'RunningSumPaymentAmountRatio_1', 'RunningSumPaymentAmountRatio_2',
    'RunningSumPaymentAmountRatio_3', 'RunningSumPaymentAmountRatio_4',
    'Person_1M_Qty', 'Person_3M_Qty', 'Person_6M_Qty',
    'Person_12M_Qty', 'Person_24M_Qty',
    'Person_1M_NetWeight', 'Person_3M_NetWeight', 'Person_6M_NetWeight',
    'Person_12M_NetWeight', 'Person_24M_NetWeight',
    'Path_1M_Qty', 'Path_3M_Qty', 'Path_6M_Qty',
    'Path_12M_Qty', 'Path_24M_Qty',
    'Path_1M_NetWeight', 'Path_3M_NetWeight', 'Path_6M_NetWeight',
    'Path_12M_NetWeight', 'Path_24M_NetWeight',
    'Zone_1M_Qty', 'Zone_3M_Qty', 'Zone_6M_Qty',
    'Zone_12M_Qty', 'Zone_24M_Qty',
    'Zone_1M_NetWeight', 'Zone_3M_NetWeight', 'Zone_6M_NetWeight',
    'Zone_12M_NetWeight', 'Zone_24M_NetWeight',
    'PersianYearFloat', 'DateSin', 'DateCos',
    *[f'H{i}' for i in range(1, 38)],   # H1 … H37
]

# FieldSpec tuple: (is_normal_path, is_embedding, raw_feature_index)
# is_normal_path=True  → feature comes from i1 / i2
# is_normal_path=False → feature comes from i3 / i4  (shopping-cart)
FieldSpec = tuple[bool, bool, int]

# ── Main interaction fields (14 groups) ──────────────────────────────────────
FIELDS: list[tuple[str, list[FieldSpec]]] = [
    ("ProductGroup",  [(True, True, 0), (True, True, 1), (True, True, 2),
                       (True, True, 3), (True, True, 4)]),
    ("Weight",        [(True, False, 0)]),
    ("Price",         [(True, False, 1), (True, False, 2), (True, False, 3)]),
    ("Customer",      [(True, True, 5), (True, False, 4), (True, True, 6),
                       (True, True, 7), (True, True, 8), (True, True, 9)]),
    ("PaymentMethod", [(True, True, 10), (True, False, 5)]),
    ("Geographical",  [(True, True, 11), (True, True, 12), (True, True, 13),
                       (True, True, 14), (True, True, 15)]),
    ("Visitor",       [(True, True, 16)]),
    ("Cheque",        [(True, False, 6), (True, False, 7), (True, False, 8),
                       (True, False, 9), (True, False, 10), (True, False, 11),
                       (True, False, 12)]),
    ("Payment",       [(True, False, 13), (True, False, 14), (True, False, 15),
                       (True, False, 16), (True, False, 17), (True, False, 18),
                       (True, False, 19), (True, False, 20)]),
    ("Customer_RFM",  [(True, False, 21), (True, False, 22), (True, False, 23),
                       (True, False, 24), (True, False, 25), (True, False, 26),
                       (True, False, 27), (True, False, 28), (True, False, 29),
                       (True, False, 30)]),
    ("Path_RFM",      [(True, False, 31), (True, False, 32), (True, False, 33),
                       (True, False, 34), (True, False, 35), (True, False, 36),
                       (True, False, 37), (True, False, 38), (True, False, 39),
                       (True, False, 40)]),
    ("Zone_RFM",      [(True, False, 41), (True, False, 42), (True, False, 43),
                       (True, False, 44), (True, False, 45), (True, False, 46),
                       (True, False, 47), (True, False, 48), (True, False, 49),
                       (True, False, 50)]),
    ("Date",          [(True, False, 51), (True, False, 52), (True, False, 53)]),
    ("Holidays",      [(True, False, i) for i in range(54, 91)]),
]

# ── Shopping-cart interaction fields (14 groups) ─────────────────────────────
# These include both normal-path features (broadcast over T) and
# shopping-cart-path features (per item).
SCC_FIELDS: list[tuple[str, list[FieldSpec]]] = [
    ("ProductGroup",          [(True, True, 0), (True, True, 1), (True, True, 2),
                               (True, True, 3), (True, True, 4)]),
    ("Weight",                [(True, False, 0)]),
    ("Price",                 [(True, False, 1), (True, False, 2), (True, False, 3)]),
    ("Customer",              [(True, True, 5), (True, False, 4), (True, True, 6),
                               (True, True, 7), (True, True, 8), (True, True, 9)]),
    ("PaymentMethod",         [(True, True, 10), (True, False, 5)]),
    ("Geographical",          [(True, True, 11), (True, True, 12), (True, True, 13),
                               (True, True, 14), (True, True, 15)]),
    ("Visitor",               [(True, True, 16)]),
    ("Cheque",                [(True, False, 6), (True, False, 7), (True, False, 8),
                               (True, False, 9), (True, False, 10), (True, False, 11),
                               (True, False, 12)]),
    ("Payment",               [(True, False, 13), (True, False, 14), (True, False, 15),
                               (True, False, 16), (True, False, 17), (True, False, 18),
                               (True, False, 19), (True, False, 20)]),
    ("Customer_RFM",          [(True, False, 21), (True, False, 22), (True, False, 23),
                               (True, False, 24), (True, False, 25), (True, False, 26),
                               (True, False, 27), (True, False, 28), (True, False, 29),
                               (True, False, 30)]),
    ("Date",                  [(True, False, 51), (True, False, 52), (True, False, 53)]),
    # ── Shopping-cart-specific fields below ──────────────────────────────────
    ("Cart_ProductGroup",     [(False, True, 0), (False, True, 1), (False, True, 2),
                               (False, True, 3), (False, True, 4)]),
    ("Cart_Weight",           [(False, False, 0)]),
    ("Cart_Price",            [(False, False, 1), (False, False, 2), (False, False, 3)]),
    ("Cart_Qty",              [(False, False, 4)]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_field_dim(
    field_spec: list[FieldSpec],
    emb_features: list[dict],
    scc_emb_features: list[dict],
) -> int:
    """
    Sum up the feature dimension contributed by each slot in a field spec.

    Rules (mirror the original inline lambda):
      (True,  True,  idx) → EMBEDDING_FEATURES[idx]['embed_dim']
      (True,  False, idx) → 1   (scalar)
      (False, True,  idx) → SCC_EMBEDDING_FEATURES[idx]['embed_dim']
      (False, False, idx) → 1   (scalar)
    """
    total = 0
    for is_normal, is_emb, idx in field_spec:
        if is_normal and is_emb:
            total += emb_features[idx]['embed_dim']
        elif is_normal and not is_emb:
            total += 1
        elif not is_normal and is_emb:
            total += scc_emb_features[idx]['embed_dim']
        else:
            total += 1
    return total


# ─────────────────────────────────────────────────────────────────────────────
# SCC Attention Network
# ─────────────────────────────────────────────────────────────────────────────


class SCCAttentionNet(nn.Module):
    """
    Item-level attention scorer for the shopping-cart digest.

    Input:  scc_digest_tensor  (B, T, scc_digest_dim)
            mask               (B, T, 1)   1 = real item, 0 = padding
    Output: attention weights  (B, T, 1)  — masked softmax over T
    """

    def __init__(self, scc_digest_dim: int, dropout: float = 0.2,
                 negative_slope: float = 0.01):
        super().__init__()

        hidden_dim = (scc_digest_dim + 1) // 2

        self.scorer = nn.Sequential(
            nn.Linear(scc_digest_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.scorer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        scc_digest_tensor: torch.Tensor,    # (B, T, scc_digest_dim)
        mask: torch.Tensor,                  # (B, T, 1)  — FIX 2: mask now received here
    ) -> torch.Tensor:
        """
        Args:
            scc_digest_tensor: (B, T, scc_digest_dim)
            mask:              (B, T, 1)  — 1 = real item, 0 = padding

        Returns:
            attention weights: (B, T, 1)  summing to 1 over real (non-padded) positions
        """
        scores = self.scorer(scc_digest_tensor)             # (B, T, 1)

        # FIX 2 — CRITICAL: apply mask BEFORE softmax.
        #
        # Old code (WRONG):
        #   weights = softmax(scores, dim=1)                # includes padding positions
        #   out = (weights * digest * mask).sum(dim=1)      # zeros padding after the fact
        #
        # Problem: softmax distributes probability over ALL T slots, including padding.
        # Zeroing after means the surviving weights sum to < 1 (e.g. 0.6 for a 3-of-5
        # cart), silently shrinking the digest vector for every padded sequence.
        #
        # Fix: set padding logits to -inf so softmax assigns them exactly 0 probability.
        # The remaining weights sum to 1 over real items only — no post-hoc mask needed.
        scores = scores.masked_fill(mask == 0, float("-inf"))
        return torch.softmax(scores, dim=1)                 # (B, T, 1), sums to 1 over real items


# ─────────────────────────────────────────────────────────────────────────────
# Head Network
# ─────────────────────────────────────────────────────────────────────────────

def _build_head_net(
    neck_dim: int,
    dropout_schedule: tuple[float, float, float] = (0.3, 0.2, 0.15),
    negative_slope: float = 0.01,
) -> nn.Sequential:
    """
    3-layer MLP that maps the neck vector to a single output score.

    Layer widths follow the original: 3/4 → 2/4 → 1/4 of neck_dim.
    L2 regularisation is applied via weight_decay in the optimizer (PyTorch
    convention) rather than per-layer, so kernel_regularizer is removed here.
    """
    d1 = int((neck_dim + 1) * 3 / 4)
    d2 = int((neck_dim + 1) * 2 / 4)
    d3 = int((neck_dim + 1) * 1 / 4)

    def _block(in_d: int, out_d: int, drop: float) -> list:
        return [
            nn.Linear(in_d, out_d, bias=False),
            nn.BatchNorm1d(out_d),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(p=drop),
        ]

    layers = (
        _block(neck_dim, d1, dropout_schedule[0]) +
        _block(d1,       d2, dropout_schedule[1]) +
        _block(d2,       d3, dropout_schedule[2]) +
        [nn.Linear(d3, 1), nn.ReLU()]
    )

    net = nn.Sequential(*layers)

    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="leaky_relu"
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm1d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    return net


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class RecSysModel(nn.Module):

    def __init__(
        self,
        interaction_dim: int,
        shopping_cart_interaction_dim: int,
        residual_embeddings_to_neck: bool,
        residual_SCC_digest_to_neck: bool,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.interaction_dim               = interaction_dim
        self.shopping_cart_interaction_dim = shopping_cart_interaction_dim
        self.residual_embeddings_to_neck   = residual_embeddings_to_neck
        self.residual_SCC_digest_to_neck   = residual_SCC_digest_to_neck

        # ── Embedding tables ─────────────────────────────────────────────────
        self.embedding_tables = nn.ModuleDict({
            feat['name']: nn.Embedding(
                num_embeddings=feat['input_dim'],
                embedding_dim=feat['embed_dim'],
                padding_idx=0,
            )
            for feat in EMBEDDING_FEATURES
        })
        self._init_embeddings()

        # ── SCC InteractionNet ───────────────────────────────────────────────
        scc_field_dims = [
            _compute_field_dim(spec, EMBEDDING_FEATURES, SCC_EMBEDDING_FEATURES)
            for _, spec in SCC_FIELDS
        ]
        self.scc_interaction_net = InteractionNet(
            field_names=[name for name, _ in SCC_FIELDS],
            field_dims=scc_field_dims,
            interaction_dim=shopping_cart_interaction_dim,
            dropout=dropout,                                # FIX (train.py BUG 3)
        )
        self.scc_digest_dim: int = self.scc_interaction_net.output_dim

        # ── SCC Attention Net ────────────────────────────────────────────────
        self.scc_attention_net = SCCAttentionNet(
            self.scc_digest_dim,
            dropout=dropout,
        )

        # ── Main InteractionNet ──────────────────────────────────────────────
        main_field_dims = [
            _compute_field_dim(spec, EMBEDDING_FEATURES, SCC_EMBEDDING_FEATURES)
            for _, spec in FIELDS
        ] + [self.scc_digest_dim]

        self.main_interaction_net = InteractionNet(
            field_names=[name for name, _ in FIELDS] + ["SCC_Digest"],
            field_dims=main_field_dims,
            interaction_dim=interaction_dim,
            dropout=dropout,
        )

        # ── Neck & Head ───────────────────────────────────────────────────────
        neck_dim = self.main_interaction_net.output_dim

        if residual_embeddings_to_neck:
            neck_dim += int(np.sum(self.main_interaction_net.field_dims[:-1]))

        if residual_SCC_digest_to_neck:
            neck_dim += self.scc_digest_dim

        self.neck_dim = neck_dim
        self.head_net = _build_head_net(neck_dim)

    # ── Embedding init ────────────────────────────────────────────────────────
    def _init_embeddings(self) -> None:
        for emb in self.embedding_tables.values():
            nn.init.trunc_normal_(emb.weight, mean=0.0, std=0.05)


    # ── Field tensor builders ─────────────────────────────────────────────────
    def _build_scc_field_tensors(
        self,
        embeddeds: list[torch.Tensor],       # list of (B, d)
        sca_feats: torch.Tensor,             # (B, 91)
        scc_embeddeds: list[torch.Tensor],   # list of (B, T, d)
        scc_sca_feats: torch.Tensor,         # (B, T, 5)
    ) -> list[torch.Tensor]:
        """
        Build the list of per-field tensors fed into the SCC InteractionNet.
        All tensors returned have shape (B, T, field_dim).

        Normal-path features (is_normal=True) have no sequence dimension —
        they are unsqueezed and expanded to T so the SCC InteractionNet
        receives consistent 3-D inputs.
        """
        T = scc_sca_feats.shape[1]
        field_tensors: list[torch.Tensor] = []

        for _, spec in SCC_FIELDS:
            parts: list[torch.Tensor] = []
            for is_normal, is_emb, idx in spec:
                if is_normal:
                    if is_emb:
                        # (B, d) → (B, 1, d) → (B, T, d)
                        parts.append(
                            embeddeds[idx].unsqueeze(1).expand(-1, T, -1)
                        )
                    else:
                        # (B,) → (B, 1, 1) → (B, T, 1)
                        parts.append(
                            sca_feats[:, idx: idx + 1]
                            .unsqueeze(1).expand(-1, T, -1)
                        )
                else:
                    if is_emb:
                        parts.append(scc_embeddeds[idx])         # (B, T, d)
                    else:
                        parts.append(scc_sca_feats[:, :, idx: idx + 1])  # (B, T, 1)

            field_tensors.append(torch.cat(parts, dim=-1))  # (B, T, field_dim)

        return field_tensors

    def _build_main_field_tensors(
        self,
        embeddeds: list[torch.Tensor],   # list of (B, d)
        sca_feats: torch.Tensor,         # (B, 91)
    ) -> list[torch.Tensor]:
        """
        Build per-field tensors for the main InteractionNet.
        All tensors have shape (B, field_dim).
        The SCC digest is appended by the caller after this returns.
        """
        field_tensors: list[torch.Tensor] = []

        for _, spec in FIELDS:
            parts: list[torch.Tensor] = []
            for is_normal, is_emb, idx in spec:
                # FIELDS contains only normal-path features (is_normal=True always)
                if is_emb:
                    parts.append(embeddeds[idx])                      # (B, d)
                else:
                    parts.append(sca_feats[:, idx: idx + 1])          # (B, 1)
            field_tensors.append(torch.cat(parts, dim=-1))            # (B, field_dim)

        return field_tensors

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        sca_feats     = inputs["i2"]
        scc_sca_feats = inputs["i4"]
        mask          = inputs["i5"].float()

        # removed dead `emb_feats` variable
        emb_feats_safe     = inputs["i1"].clamp(min=0)
        scc_emb_feats_safe = inputs["i3"].clamp(min=0)

        # ── 1. Embed normal categorical features ──────────────────────────
        embeddeds: list[torch.Tensor] = [
            self.embedding_tables[EMBEDDING_FEATURES[i]['name']](emb_feats_safe[:, i])
            for i in range(len(EMBEDDING_FEATURES))
        ]

        # ── 2. Embed SCC categorical features ────────────────────────────
        scc_embeddeds: list[torch.Tensor] = [
            self.embedding_tables[SCC_EMBEDDING_FEATURES[i]['name']](
                scc_emb_feats_safe[:, :, i]
            )
            for i in range(len(SCC_EMBEDDING_FEATURES))
        ]

        # ── 3. SCC path ───────────────────────────────────────────────────
        scc_field_tensors = self._build_scc_field_tensors(
            embeddeds, sca_feats, scc_embeddeds, scc_sca_feats
        )
        scc_digest_tensor = self.scc_interaction_net(scc_field_tensors)   # (B, T, scc_digest_dim)

        scc_att_weights   = self.scc_attention_net(scc_digest_tensor, mask)  # mask applied inside
        scc_digest_vector = (scc_att_weights * scc_digest_tensor).sum(dim=1) # no * mask here

        # ── 4. Main path ──────────────────────────────────────────────────
        main_field_tensors = self._build_main_field_tensors(embeddeds, sca_feats)
        main_field_tensors.append(scc_digest_vector)

        neck_vector = self.main_interaction_net(main_field_tensors)

        # ── 5. Residual skip-connections ──────────────────────────────────
        residual_parts: list[torch.Tensor] = [neck_vector]

        if self.residual_embeddings_to_neck:
            residual_parts.extend(main_field_tensors[:-1])

        if self.residual_SCC_digest_to_neck:
            residual_parts.append(scc_digest_vector)

        neck_with_residuals = torch.cat(residual_parts, dim=-1)

        # ── 6. Head → score ───────────────────────────────────────────────
        return self.head_net(neck_with_residuals)    # (B, 1)

    def __repr__(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        trainable    = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"RecSysModel(\n"
            f"  interaction_dim={self.interaction_dim}, "
            f"scc_interaction_dim={self.shopping_cart_interaction_dim}\n"
            f"  residual_emb={self.residual_embeddings_to_neck}, "
            f"residual_scc={self.residual_SCC_digest_to_neck}\n"
            f"  scc_digest_dim={self.scc_digest_dim}, "
            f"neck_dim={self.neck_dim}\n"
            f"  total_params={total_params:,}, trainable={trainable:,}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_recsys_model(
    interaction_dim: int,
    shopping_cart_interaction_dim: int,
    residual_embeddings_to_neck: bool,
    residual_SCC_digest_to_neck: bool,
    dropout: float = 0.15,
) -> RecSysModel:
    return RecSysModel(
        interaction_dim=interaction_dim,
        shopping_cart_interaction_dim=shopping_cart_interaction_dim,
        residual_embeddings_to_neck=residual_embeddings_to_neck,
        residual_SCC_digest_to_neck=residual_SCC_digest_to_neck,
        dropout=dropout,
    )
