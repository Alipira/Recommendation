# How to debug
on the recsys_model.py file add this to def forward(self, input):
        # ── DEBUG: validate embedding indices (remove after fixing) ──────────
        if not self.training or True:   # runs always; remove block when fixed
            for i, feat in enumerate(EMBEDDING_FEATURES):
                col = inputs["i1"][:, i]
                bad = (col < 0) | (col >= feat['input_dim'])
                if bad.any():
                    bad_vals = col[bad].tolist()[:5]
                    raise ValueError(
                        f"[i1] Feature '{feat['name']}' (col {i}): "
                        f"index out of range [0, {feat['input_dim']}). "
                        f"Offending values: {bad_vals}"
                    )
            for i, feat in enumerate(SCC_EMBEDDING_FEATURES):
                col = inputs["i3"][:, :, i]
                bad = (col < 0) | (col >= feat['input_dim'])
                if bad.any():
                    bad_vals = col[bad].tolist()[:5]
                    raise ValueError(
                        f"[i3] SCC feature '{feat['name']}' (col {i}): "
                        f"index out of range [0, {feat['input_dim']}). "
                        f"Offending values: {bad_vals}"
                    )
        # ── END DEBUG ─────────────────────────────────────────────────────────
    ...