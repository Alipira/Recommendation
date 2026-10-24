# Model architecture

This page walks through `src/recsys_model.py` from inputs to output.

## Inputs

| Name | Shape      | Type    | Content |
|------|------------|---------|---------|
| `i1` | (B, 17)    | int     | categorical indices for the row |
| `i2` | (B, 91)    | float   | numeric features for the row |
| `i3` | (B, T, 5)  | int     | product-group indices for each cart item |
| `i4` | (B, T, 5)  | float   | weight, three prices and quantity for each cart item |
| `i5` | (B, T, 1)  | float   | 1 for a real cart item, 0 for padding |

`B` is the batch size and `T` is the longest cart in the batch.

All categorical features use index 0 for "unknown / padding". Real values
start at 1, so each table has `max_index + 1` rows. The feature lists
(`EMBEDDING_FEATURES`, `SCALAR_FEATURES`, `FIELDS`, `SCC_FIELDS`) at the top of
the file are the single source of truth for sizes and order.

## Step 1: embeddings

Each of the 17 categorical features has its own embedding table. Sizes are
small (2 to 16) and grow with the number of values. The five product-group
features share their tables with the cart path, so "Group4Code = 120" means
the same thing for the target product and for a product in the cart.

## Step 2: cart summary

For every cart item the model builds 15 fields. Some come from the row
(customer, price, geography, ...) and are copied to every cart position;
the rest come from the item itself (its product groups, weight, prices,
quantity).

An `InteractionNet` mixes these fields (see below) and gives one vector per
cart item. A small attention network then scores each item. Padding slots
are set to `-inf` before the softmax, so they get exactly zero weight and the
real items' weights add up to 1. A cart with no real items gets a zero
vector instead of NaN.

The weighted sum of item vectors is the **cart digest**: one vector that
says what is already in the cart.

## Step 3: main path

The row's features are grouped into 14 fields:

ProductGroup, Weight, Price, Customer, PaymentMethod, Geographical, Visitor,
Cheque, Payment, Customer_RFM, Path_RFM, Zone_RFM, Date, Holidays.

The cart digest is added as field 15. A second `InteractionNet` mixes them.

## InteractionNet

For every pair of fields (i, j), with 15 fields that is 105 pairs, a small
block runs:

```
concat(field_i, field_j) → Linear (no bias) → BatchNorm → LeakyReLU → Dropout
                         → Linear → LeakyReLU
```

Each block outputs `interaction_dim` numbers. All outputs are joined into one
vector. This lets the model learn things like "this customer group reacts to
this price level" directly, instead of hoping one big MLP finds it.

## Step 4: neck and head

The neck is the main interaction output, plus (optionally) the raw field
tensors and the cart digest as skip connections. A 3-layer MLP with widths
3/4, 2/4 and 1/4 of the neck size gives the final score.

## Training

- **Target:** `log1p(quantity)`. Order sizes range over several orders of
  magnitude; the log keeps large orders from taking over.
- **Loss:** Huber with `delta = 0.3`. On log targets (about 0 to 6), a delta of
  1 would put most of the error of rare samples in the flat part of the
  loss, where the gradient is weak.
- **Weights:** each sample's loss is multiplied by
  `frequency weight × value weight`, both normalised per batch.
  - frequency weight = `1 / sqrt(count(retailer, product))`, so rare pairs count more.
  - value weight = `log1p(quantity) + 1`, so larger orders count more.
- **Optimizer:** Adam, lr 1e-3, weight decay 1e-3, gradient clipping at 1.0.
- **Stopping:** early stopping on validation loss; best weights are restored.
- **Tracking:** MLflow logs parameters, metrics, checkpoints, plots and
  embedding maps.

## Design choices

- **No bias before BatchNorm.** BatchNorm removes the mean anyway, so the
  bias does nothing.
- **Kaiming initialisation** for layers followed by (Leaky)ReLU.
- **LayerNorm in the cart attention scorer,** so the score does not depend on
  how many items are in the batch.
- **Mask before softmax,** not after. Masking after the softmax leaves real
  item weights summing to less than 1 and shrinks the cart digest for every
  padded cart.
- **Weighted Huber loss** to deal with the imbalance between frequent small
  orders and rare or large ones.
