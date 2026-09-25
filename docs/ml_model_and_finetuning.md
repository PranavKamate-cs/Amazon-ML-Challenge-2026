# 🤖 Machine Learning Model Architecture & Fine-Tuning Guide
## Team Epoch-alypse | Amazon ML Challenge 2026

---

## 🏗️ 1. Model Architecture & Formulation

We formulated Cross-Source Business Entity Resolution as a **Pairwise Discriminative Binary Classification** problem:

$$\hat{y} = \mathbb{I}\Big(P(\text{Match} = 1 \mid \vec{x}) \ge \theta\Big)$$

where $\vec{x} \in \mathbb{R}^{17}$ is a multi-dimensional feature vector extracted from the candidate pair $(S_1, S_{\text{cand}})$, and $\theta$ is the precision-calibrated decision threshold.

```
       Candidate Pair (S1, S_cand)
                   │
                   ▼
┌───────────────────────────────────────┐
│ 17-Dimensional Pair Feature Extractor  │
│ (RapidFuzz C++, Address & Num Logic)  │
└──────────────────┬────────────────────┘
                   │ Feature Vector x ∈ ℝ¹⁷
                   ▼
┌───────────────────────────────────────┐
│ LightGBM Gradient Boosted Decision    │
│ Trees (300 Trees, Histogram-based)    │
└──────────────────┬────────────────────┘
                   │ Match Probability P(Match | x)
                   ▼
┌───────────────────────────────────────┐
│ F0.5 Threshold Calibrator (θ = 0.65)  │
│ • If P ≥ 0.65  ──> Match (S1 ➔ S_cand)│
│ • If P < 0.65  ──> Reject Candidate   │
└───────────────────────────────────────┘
```

### Why LightGBM (Gradient Boosted Decision Trees)?
1. **Million-Scale Inference Throughput:** Fast C++ histogram binning allows scoring **100,000+ candidate pairs per second**, critical for evaluating over 11.7 Million records in hours.
2. **Non-Linear Multi-Modal Interactions:** GBDT models naturally learn complex conditional branch rules:
   - *If `address` is missing (`addr_is_missing == 1`), rely on strict `name_ratio` ($\ge 0.90$).*
   - *If `name` has a trade-name variation, but street number and PIN code match (`num_mismatch_penalty == +1`), validate the match.*
   - *If house numbers conflict (`num_mismatch_penalty == -1`), reject the match regardless of surface name similarity.*

---

## 🧬 2. The 17-Dimensional Feature Vector ($\vec{x}$)

| # | Feature Name | Formula / Logic | Purpose |
| :--- | :--- | :--- | :--- |
| 1 | `name_ratio` | $\text{Levenshtein}(S_1, S_2) / 100$ | Overall character edit similarity |
| 2 | `name_partial_ratio` | $\text{PartialSubstringRatio}(S_1, S_2) / 100$ | Catches abbreviations & prefixes |
| 3 | `name_token_sort` | $\text{TokenSortRatio}(S_1, S_2) / 100$ | Word-order invariant similarity |
| 4 | `name_token_set` | $\text{TokenSetRatio}(S_1, S_2) / 100$ | Subset-invariant similarity |
| 5 | `name_wratio` | $\text{WRatio}(S_1, S_2) / 100$ | Weighted length-adjusted ratio |
| 6 | `name_exact_match` | $\mathbb{I}(\text{clean}(S_1) == \text{clean}(S_2))$ | Exact canonical name identity |
| 7 | `name_len_diff` | $\|L_1 - L_2\|$ | Character length gap |
| 8 | `name_len_ratio` | $\min(L_1, L_2) / \max(L_1, L_2)$ | Relative length ratio |
| 9 | `addr_ratio` | $\text{Levenshtein}(\text{Addr}_1, \text{Addr}_2) / 100$ | Street & locality string distance |
| 10 | `addr_token_set` | $\text{TokenSetRatio}(\text{Addr}_1, \text{Addr}_2) / 100$ | Robust against component reordering |
| 11 | `addr_token_sort` | $\text{TokenSortRatio}(\text{Addr}_1, \text{Addr}_2) / 100$ | Sorted address token match |
| 12 | `addr_partial_ratio` | $\text{PartialRatio}(\text{Addr}_1, \text{Addr}_2) / 100$ | Substring address containment |
| 13 | `addr_is_missing` | $\mathbb{I}(\text{Addr}_1 \text{ is NaN} \lor \text{Addr}_2 \text{ is NaN})$ | Missing data flag |
| 14 | `num_common_count` | $\|N_1 \cap N_2\|$ | Count of matching digits (house/PIN) |
| 15 | `num_jaccard` | $\|N_1 \cap N_2\| / \|N_1 \cup N_2\|$ | Numerical overlap proportion |
| 16 | `num_mismatch_penalty` | $+1.0 \text{ (match)}, -1.0 \text{ (conflict)}, 0.0 \text{ (absent)}$ | Decisive address conflict check |
| 17 | `is_s3` | $\mathbb{I}(S_{\text{cand}} \in \text{Source 3})$ | Source-specific distribution bias |

---

## ⚙️ 3. Training & Regularization

```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.08,
    'num_leaves': 31,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
```

- **Leak-Free Partitioning:** Group K-Fold splitting on `source1_entity_id` guarantees that no entity's candidate pairs cross train and validation splits.
- **Regularization:** `feature_fraction=0.85` de-correlates string similarity features across decision trees, while `bagging_fraction=0.85` prevents overfitting on frequent business patterns.

---

## 🎯 4. Decision Threshold Optimization for Macro $F_{0.5}$

$$\text{F}_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- In standard binary classification, $\theta = 0.50$ is used.
- For $F_{0.5}$, **Precision is weighted 2× over Recall**, penalizing false merges heavily.
- Singletons score **1.0** when correctly identified with an empty list, and **0.0** if a single false positive is assigned.
- Sweeping $\theta \in [0.30, 0.90]$ demonstrates that **$\theta = 0.65$** achieves the optimal balance: aggressively rejecting false matches, preserving singletons, and maximizing macro $F_{0.5}$.
