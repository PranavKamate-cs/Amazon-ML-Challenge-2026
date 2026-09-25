# 🧠 Business Entity Resolution: Problem Statement & Methodology Guide
## Team Epoch-alypse | Amazon ML Challenge 2026

---

## 🎯 Part 1: Problem Statement Explained

### 1. The Core Objective
In commercial e-commerce platforms like Amazon, business information is collected from multiple fragmented, independent sources (e.g., government registrations, vendor portals, directory listings, delivery logs).
- **Source 1 (`S1-*`)**: The clean, deduplicated **reference source**.
- **Source 2 (`S2-*`) & Source 3 (`S3-*`)**: Independent, noisy, unstructured data sources.

**Your Goal:** For every business entity in **Source 1**, find all records in **Source 2** and **Source 3** that refer to the *exact same real-world business entity*.

```
Source 1 (Reference)       Source 2 (Noisy)              Source 3 (Noisy)
┌────────────────────┐    ┌──────────────────────────┐  ┌──────────────────────────┐
│ S1-965667          │───▶│ S2-681193310             │  │ S3-11291185              │
│ Maure Williams     │    │ Maure Wilblims Colombier │  │ maurewilliamscolombier   │
│ Colombier Inc      │    │ Address: NaN             │  │ Wayne Ave, Ticonderoga NY│
│ 85 Wayne Ave, NY   │    └──────────────────────────┘  └──────────────────────────┘
└────────────────────┘
```

---

### 2. Key Challenges & Real-World Noise Patterns

| Noise Category | Real Example from Challenge Dataset | Challenge / Difficulty |
| :--- | :--- | :--- |
| **Missing Addresses** | `S2: Maure Wilblims Colombier | Address: NaN` | When address is missing, match must rely solely on fuzzy name & token logic. |
| **Domain-Style Names** | `S3: maurewilliamscolombier.com` vs `S1: Maure Williams Colombier Inc` | URLs/domains must be normalized, legal suffixes stripped, and tokens segmented. |
| **Spelling & Typos** | `Wilblims` vs `Williams` \| `Wanye Ave` vs `Wayne Ave` | Substring edit distance and character n-gram similarities are required. |
| **DBA / Landmark Names** | `S3: Drxkor | 85 Wanye Ave, Ticonderoga, NY` | Completely different trade name, but identical street number & locality. |
| **Open-Set Country Shift** | Train: `{US, India}` \| Test: `{US, India, France}` | Must handle unseen languages/accents (French addresses/suffixes: `SARL`, `Rue`, `75002 Paris`) without hardcoding. |
| **Singletons (No Match)** | $\approx 5.6\%$ of S1 entities have **0 matches** | Must correctly predict empty string; false merges destroy the score. |

---

### 3. The Precision-Heavy $F_{0.5}$ Metric
$$\text{F}_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Why $\beta = 0.5$?** In real-world entity resolution, **merging two different businesses (False Positive)** is far more dangerous than missing a match (False Negative).
- **Singletons:** Correctly predicting "no match" gives a perfect **1.0** score for that entity. Predicting even 1 wrong candidate drops it to **0.0**.

---

## 🛠️ Part 2: End-to-End Solution Methodology

```
                   ┌───────────────────────────────────┐
                   │    Raw TSVs (S1, S2, S3 Records)   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │ 1. Canonical String Normalization │
                   │  • Unicode / Accent normalization │
                   │  • Legal suffix & URL stripping   │
                   │  • Address abbreviation expansion │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │ 2. Stage 1: Multi-Channel Blocker │
                   │  • Hard Country Partitioning      │
                   │  • Token & Char N-Gram Inverted Idx│
                   │  • Address Number + Locality Index│
                   │  ──> Outputs candidate_pairs.tsv  │
                   │      (99.9995% reduction ratio)   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │ 3. Stage 2: Pairwise Feature Eng  │
                   │  • 17 RapidFuzz & Numeric Signals │
                   │  • Ratio, Partial, Token Set/Sort │
                   │  • House/PIN Number intersection  │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                   ┌───────────────────────────────────┐
                   │ 4. Stage 3: GBDT Match Classifier │
                   │  • LightGBM Pair Scorer           │
                   │  • High-Precision F0.5 Calibration│
                   │  ──> Outputs matching_results.tsv │
                   └───────────────────────────────────┘
```

---

### Detailed Stage Breakdown

#### 🔹 Step 1: Canonical Normalization
- **Name Processing:** Strips web protocols (`https://`, `www.`), domain TLDs (`.com`, `.fr`, `.in`), and standardizes legal designations (`LLC`, `Corp`, `Pvt Ltd`, `SARL`, `SAS`).
- **Address Processing:** Standardizes universal abbreviations (`St` $\rightarrow$ `Street`, `Ave` $\rightarrow$ `Avenue`, `Rd` $\rightarrow$ `Road`) and normalizes accented characters (`é` $\rightarrow$ `e`, `ç` $\rightarrow$ `c`).

#### 🔹 Step 2: High-Recall Multi-Channel Candidate Blocking
Comparing 1.73M test records against 9.9M target records naively requires **$1.7 \times 10^{13}$ pairwise comparisons** (computationally impossible).
Our blocker reduces this by **$99.9995\%$**:
1. **Hard Country Partition:** Only compares entities within the same country ($100\%$ zero leakage rule verified across 7.6M training pairs).
2. **Inverted Token Index:** Indexes rare informative name tokens (IDF-weighted).
3. **Character 3-Gram Keys:** Captures typos, transliterations, and concatenated domain names.
4. **Number-Locality Keys:** Indexes pairs sharing identical street numbers, plot codes, and PIN codes.

#### 🔹 Step 3: Pairwise Discriminative Feature Engineering
For each $(S_1, S_{\text{cand}})$ candidate pair, we extract 17 features:
- **Fuzzy Name Matches:** Full Ratio, Partial Ratio, Token Sort Ratio, Token Set Ratio, Weighted Ratio (`WRatio`), Exact Match flag, Length difference & ratio.
- **Address Signals:** Address similarity ratios, Missing Address indicator (`addr_is_missing`), Common numeric token count, Numeric Jaccard similarity, and Mismatch penalty.
- **Source Indicator:** Distinguishes $S_2$ vs $S_3$ patterns.

#### 🔹 Step 4: GBDT Matcher & $F_{0.5}$ Decision Thresholding
- A **LightGBM** classifier evaluates the candidate feature vectors and outputs a calibrated match probability $P(\text{Match} \mid S_1, S_{\text{cand}})$.
- **Threshold Calibration ($\theta = 0.65$):** Because $F_{0.5}$ weights precision $2\times$ over recall, a conservative decision threshold ($\theta \ge 0.65$) is applied to filter ambiguous candidates, directly protecting singleton scores and maximizing the competition metric.
