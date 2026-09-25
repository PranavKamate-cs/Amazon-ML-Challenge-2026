# 🎯 Amazon ML Challenge 2026 — Master Execution Plan
## Team: Epoch-alypse | Problem: Business Entity Resolution

---

## 📑 Executive Summary
- **Problem Statement:** Cross-source Entity Resolution across 3 noisy sources (`Source 1`, `Source 2`, `Source 3`). Find all matching `S2-*` and `S3-*` records for every `S1-*` record.
- **Evaluation Metric:** Macro-averaged $F_{0.5}$ (Precision-weighted 2× over Recall; singletons reward 1.0 on exact empty match).
- **Core Deliverables:**
  1. `output/matching_results.tsv` (Leaderboard submission)
  2. `output/candidate_pairs.tsv` (Blocking candidates)
  3. `code/business_entity_resolution/` (Reproducible, well-commented code pipeline)
  4. `Documentation_template.md` (Comprehensive 1–2 page methodology document)

---

## 📐 1. End-to-End System Architecture

```
[Raw Sources: S1, S2, S3]
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. DATA PREPROCESSING & CANONICAL NORMALIZATION             │
│  • Unicode cleaning, case folding, punctuation stripping    │
│  • Legal suffix standardization (LLC, Pvt Ltd, Corp, Inc)   │
│  • Address abbreviations (St -> Street, Rd -> Road, etc.)   │
│  • Multilingual & country-agnostic tokenizers (US/IN/FR)    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. STAGE 1: HYBRID MULTI-INDEX BLOCKING & CANDIDATE GEN     │
│  • Country Hard-Filter: Only match within same country      │
│  • Fast BM25 / Sparse TF-IDF (Word & Char 3-5 Grams)        │
│  • Dense Multilingual Bi-Encoder (BGE-small / MiniLM-L12)   │
│  • Phonetic MinHash / Metaphone Blocking                    │
│  ──> Outputs candidate_pairs.tsv (Target: >98% Recall, K≤30)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. STAGE 2: PAIRWISE FEATURE EXTRACTION & GBDT CLASSIFIER   │
│  • String Similarity: Levenshtein, Jaro-Winkler, Token Sort │
│  • Address Matching: Pin/Postal overlap, Building/Unit check│
│  • Cross-Encoder Semantic Similarity (DeBERTa / MiniLM)     │
│  • LightGBM + CatBoost + XGBoost Pair Classification        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. STAGE 3: F0.5 THRESHOLD CALIBRATION & SINGLETON LOGIC    │
│  • Metric-specific threshold search on OOF validation       │
│  • Strict Singleton preservation (avoid false merges)       │
│  • Format verification via utils/validate_submission.py     │
│  ──> Outputs matching_results.tsv                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 🗓️ 2. 72-Hour Phase Breakdown

### 🔹 Phase 1: Setup, Baseline & Submission #1 (Hours 0 – 12)
- **Data Ingestion:** Load training & test TSVs, verify record counts, schema, and noise patterns.
- **Local Validation Strategy:**
  - Create a 5-Fold Stratified Split on `train_source1.tsv` based on match count (0 matches = singleton, 1 match, >1 matches).
  - Implement an exact macro $F_{0.5}$ metric evaluator function in Python.
- **Minimal Baseline Pipeline:**
  - Standard string normalization + Country block + TF-IDF Top-5 candidate generation.
  - Basic Jaccard threshold matching.
  - Run `utils/validate_submission.py` to ensure 0 formatting errors.
  - Generate & submit **Baseline Submission #1**.

---

### 🔹 Phase 2: Advanced Candidate Generation / Blocking (Hours 12 – 24)
- **High Recall Multi-Index Retrieval:**
  1. *Token & Character N-Gram Inverted Index (BM25 / TF-IDF)* for exact & partial token matches.
  2. *Dense Semantic Search*: Generate vector embeddings for business name + address using a lightweight multilingual transformer (`paraphrase-multilingual-MiniLM-L12-v2` or `bge-small-en-v1.5`) with FAISS cosine indexing.
  3. *Phonetic Key Blocking*: Soundex / Double Metaphone keys for noisy Indian and French names.
- **Candidate Fusion & Pruning:**
  - Combine candidates from all blocking channels with union + rank aggregation.
  - Measure **Recall Ceiling** (fraction of ground truth pairs captured) and **Reduction Ratio** on validation set. Target: Recall $\ge 98.5\%$ with $K \le 25$ pairs per S1.
  - Export verified `candidate_pairs.tsv`.

---

### 🔹 Phase 3: Deep Feature Engineering & GBDT Matchers (Hours 24 – 48)
- **Extract 40+ Discriminative Pair Features:**
  - *Name Features:* Levenshtein distance, Jaro-Winkler, Token Sort Ratio, Token Set Ratio, Longest Common Subsequence (LCS), Prefix/Suffix match, Length ratio.
  - *Address Features:* Numeric token overlap ratio (house/plot/pincode numbers), Token Jaccard, Substring containment, Street/Locality similarity.
  - *Country & Cross-lingual Features:* Language-independent normalized character n-gram cosine similarities.
  - *Bi-Encoder / Cross-Encoder Scores:* Cosine similarity of dense embeddings + Cross-Encoder classification logits.
- **Model Training:**
  - Train CatBoost, LightGBM, and XGBoost classifiers with pairwise binary cross-entropy or focal loss.
  - Measure Out-of-Fold (OOF) precision, recall, and $F_{0.5}$.
  - Deploy **Submissions #2 – #5** with progressive model enhancements.

---

### 🔹 Phase 4: Ensembling, Calibration & Post-Processing (Hours 48 – 64)
- **Model Ensembling:** Weighted blending / rank averaging across CatBoost, LightGBM, and Cross-Encoder predictions.
- **$F_{0.5}$-Optimal Decision Thresholding:**
  - Sweep threshold $\theta \in [0.1, 0.9]$ on OOF validation scores.
  - Due to $F_{0.5}$ precision weighting, optimal $\theta$ is higher ($\approx 0.65 - 0.78$), filtering marginal candidates and safeguarding singletons.
- **Submission Validation:** Benchmark and submit **Submissions #6 – #10**.

---

### 🔹 Phase 5: Final Package, Code Polish & Documentation (Hours 64 – 72)
- **Source Code Packaging:**
  - Clean modular code in `code/business_entity_resolution/src/` (`preprocess.py`, `blocking.py`, `features.py`, `train.py`, `inference.py`).
  - Strict documentation with docstrings and clean type annotations.
  - Standalone `README.md` with step-by-step reproduction instructions.
  - Pinned `requirements.txt`.
- **Methodology Document (`Documentation_template.md`):**
  - Section-by-section writeup: Executive Summary, Candidate Generation Strategy, Feature Engineering, Model Architecture, Ablation Experiments table, and Conclusions.
- **Final Packaging:** Create the final submission ZIP `<team_name>_submission.zip` matching the required structure.

---

## 🛡️ 3. Anti-Plagiarism & Quality Assurance Checklist

- [x] **Zero External Lookups:** No geocoding APIs, no external databases, no Google Maps scraping.
- [x] **100% Original Codebase:** Modular, custom algorithms developed in-house.
- [x] **Strict Output Formatting:**
  - Tab-separated `.tsv` files.
  - Exactly one row per test `source1_entity_id`.
  - Empty string for singletons.
  - Comma-separated IDs for matches (no spaces, no quotes).
  - Validation with `utils/validate_submission.py` returning `PASS (exit 0)`.
- [x] **Model Parameter & License Compliance:** Open-source architectures (Apache 2.0 / MIT), $< 8\text{B}$ parameters.

---

## ☁️ 4. AWS Scaling Strategy

| Stage | Local Development (RTX 3050 Laptop) | AWS Cloud Acceleration (EC2 `g4dn.xlarge` / `g5.xlarge`) |
| :--- | :--- | :--- |
| **EDA & Validation** | Instant Pandas / DuckDB prototyping | Not needed |
| **Stage 1 Blocking** | Fast BM25 + FAISS (CPU/GPU) | Multilingual Dense Embedding Batch Encoding |
| **Stage 2 Features** | Multi-process Polars / Joblib feature generation | High-vCPU parallel feature matrix computation |
| **Stage 3 Training** | Fast GBDT training (LightGBM/CatBoost) | DeBERTa Cross-Encoder fine-tuning |
| **Inference & Package** | End-to-end verification & packaging | Final full-dataset inference |
