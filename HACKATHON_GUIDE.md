# Amazon ML Challenge 2026 — Team Epoch-alypse Master Plan 🚀

**Team Name:** `Epoch-alypse`  
**Team Leader:** Pranav Gajanan Kamate  
**Members:** Rishab M Jain, Bhavana Holla  
**Challenge Window:** September 25, 2026, 12:00 AM IST – September 27, 2026, 11:59 PM IST (72 Hours)

---

## 📌 1. Hackathon Constraints & Golden Rules

| Category | Details / Constraints | Strategy / Action |
| :--- | :--- | :--- |
| **Submission Limit** | **Max 5 submissions per day** (Strict limit) | Never waste submissions on unvalidated code. Always benchmark against local CV first. |
| **Leaderboards** | **Public (50%) & Private (50%)** | Guard against leaderboard overfitting. Trust local Out-of-Fold (OOF) CV score. |
| **Required Artifacts** | • 1–2 page PDF documentation<br>• Clean, well-commented source code (Train & Inference)<br>• Version log of all submissions | Maintain documentation & code comments continuously as we iterate, not at the last minute. |
| **Hardware / Rules** | Single laptop per participant, no multiple logins | Local GPU: RTX 3050 (4GB) for fast prototyping + Kaggle/Colab for heavy training/inference if needed. |

---

## 🏗️ 2. Repository & Workspace Structure

```
AWS ML Challenge/
├── dataset/                    # Raw and preprocessed datasets (train.csv, test.csv, images/etc.)
│   ├── raw/
│   └── processed/
├── notebooks/                  # Exploratory Data Analysis & experiments
├── src/                        # Modular, production-grade source code
│   ├── __init__.py
│   ├── utils.py                # Seed setting, logging, metric calculations
│   ├── data_loader.py          # Data preprocessing & feature extractors
│   ├── models.py               # Model architectures (Tabular, NLP, Vision, Multimodal)
│   ├── train.py                # Cross-validation training loop & OOF tracking
│   └── inference.py            # Generates test predictions with error checks
├── models/                     # Saved model checkpoints & scalers/encoders
├── submissions/                # Version-controlled submission CSVs + metadata logs
│   ├── submission_log.csv      # Log tracking: Sub_ID, CV_Score, Model, Parameters, Notes
│   └── sub_001_baseline.csv
└── docs/                       # Final report & documentation
    └── approach_document.md    # 1-2 page final solution document
```

---

## ⏱️ 3. 72-Hour Timeline Strategy

- **Day 1 (Sept 25)**:
  - 1. Ingest problem statement, data schema, and evaluation metric.
  - 2. Implement 5-Fold Stratified / Group CV to ensure no data leakage.
  - 3. Build an end-to-end baseline pipeline (fast model -> test predictions -> format check).
  - 4. Submit Baseline (Submission #1) to verify submission pipeline.
- **Day 2 (Sept 26)**:
  - 1. Candidate Generation / Blocking (if product matching/search problem).
  - 2. Multimodal feature extraction (Text Transformers, Vision backbones, Tabular stats).
  - 3. Train diverse model families (GBDTs like CatBoost/LightGBM/XGBoost, Deep Learning/Transformers).
  - 4. Systematic submission testing (Submissions #2 - #6).
- **Day 3 (Sept 27)**:
  - 1. Model ensembling (Blending, Stacking, Rank Averaging).
  - 2. Post-processing & metric threshold optimization.
  - 3. Generate the required 1-2 page ML approach document.
  - 4. Package final code & select final best submissions.
