# ecommerce-recommendation-engine
Two-stage AI recommendation system — ALS retrieval + LightGBM ranking + automated email generation, built on the RetailRocket e-commerce dataset.
# E-Commerce Recommendation & Email Generation System

End-to-end machine learning pipeline that predicts a customer's next purchase 
and generates a personalized, deployment-ready marketing email automatically.

**Course:** Applied Machine Learning — Final Project  
**Author:** Mohammad Mahir Aziz  

---

## Overview

Most recommendation systems stop at producing a ranked list. This project closes 
the gap between ML output and marketing execution — the final deliverable is not 
just a ranking but a structured JSON payload ready for direct integration into 
platforms like Mailchimp or SendGrid.

---

## Architecture
```
Raw Events → Signal Weighting → ALS Retrieval → LightGBM Ranking → Email Generator
```

| Stage | Method | Detail |
|---|---|---|
| Candidate Generation | ALS (implicit) | Top-200 candidates per user |
| Re-ranking | LightGBM LambdaRank | Optimizes NDCG directly |
| Output | JSON + HTML email | ESP-ready payload |

---

## Dataset

[RetailRocket E-Commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

| Stat | Value |
|---|---|
| Total events | 2,755,641 |
| Unique users | 1,407,580 |
| Unique items | 235,061 |
| Train / Test split | 80% / 20% (temporal) |
| Cutoff date | 2015-08-18 |

---

## Results

Evaluated on 158 users with 5+ training events and a verified next purchase in the test window.

| System | HitRate@10 | NDCG@10 | MRR@10 |
|---|---|---|---|
| Popularity Baseline | 0.0127 | 0.0080 | 0.0063 |
| ALS Only | 0.0063 | 0.0063 | 0.0063 |
| **Two-Stage (Ours)** | **0.0443** | **0.0302** | **0.0258** |

**Candidate Recall@200: 5.7%**  
ALS retrieves the true next-purchase item in its top-200 candidates 9 out of 158 times.  
This identifies retrieval as the primary bottleneck — not ranking.

---

## Key Features

- Temporal train/test split with full leakage prevention — `cutoff_ts` enforced at every feature computation
- Implicit signal weighting — views (1.0), add-to-cart (3.0), purchases (5.0)
- LightGBM trained with LambdaRank objective — optimizes ranking quality, not classification accuracy
- Candidate Recall@200 metric — diagnoses retrieval vs ranking bottleneck
- Business validation — novelty, long-tail coverage, diversity scoring
- ESP integration format — Mailchimp and SendGrid ready output

---

## Tech Stack

- Python 3.12
- [implicit](https://github.com/benfred/implicit) — ALS collaborative filtering
- [LightGBM](https://lightgbm.readthedocs.io/) — LambdaRank re-ranking
- pandas, numpy, scipy, scikit-learn
- Jupyter Notebook (Google Colab compatible)

---

## Project Structure
```
├── Recommendation_Engine_Final.ipynb   # Main notebook — run top to bottom
├── data/
│   └── processed/                      # Generated after running Step 2
├── models/
│   ├── cf_model.pkl                    # Trained ALS model
│   └── ranker_model.txt                # Trained LightGBM model
└── outputs/
    └── recommendation_*.json           # Generated payloads
```

---

## How to Run

1. Upload `events.csv` from the RetailRocket dataset to your working directory
2. Open `Recommendation_Engine_Final.ipynb` in Jupyter or Google Colab
3. Run all cells top to bottom — each step saves its outputs to disk for the next
4. Step 12 (Multi-User Demo) and Step 13 (ESP Integration) show the final outputs

---

## Limitations

- 71.2% of users have a single interaction event — cold-start is a known constraint
- Candidate Recall@200 of 5.7% indicates the retrieval stage is the primary bottleneck
- Product names are template-generated (item_properties.csv lacked readable titles)
- LightGBM trained on 158 labeled users — a larger observation window would improve generalization
