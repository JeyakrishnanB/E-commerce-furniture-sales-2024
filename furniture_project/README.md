# E-commerce Furniture Dataset 2024 — Sales Prediction Project

Predict the number of furniture items sold (`sold`) on AliExpress from `productTitle`, `originalPrice`, `price` and `tagText`.
Follows the 7 steps in the project guide: Data Collection → Preprocessing → EDA → Feature Engineering → Model Training → Evaluation → Conclusion.

## Files
| File | Purpose |
|---|---|
| `furniture_sales_project.ipynb` | Main notebook (already executed, outputs included) |
| `furniture_sales_project.html` | Same notebook as a web page — open in any browser |
| `furniture_sales_project.py` | Plain Python script version |
| `ecommerce_furniture_dataset_2024.csv` | Original data |
| `cleaned_furniture_dataset.csv` | Cleaned data (prices numeric, shipping parsed, discount features) |
| `model_results.csv` | Model metrics table |
| `images/` | All 13 output charts |

Run it: `pip install -r requirements.txt` then `jupyter notebook furniture_sales_project.ipynb` (keep the CSV in the same folder).

## Pre-processing decisions
* `price`/`originalPrice`: `$1,294.04` → `1294.04`.
* `originalPrice` is missing for 75.6% of rows = item is **not discounted**. The guide's `dropna()` would delete most of the data, so we keep the rows and create `has_discount` and `discount_percentage`.
* 94 exact duplicate rows removed (2,000 → 1,906) to avoid train/test leakage.
* `tagText` → `shipping_type` (Free / Paid / Unknown) + numeric `shipping_cost` (the guide's version collapsed most values into "others").
* `sold` is extremely skewed (median 3, max 10,000) → models train on `log1p(sold)`.

## Key findings
* **23%** of listings have zero sales; median is 3 units; a handful of items sell thousands.
* **Cheaper sells more:** median units sold is 14 for items under $25 vs 1 for items over $200.
* **Discounts matter:** discounted listings have a median of 10 units vs 2 for non-discounted (mean 78 vs 7). Only 25% of listings are discounted, most at 40–55% off.
* **94%** of listings already offer free shipping, so it is not a differentiator. Paid-shipping items actually show a higher median (10 vs 3), but there are only 115 of them.
* Titles mentioning practical/small items (holder, shoerack, foldable, ergonomic, dustproof) go with higher sales; "farmhouse", "gray", "accent", "doors" go with lower.

## Model results (20% test set, log scale)
| Model | MSE | R² |
|---|---|---|
| Baseline (mean) | 1.715 | 0.00 |
| Linear Regression | 1.195 | 0.30 |
| **Random Forest** | **1.080** | **0.37** |
| Gradient Boosting | 1.099 | 0.36 |

Random Forest is best. Top features: `has_discount`, `log_price`, `price`, `title_len`, `discount_percentage`.

## Honest limitations
* R² ≈ 0.37 means the available columns explain only part of sales. Missing: ratings, reviews, images, seller, listing age.
* On the raw units scale all models score R² ≈ 0 because the test set contains an extreme seller (10,000 units) that dominates squared error. Use the log-scale metrics (and MAE) for comparison.
* `sold` is cumulative sales on the listing, so older listings naturally have more; listing age isn't in the data.
