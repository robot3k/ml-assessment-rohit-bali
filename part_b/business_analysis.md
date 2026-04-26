# Part B: Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### (a) ML Problem Formulation

**Target Variable:** `items_sold` (sales volume per store per month)

**Candidate Input Features:**
- Store-level: `store_id`, `location_type` (urban/semi-urban/rural), `store_size`, `competition_density`
- Calendar: `month`, `is_weekend`, `is_festival`, `day_of_week`, `is_month_end`
- Promotion: `promotion_type` (Flat Discount, BOGO, Free Gift with Purchase, Category-Specific Offer, Loyalty Points Bonus)
- Customer demographics: `footfall`, customer age/income mix (if available)

**Type of ML Problem:** This is a **multi-class classification problem**. Each month and each store, the model must recommend one of five promotion types — making this a discrete, multi-label recommendation task. Alternatively, if we want to predict the *expected items_sold for each promotion* before committing, we can frame it as five parallel regression problems (one per promotion type), then pick the promotion that yields the highest predicted volume. This "predict-then-optimise" framing is more actionable for marketing teams.

**Justification:** Classification is appropriate when the output is a fixed set of discrete choices (the five promotion types). The goal is not to predict a continuous revenue figure but to select the best action among a known option set. If interpretability is important for marketing stakeholders, a Random Forest Classifier or XGBoost is suitable; if probabilistic confidence is needed, a calibrated classifier adds value.

---

### (b) Items Sold vs. Total Sales Revenue as Target Variable

Using **items sold (sales volume)** is a more reliable and stable target variable than total sales revenue for the following reasons:

1. **Revenue is confounded by price variance.** Revenue = Price × Quantity. Different promotions (e.g., Flat Discount reduces price) artificially deflate revenue even when the promotion successfully drives volume. A model trained on revenue would incorrectly penalise volume-driving promotions that involve price cuts.

2. **Volume is directly actionable.** The marketing team controls what promotion to run — not item prices, which may fluctuate due to supplier costs, seasonal markdowns, or competitor pricing. Volume is a cleaner signal of a promotion's effectiveness at driving foot traffic and purchase behaviour.

3. **Revenue is noisier at the store level.** High-value stores can dominate revenue metrics, masking that a promotion underperformed at smaller outlets. Volume normalises this.

**Broader Principle — Target Variable Alignment:** In real-world ML projects, the target must reflect the *decision being made*, not the *metric the business tracks by convention*. Revenue may be the final KPI, but it is influenced by many factors outside the model's control. A well-designed target variable isolates the signal the model is actually responsible for predicting. This principle — choosing a target that is causally proximate to the action — is fundamental to building models that are interpretable, fair, and truly useful.

---

### (c) Against a Single Global Model — A Location-Aware Modelling Strategy

A junior analyst's suggestion of one global model across all 50 stores ignores a critical reality: **stores in urban, semi-urban, and rural locations have fundamentally different customer bases, footfall patterns, and responsiveness to promotions.** A Loyalty Points Bonus may resonate with regular urban shoppers but be irrelevant in a rural store with low repeat-visit rates.

**Proposed Alternative: Stratified or Hierarchical Modelling**

I would propose a **location-type stratified model**:
- Train a separate model for each of the three location types (urban, semi-urban, rural).
- This gives each segment its own learned parameters that capture local behaviour.
- If data per segment is insufficient, use a **hierarchical (mixed-effects) model** that learns a global prior while allowing location-specific deviations — a balanced approach between full pooling (one global model) and no pooling (50 separate models).

This approach accounts for the fact that "the same promotion behaves differently in different places" — which is precisely what the business problem states. The marginal cost of training three models instead of one is negligible; the gain in recommendation accuracy is significant.

---

## B2. Data and EDA Strategy

### (a) Joining the Four Tables

The four source tables are:
1. **Transactions** — store_id, month, promotion_type, items_sold
2. **Store Attributes** — store_id, location_type, store_size, competition_density
3. **Promotion Details** — promotion_id, promotion_type, discount_depth, mechanic
4. **Calendar** — date, is_weekend, is_festival, month, year

**Join Strategy:**
- Join `Transactions` ↔ `Store Attributes` on `store_id` (many-to-one)
- Join result ↔ `Promotion Details` on `promotion_type` (many-to-one)
- Join result ↔ `Calendar` on `month` and `year` (many-to-one)

**Grain of the Final Dataset:** One row = one store × one month. If there are multiple transactions per store per month, aggregate before joining: sum `items_sold`, take the mode of `promotion_type` (assuming one promotion per store-month as stated).

**Pre-modelling Aggregations:**
- Monthly `items_sold` per store (sum)
- Average `competition_density` per store (if it varies over time)
- Lag features: `items_sold` from the previous month (captures momentum)
- Rolling 3-month average of `items_sold` (captures seasonality trend)

---

### (b) EDA Before Modelling — Four Key Analyses

**1. Promotion Performance by Location Type (Bar Chart)**
Plot mean `items_sold` grouped by `(promotion_type, location_type)`. What to look for: whether the same promotion performs consistently across locations or has opposite effects in different settings. This directly informs whether we need stratified models and which promotions are location-specific winners.

**2. Seasonality Analysis (Line Chart)**
Plot monthly average `items_sold` across all stores over time. What to look for: clear peaks around festivals, year-end, or sale seasons. If strong seasonality exists, we need calendar features in the model and must be careful about temporal train-test splits.

**3. Store Size vs. Items Sold (Box Plot)**
Distribution of `items_sold` grouped by `store_size`. What to look for: whether large stores dominate volume to a degree that might require log-transformation of the target, or whether size needs to be interacted with promotion type in feature engineering.

**4. Correlation Between Competition Density and Promotion Lift (Scatter Plot)**
Plot `items_sold` against `competition_density`, coloured by `promotion_type`. What to look for: whether promotions like Flat Discount are more effective in high-competition areas (defensive pricing) while Loyalty Points work better in low-competition areas (relationship-building). This would justify an interaction feature `competition_density × promotion_type`.

---

### (c) Addressing 80% No-Promotion Imbalance

If 80% of transactions occurred without any promotion, the dataset is heavily imbalanced in the promotion dimension. This causes two problems:

1. **For classification:** A model predicting "which promotion to run" trained on this data would be biased toward "no promotion" as the default — it would underfit the promotional scenarios.
2. **For regression:** The model would overfit to the no-promotion baseline and underestimate the true lift from promotions.

**Steps to address this:**

- **Resampling:** Oversample promotional records (SMOTE or bootstrapping) or undersample no-promotion records to create a balanced training set focused on promotion decisions.
- **Separate models:** Train one model specifically on promoted store-months to learn promotion responsiveness, and use historical baselines separately.
- **Class weights:** For classifiers, apply `class_weight='balanced'` to penalise misclassifications of the minority promotional classes more heavily.
- **Targeted filtering:** Since the business question is *which promotion to run*, not *whether to run one*, we can filter the training data to only promoted store-months and train solely on that subset.

---

## B3. Model Evaluation and Deployment

### (a) Train-Test Setup and Evaluation Metrics

**Why random split is inappropriate:** With 3 years of monthly store-level data, a random split would train the model on future months and test it on past months — creating data leakage. Seasonality patterns and year-over-year growth would bleed from test into training, producing falsely optimistic metrics.

**Correct approach: Temporal (walk-forward) split**
- Use months 1–24 (years 1–2) as training data.
- Use month 25–36 (year 3) as the test set.
- This ensures the model is evaluated on genuinely future, unseen data — matching real deployment conditions.

**Evaluation Metrics:**

| Metric | Why it matters in this context |
|--------|-------------------------------|
| **RMSE (Root Mean Squared Error)** | Penalises large prediction errors more heavily. A model that badly mispredicts sales during a festival would be heavily penalised — appropriate since festival mis-allocations are costly. |
| **MAE (Mean Absolute Error)** | Gives the average error in absolute items sold — easy for marketing teams to understand ("on average, our predictions are off by 25 items"). |
| **Per-location-type RMSE** | Urban stores have higher volumes; reporting RMSE by segment reveals whether the model is systematically worse for rural stores despite overall decent performance. |
| **Promotion Hit Rate** | For the classification framing: what % of the time does the model recommend the promotion that would have actually yielded the highest volume? Directly measures recommendation quality. |

---

### (b) Explaining Different Recommendations for the Same Store in Different Months

The model recommends **Loyalty Points Bonus for Store 12 in December** and **Flat Discount for Store 12 in March** because it has learned that the optimal promotion changes with seasonal context.

**Using Feature Importance to Explain:**

I would use **SHAP (SHapley Additive exPlanations)** to generate a force plot for each prediction, showing which features pushed the recommendation toward each promotion type.

For Store 12 in December, I would show the marketing team:
- `is_festival = 1` and `month = 12` have high positive SHAP values for Loyalty Points — suggesting the model learned that December shoppers respond to points accumulation because they're buying gifts repeatedly.
- `competition_density` is low in December at this store, favouring retention-oriented promotions over price cuts.

For Store 12 in March:
- `month = 3`, low footfall, and high competition push SHAP values toward Flat Discount — the model learned that in off-peak months with competition nearby, price reduction is the most effective traffic driver.

**Communication to Marketing Team:**
> "The model is not being inconsistent — it is responding to the seasonal context. In December, customers at Store 12 are in a gift-buying, high-intent mode; Loyalty Points capitalise on repeat visits. In March, it's a slower month with competitive pressure nearby; a Flat Discount attracts price-sensitive customers who might otherwise shop elsewhere."

---

### (c) End-to-End Deployment Process

**Saving the Model:**
- Serialise the trained pipeline using `joblib.dump(pipeline, 'promo_model_v1.pkl')`.
- Store in a model registry (e.g., MLflow, AWS S3 with versioning) with metadata: training date, train RMSE, feature list, and data snapshot hash.

**Monthly Inference Process:**
At the start of each month, a batch scoring script:
1. Pulls the latest store attributes and calendar data (is_festival, is_weekend flags for the coming month).
2. Constructs one row per store with the relevant features.
3. Loads the saved model: `pipeline = joblib.load('promo_model_v1.pkl')`.
4. Runs `pipeline.predict(X_new)` to generate one promotion recommendation per store.
5. Outputs a CSV/dashboard with store_id → recommended_promotion for the marketing team to action.

**No retraining is needed each month** — the same model scores new inputs. Retraining is triggered only when monitoring signals degradation.

**Monitoring for Model Degradation:**

| Signal | Threshold | Action |
|--------|-----------|--------|
| **Prediction drift** | Distribution of recommended promotions shifts significantly (KL divergence > 0.2) | Investigate if store mix or context has changed |
| **RMSE on actuals** | Monthly RMSE > 1.5× training RMSE for 2 consecutive months | Trigger retraining with updated data |
| **Feature drift** | Mean of key features (competition_density, footfall) shifts >2 std from training baseline | Retrain or recalibrate |
| **Business outcome** | Actual items sold consistently underperforms recommendations by >15% | Human review + potential model rebuild |

Retraining should occur at minimum annually (to capture the most recent full year of seasonal data) and ad-hoc after major structural changes such as new store openings, a major competitor entering a market, or a new promotion type being introduced.
