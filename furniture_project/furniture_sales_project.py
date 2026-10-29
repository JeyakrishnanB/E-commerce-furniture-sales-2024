# E-commerce Furniture Dataset 2024 - full pipeline (auto-exported from the notebook)

# # 🛋️ E-commerce Furniture Dataset 2024 — Predicting Units Sold
# **Tools:** Python · pandas · scikit-learn · matplotlib · seaborn  
# **Objective:** Predict the number of furniture items sold (`sold`) from `productTitle`, `originalPrice`, `price` and `tagText`, and understand what drives sales on AliExpress.
# 
# **Steps**
# 1. Data Collection  2. Data Preprocessing  3. Exploratory Data Analysis  4. Feature Engineering  5. Model Selection & Training  6. Model Evaluation  7. Conclusion

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams.update({'figure.dpi': 110, 'savefig.bbox': 'tight'})

DATA_PATH = 'ecommerce_furniture_dataset_2024.csv'
IMG_DIR = 'images'
os.makedirs(IMG_DIR, exist_ok=True)

def save(name):
    plt.savefig(f'{IMG_DIR}/{name}.png', dpi=130)
    plt.show()

# ## 1. Data Collection

df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
print('Shape:', df.shape)
df.head()

df.info()

# ## 2. Data Preprocessing
# Checking missing values, cleaning the price columns (`$1,294.04` → `1294.04`), and handling `tagText`.

print(df.isnull().sum())
print('\nDuplicate rows:', df.duplicated().sum())
print('Missing originalPrice: %.1f%%' % (df['originalPrice'].isnull().mean()*100))

# **Decision:** exact duplicate listings are removed. If the same row appears in both the train and test sets, the test score would be artificially inflated (data leakage).

df = df.drop_duplicates().reset_index(drop=True)
print('Shape after removing duplicates:', df.shape)

# **Decision:** The PDF template uses `dropna()`, but `originalPrice` is missing for ~76% of rows (these items are simply *not discounted*), so dropping would delete most of the data. Instead we keep every row and create a `has_discount` flag.

def to_float(s):
    return pd.to_numeric(s.astype(str).str.replace(r'[\$,]', '', regex=True), errors='coerce')

df['price'] = to_float(df['price'])
df['originalPrice'] = to_float(df['originalPrice'])

# tagText: "Free shipping" or "+Shipping: $5.09" (3 rows missing -> 'Unknown')
df['tagText'] = df['tagText'].fillna('Unknown')
df['shipping_cost'] = np.where(df['tagText'].str.contains(r'\+Shipping', na=False),
                               to_float(df['tagText'].str.extract(r'\$([\d,\.]+)')[0]), 0.0)
df['shipping_type'] = np.select(
    [df['tagText'].eq('Free shipping'), df['tagText'].str.contains(r'\+Shipping'), ],
    ['Free shipping', 'Paid shipping'], default='Unknown')

print(df[['price','originalPrice','sold','shipping_cost']].describe().round(2))
print()
print(df['shipping_type'].value_counts())

# Sanity check: discounted price should be below original price
bad = df[(df['originalPrice'].notna()) & (df['originalPrice'] < df['price'])]
print('Rows where originalPrice < price:', len(bad))
df.to_csv('cleaned_furniture_dataset.csv', index=False)

# ## 3. Exploratory Data Analysis (EDA)

# ### 3.1 Distribution of `sold`
# `sold` is extremely right-skewed: most products sell only a handful of units while a few sell thousands. A log scale makes the shape visible.

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df['sold'], bins=60, kde=True, ax=ax[0], color='#4C72B0')
ax[0].set_title('Distribution of Furniture Items Sold (raw)')
sns.histplot(np.log1p(df['sold']), bins=40, kde=True, ax=ax[1], color='#55A868')
ax[1].set_title('Distribution of log(1 + sold)')
ax[1].set_xlabel('log(1 + sold)')
plt.tight_layout(); save('01_sold_distribution')
print('Products with 0 sales: %.1f%%' % ((df.sold==0).mean()*100))
print('Median sold:', df.sold.median(), '| Mean sold: %.1f' % df.sold.mean(), '| Max:', df.sold.max())

# ### 3.2 Distribution of `price`

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(df['price'], bins=60, kde=True, ax=ax[0], color='#C44E52')
ax[0].set_title('Price distribution')
sns.histplot(np.log1p(df['price']), bins=40, kde=True, ax=ax[1], color='#8172B2')
ax[1].set_title('Distribution of log(1 + price)'); ax[1].set_xlabel('log(1 + price)')
plt.tight_layout(); save('02_price_distribution')

# ### 3.3 Shipping tag (`tagText`)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
order = df['shipping_type'].value_counts().index
sns.countplot(x='shipping_type', data=df, order=order, ax=ax[0])
ax[0].set_title('Listings by shipping type')
for c in ax[0].containers: ax[0].bar_label(c)
sns.boxplot(x='shipping_type', y=np.log1p(df['sold']), data=df, order=order, ax=ax[1])
ax[1].set_ylabel('log(1 + sold)'); ax[1].set_title('Sales by shipping type')
plt.tight_layout(); save('03_shipping_type')
print(df.groupby('shipping_type')['sold'].agg(['count','mean','median','sum']).round(1))

# ### 3.4 Price vs Sold

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
sns.scatterplot(x='price', y='sold', data=df, alpha=.5, ax=ax[0])
ax[0].set_title('Price vs Sold (raw)')
sns.scatterplot(x='price', y='sold', data=df, alpha=.5, ax=ax[1])
ax[1].set_xscale('log'); ax[1].set_yscale('symlog')
ax[1].set_title('Price vs Sold (log scales)')
plt.tight_layout(); save('04_price_vs_sold')

df['price_band'] = pd.cut(df['price'], bins=[0,25,50,100,200,500,10000],
                          labels=['<$25','$25-50','$50-100','$100-200','$200-500','>$500'])
band = df.groupby('price_band', observed=True)['sold'].agg(['count','mean','median','sum'])
print(band.round(1))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(x=band.index, y=band['mean'], ax=ax[0], color='#4C72B0'); ax[0].set_title('Average units sold by price band'); ax[0].set_ylabel('mean sold')
sns.barplot(x=band.index, y=band['median'], ax=ax[1], color='#DD8452'); ax[1].set_title('Median units sold by price band'); ax[1].set_ylabel('median sold')
plt.tight_layout(); save('05_price_band_sales')

# ### 3.5 Impact of discounts

df['has_discount'] = df['originalPrice'].notna().astype(int)
df['discount_percentage'] = ((df['originalPrice'] - df['price']) / df['originalPrice'] * 100).fillna(0).clip(lower=0)

print(df.groupby('has_discount')['sold'].agg(['count','mean','median']).round(2))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
sns.boxplot(x='has_discount', y=np.log1p(df['sold']), data=df, ax=ax[0])
ax[0].set_title('Sales: discounted vs not'); ax[0].set_xticklabels(['No discount','Discounted']); ax[0].set_ylabel('log(1 + sold)')
sns.histplot(df.loc[df.has_discount==1,'discount_percentage'], bins=30, ax=ax[1], color='#55A868')
ax[1].set_title('Discount % (discounted items)')
d = df[df.has_discount==1]
sns.scatterplot(x='discount_percentage', y=np.log1p(d['sold']), data=d, alpha=.6, ax=ax[2])
ax[2].set_title('Discount % vs log(1 + sold)'); ax[2].set_ylabel('log(1 + sold)')
plt.tight_layout(); save('06_discount_analysis')

# ### 3.6 Which words in titles go with higher sales?

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english', min_df=25, binary=True)
Xw = cv.fit_transform(df['productTitle'].str.lower())
words = cv.get_feature_names_out()
rows = []
for i, w in enumerate(words):
    m = Xw[:, i].toarray().ravel() == 1
    rows.append((w, m.sum(), df.loc[m,'sold'].median(), np.log1p(df.loc[m,'sold']).mean()))
wd = pd.DataFrame(rows, columns=['word','n_products','median_sold','mean_log_sold']).sort_values('mean_log_sold', ascending=False)
top = wd.head(12); bot = wd.tail(12)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(y='word', x='mean_log_sold', data=top, ax=ax[0], color='#55A868'); ax[0].set_title('Title words linked to HIGHER sales')
sns.barplot(y='word', x='mean_log_sold', data=bot, ax=ax[1], color='#C44E52'); ax[1].set_title('Title words linked to LOWER sales')
for a in ax: a.set_xlabel('mean log(1 + sold) of products containing the word')
plt.tight_layout(); save('07_title_keywords')
wd.head(12)

# ### 3.7 Top sellers and correlations

top10 = df.nlargest(10, 'sold')[['productTitle','price','sold','tagText']].copy()
top10['productTitle'] = top10['productTitle'].str.slice(0, 70) + '...'
fig, ax = plt.subplots(figsize=(11, 5))
sns.barplot(y=top10['productTitle'], x=top10['sold'], ax=ax, color='#4C72B0')
ax.set_title('Top 10 best-selling products'); ax.set_xlabel('Units sold'); ax.set_ylabel('')
plt.tight_layout(); save('08_top10_products')
top10

corr_df = df[['price','sold','has_discount','discount_percentage','shipping_cost']].copy()
corr_df['log_sold'] = np.log1p(corr_df['sold']); corr_df['log_price'] = np.log1p(corr_df['price'])
plt.figure(figsize=(7.5, 6))
sns.heatmap(corr_df.corr(method='spearman'), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Spearman correlation matrix'); plt.tight_layout(); save('09_correlation_heatmap')

# ## 4. Feature Engineering
# * `discount_percentage` and `has_discount` (from `originalPrice` and `price`)
# * `log_price`, `shipping_cost`, one-hot shipping type
# * Title length and word count
# * **TF-IDF** (100 features) from `productTitle`
# 
# **Target:** `sold` is heavily skewed (max 10,000 vs median 3), so we train on `log1p(sold)` and convert predictions back with `expm1` for reporting.

df['title_len'] = df['productTitle'].str.len()
df['title_words'] = df['productTitle'].str.split().str.len()
df['log_price'] = np.log1p(df['price'])
ship = pd.get_dummies(df['shipping_type'], prefix='ship', dtype=int)

tfidf = TfidfVectorizer(max_features=100, stop_words='english')
T = tfidf.fit_transform(df['productTitle'])
tfidf_df = pd.DataFrame(T.toarray(), columns=['tf_' + w for w in tfidf.get_feature_names_out()], index=df.index)

num_cols = ['price','log_price','has_discount','discount_percentage','shipping_cost','title_len','title_words']
X = pd.concat([df[num_cols], ship, tfidf_df], axis=1)
y = df['sold']; y_log = np.log1p(y)
print('Feature matrix:', X.shape)

# ## 5. Model Selection & Training
# 80/20 train-test split (`random_state=42`). Models: Linear Regression and Random Forest (as in the guide), plus Gradient Boosting and a mean-predictor baseline for context.

X_train, X_test, ytr_log, yte_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
y_test = np.expm1(yte_log)

models = {
    'Baseline (mean)': DummyRegressor(strategy='mean'),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=.8, random_state=42),
}
preds = {}
for name, m in models.items():
    m.fit(X_train, ytr_log)
    preds[name] = m.predict(X_test)
print('Trained:', list(models))

# ## 6. Model Evaluation
# Metrics are reported on the **log scale** (the scale the models are trained on) and on the **original units-sold scale**. The raw-scale MSE is dominated by a few extreme sellers (e.g. 10,000 units), so the log-scale numbers are the fairer comparison.

rows = []
for name, p in preds.items():
    p_units = np.clip(np.expm1(p), 0, None)
    rows.append({
        'Model': name,
        'MSE (log)': mean_squared_error(yte_log, p),
        'RMSE (log)': np.sqrt(mean_squared_error(yte_log, p)),
        'MAE (log)': mean_absolute_error(yte_log, p),
        'R2 (log)': r2_score(yte_log, p),
        'MSE (units)': mean_squared_error(y_test, p_units),
        'MAE (units)': mean_absolute_error(y_test, p_units),
        'R2 (units)': r2_score(y_test, p_units),
    })
results = pd.DataFrame(rows).set_index('Model').round(3)
results.to_csv('model_results.csv')
results

print(f"Linear Regression  -> MSE(log): {results.loc['Linear Regression','MSE (log)']}, R2(log): {results.loc['Linear Regression','R2 (log)']}")
print(f"Random Forest      -> MSE(log): {results.loc['Random Forest','MSE (log)']}, R2(log): {results.loc['Random Forest','R2 (log)']}")
print(f"Gradient Boosting  -> MSE(log): {results.loc['Gradient Boosting','MSE (log)']}, R2(log): {results.loc['Gradient Boosting','R2 (log)']}")

plot_res = results.drop('Baseline (mean)')
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
plot_res['MSE (log)'].plot(kind='bar', ax=ax[0], color=['#4C72B0','#55A868','#DD8452']); ax[0].set_title('MSE on log scale (lower is better)')
plot_res['R2 (log)'].plot(kind='bar', ax=ax[1], color=['#4C72B0','#55A868','#DD8452']); ax[1].set_title('R² on log scale (higher is better)')
ax[1].axhline(0, color='k', lw=.8)
for a in ax: a.set_xlabel(''); a.tick_params(axis='x', rotation=20)
plt.tight_layout(); save('10_model_comparison')

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
for a, name in zip(ax, ['Linear Regression','Random Forest','Gradient Boosting']):
    a.scatter(yte_log, preds[name], alpha=.5, s=18)
    lim = [0, max(yte_log.max(), preds[name].max())]
    a.plot(lim, lim, 'r--', lw=1)
    a.set_title(f'{name}: actual vs predicted'); a.set_xlabel('Actual log(1 + sold)'); a.set_ylabel('Predicted log(1 + sold)')
plt.tight_layout(); save('11_actual_vs_predicted')

best = results.drop('Baseline (mean)')['R2 (log)'].idxmax()
print('Best model by log-scale R²:', best)
res = yte_log - preds[best]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(res, bins=40, kde=True, ax=ax[0]); ax[0].set_title(f'{best}: residual distribution')
ax[1].scatter(preds[best], res, alpha=.5, s=18); ax[1].axhline(0, color='r', ls='--')
ax[1].set_title('Residuals vs predicted'); ax[1].set_xlabel('Predicted'); ax[1].set_ylabel('Residual')
plt.tight_layout(); save('12_residuals')

rf = models['Random Forest']
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
plt.figure(figsize=(8, 6))
sns.barplot(x=imp.values, y=imp.index, color='#55A868')
plt.title('Random Forest — top 20 feature importances'); plt.xlabel('Importance')
plt.tight_layout(); save('13_feature_importance')
imp.round(4).to_frame('importance')

# ## 7. Conclusion

bl = results.loc['Baseline (mean)','R2 (log)']
print('Findings')
print('-'*60)
print(f"1. {(df.sold==0).mean()*100:.0f}% of listings have zero sales; median sold = {df.sold.median():.0f}, mean = {df.sold.mean():.1f} (driven by a few viral items).")
print(f"2. {(df.shipping_type=='Free shipping').mean()*100:.0f}% of listings offer free shipping.")
print(f"3. Discounted listings: median sold {df[df.has_discount==1].sold.median():.0f} vs {df[df.has_discount==0].sold.median():.0f} for non-discounted.")
print(f"4. Best model: {best} (log-scale R² = {results.loc[best,'R2 (log)']}, baseline = {bl}).")
print(f"5. Most important features: {', '.join(imp.index[:5])}")

