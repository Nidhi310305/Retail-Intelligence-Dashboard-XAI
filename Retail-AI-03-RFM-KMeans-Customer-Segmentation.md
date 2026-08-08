# 📊 Day XX — Customer Segmentation with K-Means & RFM

> *Today's focus wasn't just running K-Means. It was understanding **why every preprocessing step exists** before clustering customers.*

---

## 🎯 Goal

Build a customer segmentation pipeline using **RFM Analysis + K-Means Clustering**, while understanding every preprocessing decision instead of treating it as a black box.

---

# 🧩 Step 1 — Creating RFM Features

Instead of clustering raw transaction records, I learned why businesses first convert transactional data into customer behaviour.

### RFM Metrics

| Metric | Meaning | Business Interpretation |
|---------|----------|------------------------|
| **Recency** | Days since last purchase | Lower is better |
| **Frequency** | Number of purchases | Higher is better |
| **Monetary** | Total spending | Higher is better |

---

### Building the RFM Table

I understood how Pandas creates one record per customer.

```python
dataset.groupby("Customer Name").agg(...)
```

Instead of storing every order separately,

```
Alice
Order 1
Order 2
Order 3
```

becomes

```
Alice

Recency
Frequency
Monetary
```

which is exactly what K-Means needs.

---

# 📅 Understanding Reference Date

One thing that confused me initially was **why we create a reference date**.

```python
reference_date = dataset["Order Date"].max() + pd.Timedelta(days=1)
```

### My understanding

The reference date is **NOT today's date.**

Instead,

it's **one day after the latest purchase in the dataset**.

That means every customer's Recency is calculated from the **same reference point**, making the notebook reproducible even if it's run months later.

---

# 🧠 Understanding Lambda

One of the biggest "aha!" moments today was finally understanding this line.

```python
lambda x: (reference_date - x.max()).days
```

Instead of memorizing it, I understood that:

- `x` contains all order dates for one customer.
- `x.max()` picks the latest purchase.
- The difference from the reference date becomes Recency.
- `.days` converts the time difference into integer days.

---

# 📈 Before Scaling — Checking Skewness

Before jumping into StandardScaler, I learned that **feature distributions matter.**

```python
rfm.skew()
```

Output

| Feature | Skewness |
|----------|-----------|
| Recency | 2.27 |
| Frequency | 0.36 |
| Monetary | 2.47 |

---

### Rule of Thumb

- -0.5 → 0.5 → Approximately symmetric
- >1 → Highly skewed

This immediately explained why:

✅ Recency → Log Transform

✅ Monetary → Log Transform

❌ Frequency → Already symmetric → No transformation needed

This was a much better preprocessing pipeline than blindly transforming every feature.

---

# 📉 Why Log Transformation?

Initially I thought log transformation was just another preprocessing step.

Now I understand that its purpose is to:

- reduce positive skew
- compress extremely large values
- create more balanced distributions

Instead of:

```
100
500
10000
100000
```

Log transformation compresses large values while preserving order.

---

# ⚖️ Why StandardScaler Instead of MinMaxScaler?

This was another important concept.

K-Means calculates **distance**.

If one feature has much larger values,

```
Recency → 120

Frequency → 10

Monetary → ₹15,000
```

then Monetary dominates the distance calculation.

StandardScaler fixes this by making every feature have

- Mean ≈ 0
- Standard Deviation ≈ 1

allowing all three RFM features to contribute fairly.

---

# 🔍 Understanding Scaled Data

After scaling,

I checked

```python
rfm_scaled.describe()
```

Things I learned:

- Mean becomes approximately 0
- Standard deviation becomes approximately 1
- Negative values are completely normal

Example

```
Monetary_scaled = -5.52
```

does **NOT** mean the customer spent negative money.

It means the customer spent **far below the average customer.**

That was a nice conceptual clarification.

---

# 📉 Choosing the Number of Clusters

Instead of randomly selecting K,

I learned two evaluation techniques.

## 1️⃣ Elbow Method

Measures

```
WCSS (Inertia)
```

Lower WCSS means tighter clusters.

The optimal K is chosen where improvements begin slowing down.

---

## 2️⃣ Silhouette Score

Measures

- cohesion within a cluster
- separation between clusters

Range

```
-1 → +1
```

Closer to +1 indicates better clustering.

---

# 🤔 Interesting Observation

My Silhouette scores suggested

```
K = 2
```

while the Elbow Method suggested

```
K ≈ 4
```

Instead of blindly choosing the highest Silhouette score,

I learned something more valuable.

## Machine Learning isn't only about metrics.

Business interpretation matters too.

K = 2 would only produce

- Good Customers
- Bad Customers

which isn't very actionable.

K = 4 creates much richer customer segments.

---

# 🚀 Final Customer Segments

After fitting the final K-Means model,

the average RFM values helped interpret each cluster.

| Segment | Interpretation |
|-----------|----------------|
| 🏆 Champions | Purchased recently with good engagement |
| 💎 Loyal High-Value | Highest spenders and most frequent buyers |
| ⚠️ At-Risk | Valuable customers who haven't purchased recently |
| ❌ Lost / Churned | Low engagement and inactive customers |

One thing I found interesting is that **K-Means itself never knows these names.**

It only assigns

```
Cluster 0
Cluster 1
Cluster 2
Cluster 3
```

The business analyst studies the average RFM values and then gives meaningful names.

---

# 📊 Visualizing Customer Segments

Finally,

I created an interactive **3D Plotly visualization** using

- Recency
- Frequency
- Monetary

Each point represents one customer,

and the colour indicates the customer segment.

This makes customer behaviour much easier to interpret compared to looking at raw tables.

---

# 💡 Biggest Takeaways

✔ Customer segmentation starts with behaviour, not raw transactions.

✔ Reference Date keeps Recency calculations reproducible.

✔ Log transformation should only be applied when the feature is actually skewed.

✔ StandardScaler prevents one feature from dominating distance calculations.

✔ Elbow Method and Silhouette Score are guidelines, not absolute rules.

✔ Business understanding is just as important as mathematical metrics.

✔ K-Means gives clusters, while humans give those clusters business meaning.

---

## 📚 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- K-Means Clustering
- RFM Analysis

---

## 🌱 Reflection

Today's session changed how I look at preprocessing.

Earlier, I used to think scaling, log transformation, and K-Means were simply steps to execute.

Now I understand **why each preprocessing decision is made**, **how it affects clustering**, and **how machine learning outputs become business decisions instead of just model predictions.**
