# 🛒 E-commerce Customer Churn Prediction

## 📌 Overview

This project predicts whether a customer will churn using machine learning models based on behavioral, transactional, and engagement data.

👉 Goal:
Build a reliable classification model to identify customers at risk of leaving.

---

## 🚀 Final Result

- **Best Model:** Logistic Regression (Balanced)
- **Accuracy:** ~0.97
- **Recall (Churn):** **0.97** ✅
- **F1 Score:** ~0.91

👉 Optimized for **maximum churn detection (high recall)**

---

## 🧠 Key Learnings

Through this project, I learned:

- Data cleaning and preprocessing
- Handling categorical encoding issues (case sensitivity, spaces)
- Dealing with imbalanced datasets
- Model comparison and evaluation
- Feature importance interpretation
- Business-driven model selection

---

## 📂 Dataset

Dataset link: [text](https://www.kaggle.com/datasets/vishardmehta/e-commerce-customer-churn-prediction-dataset)
The dataset consists of two files:

- `ecommerce_customer_features.csv` → Customer behavior data
- `ecommerce_customer_targets.csv` → Churn labels

After merging:

- Total samples: **6000**
- Features: **14**
- Target: **churned**

---

## ⚙️ Project Workflow (Step-by-Step)

### 1️⃣ Data Loading

- Loaded features and target datasets
- Merged using `Customer_ID`

---

### 2️⃣ Data Cleaning

- Removed unnecessary column (`Customer_ID`)
- Fixed categorical values:
  - Lowercased
  - Removed extra spaces

---

### 3️⃣ Encoding

Converted categorical features:

```python
churned → {no: 0, yes: 1}
loyalty_member → {no: 0, yes: 1}
```

---

### 4️⃣ Handling Missing Values

- Checked missing values using `.isnull().sum()`
- Filled missing numerical values with **median**

---

### 5️⃣ Train-Test Split

```python
train_test_split(test_size=0.2, random_state=42)
```

- Train: 4800 samples
- Test: 1200 samples

---

### 6️⃣ Models Used

### 🔹 Logistic Regression

- Baseline model
- Simple and interpretable

---

### 🔹 Logistic Regression (Balanced)

- Used `class_weight="balanced"`
- Handles class imbalance
- Best recall performance

---

### 🔹 Random Forest

- Captures non-linear patterns
- High precision model

---

### 🔹 XGBoost

- Gradient boosting model
- Balanced performance across metrics

---

### 7️⃣ Model Evaluation

Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score

---

### 📊 Model Performance

| Model               | Precision | Recall   | F1 Score |
| ------------------- | --------- | -------- | -------- |
| Logistic (balanced) | 0.85      | **0.97** | 0.91     |
| Random Forest       | **0.95**  | 0.87     | 0.90     |
| XGBoost             | 0.91      | 0.90     | 0.90     |

---

## 🏆 Final Model Selection

**Selected Model: Logistic Regression (Balanced)**

### Why?

- Maximizes **recall (97%)**
- Detects almost all churners
- Ideal for retention-focused business strategy

---

## 💼 My Final Insight

I compared multiple models and selected Logistic Regression with class balancing because it maximizes recall (97%), ensuring minimal customer loss.

---

## 📊 Feature Importance (Random Forest)

Top features influencing churn:

1. **engagement_score** (~43%)
2. **days_since_last_purchase** (~39%)
3. satisfaction_score
4. browsing_frequency_per_week
5. product_review_score_avg

👉 These two features explain over **80% of churn behavior**

---

## 💼 Business Insights

- Customers with **low engagement** are highly likely to churn
- **Inactive customers** (long time since last purchase) are at highest risk
- Satisfaction and browsing behavior also influence churn

---

## 📁 Project Structure

```text
ecommerce-churn-predictor/
│
├── data/
│   └── raw/
│       ├── ecommerce_customer_features.csv
│       └── ecommerce_customer_targets.csv
│
├── models/ e-commerce_churn_predictor.py
├── README.md
└── notebooks/e-commerce_churn_predictor.ipynb
```

---

## ▶️ How to Run

```bash
pip install pandas scikit-learn xgboost

py ./models/e-commerce_churn_predictor.py
```

---

## 🧠 Key Insights

- Recall is more important than accuracy in churn problems
- Class imbalance must be handled carefully
- Feature importance reveals real business drivers
- Simpler models can outperform complex ones when tuned properly

---

## 🙌 Conclusion

This project demonstrates a complete ML workflow:

```text
Data → Cleaning → Encoding → Modeling → Evaluation → Business Insight
```

---

## 🔗 Author

**[MD Shafayetur Rahman]**
