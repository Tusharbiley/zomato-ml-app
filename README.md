# 🍽 Restaurant Profitability Prediction & Customer Segmentation System

### End-to-End Machine Learning & Business Analytics Platform using Zomato Dataset

---

# 📌 Project Overview

This project is a complete end-to-end Machine Learning application designed to simulate a real-world restaurant analytics platform similar to Zomato or Swiggy.

The system combines:

- 📊 Exploratory Data Analysis (EDA)
- 🤖 Supervised Machine Learning
- 🔵 Unsupervised Clustering
- ⚙ Hyperparameter Optimization
- 💰 Business Profitability Scoring
- 🌐 Interactive Streamlit Dashboard

The main goal is to help identify:
- High-performing restaurants
- Low-visibility businesses
- Market segments
- Profitability potential
- Business growth opportunities

---

# 🎯 Problem Statement

Restaurant businesses generate large amounts of operational and customer data. However, converting this data into actionable business intelligence remains a challenge.

This system solves that problem by:

1. Predicting restaurant ratings using Machine Learning
2. Computing a custom Profitability Score (0–100)
3. Segmenting restaurants into business clusters
4. Providing insights and recommendations for revenue growth

---

# 🧠 Complete ML Workflow

```text
Raw Dataset
     ↓
Data Cleaning & Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Hyperparameter Optimization
     ↓
K-Means Clustering
     ↓
Profitability Scoring
     ↓
Interactive Streamlit Deployment
```

---

# 📁 Project Structure

```text
zomato_ml/
│
├── app.py                  # Streamlit Web Application
├── main.py                 # Main ML pipeline
│
├── data_loading.py         # Dataset loading & synthetic dataset generation
├── preprocessing.py        # Cleaning, encoding, scaling
├── eda.py                  # Exploratory Data Analysis
├── models.py               # ML model training
├── evaluation.py           # Metrics & visual evaluation
├── clustering.py           # K-Means clustering
├── optimization.py         # GridSearchCV optimization
├── profitability.py        # Profitability scoring system
│
├── models/                 # Saved trained models
├── plots/                  # Generated visualization images
├── requirements.txt
└── README.md
```

---

# 📊 Dataset Information

The project uses the Zomato Restaurant Dataset.

### Dataset Features

| Feature | Description |
|---|---|
| city | Restaurant city |
| location | Restaurant locality |
| cuisines | Cuisine type |
| restaurant_type | Dining category |
| cost_for_two | Average cost |
| votes | Customer votes |
| online_order | Online ordering availability |
| book_table | Table booking availability |
| rating | Restaurant rating (target variable) |

---

# 🧹 Data Cleaning & Preprocessing

The preprocessing pipeline performs:

- Removal of duplicate records
- Missing value handling
- Median imputation for numeric features
- Mode imputation for categorical features
- Binary encoding of Yes/No columns
- One-Hot Encoding for categorical variables
- Feature scaling using `StandardScaler`
- Train/Test splitting (80/20)

### Additional Optimizations

- High-cardinality categorical reduction
- Outlier handling using IQR-based capping
- Final NaN sweep before training

---

# 📈 Exploratory Data Analysis (EDA)

The project contains multiple business-oriented visualizations including:

### Distribution Analysis
- Rating Distribution
- Cost-for-Two Distribution
- Votes Distribution

### Category Analysis
- Top Cuisines
- Restaurant Types
- City-wise Restaurant Counts

### Relationship Analysis
- Rating vs Cost
- Votes vs Rating
- Online Order vs Rating
- Table Booking vs Rating

### Statistical Analysis
- Correlation Heatmap
- Trend Analysis

---

# 🤖 Machine Learning Models

The following regression models were implemented and compared:

| Model | Purpose |
|---|---|
| Linear Regression | Baseline model |
| Ridge Regression | Regularized linear model |
| Decision Tree Regressor | Non-linear learning |
| Random Forest Regressor | Ensemble learning |
| Gradient Boosting Regressor | Boosted ensemble |

---

# 🏆 Best Model

### ✅ Random Forest Regressor

The Random Forest model achieved the best overall performance and was selected as the production model.

### Evaluation Metrics

| Metric | Value |
|---|---|
| MAE | ~0.17 |
| RMSE | ~0.27 |
| R² Score | ~0.50 |
| Overfitting | Controlled |

---

# 📉 Model Evaluation Features

The system includes:

- MAE comparison
- RMSE comparison
- R² comparison
- Train vs Test R² analysis
- Actual vs Predicted visualization
- Feature importance visualization
- Overfitting detection

---

# ⚙ Hyperparameter Optimization

GridSearchCV was used to optimize the Random Forest model.

### Optimization Includes:
- Number of estimators
- Maximum depth
- Minimum samples split
- Minimum samples leaf

### Validation Method
- 5-Fold Cross Validation

---

# 🔵 Unsupervised Learning — Restaurant Clustering

The project uses K-Means Clustering to identify restaurant market segments.

---

# 🧩 Cluster Analysis Pipeline

1. Feature selection
2. Standard scaling
3. Elbow method
4. Silhouette analysis
5. K-Means clustering
6. PCA dimensionality reduction
7. Cluster business profiling

---

# 📌 Business Cluster Labels

| Cluster Type | Description |
|---|---|
| 🌟 Premium Performers | High rating + high pricing |
| 💚 Budget Stars | Affordable & highly rated |
| ⚠ Underperformers | Popular but weak ratings |
| 🆕 Low Visibility | Low engagement restaurants |
| 🍽 Mid-Tier Mainstream | Average market performers |

---

# 💰 Profitability Prediction System

One of the major highlights of the project is the custom Profitability Scoring Engine.

The system computes a **Profitability Score (0–100)** using business signals.

---

# 📊 Profitability Formula

| Component | Weight |
|---|---|
| ⭐ Predicted Rating | 40% |
| 📈 Customer Demand (Votes) | 25% |
| 💰 Price Positioning | 20% |
| 📱 Digital Presence | 15% |

---

# 🟢 Profitability Tiers

| Score Range | Tier |
|---|---|
| 80–100 | High Profit Potential |
| 60–79 | Moderate Potential |
| 40–59 | Low Potential |
| 0–39 | At Risk |

---

# 🌐 Streamlit Web Application

The project is deployed as a fully interactive Streamlit dashboard.

### Features

✅ Dark-themed professional UI  
✅ Interactive ML prediction system  
✅ Real-time profitability prediction  
✅ Business recommendations  
✅ Dynamic EDA visualizations  
✅ Cluster analysis dashboard  
✅ Model comparison system  
✅ KPI cards & analytics panels

---

# 📷 Dashboard Modules

### 🏠 Home
- Project overview
- KPI cards
- Best model summary
- Sample dataset preview

### 📊 EDA & Insights
- Distribution analysis
- Correlation analysis
- Business insights

### 🧠 Model Results
- Model comparison table
- Overfitting analysis
- Actual vs predicted plots

### 🔵 Clusters
- PCA visualization
- Elbow analysis
- Cluster profiles

### 💰 Profitability Predictor
- Live prediction interface
- Profitability scoring
- Revenue estimation
- Recommendations

---

# 📦 Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| ML Libraries | Scikit-Learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Optimization | GridSearchCV |
| Deployment | Streamlit Cloud |
| Version Control | Git & GitHub |

---

# 🚀 How to Run Locally

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 3️⃣ Run Complete ML Pipeline

```bash
python main.py
```

---

## 4️⃣ Skip Slow Hyperparameter Tuning

```bash
python main.py --skip-tune
```

---

# 📌 Key Learning Outcomes

This project demonstrates:

- End-to-end ML pipeline development
- Business-oriented analytics
- Regression modeling
- Ensemble learning
- Hyperparameter optimization
- Clustering techniques
- PCA visualization
- Model explainability
- Dashboard deployment
- Real-world problem solving

---

# 🔮 Future Improvements

- NLP-based sentiment analysis on reviews
- Real-time API integration
- Deep Learning models
- Restaurant recommendation system
- Time-series demand forecasting
- Cloud database integration
- User authentication system

---

# 👨‍💻 Author

**Tushar Biley**  
Machine Learning Project