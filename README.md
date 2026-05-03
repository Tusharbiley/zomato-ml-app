# 🍽 Restaurant Profitability Prediction & Customer Segmentation
### Machine Learning System using Zomato Dataset

---

## 📌 Project Overview

This is a complete end-to-end Machine Learning application that simulates a real-world restaurant analytics system for a food delivery platform like Zomato.

**Objective:** Predict restaurant ratings (proxy for profitability) and segment restaurants into meaningful customer clusters to enable data-driven decisions.

---

## 🧠 ML Pipeline (4 Phases)

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `data_loading.py` + `preprocessing.py` + `eda.py` | Data understanding, cleaning, EDA |
| 2 | `models.py` + `evaluation.py` | Baseline & advanced model training |
| 3 | `clustering.py` + `optimization.py` | Unsupervised segmentation + hyperparameter tuning |
| 4 | `main.py` | Final pipeline, model comparison, prediction |

---

## 📁 Project Structure

```
zomato_ml/
├── data_loading.py      # Dataset loading (real CSV or synthetic)
├── preprocessing.py     # Cleaning, encoding, scaling, train/test split
├── eda.py               # 10 EDA visualisations with insights
├── models.py            # Linear Reg, Ridge, Decision Tree, RF, GBM
├── evaluation.py        # MAE, RMSE, R², overfitting analysis, plots
├── clustering.py        # K-Means, Elbow Method, PCA visualisation
├── optimization.py      # GridSearchCV for Random Forest
├── main.py              # Full pipeline + CLI predictor
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── plots/               # All generated visualisation PNGs
└── models/              # Saved .pkl model files
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python main.py
```

### 3. Run pipeline (skip slow GridSearchCV)
```bash
python main.py --skip-tune
```

### 4. Use your own Zomato CSV dataset
```bash
python main.py --dataset path/to/zomato.csv
```

### 5. Run interactive predictor only
```bash
python main.py --predict
```

---

## 📊 Dataset

The system works with or without a real dataset:
- **With real data:** Download from [Kaggle – Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) and pass via `--dataset`
- **Without real data:** A synthetic 1,500-row dataset is auto-generated that mirrors the Zomato schema

**Features used:**
| Feature | Type | Description |
|---------|------|-------------|
| `city` | categorical | City of the restaurant |
| `location` | categorical | Neighbourhood |
| `cuisines` | categorical | Primary cuisine |
| `cost_for_two` | numeric (₹) | Average meal cost |
| `online_order` | binary | Online ordering available |
| `book_table` | binary | Table booking available |
| `votes` | numeric | Number of user ratings |
| `restaurant_type` | categorical | QSR / Casual / Fine Dining etc. |
| **`rating`** | **target** | **Aggregate rating (1.0–5.0)** |

---

## 🤖 Models Implemented

| Model | Type | Purpose |
|-------|------|---------|
| Linear Regression | Baseline | Interpretable, fast |
| Ridge Regression | Regularised linear | Prevents overfitting |
| Decision Tree | Tree-based | Non-linear patterns |
| Random Forest | Ensemble | Primary production model |
| Gradient Boosting | Boosted ensemble | Advanced accuracy |
| Random Forest (Tuned) | GridSearchCV | Optimised best model |

---

## 📈 Evaluation Metrics

- **MAE** – Mean Absolute Error
- **RMSE** – Root Mean Squared Error  
- **R²** – Coefficient of Determination (main ranking metric)
- **Train R² vs Test R²** – Overfitting detection

---

## 📦 Plots Generated

| # | File | Description |
|---|------|-------------|
| 01 | `rating_distribution.png` | Histogram of ratings |
| 02 | `cost_distribution.png` | Cost-for-two histogram |
| 03 | `top_cuisines.png` | Top-10 cuisine bar chart |
| 04 | `city_counts.png` | Restaurant count by city |
| 05 | `online_order_vs_rating.png` | Boxplot: online order vs rating |
| 06 | `booking_vs_rating.png` | Boxplot: table booking vs rating |
| 07 | `rating_vs_cost.png` | Scatter: rating vs cost |
| 08 | `votes_vs_rating.png` | Scatter: votes vs rating |
| 09 | `correlation_heatmap.png` | Numeric feature correlations |
| 10 | `restaurant_type.png` | Pie chart of restaurant types |
| 11 | `model_comparison.png` | MAE/RMSE/R² bar comparison |
| 12 | `overfitting_analysis.png` | Train vs test R² |
| 13 | `actual_vs_predicted.png` | Best model predictions |
| 14 | `feature_importance.png` | Top-20 features (RF) |
| 15 | `elbow_silhouette.png` | K-Means optimal k selection |
| 16 | `cluster_pca.png` | PCA 2-D cluster visualisation |
| 17 | `cluster_profiles.png` | Per-cluster average stats |
| 18 | `optimization_comparison.png` | Before/after GridSearchCV |
| 19 | `cv_results.png` | GridSearchCV score distribution |

---

## 🧩 Business Cluster Labels

| Cluster | Label | Description |
|---------|-------|-------------|
| A | 🌟 Premium High-Performers | High rating + high cost, fine dining |
| B | 💚 Budget Stars | High rating + affordable, best value |
| C | ⚠ Popular but Underperforming | High votes but mediocre rating |
| D | 🆕 New / Low Visibility | Low votes, needs marketing |
| E | 🍽 Mid-Tier Mainstream | Average on all dimensions |

---

## 📝 Academic Submission

- **Report:** `FirstName_RollNo_MLProject.pdf`  
- **Code zip:** `FirstName_RollNo_MLProject_Code.zip`

---

## 🔮 Future Scope

1. Deploy as a Streamlit web app for real-time predictions
2. Incorporate NLP on customer reviews (sentiment analysis)
3. Add time-series forecasting for seasonal demand
4. Integrate real-time Zomato API data
5. Build a restaurant recommendation engine using collaborative filtering
