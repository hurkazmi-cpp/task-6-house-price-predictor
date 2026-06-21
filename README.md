# Task 6: House Price Prediction

A professional machine learning regression repository that predicts house prices based on property features such as room counts, local tax rates, and neighborhood socio-economic metrics using a classic real estate dataset. Features automated data preprocessing, feature scaling, an ensemble gradient boosting pipeline, and descriptive visualization.

## 📌 Project Objective
The core objective of this project is to construct a predictive regression model capable of estimating median housing valuations. The workflow focuses on standard engineering practices including clean feature scaling, data splitting to eliminate validation bias, training a robust ensemble learning regressor, and evaluating performance using industry-standard error tracking metrics.

## 🛠️ Tech Stack & Tools
* **Programming Language:** Python 3.x
* **Machine Learning Framework:** Scikit-Learn
* **Data Engineering Libraries:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab

## 📈 Machine Learning Workflow Pipeline
The project implements a classic end-to-end data science pipeline layout:
1. **Data Ingestion:** Fetching structural housing features from a verified public data matrix mirror.
2. **Preprocessing & Isolation:** Segregating architectural attributes (independent parameters) from valuation metrics (dependent target) and sharding rows into an 80/20 train-test ratio.
3. **Feature Scaling:** Implementing standard variance-normalization (`StandardScaler`) to eliminate value scale discrepancies across numeric arrays.
4. **Ensemble Modeling:** Initializing and training a sequentially optimizing `GradientBoostingRegressor`.
5. **Statistical Evaluation:** Tracking global error deviations using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## 📁 Repository Structure
```text
task-6-house-prediction/
│
├── House_Price_Prediction.ipynb   # Interactive Google Colab notebook with step-by-step markdown cells
└── README.md                      # Comprehensive documentation and engineering summary
