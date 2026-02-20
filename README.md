# 🌦️ Delhi Weather Forecasting & AQI Predictive Modeling
> **An end-to-end Machine Learning pipeline utilizing Linear Regression to predict next-day temperature fluctuations in Delhi (2025).**

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Data-Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Complete-success)]()

---

## 📖 Project Overview
This repository demonstrates a **Supervised Learning** approach to meteorology. By analyzing historical atmospheric data—including **AQI (Air Quality Index)**, humidity, and pressure—this model identifies linear relationships to forecast the next day's thermal state. 

This project goes beyond simple curve fitting; it implements a full **Data Engineering cycle**, from temporal sorting to lead-target feature generation.

---

## 🧠 Architectural Workflow



### 1. Data Engineering & Preprocessing
* **Temporal Alignment**: Converted raw `date_ist` strings into high-precision `datetime` objects and performed an ascending sort to maintain chronological integrity.
* **Lead-Target Generation**: Utilized the `.shift(-1)` operator to create a "look-ahead" target variable (`tomorrow_temp`), transforming a static dataset into a time-series forecasting problem.
* **Feature Selection**: Optimized the model using a 5-dimensional feature space:
    - `temp_c`: Current temperature baseline.
    - `aqi_index`: Pollution-level correlation.
    - `humidity` & `pressure_mb`: Atmospheric stability indicators.
    - `windspeed_kph`: Convection factor.

### 2. Model Architecture
* **Algorithm**: Ordinary Least Squares (OLS) Linear Regression.
* **Train/Test Split**: 80-20 stratified-style split (via `random_state=42` for reproducibility).
* **Validation Strategy**: Evaluation via **Mean Absolute Error (MAE)** to quantify the average degrees of variance in Celsius.

---

## 📊 Performance Analysis
The model's accuracy is visualized through an **Actual vs. Predicted** scatter plot. The proximity of data points to the identity line (Red) indicates the high reliability of the Linear Regression model in stable atmospheric conditions.



| Metric | Score |
| :--- | :--- |
| **Model Type** | Linear Regression |
| **Primary Metric** | Mean Absolute Error (MAE) |
| **Training Split** | 80% (Historical Data) |
| **Testing Split** | 20% (Unseen Data) |

---

## 🛠️ Installation & Execution

### 1. Clone & Setup
```bash
git clone [https://github.com/jaweriashakoor/Delhi-Weather-AQI-Forecasting.git](https://github.com/jaweriashakoor/Delhi-Weather-AQI-Forecasting.git)
cd Delhi-Weather-AQI-Forecasting
