# life-factors-and-student-performace

This project explores how high school students’ lifestyle and personal habits relate to academic success or risk. Using the UCI Student Performance dataset, we apply machine learning to predict whether a student is **at risk of underperforming** based on non-academic indicators like **alcohol consumption**, **free time**, **absences**, and more.

## 🗂 Project Structure
```
├── data/ # Raw, interim, and processed data files
├── notebooks/ # EDA and model development notebooks
├── src/ # Python modules for ETL, features, training, etc.
├── models/ # Serialized models (.pkl)
├── reports/ # Evaluation results and visualizations
├── README.md # You're here!
├── pyproject.toml # Dependencies and project metadata
└── uv.lock # Lock file managed by uv
```


## Dataset

- **Source:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Size:** 649 student records from two Portuguese secondary schools
- **Features:** 33 variables including:
  - Demographics (age, gender, parental education)
  - Lifestyle factors (alcohol use, free time, going out)
  - School-related (study time, absences, support programs)
- **Target:** Final course grade (G3), binned into classes for classification

## Goals

- Identify students at academic risk using interpretable ML models
- Explore lifestyle and behavioral predictors of poor performance
- Compare baseline (logistic regression) with advanced models (Random Forest, XGBoost)
- Optionally deploy as a FastAPI service for inference

