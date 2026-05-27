![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)

# public_transport

A university data analytics mini project analysing public transport stop-level delay data.
The dataset contains planned and actual arrival/departure times per stop,
allowing delay prediction.

---

#### Tasks

**[Task 1 – Data Understanding & Cleanup](./task1_data_understanding/)**
Load raw CSV stop records, remove duplicates and outliers, include visualization and giving an insight into its most important features.
**[Task 2 – Model Comparison](./task2_model_comparison/)**
Compare different models and explain which model you have chosen and why it’s the best suited one for your project.

---

#### Dataset

Raw data lives in `dataset/` as `.csv` files.
Each row is a single planned stop event with the following key columns:

| Column | Description |
|---|---|
| `station_id` / `station_name` | Stop identifier and human-readable name |
| `line_id` | Transit line |
| `planned_arr_time` / `planned_dep_time` | Scheduled times |
| `arr_delay` / `dep_delay` | Actual delay in minutes (target variable: `dep_delay`) |

---

#### Model results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 0.4740 | 1.7996 | 0.9002 |
| Ridge Regression | 0.4740 | 1.7996 | 0.9002 |
| Decision Tree | 0.4276 | 1.1686 | 0.9579 |
| Random Forest | 0.4022 | 1.1013 | 0.9626 |
| **Gradient Boosting** ✓ | **0.4045** | **1.0225** | **0.9678** |

Gradient Boosting was chosen for its lowest RMSE and best R², handling the zero-inflated,
right-skewed target well. Top predictors: `arr_delay`, `hour`, `station_id`.

<img width="2384" height="782" alt="model_comparison" src="https://github.com/user-attachments/assets/5a178b3e-4805-4d8f-bcce-e61048e851cf" />

---

#### How to run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r task1_data_understanding/requirements.txt
```

Run the tasks in order — Task 2 depends on the cleaned dataset produced by Task 1:

```bash
python task1_data_understanding/main.py
python task2_model_comparison/main.py
```

Output files are saved to each task's `outputs/` directory.
