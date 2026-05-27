#### Task
Model Comparison: Train and evaluate multiple regression models to predict public transport
departure delays. Compare their performance using cross-validation and hold-out test metrics,
and select the best model with justification.

#### Prerequisite
This task depends on the output of `task1_data_understanding`. Run Task 1 first to generate
`task1_data_understanding/outputs/cleaned_dataset.csv`.

#### How to run
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r ../task1_data_understanding/requirements.txt
```

```python
python main.py
```

#### The script will generate:

- visualizations in `outputs/` directory: `model_comparison.png` (test-set MAE/RMSE/R² bar charts), `cv_comparison.png` (5-fold cross-validation MAE per model), `feature_importance.png` (top feature importances for the chosen Gradient Boosting model), and `residual_analysis.png` (predicted vs. residual scatter and residual distribution).
- a CSV summary of all model metrics: `outputs/model_comparison_results.csv`.
- a console report including problem definition, data preparation steps, per-model evaluation results, and a final conclusion with justification for the chosen model.

#### Models compared

- Linear Regression
- Ridge Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor *(chosen model)*

#### Chosen model: Gradient Boosting Regressor

Selected for achieving the best MAE, RMSE, and R² on the hold-out test set. It handles the zero-inflated, right-skewed delay distribution without normality assumptions and captures non-linear interactions (e.g. rush-hour × station).
