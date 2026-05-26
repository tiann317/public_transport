#### Task
Data Understanding and Cleanup: Describe your dataset and give an insight into the
data you’re working with. This should also include visualization and giving an insight
into its most important features.

#### How to run
```bash
python3.12 -m venv .venv
activate .venv/bin/activate
pip install -r requirements.txt
```

```python
python main.py
```

#### The script will generate:

- visualizations in `outputs/` directory: correlation_matrix, delay_by_hour, delay_by_weekday, delay_distribution and top_delayed_stations.
- a cleaned up version of a dataset: `outputs/cleaned_dataset.csv`.
- report of the actions in a console including memory optimization, statistics, correlation and final insights about the dataset in general.
