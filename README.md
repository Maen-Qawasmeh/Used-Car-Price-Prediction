# Used Car Price Prediction with Machine Learning

A machine learning regression project that predicts used car prices from vehicle details and location information. The project compares eight regression models, evaluates their performance, and generates a CSV file containing predictions for a separate competition test set.

The complete workflow is implemented in **`Used Car Price Prediction Competition.ipynb`**.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Project Workflow](#project-workflow)
- [Models and Results](#models-and-results)
- [Generated Outputs](#generated-outputs)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)

## Project Overview

Used car prices depend on factors such as manufacturing year, mileage, manufacturer, vehicle model, and transmission. This project explores how different regression algorithms use these features to estimate a vehicle's price.

The main goals are to:

- Clean a large dataset of used car listings.
- Handle missing values and extreme values.
- Convert categorical features into numerical values.
- Compare linear, distance-based, tree-based, and ensemble models.
- Evaluate predictions using regression metrics.
- Export predicted prices for the competition test set.

This is an experimental, notebook-based project. The final prediction section uses **AdaBoost**, while **Random Forest** achieves the same displayed R² score in the saved comparison results.

## Dataset

The notebook expects two CSV files in its working directory:

| File | Purpose |
| --- | --- |
| `Train_Set.csv` | Training data, including the target column `price`. |
| `Test-Set.csv` | Competition test data used to generate price predictions. |

**These CSV files were not included with the supplied notebook.** Obtain them from the original competition or dataset provider before running the project. The notebook does not identify a source URL, competition platform, or dataset license.

### Expected Columns

| Column | Description |
| --- | --- |
| `id` | Listing identifier. |
| `year` | Vehicle manufacturing year. |
| `manufacturer` | Vehicle manufacturer. |
| `model` | Vehicle model. |
| `condition` | Reported vehicle condition; removed during preprocessing. |
| `cylinders` | Engine cylinder information. |
| `fuel` | Fuel type. |
| `odometer` | Vehicle mileage reading. |
| `title_status` | Vehicle title status. |
| `transmission` | Transmission type. |
| `drive` | Drive configuration. |
| `size` | Vehicle size category; removed during preprocessing. |
| `type` | Vehicle type or body style. |
| `paint_color` | Exterior color; removed during preprocessing. |
| `state` | Listing state. |
| `lat`, `long` | Geographic coordinates. |
| `posting_date` | Listing date; removed during preprocessing. |
| `price` | Target value, required in the training data. |

Columns containing `Unnamed` in their names are removed when the CSV files are loaded. The notebook assumes that the remaining columns match the expected schema and order. Currency and mileage units are not documented in the notebook.

## Project Structure

Place the files in the same project folder:

```text
used-car-price-prediction/
├── README.md
├── Used Car Price Prediction Competition.ipynb
├── Train_Set.csv                 # Required input; obtain separately
├── Test-Set.csv                  # Required input; obtain separately
├── Overall-Performance.jpg       # Generated after execution
└── price_predict.csv             # Generated after execution
```

The notebook is the main entry point. There is no standalone application, saved model, or inference API in the supplied project.

## Requirements

- Python. The supplied notebook metadata records **Python 3.12.12**.
- JupyterLab or another environment that can run Jupyter notebooks.
- The following Python packages:

| Package | Use |
| --- | --- |
| `numpy` | Numerical operations and percentile calculations. |
| `pandas` | CSV loading, data cleaning, and prediction export. |
| `matplotlib` | Model performance visualization. |
| `seaborn` | Imported visualization library. |
| `scikit-learn` | Preprocessing, regression models, and evaluation metrics. |
| `jupyterlab` | Opening and running the notebook locally. |

Exact package versions are not recorded in the notebook, and a dependency lock file is not supplied. Its use of `BaggingRegressor(estimator=...)` requires a scikit-learn version that supports that parameter. Several pandas operations use chained `fillna(..., inplace=True)` calls; see the compatibility note below.

## How to Run

### 1. Download the Project

Download the notebook and this README from GitHub into a local folder. Place `Train_Set.csv` and `Test-Set.csv` in that same folder.

Open a terminal in the project folder.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyterlab
```

This installs the dependencies without reproducing the original package versions. If a newer pandas version does not apply an in-place fill as expected, replace calls such as:

```python
trainData['fuel'].fillna(fuel_mode, inplace=True)
```

with explicit assignment:

```python
trainData['fuel'] = trainData['fuel'].fillna(fuel_mode)
```

Apply the same change to equivalent calls in the training and test preprocessing sections.

### 4. Start JupyterLab

```bash
jupyter lab
```

Open **`Used Car Price Prediction Competition.ipynb`**, select the Python environment containing the installed packages, and run the cells from top to bottom. Restart the kernel before a complete rerun to avoid reusing modified data or model state.

The CSV paths are relative to the notebook's working directory. Keep the filenames exactly as shown, including the underscore in `Train_Set.csv` and the hyphen in `Test-Set.csv`.

### 5. Review the Results

After successful execution, inspect the printed metrics, model comparison plot, and generated `price_predict.csv` file.

Training multiple ensembles on hundreds of thousands of rows can take substantial time and memory. Bagging and Random Forest use `n_jobs=-1`, allowing them to use all available CPU cores.

## Project Workflow

### 1. Load and Clean the Training Data

- Read `Train_Set.csv` and remove `Unnamed` columns.
- Drop `size`, `condition`, `paint_color`, and `posting_date`.
- Remove rows with more than four missing values after those columns are dropped.
- Remove duplicate rows and reset the index.

The actual code retains `id` as a model feature, even though an earlier markdown heading also mentions dropping it.

### 2. Handle Missing Values

- **Manufacturer:** infer from vehicle model where possible, then use the overall mode.
- **Drive and cylinders:** use the most frequent value for each vehicle model, with an overall mode fallback.
- **Vehicle type:** infer from a model-to-type mapping, then use the overall mode.
- **Fuel and title status:** fill with the overall mode.
- **Latitude and longitude:** fill with state-level means, followed by overall means for remaining missing values in the training data.

### 3. Filter Extreme Values

- Remove listings with an odometer reading of zero or the maximum observed reading.
- Keep odometer readings between the **0.5th and 99.9th percentiles**.
- Keep prices between the **8th and 99.9th percentiles**.

These thresholds are calculated from the loaded training data, so their numeric values depend on the dataset.

### 4. Encode and Scale

The notebook applies label encoding to `manufacturer`, `model`, `cylinders`, `fuel`, `title_status`, `transmission`, `drive`, `type`, and `state`.

It standardizes **odometer** and the target **price** using `StandardScaler`. Other numerical features are not standardized. The same scaler variable is fitted first on odometer and then refitted on price; its final state is therefore used to convert predicted prices back to the original target scale.

### 5. Split and Train

The cleaned data is split into:

- **80% training data**.
- **20% held-out validation data**, stored as `X_test` and `y_test`.
- A split seed of **`random_state=42`**.

This held-out subset is separate from the competition file `Test-Set.csv`. The notebook also trains an initial 20-tree Random Forest baseline before the main eight-model comparison.

### 6. Evaluate the Models

The notebook reports:

- **R²:** how well predictions explain variation in the target; higher is better.
- **MAE:** mean absolute prediction error; lower is better.
- **MSE:** mean squared prediction error; lower is better.
- **RMSE:** square root of MSE; lower is better.

### 7. Generate Competition Predictions

The final section loads and preprocesses `Test-Set.csv`, trains AdaBoost on `X_train` and `y_train`, predicts standardized prices, converts them back to the original price scale, and exports `price_predict.csv`.

The final model uses the 80% training subset rather than being refitted on all cleaned training rows.

## Models and Results

The following values come from the **saved outputs in the supplied notebook**. They have not been independently reproduced.

| Model | R² | MSE | MAE | RMSE |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 0.4419 | 0.55 | 0.51 | 0.74 |
| Ridge Regression | 0.4419 | 0.55 | 0.51 | 0.74 |
| K-Nearest Neighbors | 0.0258 | 0.97 | 0.74 | 0.98 |
| Bagging Regressor | 0.8650 | 0.13 | 0.18 | 0.37 |
| Lasso Regression | 0.4419 | 0.55 | 0.51 | 0.74 |
| AdaBoost Regressor | **0.8834** | **0.12** | **0.13** | **0.34** |
| Decision Tree Regressor | 0.7612 | 0.24 | 0.20 | 0.49 |
| Random Forest Regressor | **0.8834** | **0.12** | 0.16 | **0.34** |

**Interpretation:** AdaBoost and Random Forest share the highest displayed R². AdaBoost has the lower displayed MAE and is used for the final CSV export.

All error metrics above are calculated on **standardized prices**. MAE and RMSE are not errors in the original currency, and MSE is expressed in squared standardized units.

The notebook labels `R² × 100` as `Accuracy(%)` in its results table and plot. This is a percentage presentation of R², **not classification accuracy**. An R² of 0.8834 should not be described as “88.34% prediction accuracy.”

### Main Ensemble Settings

| Model | Settings used in the comparison |
| --- | --- |
| Bagging | 50 decision trees, base tree `max_depth=20`, `random_state=1`. |
| AdaBoost | 200 estimators, base tree `max_depth=24`, `learning_rate=0.1`. |
| Random Forest | 150 trees, `max_features=0.3`, `min_samples_leaf=1`, `random_state=0`, `oob_score=True`. |

Ridge uses `alpha=20.336`, Lasso uses `alpha=0.0001`, and KNN uses five neighbors. Some models have no fixed random seed, so results may vary between runs.

## Generated Outputs

| Output | Description |
| --- | --- |
| `score_data` | In-memory DataFrame containing the eight-model comparison. |
| `Overall-Performance.jpg` | Performance chart saved at 600 DPI; displays R² × 100 under the notebook's `Accuracy(%)` label. |
| `price_predict.csv` | Predictions in the original price scale, with `id` as the exported index column and `price` as the prediction column. |

The prediction file has this column structure:

```csv
id,price
```

No trained model file is saved. To regenerate predictions, rerun the notebook.

## Current Limitations

The notebook demonstrates a complete experimental workflow, but several implementation details affect the reliability of its evaluation and exported predictions:

- **Preprocessing before the split:** imputation, encoding, and scaling are performed before the validation split. Validation data therefore influences the learned preprocessing statistics, which can make the evaluation optimistic.
- **Separate category encodings:** categorical encoders are fitted independently on training and competition test data. The same category can receive different numeric codes in the two datasets.
- **Different odometer scales:** training odometer values are standardized, but competition test odometer values remain on their original scale. Final predictions therefore receive inputs that differ from those used during training.
- **Separate test imputation:** the competition test preprocessing learns its own mappings and fill values instead of reusing those learned from training. It also lacks the training section's overall coordinate fallback.
- **Identifier as a feature:** `id` is retained during training. Its contribution should be checked because it is a listing identifier rather than a vehicle attribute.
- **Incomplete fallback coverage:** the notebook does not provide general imputation for missing `year`, `model`, `odometer`, or `transmission` values. Different input data may require additional handling.
- **Unpinned dependencies:** exact package versions are unavailable, and warnings are globally suppressed. Some pandas calls may need explicit assignment for compatibility.

The saved validation scores do not establish the quality of the exported competition predictions. Correct the shared preprocessing workflow and rerun evaluation before treating the CSV as a reliable submission.

## Future Improvements

- Split the data before learning preprocessing statistics.
- Use a shared `Pipeline` and `ColumnTransformer` for training, validation, and inference.
- Keep separate feature and target scalers, and reuse the fitted feature transformations for new data.
- Use categorical encoding that handles unseen categories consistently.
- Compare performance with and without `id`.
- Evaluate MAE and RMSE after converting prices back to their original scale.
- Use cross-validation and hyperparameter search; these utilities are imported but are not used in the current notebook.
- Add feature importance analysis and more exploratory plots.
- Refit the selected workflow on all training data after validation.
- Save the fitted preprocessing objects and model for reusable inference.
- Record package versions and add the dataset source and licensing details when available.

## Project Status

**Experimental notebook project:** data cleaning, model comparison, visualization, and CSV export are implemented. Shared preprocessing and reproducible evaluation are the main next steps.
