# Employee Productivity Prediction Using Machine Learning

## A Supervised Regression Study Using Baseline and Ensemble Models

This project develops an end-to-end **supervised machine learning regression pipeline** for predicting employee `productivity_score` from workforce and remote-work characteristics.

The project uses a **Remote Work Productivity Dataset containing 50,000 records and 15 features**. It compares three baseline regression models with three ensemble methods and then performs hyperparameter tuning on the strongest ensemble using `GridSearchCV`.

The analysis focuses on:

- Data cleaning
- Outlier detection and treatment
- Categorical encoding
- Feature scaling
- Train/test splitting
- Baseline regression models
- Ensemble regression models
- 5-fold cross-validation
- Hyperparameter tuning
- Model comparison
- Regression diagnostics
- Feature importance
- Supplementary classification analysis

---

# Project Objective

The main objective is to predict the continuous `productivity_score` of employees using available workforce, wellbeing, work-pattern, and compensation variables.

The project investigates whether employee productivity can be predicted from factors such as:

- Age
- Country
- Industry
- Work mode
- Weekly working hours
- Tasks completed
- Sleep hours
- Exercise frequency
- Mental health
- Burnout level
- Job satisfaction
- Salary

The project also compares simple regression approaches against tree-based ensemble methods to determine how much predictive improvement can be achieved through non-linear modelling and ensemble learning.

---

# Problem Statement

Employee productivity in remote and hybrid work environments can be influenced by multiple interacting factors, including workload, working hours, wellbeing, burnout, sleep, and job satisfaction.

A purely linear relationship may not adequately represent these interactions.

This project therefore formulates employee productivity prediction as a **supervised regression problem**, with:

```text
Input:
Employee and workforce characteristics

        ↓

Machine Learning Regression Model

        ↓

Output:
Continuous productivity_score
```

The objective is to build a model that provides strong predictive performance while also allowing the important productivity drivers to be interpreted.

---

# Dataset

The project uses the **Remote Work Productivity Dataset**.

### Dataset Size

- **50,000 records**
- **15 columns**
- Continuous target variable: `productivity_score`

## Features

| Column | Type | Description |
|---|---|---|
| `employee_id` | ID | Unique employee identifier; removed before modelling |
| `age` | Numeric | Employee age |
| `country` | Categorical | Employee country |
| `industry` | Categorical | Industry sector |
| `work_mode` | Categorical | Remote, Hybrid, or On-site |
| `year` | Numeric | Year of record |
| `weekly_hours_worked` | Numeric | Total weekly working hours |
| `tasks_completed` | Numeric | Number of completed tasks |
| `productivity_score` | Numeric | Continuous productivity score; prediction target |
| `sleep_hours_avg` | Numeric | Average sleep hours |
| `exercise_frequency_per_week` | Numeric | Weekly exercise frequency |
| `mental_health_score` | Numeric | Mental health rating |
| `burnout_level` | Categorical | Low, Medium, or High |
| `job_satisfaction` | Numeric | Job satisfaction score |
| `salary_usd` | Numeric | Employee salary in USD |

The target `productivity_score` is a continuous value on approximately a **60–100 scale**, making the dataset appropriate for supervised regression.

---

# Why This Dataset?

The dataset provides:

### Large Sample Size

With 50,000 records, the dataset supports:

- An 80/20 train-test split
- 5-fold cross-validation
- Reliable comparison of multiple regression models

### Mixed Feature Types

The dataset contains:

- Demographic variables
- Work-structure variables
- Productivity/output variables
- Wellbeing variables
- Categorical variables
- Numerical variables

### Continuous Target

`productivity_score` is continuous, satisfying the requirements of a regression problem.

### Potential Non-Linear Relationships

Variables such as:

- `weekly_hours_worked`
- `mental_health_score`
- `burnout_level`
- `job_satisfaction`
- `tasks_completed`

may interact in non-linear ways, making tree-based ensemble models suitable candidates.

---

# Machine Learning Pipeline

```text
                    Remote Work Dataset
                            │
                            ▼
                  Data Quality Analysis
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          Missing Values          Duplicates
                 │                     │
                 └──────────┬──────────┘
                            ▼
                  Identifier Removal
                            │
                            ▼
                  Outlier Detection
                            │
                            ▼
                  Outlier Treatment
                            │
                            ▼
               Categorical Encoding
                            │
                            ▼
                  Train/Test Split
                       80 / 20
                            │
                            ▼
                   Feature Scaling
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
          Baseline Models        Ensemble Models
                │                       │
                ▼                       ▼
          Model Evaluation       Model Evaluation
                │                       │
                └───────────┬───────────┘
                            ▼
                    Model Comparison
                            │
                            ▼
                    GridSearchCV
                            │
                            ▼
                    Tuned XGBoost
                            │
                            ▼
             Diagnostics & Interpretation
```

---

# Step 1 — Data Cleaning

The data-cleaning stage checks:

- Missing values
- Duplicate records
- Identifier columns
- Numerical outliers

## Missing Values

The notebook checks every column for missing values.

If missing values are found, they are reported before modelling.

## Duplicate Records

Duplicate rows are detected and removed if present.

## Identifier Removal

Non-predictive identifiers such as:

```text
employee_id
Employee_ID
ID
Name
```

are removed when present.

This prevents identifiers from being incorrectly interpreted as predictive variables.

---

# Outlier Detection

Outliers are detected using the **Interquartile Range (IQR)** method.

The IQR is calculated as:

```text
IQR = Q3 - Q1
```

The standard boundaries are:

```text
Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Observations outside these boundaries are identified as potential outliers.

The notebook also calculates feature skewness to determine the most appropriate treatment.

---

# Outlier Treatment

The project uses a skewness-driven two-stage strategy.

## Highly Skewed Features

For features where:

```text
|skewness| > 1
```

IQR capping is applied.

Extreme values are clipped to the IQR boundaries rather than removing entire observations.

## Approximately Symmetric Features

For features where:

```text
|skewness| ≤ 1
```

Winsorisation-style IQR boundary treatment is applied.

This preserves the observations while limiting the influence of extreme values.

## Integer-Valued Features

The following features are rounded after treatment:

- `tasks_completed`
- `mental_health_score`
- `job_satisfaction`

This preserves their intended integer-like representation.

---

# Second Outlier Treatment

A second treatment is applied to:

```text
job_satisfaction
```

The notebook uses **5th–95th percentile capping** to force convergence after the first IQR-based treatment.

The resulting values are rounded to preserve the feature's score-like representation.

---

# Step 2 — Data Preprocessing

The preprocessing pipeline contains:

1. Categorical encoding
2. Feature/target separation
3. 80/20 train-test split
4. Feature scaling

---

# Categorical Encoding

The categorical variables are converted into numerical representations using `LabelEncoder`.

The relevant categorical features include:

- `country`
- `industry`
- `work_mode`
- `burnout_level`

This produces numerical inputs suitable for the machine-learning models.

---

# Target Variable

The target variable is:

```python
productivity_score
```

The feature matrix is created by removing the target:

```python
X = df.drop("productivity_score", axis=1)
```

The target vector is:

```python
y = df["productivity_score"]
```

---

# Train/Test Split

The dataset is divided using:

```text
Training Data = 80%
Testing Data  = 20%
```

with:

```python
random_state = 42
```

This provides reproducibility across experiments.

---

# Feature Scaling

`StandardScaler` is used for scale-sensitive models.

The scaler is fitted only on the training data:

```python
scaler.fit(X_train)
```

and then applied to both training and test data:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

This prevents information leakage from the test set into model training.

Tree-based models are trained on the unscaled features because tree-based algorithms are not dependent on feature scale.

---

# Evaluation Metrics

The project evaluates regression models using:

## R-squared

R-squared measures the proportion of target variance explained by the model.

```text
R² = 1 - SSres / SStot
```

Higher values indicate stronger explanatory performance.

## RMSE

Root Mean Squared Error measures the average magnitude of prediction errors while giving greater weight to larger errors.

```text
RMSE = √(Mean Squared Error)
```

Lower values are better.

## MAE

Mean Absolute Error measures the average absolute prediction error.

```text
MAE = Mean(|actual - predicted|)
```

Lower values are better.

## 5-Fold Cross-Validation

The project uses:

```text
K = 5
```

folds to evaluate the robustness of model performance.

The notebook reports:

- Mean cross-validated R-squared
- Standard deviation of cross-validated R-squared

---

# Step 3 — Baseline Regression Models

Three non-ensemble models are evaluated.

## 1. Linear Regression

Linear Regression provides the simplest reference model.

It assumes that the target can be represented as a linear combination of the input features.

Advantages:

- Highly interpretable
- Simple
- Fast
- Useful as a reference point

Limitation:

- Cannot naturally capture complex non-linear relationships.

---

## 2. K-Nearest Neighbors Regression

KNN predicts a target based on nearby observations in feature space.

Configuration:

```text
n_neighbors = 5
```

Advantages:

- Non-parametric
- Can capture local patterns
- Does not assume a specific functional form

Limitations:

- Sensitive to feature scaling
- Computationally more expensive during prediction
- Can suffer from the curse of dimensionality

---

## 3. Decision Tree Regression

Decision Tree Regression captures:

- Non-linear relationships
- Feature interactions
- Rule-based patterns

Configuration:

```text
max_depth = 10
random_state = 42
```

The tree provides a natural transition from simple baseline models to ensemble tree methods.

---

# Baseline Results

| Model | R² | RMSE | MAE | CV Mean R² | CV Std |
|---|---:|---:|---:|---:|---:|
| **Decision Tree** | **0.7675** | **3.3982** | **2.2526** | **0.7693** | 0.0034 |
| Linear Regression | 0.7115 | 3.7858 | 2.9809 | 0.7158 | 0.0046 |
| KNN | 0.6980 | 3.8734 | 2.6240 | 0.7033 | 0.0056 |

The Decision Tree is the strongest baseline model based on the reported hold-out R².

---

# Step 4 — Ensemble Regression Models

Three ensemble approaches are evaluated:

1. Gradient Boosting
2. XGBoost
3. Voting Regressor

Ensemble methods combine multiple learners to improve predictive performance and/or reduce variance.

---

# Gradient Boosting

Gradient Boosting trains trees sequentially.

Each new tree focuses on reducing the errors made by the previous ensemble.

Initial configuration:

```text
n_estimators = 200
learning_rate = 0.05
random_state = 42
```

Gradient Boosting is included as a standard boosting baseline for tabular regression.

---

# XGBoost

XGBoost is a regularised gradient-boosting implementation designed for high-performance tabular machine learning.

Initial configuration:

```text
n_estimators = 250
learning_rate = 0.05
max_depth = 6
random_state = 42
objective = reg:squarederror
```

XGBoost was expected to perform strongly because the dataset contains structured numerical and categorical features with potential non-linear relationships.

---

# Voting Regressor

The Voting Regressor combines predictions from different model families.

The implementation combines:

- Linear Regression
- Decision Tree
- Random Forest

The final prediction is obtained by averaging the component predictions.

The purpose is to test whether combining structurally different models can reduce prediction error.

---

# Ensemble Results

| Model | R² | RMSE | MAE | CV Mean R² | CV Std |
|---|---:|---:|---:|---:|---:|
| **XGBoost** | **0.7985** | **3.1634** | **2.1364** | **0.8025** | 0.0050 |
| Gradient Boosting | 0.7969 | 3.1763 | 2.1937 | 0.8002 | 0.0054 |
| Voting Ensemble | 0.7848 | 3.2693 | 2.3644 | 0.7904 | 0.0040 |

XGBoost is the strongest model in the initial ensemble comparison.

---

# Hyperparameter Tuning

The strongest initial ensemble, **XGBoost**, is tuned using `GridSearchCV`.

## Search Strategy

The tuning process uses:

- `GridSearchCV`
- 3-fold cross-validation inside the grid search
- R-squared as the optimisation metric
- `n_jobs=-1` for parallel execution

The grid contains:

```text
n_estimators:
200, 300, 500

max_depth:
4, 6, 8

learning_rate:
0.03, 0.05, 0.1
```

This results in:

```text
27 parameter combinations
×
3 CV folds
=
81 model fits
```

---

# Best XGBoost Hyperparameters

The grid search selected:

| Hyperparameter | Best Value |
|---|---:|
| `n_estimators` | **500** |
| `learning_rate` | **0.03** |
| `max_depth` | **4** |

Best grid-search cross-validation R²:

```text
0.8031
```

---

# Tuned XGBoost Performance

| Metric | Result |
|---|---:|
| R² | **0.8006** |
| RMSE | **3.1473** |
| MAE | **2.1409** |
| CV Mean R² | **0.8046** |
| CV Std R² | **0.0047** |

---

# Final Model Comparison

| Rank | Model | Category | R² | RMSE | MAE | CV Mean R² | CV Std |
|---:|---|---|---:|---:|---:|---:|---:|
| **1** | **XGBoost (tuned)** | Ensemble | **0.8006** | **3.1473** | 2.1409 | **0.8046** | 0.0047 |
| 2 | XGBoost | Ensemble | 0.7985 | 3.1634 | **2.1364** | 0.8025 | 0.0050 |
| 3 | Gradient Boosting | Ensemble | 0.7969 | 3.1763 | 2.1937 | 0.8002 | 0.0054 |
| 4 | Voting Ensemble | Ensemble | 0.7848 | 3.2693 | 2.3644 | 0.7904 | 0.0040 |
| 5 | Decision Tree | Baseline | 0.7675 | 3.3982 | 2.2526 | 0.7693 | 0.0034 |
| 6 | Linear Regression | Baseline | 0.7115 | 3.7858 | 2.9809 | 0.7158 | 0.0046 |
| 7 | KNN | Baseline | 0.6980 | 3.8734 | 2.6240 | 0.7033 | 0.0056 |

The notebook identifies **tuned XGBoost** as the overall best model according to the final model comparison.

---

# Baseline vs Ensemble Analysis

The results show that all three ensemble approaches outperform the baseline models on hold-out R².

The weakest ensemble, Voting Regressor:

```text
R² = 0.7848
```

still exceeds the strongest baseline, Decision Tree:

```text
R² = 0.7675
```

The tuned XGBoost model further improves the result:

```text
R² = 0.8006
RMSE = 3.1473
CV Mean R² = 0.8046
```

This supports the use of ensemble methods for the structured workforce dataset.

---

# Regression Diagnostics

The notebook performs two standard diagnostics for the final model.

## Actual vs Predicted

The actual-vs-predicted plot compares:

```text
Actual productivity_score
            vs
Predicted productivity_score
```

The ideal relationship is represented by:

```text
y = x
```

Points closer to the diagonal indicate more accurate predictions.

---

## Residual Plot

Residuals are calculated as:

```text
Residual = Actual - Predicted
```

The residual plot is used to inspect:

- Systematic prediction bias
- Non-linear patterns
- Heteroscedasticity
- Model misspecification

Ideally, residuals should be distributed around zero without a strong systematic pattern.

---

# Feature Importance

The final model exposes feature importance values.

The top five features reported by the notebook are:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `mental_health_score` | **0.529436** |
| 2 | `weekly_hours_worked` | **0.154294** |
| 3 | `tasks_completed` | **0.124892** |
| 4 | `sleep_hours_avg` | **0.122799** |
| 5 | `burnout_level` | **0.049395** |

The results indicate that `mental_health_score` has the largest model-derived feature importance in the final model, followed by weekly working hours, tasks completed, sleep hours, and burnout level.

These are **model-derived importance measures**, not causal effects.

---

# Supplementary Classification Analysis

The notebook also includes an optional classification-style analysis.

This does **not replace the primary regression evaluation**.

The continuous `productivity_score` prediction is converted into a binary high/low classification using the median of the training target.

## Threshold

```text
Median of y_train = 98.0
```

Predictions are converted as:

```text
productivity_score >= 98
    → High

productivity_score < 98
    → Low
```

---

# Supplementary Classification Results

| Metric | Result |
|---|---:|
| Accuracy | **0.8533** |
| Precision | **0.9195** |
| Recall | **0.7848** |
| F1 Score | **0.8468** |

A confusion matrix is also generated for this supplementary classification view.

This analysis can support decision-support scenarios where productivity is framed as a high/low category rather than a continuous score.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

### Machine Learning Algorithms

- Linear Regression
- K-Nearest Neighbors Regression
- Decision Tree Regression
- Gradient Boosting Regression
- XGBoost Regression
- Voting Regression

### Machine Learning Techniques

- Train/Test Split
- Standard Scaling
- Label Encoding
- IQR Outlier Detection
- IQR Capping
- Winsorisation
- Percentile Capping
- K-Fold Cross-Validation
- GridSearchCV
- Feature Importance
- Regression Diagnostics

---

# Reproducibility

The notebook uses:

```text
random_state = 42
```

for the main train/test split and relevant models.

The main workflow is:

```text
1. Load dataset
2. Check missing values
3. Check duplicate records
4. Remove identifier columns
5. Detect numerical outliers
6. Treat outliers
7. Encode categorical variables
8. Separate features and target
9. Split data 80/20
10. Scale features for scale-sensitive models
11. Train baseline regressors
12. Evaluate baseline models
13. Train ensemble models
14. Compare ensemble performance
15. Tune XGBoost with GridSearchCV
16. Perform final model comparison
17. Generate regression diagnostics
18. Analyse feature importance
19. Perform supplementary classification analysis
```

---

# Project Structure

A recommended repository structure is:

```text
employee-productivity-prediction/
│
├── README.md
│
├── data/
│   └── remote_work_productivity_dataset.csv
│
├── notebooks/
│   └── productivity_regression.ipynb
│
├── results/
│   ├── model_comparison/
│   ├── diagnostics/
│   └── feature_importance/
│
├── models/
│
├── requirements.txt
│
└── docs/
```

---

# Requirements

A suitable `requirements.txt` can include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

# How to Run

## 1. Clone the Repository

```bash
git clone <repository-url>
cd employee-productivity-prediction
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Place the Dataset

Place:

```text
remote_work_productivity_dataset.csv
```

inside the appropriate `data/` or notebook working directory.

## 4. Open the Notebook

```bash
jupyter notebook
```

Then open:

```text
productivity_regression.ipynb
```

## 5. Run the Notebook

Execute the cells sequentially to reproduce:

- Data cleaning
- Outlier treatment
- Preprocessing
- Baseline models
- Ensemble models
- Hyperparameter tuning
- Model comparison
- Diagnostics
- Feature importance
- Supplementary classification metrics

---

# Key Results

The final tuned XGBoost model achieved:

```text
R²              = 0.8006
RMSE            = 3.1473
MAE             = 2.1409
CV Mean R²      = 0.8046
CV Std R²       = 0.0047
```

Best hyperparameters:

```text
n_estimators    = 500
learning_rate   = 0.03
max_depth       = 4
```

Top model-derived features:

```text
mental_health_score
weekly_hours_worked
tasks_completed
sleep_hours_avg
burnout_level
```

---

# Important Interpretation

The model results should be interpreted as **predictive associations rather than causal relationships**.

For example, the high feature importance of `mental_health_score` indicates that this feature contributes strongly to the model's predictions. It does not by itself establish that changing mental health scores will cause a specific change in productivity.

Similarly, the model should be used as a predictive and analytical tool rather than as the sole basis for employment decisions.

---

# Limitations

- The project uses a single workforce dataset.
- Model performance depends on the quality and representativeness of the dataset.
- Label encoding can impose numerical representations on categorical values.
- Tree ensembles provide strong predictive performance but are less directly interpretable than Linear Regression.
- Feature importance does not establish causality.
- Historical data may not fully represent future workforce behaviour.
- The supplementary high/low classification is derived from a median threshold and is not the primary modelling objective.
- Real-world deployment would require additional validation, monitoring, privacy considerations, and responsible-use controls.

---

# Conclusion

This project delivers a complete supervised regression pipeline for predicting employee `productivity_score` using workforce and remote-work data.

The project compares:

```text
Baseline Models
├── Linear Regression
├── KNN
└── Decision Tree

Ensemble Models
├── Gradient Boosting
├── XGBoost
└── Voting Regressor
```

The initial ensemble comparison identified XGBoost as the strongest model. GridSearchCV further tuned XGBoost using 27 parameter combinations and selected:

```text
n_estimators = 500
learning_rate = 0.03
max_depth = 4
```

The tuned model achieved:

```text
R² = 0.8006
RMSE = 3.1473
CV Mean R² = 0.8046
```

The final analysis also identifies `mental_health_score`, `weekly_hours_worked`, `tasks_completed`, `sleep_hours_avg`, and `burnout_level` as the five most important features according to the trained model.

Overall, the project demonstrates an end-to-end machine-learning workflow for structured workforce regression, from data preparation and outlier treatment through model benchmarking, ensemble learning, hyperparameter tuning, diagnostics, and interpretation.
