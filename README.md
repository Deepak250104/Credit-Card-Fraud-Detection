
# Credit Card Fraud Detection Project

This project implements machine learning techniques to detect fraudulent credit card transactions from a highly imbalanced dataset. The workflow includes data acquisition, preprocessing, feature engineering, addressing class imbalance, model selection, hyperparameter tuning, and final evaluation.

## File Structure

```plaintext
Credit-Card-Fraud-Detection/
├── Data/
│   ├── archive.zip
├── Models/
│   ├── best_model.joblib
│   ├── gradient_boosting_tuned.joblib
│   ├── lightgbm_tuned.joblib
│   ├── logistic_regression_tuned.joblib
│   ├── random_forest_tuned.joblib
│   └── scaler.joblib
├── Notebooks/
│   ├── EDA.ipynb
│   ├── Evaluation.ipynb
│   ├── FeatureEngineering.ipynb
│   └── ModelSelection.ipynb
├── src/
│   ├── pycache/
│   ├── models.py
│   ├── preprocessing.py
│   └── download_dataset.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Steps to Run the Project

1. **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd Credit-Card-Fraud-Detection
    ```

2. **Set Up Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate  # On Windows
    ```

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download Dataset:**
    The dataset is downloaded using the `src/download_dataset.py` script, which requires the Kaggle API to be installed and configured.
    ```bash
    python src/download_dataset.py
    ```
    **Dataset Source:** [Credit card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data)

    The `creditcard.csv` file will be saved in the `Data/` directory.

5. **Execute Jupyter Notebooks:**
    Run the notebooks in the following sequential order:
    - **EDA.ipynb:** Conducts Exploratory Data Analysis to understand the dataset.
    - **FeatureEngineering.ipynb:** Performs data preprocessing, feature scaling, and handles class imbalance using SMOTE on the training data. The processed data and the scaler are saved.
    - **ModelSelection.ipynb:** Trains and performs hyperparameter tuning for Logistic Regression, Random Forest, Gradient Boosting, and LightGBM using `RandomizedSearchCV`. The best models are saved.
    - **Evaluation.ipynb:** Loads the best trained model and evaluates its performance on the test dataset, generating classification reports and ROC curves.

## Handling Class Imbalance

The dataset exhibits a significant class imbalance. This was addressed in the `FeatureEngineering.ipynb` notebook by applying **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data. SMOTE creates synthetic samples of the minority (fraudulent) class to provide a more balanced dataset for training the machine learning models.

## Hyperparameter Tuning

Hyperparameter tuning was performed using `RandomizedSearchCV` to efficiently search for the optimal hyperparameters for each model. The best cross-validation ROC AUC scores achieved during tuning were:

- **Random Forest:** `0.99994430332102`
- **Logistic Regression:** `0.9916568588276671`
- **Gradient Boosting:** `0.9998275846550692`
- **LightGBM:** `0.9999481089281613`

## Best Model and Evaluation

Based on the hyperparameter tuning results, **LightGBM** achieved the highest cross-validation ROC AUC score and was selected as the best model. The tuned LightGBM model is saved as `best_model.joblib` in the `Models/` directory.

The evaluation of the best model on the unseen test data in `Evaluation.ipynb` yielded the following results:

**ROC AUC Score on Test Set:** 0.91

### Classification Report

```plaintext
          precision    recall  f1-score   support

       0       1.00      1.00      1.00     56864
       1       0.63      0.82      0.71        98

accuracy                           1.00     56962
macro avg       0.82      0.91      0.86     56962
weighted avg       1.00      1.00      1.00     56962
```

The evaluation metrics indicate a strong performance in detecting fraudulent transactions, with a high recall for the fraud class and a good overall AUC score.

## License

This project is open-source and available under the [MIT License](https://opensource.org/license/MIT).

## Author 

Deepak250104
