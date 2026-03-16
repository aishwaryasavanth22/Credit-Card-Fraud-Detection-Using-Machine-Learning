### Credit-Card-Fraud-Detection-Using-Machine-Learning
A machine learning project that detects fraudulent credit card transactions using classification algorithms and advanced evaluation metrics. The project demonstrates data preprocessing, class imbalance handling using SMOTE, model building, ROC-AUC evaluation, and feature importance analysis.

#### 🚀 Project Overview

Credit card fraud has become a major concern for financial institutions and online payment systems. Detecting fraudulent transactions accurately and quickly is essential to minimize financial losses and improve transaction security.

The objective of this project is to build machine learning models capable of identifying fraudulent transactions from genuine ones using historical transaction data.

This project applies machine learning classification algorithms including Decision Tree and Random Forest to detect fraudulent activities. The models are evaluated using multiple performance metrics such as accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC score to ensure reliable fraud detection.

#### 2. Dataset Description

The dataset contains information about credit card transactions categorized as fraudulent or genuine.

Key characteristics of the dataset include:

- Target Variable: Class

    - 0 → Genuine Transaction
    - 1 → Fraudulent Transaction

- Transaction Amount: Amount represents the total money involved in the transaction.

Dataset Statistics :

- Total records: 11,620 transactions
- Genuine transactions: 11,571
- Fraudulent transactions: 49
- Fraud transaction percentage: ~0.42%

⚠️ The dataset is highly imbalanced, which is common in fraud detection problems.

#### 3. Importing Required Libraries

The following Python libraries were used in this project:

- NumPy – numerical computations
- Pandas – data manipulation
- Matplotlib & Seaborn – data visualization
- Scikit-Learn – machine learning models and evaluation metrics

#### 4. Data Loading and Initial Inspection

The dataset was loaded using the Pandas library.

Initial data inspection was performed to understand the dataset structure using the following steps:

- Displaying the first five rows using .head()
- Checking dataset shape
- Identifying column data types using .info()
- Listing column names
- Generating statistical summaries using .describe()

These steps helped understand the data distribution and structure before proceeding with preprocessing.

#### 5. Data Cleaning

Data quality checks were performed to ensure the reliability and Integrity of the dataset.

The following issues were identified:

| Issue             | Count |
| ----------------- | ----- |
| Missing Values    | 19    |
| Duplicate Records | 44    |


Since the number of missing and duplicate values was very small compared to the total dataset size, these records were removed to maintain data quality.

#### 6. Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand the distribution of transactions and detect potential patterns.

Key observations include:

- The target variable Class is highly imbalanced, with the majority of transactions being genuine.
- Genuine transactions significantly outnumber fraudulent transactions.
- Fraud transactions represent only 0.42% of the dataset, which highlights the need for special techniques to handle class imbalance.

Visualizations performed

- Bar plot showing distribution of genuine vs fraudulent transactions

EDA confirmed that special techniques are required to handle the class imbalance problem.

#### Visualizations :
![Genuine Vs Fraud Detection](https://github.com/aishwaryasavanth22/Credit-Card-Fraud-Detection-Using-Machine-Learning/blob/600d78dbc3a6165f07ddec9edb074384fad85101/Visualizations/Genuine%20Vs%20Fraud%20Transactions.png)


#### 7. Data Preprocessing

**Feature Scaling :**

The `Amount` feature was normalized using StandardScaler to ensure that the feature values are on a comparable scale, which helps machine learning models learn more effectively.

**Handling Class Imbalance :**

Since fraud detection datasets typically suffer from extreme class imbalance, the **SMOTE (Synthetic Minority Oversampling Technique)** was applied.

SMOTE generates synthetic samples for the minority class (fraudulent transactions), enabling the models to learn meaningful fraud patterns.

This step significantly improves the model's ability to detect fraudulent activities.

#### 8. Model Development

Two machine learning classification models were implemented to detect fraudulent transactions:

1. Decision Tree Classifier

Decision Tree is a supervised learning algorithm that splits the dataset into branches based on feature values to classify transactions as fraud or genuine.

2. Random Forest Classifier

Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

#### 9. Model Evaluation

Both models were evaluated using standard classification metrics.

Evaluation metrics used include:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC Score & ROC Curve.
- Feature Importance.

These metrics provide a comprehensive understanding of the model's performance, especially in detecting fraudulent transactions.

Since fraud detection datasets are highly imbalanced, accuracy can be misleading. A model can achieve very high accuracy simply by predicting all transactions as genuine. Therefore, I used ROC-AUC score to evaluate how well the model distinguishes between fraudulent and legitimate transactions across different classification thresholds.

#### 9. Decision Tree Classifier

The Decision Tree model builds a tree-like structure where features are used to split the data into branches to classify transactions.

**Model Performance :**

- Accuracy: ~0.99
- Precision: ~0.99
- Recall: ~0.99
- F1 Score: ~0.99

**Confusion Matrix Interpretation :**

![Decision Tree Confusion Matrix](https://github.com/aishwaryasavanth22/Credit-Card-Fraud-Detection-Using-Machine-Learning/blob/29aa4a22fe47c448b1fa4b1a5d008e905704df91/Visualizations/Decision%20Tree%20Confusion%20Matrix.png)

This means :

- 3440 genuine transactions correctly identified
- 3490 fraud transactions correctly detected
- 9 genuine transactions incorrectly flagged as fraud
- 4 fraud transactions missed by the model

The confusion matrix was visualized using a heatmap, which clearly shows the prediction performance of the model.

**ROC Curve Analysis :**

The ROC curve illustrates the trade-off between True Positive Rate and False Positive Rate.

The model achieved an ROC-AUC score close to 1.00, indicating excellent classification ability.

Deision Tree ROC-AUC Curve
![Decision Tree ROC-AUC Curve](https://github.com/aishwaryasavanth22/Credit-Card-Fraud-Detection-Using-Machine-Learning/blob/29aa4a22fe47c448b1fa4b1a5d008e905704df91/Visualizations/DT%20ROC-AUC%20Curve.png)

Interpretation

- High True Positive Rate → fraud transactions detected successfully
- Low False Positive Rate → fewer genuine transactions flagged as fraud
- Curve close to top-left corner → strong classification performance.

**Feature Importance (Decision Tree):**

Feature importance analysis was performed to identify which variables contribute the most to fraud detection.

DT Feature Importance
![Decision Tree Feature Importance](https://github.com/aishwaryasavanth22/Credit-Card-Fraud-Detection-Using-Machine-Learning/blob/29aa4a22fe47c448b1fa4b1a5d008e905704df91/Visualizations/DT%20Feature%20Importance.png)

Insights :

- The model assigns higher importance to features that significantly influence transaction classification.
- These features help identify suspicious transaction patterns.

Understanding feature importance helps financial institutions interpret why a transaction was flagged as fraudulent.

#### 10. Random Forest Model Performance

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

Performance Metrics

- Accuracy: ~0.99
- Precision: ~0.99
- Recall: 1.00
- F1 Score: ~0.99

**Confusion Matrix Interpretation :**

![Random Forest Confusion Matrix](https://github.com/aishwaryasavanth22/Credit-Card-Fraud-Detection-Using-Machine-Learning/blob/29aa4a22fe47c448b1fa4b1a5d008e905704df91/Visualizations/RF%20Confusion%20Matrix.png)

This indicates:

- 3444 genuine transactions correctly classified
- 3494 fraud transactions correctly detected
- 5 genuine transactions incorrectly predicted as fraud
- No fraud transactions were missed

Importantly, the model did not miss any fraudulent transactions, which is a critical requirement in fraud detection systems.

**ROC Curve and ROC-AUC Score (Random Forest) :**

The Random Forest model achieved an ROC-AUC score close to 1.00, demonstrating excellent ability to distinguish fraudulent transactions from genuine ones.

RF ROC-AUC Curve 
![Random Forest ROC-AUC Curve](https://github.com/aishwaryasavanth22/Credit-Card-Fraud-Detection-Using-Machine-Learning/blob/29aa4a22fe47c448b1fa4b1a5d008e905704df91/Visualizations/RF%20ROC-AUC%20Curve.png)

Interpretation

- High ROC-AUC indicates strong classification capability
- The model maintains high fraud detection while minimizing false alerts
- This is critical in real-world financial systems.

**Feature Importance (Random Forest) :**

Feature importance analysis was also performed for the Random Forest model.

RF Feature Importance
![Random Forest Feature Importance](https://github.com/aishwaryasavanth22/Credit-Card-Fraud-Detection-Using-Machine-Learning/blob/29aa4a22fe47c448b1fa4b1a5d008e905704df91/Visualizations/RF%20Feature%20Importance.png)


Insights :

- Random Forest identifies key transaction features contributing to fraud detection.
- Ensemble models provide more stable feature importance compared to single decision trees.

This analysis helps explain the model's decision-making process and improves model interpretability.

#### 11. Model Comparison

| Model         | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------- | -------- | --------- | ------ | -------- | ------- |
| Decision Tree | 0.99     | 0.99      | 0.99   | 0.99     | ~1.00   |
| Random Forest | 0.99     | 0.99      | 1.00   | 0.99     | ~1.00   |


**Key Observation :**

The Random Forest model performed slightly better because:

- It detected all fraudulent transactions
- It reduced the chances of fraud being missed

#### 12. Key Insights from the Project

- The dataset is extremely imbalanced, with fraud transactions representing only 0.42% of total transactions.
- Applying SMOTE oversampling significantly improved the model's ability to detect fraud patterns.
- Both Decision Tree and Random Forest performed well, achieving ~99% accuracy.
- Random Forest showed better fraud detection capability due to its ensemble learning approach.
- ROC-AUC analysis confirmed that the models can effectively distinguish between fraudulent and legitimate transactions.

#### 13. Conclusion

This project demonstrates the use of machine learning techniques to detect fraudulent credit card transactions.

By applying preprocessing techniques such as feature scaling and SMOTE oversampling, the dataset imbalance was effectively addressed.

Two machine learning models were developed and evaluated using multiple performance metrics.

The Random Forest classifier emerged as the best performing model, achieving high accuracy, strong fraud detection capability, and excellent ROC-AUC performance.

Such systems can assist financial institutions in identifying suspicious transactions, preventing fraud, and improving overall payment security.

## Author

Aishwarya Savanth
Aspiring Data Scientist | Machine Learning Enthusiast



