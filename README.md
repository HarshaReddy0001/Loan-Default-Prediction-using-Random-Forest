## Project README

### 1. Introduction

*   **What is this project about?** This project focuses on building a classification model to predict whether a borrower will pay back their loan in full.
*   **What problem does it solve?** The project addresses the real-world problem faced by investors in platforms like LendingClub: how to predict the likelihood of a borrower defaulting on a loan to make better investment decisions and mitigate risk.
*   **Key takeaways/results:** This project developed and compared a Decision Tree and a Random Forest model for loan repayment prediction. The Random Forest model showed higher overall accuracy (0.85) compared to the Decision Tree (0.76), although the Decision Tree had better recall and F1-score for the minority class (not fully paid).

### 2. Dataset

*   **Source:** The data is publicly available data from LendingClub.com. It's provided as a CSV file named `loans.csv`. The specific source link was not provided in the notebook, but the data is described as being from 2007-2010.
*   **Description:** The dataset contains information about loans and borrowers from Lending Club. After dropping rows with missing values, it includes 9513 entries and 15 columns. Key variables include `credit.policy`, `purpose`, `int.rate`, `installment`, `log.annual.inc`, `dti`, `days.with.cr.line`, `revol.bal`, `revol.util`, `inq.last.6mths`, `delinq.2yrs`, `pub.rec`, `fico`, and the target variable `not.fully.paid`. There is also a `customer.id` column.
*   **Preprocessing:** Several preprocessing steps were performed:
    *   Handling missing values by dropping rows with `dropna()`.
    *   Converting object type columns (`credit.policy`, `dti`, `revol.util`, `delinq.2yrs`, `pub.rec`) to appropriate numeric types (int64 and float64) by replacing non-numeric values and using `pd.to_numeric` with `errors='coerce'` and `fillna(0)`.
    *   Creating dummy variables for the categorical 'purpose' column using `pd.get_dummies` with `drop_first=True`.

### 3. Methodology

*   **Approach:** The project follows a typical supervised learning approach, starting with Exploratory Data Analysis (EDA) to understand the data, followed by data preprocessing and then training and evaluating classification models.
*   **Techniques/Models:** Two classification algorithms were employed:
    *   Decision Tree Classifier (`DecisionTreeClassifier`)
    *   Random Forest Classifier (`RandomForestClassifier`)
*   **Tools/Libraries:** The project utilized the following Python libraries:
    *   pandas for data manipulation and analysis.
    *   numpy for numerical operations.
    *   matplotlib.pyplot and seaborn for data visualization.
    *   sklearn (scikit-learn) for splitting data (`train_test_split`), building models (`DecisionTreeClassifier`, `RandomForestClassifier`), and evaluating models (`confusion_matrix`, `classification_report`).

### 4. Analysis & Results

*   **Key Findings/Insights:**
    *   The majority of customers in the dataset met the credit underwriting criteria (`credit.policy` = 1).
    *   The most common loan purpose is 'debt\_consolidation', followed by 'all\_other' and 'credit\_card'.
    *   The distribution of interest rates is skewed, with a peak around 0.11-0.15.
    *   There are differences in the distribution of loan purposes between those who fully paid and those who did not.
    *   There appears to be a negative correlation between FICO score and interest rate, with higher FICO scores associated with lower interest rates.
*   **Model Performance:**
    *   **Decision Tree Model:**
    *   print("Decision Tree Model Performance:")
print(classification_report(y_test, dtree_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, dtree_predictions))

    *   **Random Forest Model:**
    *   print("Random Forest Model Performance:")
print(classification_report(y_test, rfc_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rfc_pred))

*   **Visualizations:** The notebook includes visualizations such as countplots of `credit.policy` and `purpose` (with `not.fully.paid` hue), and a jointplot of `fico` vs `int.rate`. A confusion matrix heatmap for the Random Forest model is also included.

### 5. Conclusion & Future Work

*   **Summary of Achievements:** This project successfully loaded, cleaned, and preprocessed the LendingClub loan data. It then trained and evaluated a Decision Tree and a Random Forest model to predict loan repayment, providing insights into their performance.
*   **Limitations:** A key limitation is the class imbalance in the target variable (`not.fully.paid`), where the number of loans not fully paid is significantly smaller than those fully paid. This affects the performance metrics, particularly for the minority class. The Random Forest model, while having higher overall accuracy, shows very low recall and F1-score for the 'not fully paid' class, indicating it struggles to correctly identify these instances.
*   **Future Enhancements:** Future work could involve addressing the class imbalance using techniques like oversampling (e.g., SMOTE) or undersampling. Exploring other classification algorithms (e.g., Logistic Regression, Gradient Boosting) and hyperparameter tuning for the current models could also improve performance, especially for the minority class. Further feature engineering and selection might also yield better results.

### 6. How to Run/Reproduce

*   **Environment Setup:**
    1.  Ensure you have Python installed (version 3.6 or higher recommended).
    2.  Install the necessary libraries using pip:
