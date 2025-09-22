# Customer-Risk-Intelligence-and-Proactive-Retention-Framework

# Overview

This repository contains 500 records of customers from a financial institution. It includes demographic information (age, gender, income), financial behaviour (spending score, credit score, loan amount), historical risk (previous defaults), marketing engagement (marketing spend), and critical outcomes (customer churn, defaults). 

The primary goal is to leverage this data to mitigate risk, improve customer retention, and optimise marketing strategies.


# Analysis and Business Questions

The analysis is divided into 4 key areas and seeks to answer the following business questions:

1. Risk Assessment & Default Prediction:

	**What are the key factors that correlate with a customer defaulting on a loan?**

	**Can we build a model to predict the probability of default for a new applicant based on their profile?**

2. Customer Churn Analysis:

	**Why are customers leaving? What patterns distinguish customers who churn from those who stay?**

	**Is there a relationship between loan approval amounts, credit score, and customer churn?**

3. Customer Segmentation for Marketing:

	**Can we segment customers into groups (e.g., "high-value low-risk," "high-risk," "credit builders") to tailor marketing offers and loan products?**

	**How does marketing spend correlate with purchase frequency or new loan uptake? Is the marketing budget being spent effectively?**

4. Financial Product Analysis:

	**What is the typical loan amount granted based on income and credit score?**

	**How does spending behaviour (spending score) relate to income and creditworthiness?**


# Methods

1.	Descriptive Analytics: Summarising key metrics (average loan size, default rate, churn rate). 
2.	Correlation Analysis: Identifying relationships between variables (e.g., Income vs. Loan Amount, Credit Score vs. Defaulted).
3.	Predictive Modelling: Using historical data to predict future outcomes (e.g., Logistic Regression to predict Defaulted or Customer Churn). 
4.	Clustering: Using algorithms like K-Means to segment customers into distinct groups based on multiple characteristics.


# Tools and their purpose 

•	Excel: Initial data exploration, using filters and pivot tables to get a quick sense of distribution (e.g., average income of defaulters vs. non-defaulters).

•	SQL: To query the database and prepare specific datasets for analysis.

•	Python (Pandas, Scikit-learn, Seaborn): The primary tool for deep analysis.

    o	Pandas: For robust data cleaning and manipulation.
    
    o	Seaborn/Matplotlib: For creating visualisations like boxplots (Income by Default status) and correlation heatmaps.
    
    o	Scikit-learn: To build a classification model to predict default using features like Age, Income, Credit Score, and Previous Defaults.

# DATA PROCESSING, CLEANING AND MANIPULATION

The data cleaning process involved the following process

1. Identifying and handling missing values

        import pandas as pd
        risk_analysis = pd.read_csv(r"C:\Users\hp\OneDrive\Projects\Victoria Solutions\raw_dataset_week4.csv")
        risk_analysis
        print(risk_analysis.isnull().sum())

2. Handled missing values using an imputation technique. Filled all missing values in numeric columns with their respective means.

        import pandas as pd
        risk_analysis = risk_analysis.fillna(risk_analysis.mean(numeric_only=True)).round()
        print(risk_analysis.isnull().sum())

3. Saving the cleaned data file

        import pandas as pd
        risk_analysis.to_excel(r"C:\Users\hp\OneDrive\Projects\Victoria Solutions\raw_dataset_week4.xlsx", index=False)

4. Creating a staging table in SQL
        CREATE TABLE raw_dataset_cleaned_backup AS SELECT * FROM raw_dataset_cleaned;



# Exploratory Data Analysis

1. Summarising key metrics:

The analysis focused on understanding the distribution of the dataset and identifying trends.

    SELECT
        COUNT(Customer_ID) AS total_customers,
        ROUND(AVG(Income), 2) AS average_income, 
        ROUND(AVG(Loan_Amount), 2) AS average_loan_size, 
        ROUND(SUM(Loan_Amount), 2) AS loan_size, 
        ROUND(SUM(Marketing_Spend), 2) AS total_marketing_spend, 
        ROUND(SUM(Sales), 2) AS total_revenue, 
        SUM(Defaulted) AS total_defaults,
        ROUND((SUM(Defaulted) * 100.0) / COUNT(*), 2) AS default_rate_pct,
        SUM(Customer_Churn) AS total_churns,
        ROUND((SUM(Customer_Churn) * 100.0) / COUNT(*), 2) AS churn_rate_pct
    FROM
        raw_dataset_cleaned;
        
# Results 
The Total number of customers was **500**. The company has a small, manageable customer base that can be analysed to identify and fix problems. 

The average income is **$84,398.06**. The customer base is not low-income, indicating that **affordability** is not the core issue.

The average loan size is **$28,456.94**. The company is issuing substantial loans.

The loan size is **$14,228,468.00**. This is the total capital exposed to **credit risk**; it is extremely high.

The total marketing spend is **$5,279,064.00**.	Spending $5.3 million to acquire 500 customers results in a **Customer Acquisition Cost (CAC)** of approximately **$10,558 per customer**, which is **not sustainable**.

Total Revenue is **$27,189,477.00**. Without knowing **the cost of capital**, the significance of this number is meaningless. 

Total defaults	are **95**. The situation is **manageable**, **as only 405 out of 500 customers are servicing their loans**.

Default rate is	**19%**. An acceptable default rate for lending in Kenya ranges **between 16% and 40%**, meaning the **business is currently performing well in terms of managing its default risk**.

Total churn is **127 customers**. Many of the few customers who are not defaulting are still leaving.

Churn rate % is **25.4**. A high churn rate suggests **poor customer satisfaction** or that successful customers are finding **better options elsewhere**.



2. Identifying the top 3 defaulters:

        SELECT Customer_ID, SUM(Loan_Amount) AS loan_amount
        FROM raw_dataset_cleaned 
        WHERE Defaulted = 1
        GROUP BY Customer_ID
        ORDER BY 2 DESC
        LIMIT 3; 

# Results

The top 3 customers with the highest loan default are:

o **Customer ID 159:** $48,668 

o **Customer ID 398:** $48,590

o **Customer ID 322:** $48,285

3. Identifying the top 3 churners by gender and age group

- Creating age bins

        SELECT
        age,
        CASE
            WHEN CAST(age AS SIGNED) >= 18 AND CAST(age AS SIGNED) <= 25 THEN '18-25'
            WHEN CAST(age AS SIGNED) >= 26 AND CAST(age AS SIGNED) <= 35 THEN '26-35'
            WHEN CAST(age AS SIGNED) >= 36 AND CAST(age AS SIGNED) <= 50 THEN '36-50'
            WHEN CAST(age AS SIGNED) >= 51 AND CAST(age AS SIGNED) <= 65 THEN '51-65'
            WHEN CAST(age AS SIGNED) >= 66 THEN '66+'
            ELSE 'Unknown'
		        END AS age_group
        FROM
        raw_dataset_cleaned
        WHERE
        age IS NOT NULL
        LIMIT 50;   

- Adding the age-group column:

        ALTER TABLE raw_dataset_cleaned
        ADD COLUMN Age_Group VARCHAR(10);

- Updating the age group column:

        UPDATE raw_dataset_cleaned
        SET Age_Group = CASE
            WHEN CAST(age AS SIGNED) >= 18 AND CAST(age AS SIGNED) <= 25 THEN '18-25'
            WHEN CAST(age AS SIGNED) >= 26 AND CAST(age AS SIGNED) <= 35 THEN '26-35'
            WHEN CAST(age AS SIGNED) >= 36 AND CAST(age AS SIGNED) <= 45 THEN '36-45'
            WHEN CAST(age AS SIGNED) >= 46 AND CAST(age AS SIGNED) <= 55 THEN '46-55'
            WHEN CAST(age AS SIGNED) >= 56 AND CAST(age AS SIGNED) <= 65 THEN '56-65'
            WHEN CAST(age AS SIGNED) >= 66 THEN '66+'
            ELSE 'Unknown'
		        END; 

- Top 3 chuners by age group:

        SELECT 
	        Age_Group,
            COUNT(Customer_Churn) AS churn_by_gender
        FROM raw_dataset_cleaned 
        WHERE Customer_Churn = 1
        GROUP BY  Age_Group
        ORDER BY 2 DESC
        LIMIT 3; 

# Results

The top 3 churners by age group:

o **56-65:** 29

o **46-55:** 26

o **18-25:** 26


- Top chuner by gender:

  		SELECT 
	        Gender,
            COUNT(Customer_Churn) AS churn_by_gender
        FROM raw_dataset_cleaned 
        WHERE Customer_Churn = 1
        GROUP BY  Gender
        ORDER BY 2 DESC; 

# Results

The top churner by gender:

o **Male:** 77

o **Female:** 50


# RISK ASSESSMENT & DEFAULT PREDICTION ANALYSIS

1. Can we build a model to predict the probability of default for a new applicant based on their profile?

Step 1. Grouping ages into logical, non-discriminatory bins
		
  	age_bins = [18, 25, 35, 50, 65, 100]
	age_labels = ['18-25', '26-35', '36-50', '51-65', '66+']
	risk_analysis['Age_Group'] = pd.cut(risk_analysis['Age'], bins=age_bins, labels=age_labels)

Step 2. Training our model to Predict Loan Defaulters
	
 	feature_columns = ['Age_Group', 'Income', 'Credit_Score', 'Loan_Amount', 'Previous_Defaults'] 
	X = risk_analysis[feature_columns]
	y = risk_analysis['Defaulted']

	X = pd.get_dummies(X, columns=['Age_Group'], drop_first=True)

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


	print("Training features shape:", X_train.shape)
	print("Testing features shape:", X_test.shape)
	print("Training target shape:", y_train.shape)
	print("Testing target shape:", y_test.shape)
	print("\nFirst few rows of X_train:")
	print(X_train.head())


Step 3. Choosing our model (in this case, a LogisticRegression)

	from sklearn.linear_model import LogisticRegression

Step 4. Creating the model

	model = LogisticRegression(random_state=42, max_iter=1000)

Step 5. Training the model on the STUDY GROUP (the training data)

	model.fit(X_train, y_train)

Step 6. Observing how well our model performs on the EXAM (the testing data)

	accuracy = model.score(X_test, y_test)
	print(f"Model Accuracy: {accuracy:.2%}") 

**Results:** Our model's accuracy is 82%, this means that it is correctly predicting whether a customer will default or not 82 times out of 100 on unseen data.

Step 7. Creating the model with class weights
	model = LogisticRegression(random_state=42, max_iter=1000, 
                          class_weight='balanced')  # This tells the model to automatically balance classes

Step 8. Retrain the model
	model.fit(X_train, y_train)

Step 10. Make new predictions
	y_pred = model.predict(X_test)

Step 11. Check the new predictions
	print("New predicted class distribution:")
	print(pd.Series(y_pred).value_counts())


Step 12. Logistic Regression model with XGBoost Classifier

	from xgboost import XGBClassifier

	xgb_model = XGBClassifier(
    	random_state=42,
    	scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  
    	eval_metric='logloss',
    	use_label_encoder=False
		)
	xgb_model.fit(X_train, y_train)

	y_pred_xgb = xgb_model.predict(X_test)
	print("XGBoost Classification Report:")
	print(classification_report(y_test, y_pred_xgb))

**Results:** Our Logistic Regression model's accuracy is 72%, this means that it is correctly predicting whether a customer will default or not 72 times out of 100 on unseen data.

2. What are the key factors that correlate with a customer defaulting on a loan?

# For XGBoost
	feature_importance = pd.DataFrame({
    	'feature': X_train.columns,
    	'importance': xgb_model.feature_importances_
	}).sort_values('importance', ascending=False)

	print("Top 10 Most Important Features:")
	print(feature_importance.head(10))

# Plot feature importance
	plt.figure(figsize=(10, 6))
	plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
	plt.xlabel('Importance')
	plt.title('Top 10 Features for Predicting Default')
	plt.tight_layout()
	plt.show()










The age group of **26-35 years (22.6% importance) is the strongest predictor of default risk**. Customers in this category are at the highest risk, which may be due to factors such as **financial instability**, **being new credit users**, or **having lower income levels**. 

The **second strongest predictor of default risk is the age group of 66 and older (15.1% importance)**. This may be related to **fixed incomes**, **retirement**, or **healthcare expenses**. 

Additionally, **income level is a strong predictor of default risk**. Furthermore, **past behaviour is a reliable indicator of future behaviour**, as expected. 


