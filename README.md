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


# RISK ASSESSMENT & DEFAULT PREDICTION ANALYSIS

1. Identifying the top 3 defaulters:

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

2. Can we build a model to predict the probability of default for a new applicant based on their profile?

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

**Results:** From to the classification report, **the model achieves a 76% accuracy rate in predicting loan defaults**. It demonstrates a strong capability to identify **creditworthy customers**, with a **recall rate of 83-89% for class 0**. However, the model **struggles to accurately predict defaulters**, reflected in its **low precision and recall rates for class 1.** This pattern is quite common in credit risk modelling, as defaults are rare and difficult to predict.

3. What are the key factors that correlate with a customer defaulting on a loan?

	<img width="940" height="496" alt="image" src="https://github.com/user-attachments/assets/0ff08546-801a-492f-bccc-4a96dfff4356" />


For XGBoost:

	feature_importance = pd.DataFrame({
    	'feature': X_train.columns,
    	'importance': xgb_model.feature_importances_
	}).sort_values('importance', ascending=False)

	print("Top 10 Most Important Features:")
	print(feature_importance.head(10))

Plot feature importance:

	plt.figure(figsize=(10, 6))
	plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
	plt.xlabel('Importance')
	plt.title('Top 10 Features for Predicting Default')
	plt.tight_layout()
	plt.show()

**Results:**

The age group of **26-35 years (22.6% importance) is the strongest predictor of default risk**. Customers in this category are at the highest risk, which may be due to factors such as **financial instability**, **being new credit users**, or **having lower income levels**. 

The **second strongest predictor of default risk is the age group of 66 and older (15.1% importance)**. This may be related to **fixed incomes**, **retirement**, or **healthcare expenses**. 

Additionally, **income level is a strong predictor of default risk**. Furthermore, **past behaviour is a reliable indicator of future behaviour**, as expected. 


# CUSTOMER CHURN ANALYSIS 

1. Identifying the top 3 churners by gender and age group

	<img width="941" height="612" alt="image" src="https://github.com/user-attachments/assets/f8e0b1f4-fad8-4264-b2e5-49fe3cf1600d" />


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

2. Why are customers leaving? What patterns distinguish customers who churn from those who stay?

- Checking the overall churn rate
  
		import pandas as pd
		import numpy as np
		import matplotlib.pyplot as plt
		import seaborn as sns

		churn_rate = df['churn'].mean() * 100
		print(f"Overall Churn Rate: {churn_rate:.2f}%")

		plt.figure(figsize=(6,4))
		sns.countplot(x='churn', data=df)
		plt.title('Customer Churn Distribution')
		plt.show()

		for p in ax.patches:
   			 ax.annotate(f'{p.get_height():.0f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', 
                xytext=(0, 5), 
                textcoords='offset points',
                fontsize=12)

		plt.show()

	<img width="940" height="402" alt="image" src="https://github.com/user-attachments/assets/02bd2fa8-7704-4f0c-b227-637fc7963998" />

The churn rate is **25.40%**, which means that **25% (n = 127) of the customers have left**. Approximately 373 customers remained with the company. 

A churn rate of 25.4% indicates that 1 in 4 customers are leaving, **representing a significant risk to revenue**. 

This level of churn highlights an urgent need for effective retention strategies. 

Common reasons for high churn rates include: **Competitive pressure**, **Poor customer experience**, **Pricing issues**, and **Products or services not meeting customer expectations**


**Recommendations:**

1. Investigate primary reasons for churn through:
   
•	Exit interviews

•	Customer satisfaction surveys

•	Usage pattern analysis

2. Develop targeted retention programs for high-risk segments

- Comparing churn rates across various age groups
  
		segment_churn = risk_analysis.groupby('Age_Group', observed=True)['Customer_Churn'].mean().sort_values(ascending=False) * 100

		print("Churn Rate by Age Group:")
		print(segment_churn)

		plt.figure(figsize=(10,6))
		segment_churn.plot(kind='bar')
		plt.title('Churn Rate by Age Group')
		plt.ylabel('Churn Rate (%)')
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.show()

	<img width="940" height="560" alt="image" src="https://github.com/user-attachments/assets/75c9b1af-99ae-40a2-8401-be935a179b3f" />

Age_Group:

	18-25    30.434783
	51-65    27.083333
	36-50    25.000000
	66+      21.428571
	26-35    17.808219
	Name: Customer_Churn, dtype: float64

**Results:**

Young customers, particularly those **aged 18 to 25, are the most likely to leave**. This trend can be attributed to several factors, including **financial instability**, **a desire to explore different options**, **a lack of loyalty**, and s**ignificant life transitions such as college or starting their first job**. 

In contrast, t**he most loyal customers tend to be in the age group of 26 to 35 years**. This loyalty is often related to **career stability**, **established banking relationships**, and **family financial responsibilities**. 

Meanwhile, **senior citizens**, **aged 66 and above**, have a **relatively low churn rate of 21%**. They are **generally more stable in their banking choices**, as **they are less likely to switch providers**and **often adhere to more traditional banking habits**.


3. Is there a relationship between loan approval amounts, credit score, and customer churn?

- Analyzing Churn by Numerical Features (e.g., Income, Loan Amount, Credit Score)
  
		numerical_features = risk_analysis[['Income', 'Loan_Amount', 'Credit_Score', 'Age', Previous_Defaults']]

		fig, axes = plt.subplots(2, 3, figsize=(15, 10)) 
		axes = axes.flat

		for i, col in enumerate(numerical_features):
    	if col in risk_analysis.columns:
       	 	sns.boxplot(x='Customer_Churn', y=col, data= risk_analysis, ax=axes[i])
        	axes[i].set_title(f'Churn vs. {col.title()}')

		for j in range(i+1, len(axes)):
    	axes[j].set_visible(False)

		plt.tight_layout()
		plt.show()

	<img width="940" height="313" alt="image" src="https://github.com/user-attachments/assets/45cbb675-1885-4ad2-a2a4-a9844eed0306" />

	<img width="940" height="315" alt="image" src="https://github.com/user-attachments/assets/c03854ed-3721-4800-9eae-c142f2565e4d" />

**Income vs. Churn**

Customers who churn tend to **have slightly lower median incomes than those who remain**. This suggests that **lower-income customers may be more price-sensitive or financially strained**, making them more likely to seek better offers elsewhere.

**Loan Amount vs. Churn**

Customers with **smaller loan amounts are more likely to churn**. Those with smaller loans **may feel less committed to the relationship** or **are possibly testing the service**.

**Credit Score vs. Churn**

Customers who churn generally **have significantly lower credit scores**. Customers with lower credit scores **may be dissatisfied with the terms or rates**, or **they might be looking for better offers**that they now qualify for.

**Age vs. Churn**

Younger customers **are more likely to churn compared to older customers**. This may be because **younger customers tend to be less loyal**, are **more inclined to shop around**, or **have changing financial needs**. Age is a significant factor in the probability of churn.

**Previous Defaults vs. Churn**

Customers with **previous defaults are more likely to churn**. This could indicate **dissatisfaction with service terms**, or it might suggest that **these customers are being "managed out" of the service**.

# CUSTOMER SEGMENTATION FOR MARKETING 

1. Can we segment customers into groups (e.g., "high-value low-risk," "high-risk," "credit builders") to tailor marketing offers and loan products?



2. How does marketing spend correlate with purchase frequency or new loan uptake? Is the marketing budget being spent effectively?



# FINANCIAL PRODUCT ANALYSIS

1. What is the typical loan amount granted based on income and credit score?

- Creating Groups Based on Income Brackets and Credit Score Tiers

- Income Brackets:
  
	SELECT
    	Income,  
   		CASE
        	WHEN Income >= 0 AND Income <= 50000 THEN '$0-50k'
        	WHEN Income > 50000 AND Income <= 75000 THEN '$50k-75k'
        	WHEN Income > 75000 AND Income <= 100000 THEN '$75k-100k'
        	WHEN Income > 100000 THEN '$100k+'
        	ELSE 'Unknown'
    	END AS Income_Bracket
	FROM
    	raw_dataset_cleaned;


	ALTER TABLE raw_dataset_cleaned
	ADD COLUMN Income_Bracket VARCHAR(10);

 
	UPDATE raw_dataset_cleaned
	SET Income_Bracket = CASE
 	WHEN Income >= 0 AND Income <= 50000 THEN '$0-50k'
        WHEN Income > 50000 AND Income <= 75000 THEN '$50k-75k'
        WHEN Income > 75000 AND Income <= 100000 THEN '$75k-100k'
        WHEN Income > 100000 THEN '$100k+'
        ELSE 'Unknown'
    END;

- Credit Score Tiers

 		SELECT                
    		Credit_Score,  
    		CASE
       			WHEN Credit_Score >= 0 AND Credit_Score <= 579 THEN 'Poor'
        		WHEN Credit_Score > 580 AND Credit_Score <= 669 THEN 'Fair'
        		WHEN Credit_Score > 670 AND Credit_Score <= 739 THEN 'Good'
        		WHEN Credit_Score > 740 AND Credit_Score <= 799 THEN 'Very Good'
        		WHEN Credit_Score > 800 THEN 'Excellent'
        		ELSE 'Unknown'
    		END AS Credit_Score_Tier
		FROM
    		raw_dataset_cleaned;

		ALTER TABLE raw_dataset_cleaned
		ADD COLUMN Credit_Score_Tiers VARCHAR(30);

		UPDATE raw_dataset_cleaned
		SET Credit_Score_Tiers = CASE
   			WHEN Credit_Score >= 0 AND Credit_Score <= 579 THEN 'Poor'
        	WHEN Credit_Score > 580 AND Credit_Score <= 669 THEN 'Fair'
        	WHEN Credit_Score > 670 AND Credit_Score <= 739 THEN 'Good'
        	WHEN Credit_Score > 740 AND Credit_Score <= 799 THEN 'Very Good'
       	 	WHEN Credit_Score > 800 THEN 'Excellent'
        	ELSE 'Unknown'
    	END;

- Using a CTE to calculate the loan amount based on income brackets and credit score tiers

		WITH median_calc AS (
    	SELECT
          	t.Income_Bracket,
        	t.Credit_Score_Tiers,
        	AVG(t.Loan_Amount) as median_loan_amount
    	FROM (
        SELECT 
            Income_Bracket,
            Credit_Score_Tiers,
            Loan_Amount,
            COUNT(*) OVER (PARTITION BY Income_Bracket, Credit_Score_Tiers) as group_count,
            ROW_NUMBER() OVER (PARTITION BY Income_Bracket, Credit_Score_Tiers ORDER BY Loan_Amount) as row_num
       	FROM raw_dataset_cleaned
        WHERE Loan_Amount IS NOT NULL
   		 ) t
    	WHERE row_num BETWEEN group_count/2.0 AND group_count/2.0 + 1
    	GROUP BY Income_Bracket, Credit_Score_Tiers
		),
		mode_calc AS (
    	SELECT 
        	Income_Bracket,
        	Credit_Score_Tiers,
        	Loan_Amount as mode_loan_amount
    	FROM (
       	SELECT 
            Income_Bracket,
            Credit_Score_Tiers,
            Loan_Amount,
            COUNT(*) as frequency,
            ROW_NUMBER() OVER (PARTITION BY Income_Bracket, Credit_Score_Tiers ORDER BY COUNT(*) DESC) as rn
        FROM raw_dataset_cleaned
        WHERE Loan_Amount IS NOT NULL
        GROUP BY Income_Bracket, Credit_Score_Tiers, Loan_Amount
    	) freq_table
    	WHERE rn = 1
		)
		SELECT 
    		t.Income_Bracket,
    		t.Credit_Score_Tiers,
    		COUNT(t.Loan_Amount) as loan_count,
    		ROUND(AVG(t.Loan_Amount), 2) as mean_loan_amount,
    		MAX(mc.median_loan_amount) as median_loan_amount,  -- Using MAX() aggregate function
    		MAX(moc.mode_loan_amount) as mode_loan_amount,     -- Using MAX() aggregate function
    		MIN(t.Loan_Amount) as min_loan_amount,
    		MAX(t.Loan_Amount) as max_loan_amount,
    		ROUND(STDDEV(t.Loan_Amount), 2) as std_dev_loan_amount
		FROM raw_dataset_cleaned as t
		LEFT JOIN median_calc mc ON t.Income_Bracket = mc.Income_Bracket 
                        AND t.Credit_Score_Tiers = mc.Credit_Score_Tiers
		LEFT JOIN mode_calc moc ON t.Income_Bracket = moc.Income_Bracket 
                        AND t.Credit_Score_Tiers = moc.Credit_Score_Tiers
		WHERE t.Loan_Amount IS NOT NULL
		GROUP BY t.Income_Bracket, t.Credit_Score_Tiers
		ORDER BY 
    		CASE t.Income_Bracket
        		WHEN '$0-50k' THEN 1
        		WHEN '$50k-75k' THEN 2
        		WHEN '$75k-100k' THEN 3
        		WHEN '$100k+' THEN 4
  			ELSE 5
    		END,
   		 	CASE t.Credit_Score_Tiers
        		WHEN 'Poor' THEN 1
        		WHEN 'Fair' THEN 2
        		WHEN 'Good' THEN 3
        		WHEN 'Very Good' THEN 4
        		WHEN 'Excellent' THEN 5
        	ELSE 6
    		END;

	<img width="940" height="841" alt="image" src="https://github.com/user-attachments/assets/b52b9868-24e2-4cd0-b39f-01cc82885035" />

Key insights and interpretations regarding loan amounts based on income brackets and credit score tiers:

•	**Borrowers with very good credit generally receive higher loan amounts across most income brackets**. For example, customers in the $0–50k income bracket received an average loan amount of $29,809.60. 

•	**Poor credit borrowers consistently receive lower loan amounts**, regardless of their income level.

•	**Higher income does not always correlate with higher loan amounts**. Interestingly, some **lower-income brackets sometimes have higher average loan amounts than higher-income** brackets within the same credit tier.

•	The **$75k–100k income bracket exhibits the most variability in lending patterns** across different credit tiers.

•	**Data Quality Issues:** Several entries show minimum values higher than maximum values (e.g., $0–50k/Poor: min 10,688 > max 9,795), indicating potential data errors.

•	**Standard Deviation:** Higher standard deviations suggest greater variability in loan amounts within certain groups, particularly in the lower income brackets.


2. How does spending behaviour (spending score) relate to income and creditworthiness?

	import pandas as pd
	from sklearn.linear_model import LinearRegression
	X = df[[Income', Credit_Score']]
	y = df[Spending_Score]
	model = LinearRegression ()
	model.fit(X, y)
	print(model.coef_, model.intercept_)

	<img width="880" height="66" alt="image" src="https://github.com/user-attachments/assets/da662135-510f-44f0-a082-d0fc7dff3f93" />

The multiple linear regression (MLR) model is:

	Spending Behaviour = 50.1481 + (-4.7341 * Income) + (8.2135 * Credit Score)

A **moderate negative correlation exists between income and spending (borrowing) behaviour**. Thus, for each additional increase in income, customer borrowing behaviour decreases by 4.7341 units, assuming the credit score remains constant. This indicates that **a rise in income does not necessarily lead to an increase in client borrowing**.

On the other hand, **a very strong positive relationship is evident between credit score and spending behaviour**. The analysis indicates that for every one-unit increase in credit score, customer borrowing behaviour increases by 8.2135 units, provided income remains constant. This suggests that **customers with higher credit scores tend to borrow more, and vice versa**.
