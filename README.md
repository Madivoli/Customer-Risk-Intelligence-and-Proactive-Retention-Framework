# Customer-Risk-Intelligence-and-Proactive-Retention-Framework

# Overview

This repository contains 500 records of customers from a financial institution. It includes demographic information (age, gender, income), financial behaviour (spending score, credit score, loan amount), historical risk (previous defaults), marketing engagement (marketing spend), and critical outcomes (customer churn, defaults). 

The primary goal is to leverage this data to mitigate risk, improve customer retention, and optimise marketing strategies.


# Analysis and Business Questions

The analysis is divided into 4 key areas and seeks to answer the following business questions:
1. Risk Assessment & Default Prediction:
Question: What are the key factors that correlate with a customer defaulting on a loan?
Question: Can we build a model to predict the probability of default for a new applicant based on their profile?

2. Customer Churn Analysis:
Question: Why are customers leaving? What patterns distinguish customers who churn from those who stay?
Question: Is there a relationship between loan approval amounts, credit score, and customer churn?

3. Customer Segmentation for Marketing:
Question: Can we segment customers into groups (e.g., "high-value low-risk," "high-risk," "credit builders") to tailor marketing offers and loan products?
Question: How does marketing spend correlate with purchase frequency or new loan uptake? Is the marketing budget being spent effectively?

4. Financial Product Analysis:
Question: What is the typical loan amount granted based on income and credit score?
Question: How does spending behaviour (spending score) relate to income and creditworthiness?


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

The analysis focused on understanding the distribution of the dataset and identifying trends.

    SELECT
        COUNT(Customer_ID) AS total_customers,
        ROUND(AVG(Income), 2) AS average_income, 
        ROUND(AVG(Loan_Amount), 2) AS average_loan_size, 
        ROUND(SUM(Loan_Amount), 2) AS loan_size, 
        ROUND(SUM(Marketing_Spend), 2) AS total_marketing_spend, 
        ROUND(SUM(Sales), 2) AS total_revenue, 
        SUM(Previous_Defaults) AS total_defaults,
        ROUND((SUM(Previous_Defaults) * 100.0) / COUNT(*), 2) AS default_rate_pct,
        SUM(Customer_Churn) AS total_churns,
        ROUND((SUM(Customer_Churn) * 100.0) / COUNT(*), 2) AS churn_rate_pct
    FROM
        raw_dataset_cleaned;
        
# Results 
The Total number of customers was **500**. The company has a small, manageable customer base that can be analysed to identify and fix problems. 


**Metric**	                    **Figure**	                **Interpretation**
Total Customers	                **500**	                    The company has a small, manageable customer base that can be analyzed to identify and fix problems.
Average Income	                **$84,398.06**	            The customer base is not low-income, indicating that affordability is not the core issue.
Average Loan Size	            **$28,456.94**              The company is issuing substantial loans.
Loan Size	14,228,468.00	The total capital exposed to credit risk is extremely high.
Total Marketing Spend	5,279,064.00	Spending $5.3 million to acquire 500 customers results in a Customer Acquisition Cost (CAC) of approximately $10,558 per customer, which is not sustainable.
Total Revenue	27,189,477.00	Without knowing the cost of capital, the significance of this number is meaningless. With a default rate of 97.4%, the revenue generated is almost entirely wiped out by losses.
Total Defaults	487	The situation is catastrophic, as only 13 out of 500 customers are servicing their loans.
Default Rate %	97.4	Extremely high. An acceptable default rate for lending in Kenya ranges between 16% and 40%, meaning the business is currently losing money.
Total Churns	127	Many of the few customers who are not defaulting are still leaving.
Churn Rate %	25.4	A high churn rate that suggests poor customer satisfaction or that successful customers are finding better options elsewhere.

    
