/*
CREATING A STAGING TABLE
*/
CREATE TABLE raw_dataset_cleaned_backup AS SELECT * FROM raw_dataset_cleaned;

-------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------
UPDATE raw_dataset_cleaned
SET
    Age = CAST(REPLACE(REPLACE(Age, ',', ''), ' ', '') AS UNSIGNED),
    Gender = REPLACE(REPLACE(Gender, ',', ''), ' ', ''),
    Spending_Score = CAST(REPLACE(REPLACE(Spending_Score, ',', ''), ' ', '') AS DECIMAL(10,2)),
    Credit_Score = REPLACE(REPLACE(Credit_Score, ',', ''), ' ', ''),
    Income = CAST(REPLACE(REPLACE(Income, ',', ''), ' ', '') AS DECIMAL(10,2)),
    Loan_Amount = CAST(REPLACE(REPLACE(Loan_Amount, ',', ''), ' ', '') AS DECIMAL(10,2)),
    Marketing_Spend = CAST(REPLACE(REPLACE(Marketing_Spend, ',', ''), ' ', '') AS DECIMAL(10,2)),
    Sales = CAST(REPLACE(REPLACE(Sales, ',', ''), ' ', '') AS DECIMAL(10,2)),
    Previous_Defaults = CAST(REPLACE(REPLACE(Previous_Defaults, ',', ''), ' ', '') AS UNSIGNED),
    Purchase_Frequency = CAST(REPLACE(REPLACE(Purchase_Frequency, ',', ''), ' ', '') AS DECIMAL(5,2)),
    Seasonality = REPLACE(REPLACE(Seasonality, ',', ''), ' ', ''),
    Customer_Churn = CAST(REPLACE(REPLACE(Customer_Churn, ',', ''), ' ', '') AS UNSIGNED),
    Defaulted = CAST(REPLACE(REPLACE(Defaulted, ',', ''), ' ', '') AS UNSIGNED);

-------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------
/*
DESCRIPTIVE ANALYTICS
*/
# Summarizing key metrics (average loan size, default rate, churn rate)

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
 
------------------------------------------------------------------------------------------------------------------------------------------------- 
-------------------------------------------------------------------------------------------------------------------------------------------------    
# Identifying the top 3 defaulters

SELECT Customer_ID, SUM(Loan_Amount) AS loan_amount
FROM raw_dataset_cleaned 
WHERE Defaulted = 1
GROUP BY Customer_ID
ORDER BY 2 DESC
LIMIT 3; 


-------------------------------------------------------------------------------------------------------------------------------------------------
/*
IDENTIFYING TOP THE 3 CHURNERS BY GENDER AND AGE-GROUP
*/

# CREATING AGE BINS
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

------------------------------------------------------------------------------------------------------------------------------------------------
# ADDING AND UPDATING THE AGE-GROUP COLUMN 

ALTER TABLE raw_dataset_cleaned
ADD COLUMN Age_Group VARCHAR(10);

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

-------------------------------------------------------------------------------------------------------------------------------------------------
# ANALYSING CHURNERS BASED ON AGE-GROUP AND GENDER
SELECT 
	Age_Group,
    COUNT(Customer_Churn) AS churn_by_gender
FROM raw_dataset_cleaned 
WHERE Customer_Churn = 1
GROUP BY  Age_Group
ORDER BY 2 DESC
LIMIT 3; 



SELECT 
	Gender,
    COUNT(Customer_Churn) AS churn_by_gender
FROM raw_dataset_cleaned 
WHERE Customer_Churn = 1
GROUP BY  Gender
ORDER BY 2 DESC; 
-------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------
 /*
 FINANCIAL PRODUCT ANALYSIS 
 */
 # What is the typical loan amount granted based on income and credit score?
 # creating segments based on income and credit score
 
 
# CREATE GROUPS BASED ON INCOME BRACKETS 
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
 
  
# CREATE GROUPS BASED ON CREDIT SCORE TIERS
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
 
ALTER TABLE raw_dataset_cleaned
DROP COLUMN Credit_Score_Tiers;
 
UPDATE raw_dataset_cleaned
SET Credit_Score_Tiers = CASE
    WHEN Credit_Score >= 0 AND Credit_Score <= 579 THEN 'Poor'
        WHEN Credit_Score > 580 AND Credit_Score <= 669 THEN 'Fair'
        WHEN Credit_Score > 670 AND Credit_Score <= 739 THEN 'Good'
        WHEN Credit_Score > 740 AND Credit_Score <= 799 THEN 'Very Good'
        WHEN Credit_Score > 800 THEN 'Excellent'
        ELSE 'Unknown'
    END;
 
 -----------------------------------------------------------------------------------------------------------------------------------------------
 # CALCULATING KEY STATISTICS FOR THE LOAN AMOUNT BASED ON INCOME BRACKETS AND CREDIT SCORE TIER USING A COMMON TABLE EXPRESSION (CTE)
 
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
 
-------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------------
/*
CUSTOMER SEGMENTATION FOR MARKETING 

Can we segment customers into groups (e.g., "high-value low-risk," "high-risk," "credit builders") to tailor marketing offers and loan products?
*/

# Segmenting Customers Based on Risk and Value using a CASE STATEMENT

SELECT 
    Customer_ID,
    Income_Bracket,
    Credit_Score_Tiers,
   CASE 
        WHEN Credit_Score_Tiers IN ('Good', 'Very Good', 'Excellent') 
             AND Income_Bracket IN ('$75k-100k', '$100k+') 
             THEN 'High-Value, Low-Risk'
        
        WHEN Credit_Score_Tiers IN ('Fair', 'Poor') 
             AND Income_Bracket IN ('$75k-100k', '$100k+') 
             THEN 'High-Value, High-Risk'
        
        WHEN Credit_Score_Tiers IN ('Good', 'Very Good', 'Excellent') 
             AND Income_Bracket IN ('$0-50k', '$50k-75k') 
             THEN 'Low-Value, Low-Risk'
        
        WHEN Credit_Score_Tiers IN ('Fair', 'Poor') 
             AND Income_Bracket IN ('$0-50k', '$50k-75k') 
             THEN 'Low-Value, High-Risk'
        
        ELSE 'Needs Review'
    END AS Risk_Value_Segment,

    Loan_Amount,
    Marketing_Spend
    
FROM raw_dataset_cleaned;

----------------------------------------------------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------------------------------------------------
