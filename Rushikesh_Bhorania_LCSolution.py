import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
import statsmodels.api as sm
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import t
import warnings
from scipy.stats import ttest_1samp

# Problem statement

#LendingClub is online loan markplace providing loan to customer through fast online interface. For each loan application company has to make either of two decision:
#1. Loan accepted: If the company approves the loan, there are 3 possible scenarios described below:
#    Fully paid: Applicant has fully paid the loan (the principal and the interest rate)
#    Current: Applicant is in the process of paying the instalments, i.e. the tenure of the loan is not yet completed. These candidates are not labelled as 'defaulted'.
#    Charged-off: Applicant has not paid the instalments in due time for a long period of time, i.e. he/she has defaulted on the loan 
#2. Loan rejected: The company had rejected the loan (because the candidate does not meet their requirements etc.). Since the loan was rejected, there is no transactional history of those applicants with the company and so this data is not available with the company (and thus in this dataset)

#Lending business has its own risk which result in credit loss. 

#### Credit loss is amount of money lost by lender when borower do not payback the loan. In this case study such loans are identified as 'charged-off' which ideally would be called as 'defaulters'
#Scope of the project
### To identify risky loan applicants, then such loans can be reduced thereby cutting down the amount of credit loss. 
#### 1. Identification of such applicants using EDA is the aim of this case study.
#### 2. In other words, the company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  
#### 3. The company can utilise this knowledge for its portfolio and risk assessment. 

#import data for loan
#lending_club = loan_raw_data file
loan_raw_data = pd.read_csv("loan.csv")

# code to identify total number of row and column in database
loan_raw_data.shape

# code for obtaining column name in list. # This list will be used to further manipulate the dataframe by either adding or removing columns
column_names = loan_raw_data.columns.tolist()
print("Column id:" , column_names)


### Data Cleaning
#1. To identify columns with 100% missing data. This is clubbed with identifying columns with more than 30% missing data.
#2. To identify columns haivng only unique values
#Since the loan data has huge number of columns with missing data, all these columns needs to be removed to reduce the data for analysis. List comprehension is created to identify all columns with more than 30% - 100% of data missing using following code.
#Subset of data is created on which further analysis will be done. 
#New dataframe with name "Loan_Review_data" is created by dropping the columns with 100% missing data.

# To identify the columns which are blanks. 
# Since data is large, display options will need to be set to max for columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# identify missing values in each column
missing_data = 100 * loan_raw_data.isnull().mean()
print("Missing columns:", missing_data)

# code for filtering columns with more than 30% data missing.
missing_percent = loan_raw_data.isnull().mean() * 100
columns_missing = [column for column in loan_raw_data.columns if missing_percent[column] > 30]

# code for removing columns having more than 30% missing data (i.e. it identifies columns which have 30% data missing)
loan_review_subdata = loan_raw_data.drop(columns=columns_missing)

# Identify columns which do not have duplicate values
columns_with_duplicates = loan_review_subdata.columns[loan_review_subdata.apply(lambda x: x.nunique() < len(loan_review_subdata))]

# Create a subset DataFrame by dropping columns without duplicates
loan_subdata = loan_review_subdata.drop(columns=set(loan_review_subdata.columns) - set(columns_with_duplicates), axis=1)

# list of columns that can be removed from analysis
extra_columns = ['emp_title', 'emp_length', 'pymnt_plan', 'title', 'initial_list_status', 'last_pymnt_d', 'last_credit_pull_d', 'collections_12_mths_ex_med', 'policy_code', 'application_type', 'acc_now_delinq', 'chargeoff_within_12_mths', 'delinq_amnt', 'pub_rec_bankruptcies', 'tax_liens']

# Drop the specified columns. Resulting dataframe will be used as final dataframe for analysis
loan_data = loan_subdata.drop(columns=extra_columns)

update_column_names = loan_data.columns.tolist()
print("Updated Column Name:" , update_column_names)
len(column_names)

#### Data processing and arranging
#1. Data is arrange in list with two classification. (a) Data with more than 30 unique value and (b) data with less than equal to 30 unique value
#2. Data is arranged in list with two classification. (a) Numerical data and (b) categorical data. To do this data with less than equal to 30 unique value is considered as categorical data
#3. Interest rate data is converted into numberical values.
#4. 2 columns are added by converting loan status and loan verified data into numerical values

loan_data.nunique()

# Converting loan status data into numerical values and adding column to dataset
def map_loan_status(status):
    if status == "Fully Paid":
        return 1
    elif status == "Current":
        return 2
    elif status == "Charged Off":
        return 3
    else:
        return None

# Adding new column 'loan_status_numerical' using the apply function
loan_data['loan_status_numerical'] = loan_data['loan_status'].apply(map_loan_status)

loan_data.head(3)

# Converting loan verified data into numerical values and adding column to dataset
def map_verification_status(status):
    if status == "Source Verified":
        return 1
    elif status == "Verified":
        return 1
    elif status == "Not Verified":
        return 0
    else:
        return None 

loan_data['loan_verified_numerical'] = loan_data['verification_status'].apply(map_verification_status)


# Count of total number of verified loans and count of loans that have status charged off
num_verified_loans = len(loan_data[loan_data['verification_status'] == 'Verified'])
num_verified_charged_off = len(loan_data[(loan_data['verification_status'] == 'Verified') & (loan_data['loan_status'] == 'Charged Off')])
print("Number of Verified Loans:", num_verified_loans)
print("Number of Charged Off Verified Loans:", num_verified_charged_off)

# Count of total number of source verified loans and count of loans that have status charged off
num_source_verified_loans = len(loan_data[loan_data['verification_status'] == 'Source Verified'])
num_source_verified_charged_off = len(loan_data[(loan_data['verification_status'] == 'Source Verified') & (loan_data['loan_status'] == 'Charged Off')])
print("Number of Source Verified Loans:", num_source_verified_loans)
print("Number of Charged Off Source Verified Loans:", num_source_verified_charged_off)

# Count of total number of non verified loans and count of loans that have status charged off
num_non_verified_loans = len(loan_data[loan_data['verification_status'] == 'Not Verified'])
num_non_verified_charged_off = len(loan_data[(loan_data['verification_status'] == 'Not Verified') & (loan_data['loan_status'] == 'Charged Off')])
print("Number of Non-Verified Loans:", num_non_verified_loans)
print("Number of Charged Off Non-Verified Loans:", num_non_verified_charged_off)

# Display the DataFrame to verify the new column
loan_data.head(3)


# to identify the columns which have more than 30 unique value. These columns will be used for categorical data analysis
unique_columns = loan_data.nunique()
columns_morethan_30unique = unique_columns[unique_columns>30].index.tolist()
columns_lessthan_30unique = unique_columns[unique_columns<=30].index.tolist()
print("Columns with more than 30 unique:",columns_morethan_30unique)
print("Columns with less than 30 unique:",columns_lessthan_30unique)
len(extra_columns)

# coverting interset value from str to float
loan_data["int_rate"] = loan_data["int_rate"].str.replace("%", "").astype(float)

# create list of columns containing integer dtypes
numerical = loan_data.select_dtypes(include=['int','float'])

final_numerical = numerical.columns.tolist()
print("Numerical Columns:" , final_numerical)

# creat list of columns containing strng or object dtypes
category = loan_data.select_dtypes(include=['object'])

final_categorical = category.columns.tolist()
print("Category Columns:", final_categorical)

# List of colimns for numerical and categorical analysis
numerical_columns = ['loan_amnt', 'int_rate', 'installment', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'revol_bal', 'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt']
category_columns = ['term', 'grade', 'sub_grade', 'home_ownership', 'verification_status', 'loan_status', 'purpose']

### Performing EDA:
#1. Preparing plot of categorical data to evaluate any trend in data on single column
#2. Prepare plot of numerical data to evaluate any trend in numerical data in single column
#3. Preparing plot of categorical and numberical data to see any correlation between the data
#4. Prepare correlation plot of numberical data with numerical data to evaluate any trend between the data

#code for categorical analysis
for i in category_columns:
    plt.figure(figsize=(14, 8))
    print("Countplot of", i)
    
    # Sort the unique values of the column
    sorted_values = loan_data[i].sort_values().unique()
    
    ax = sns.countplot(x=loan_data[i], order=sorted_values)
    
    # Calculate the percentage of each category
    total = float(len(loan_data[i]))
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + total * 0.01,  # Adjust the vertical position here
                '{:.2f}%'.format((height / total) * 100),
                ha='center')
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels
    plt.show()

# code for numerical analysis
numerical_col = ['loan_amnt', 'int_rate', 'installment', 'dti']
for i in numerical_col:
    plt.figure(figsize=(10, 8))
    print("Histplot of", i)
    ax = sns.histplot(x=loan_data[i])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels
    plt.show()

# Code for comparative analysis
numerical_col = ['loan_amnt', 'int_rate']
category_col = ['term', 'grade', 'verification_status', 'loan_status']

for col1 in category_col:
    for col2 in numerical_col:
        print("Boxplot of", col1, "vs", col2)
        sorted_data = loan_data.sort_values(by=col1)
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=sorted_data[col1], y=sorted_data[col2])
        plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
        plt.show()

# Co-relation analysis of data
plt.figure(figsize=(16,14))
sns.heatmap(loan_data[['loan_amnt', 'int_rate', 'installment','loan_status_numerical', 'loan_verified_numerical']].corr(),annot=True)
plt.show()

## Statistical analysis
### Analysis 1: To detrmine if there is any difference in interest rate of loan that are fully paid vs loans that are charged off.
#Approach:
#1. Interest rate data is divided into 2 parts. (a) interest rate of loan that are fully paid vs (b) interest rate of loan that is charged off.
#2. 1 pair t-test is performed on this data to evaluate if there is any difference between these two data.
#### Rational for using 1-pair t-test is because this is qualititative analysis to determine if there is any statistical difference between interest rate of fully paid loan vs charged off loan

# Subseting the data for fully paid and charged off loans
fully_paid_interest_rates = loan_data.loc[loan_data['loan_status'] == 'Fully Paid', 'int_rate']
charged_off_interest_rates = loan_data.loc[loan_data['loan_status'] == 'Charged Off', 'int_rate']

# independent 1 pair t-test
t_stat, p_value = ttest_ind(fully_paid_interest_rates, charged_off_interest_rates)

# Interpreting p value
if p_value < 0.05:
    print("There is a significant difference in interest rates between fully paid and charged off loans.")
else:
    print("There is no significant difference in interest rates between fully paid and charged off loans.")

# calculating correlation
correlation = loan_data['int_rate'].corr(loan_data['loan_status_numerical'])
print("Correlation coefficient between interest rate and loan status:", correlation)

# correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(loan_data[['int_rate', 'loan_status_numerical']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# correlation boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([loan_data['int_rate'],loan_data['loan_status_numerical']])
plt.title('Correlation between Interest Rate and Loan Status')
plt.xlabel('Interest Rate')
plt.ylabel('Loan Status')
plt.grid(True)
plt.show()

print("Correlation between Interest Rate and Loan Status:", correlation)

### Analysis 2: To detrmine if there is any difference in default rate of loan that are verified or source verified vs. unverified laons.
#Approach:
#1. Default rate is calculated from loan status and loan verified data. Loan verified data is convered in numberical data to have quantitative evaluation. From this data default rate of verified loan vs unverified loan is determined.
#2. From this data statistical parameter like confidence interval, sample size, pooled standard deviaiton, margin of error are calculated and 2 pair t-test is performed on this data to evaluate if there is any difference between these two data. 

# Calculating default rates
num_verified_charged_off = len(loan_data[(loan_data['loan_verified_numerical'] == 1) & (loan_data['loan_status'] == 'Charged Off')])
num_verified_loans = len(loan_data[loan_data['loan_verified_numerical'] == 1])
verified_default_rate = num_verified_charged_off / num_verified_loans

num_unverified_charged_off = len(loan_data[(loan_data['loan_verified_numerical'] == 0) & (loan_data['loan_status'] == 'Charged Off')])
num_unverified_loans = len(loan_data[loan_data['loan_verified_numerical'] == 0])
unverified_default_rate = num_unverified_charged_off / num_unverified_loans

# Ploting count of loan status vs verification status
plt.figure(figsize=(10, 6))
sns.countplot(x='loan_status', hue='verification_status', data=loan_data)
plt.title('Count of Loan Status vs Verification Status')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.legend(title='Verification Status')
plt.show()

# Calculating sample standard deviations
std_verified = np.std(loan_data.loc[loan_data['loan_verified_numerical'] == 1, 'loan_status_numerical'], ddof=1)
std_unverified = np.std(loan_data.loc[loan_data['loan_verified_numerical'] == 0, 'loan_status_numerical'], ddof=1)

# Calculating sample sizes
n_verified = len(loan_data.loc[loan_data['loan_verified_numerical'] == 1])
n_unverified = len(loan_data.loc[loan_data['loan_verified_numerical'] == 0])

# Calculating pooled standard deviation
pooled_std = np.sqrt(((n_verified - 1) * std_verified ** 2 + (n_unverified - 1) * std_unverified ** 2) / (n_verified + n_unverified - 2))

# Performing independent two-tailed t-test
t_stat, p_value = ttest_ind(
    loan_data.loc[loan_data['loan_verified_numerical'] == 1, 'loan_status_numerical'],
    loan_data.loc[loan_data['loan_verified_numerical'] == 0, 'loan_status_numerical']
)

# Calculating degrees of freedom
degrees_of_freedom = len(loan_data[(loan_data['loan_verified_numerical'] == 1) & (loan_data['loan_status'] == 'Charged Off')]) + len(loan_data[(loan_data['loan_verified_numerical'] == 0) & (loan_data['loan_status'] == 'Charged Off')]) - 2

# Calculating confidence level
confidence_level = 0.95

# Calculating confidence interval
ci_lower, ci_upper = t.interval(confidence_level, df=degrees_of_freedom, loc=t_stat, scale=pooled_std)

# Calculating alpha
alpha = 1 - confidence_level

# Printing the results
print("Verified Default Rate:", verified_default_rate)
print("Unverified Default Rate:", unverified_default_rate)
print("t-statistic:", t_stat)
print("p-value:", p_value)
print("Confidence Level:", confidence_level)
print("Confidence Interval ({}%): ({}, {})".format(confidence_level * 100, ci_lower, ci_upper))
print("Alpha value:", alpha)

#Interpreting the result
if p_value < alpha:
    print("There is a significant difference in default rates between verified and unverified loans at", confidence_level * 100, "% confidence level.")
    print("The probability of a difference in default rates is statistically significant.")
else:
    print("There is no significant difference in default rates between verified and unverified loans at", confidence_level * 100, "% confidence level.")
    print("The probability of a difference in default rates is not statistically significant.")

### Analysis 3: To detrmine if there is any difference in default rate of loan based on dti? DTI is debt to income ratio.
#Approach:
#1. DTI value is divided into 2 parts. Loan with DTI of greater than 20 and loan with DTI or less than equal to 20. Loan having DTI more than 20 is classified as high DTI loan and remaining loans as low DTI loan.
#2. Based on DTI values default rate is calculated as high DTI default rate vs low DTI defualt rate.
#3. From this data statistical parameter like confidence interval, sample size, pooled standard deviaiton, margin of error are calculated and 2 pair t-test is performed on this data to evaluate if there is any difference between these two data. 

# Calculating high and low DTI. 
num_high_dti_charged_off = len(loan_data[(loan_data['dti'] > 20) & (loan_data['loan_status'] == 'Charged Off')])
num_high_dti_loans = len(loan_data[loan_data['dti'] > 20])

if num_high_dti_loans != 0:
    high_dti_default_rate = num_high_dti_charged_off / num_high_dti_loans
else:
    high_dti_default_rate = 0

num_low_dti_charged_off = len(loan_data[(loan_data['dti'] <= 20) & (loan_data['loan_status'] == 'Charged Off')])
num_low_dti_loans = len(loan_data[loan_data['dti'] <= 20])

if num_low_dti_loans != 0:
    low_dti_default_rate = num_low_dti_charged_off / num_low_dti_loans
else:
    low_dti_default_rate = 0

# Ploting count of loan status vs DTI
plt.figure(figsize=(10, 6))
sns.countplot(x='loan_status', hue=pd.cut(loan_data['dti'], bins=[0, 20, np.inf], labels=['Low', 'High']), data=loan_data)
plt.title('Count of Loan Status vs DTI')
plt.xlabel('Loan Status')
plt.ylabel('Count')
plt.legend(title='DTI')
plt.show()

# Calculating sample sizes
n_high_dti = len(loan_data.loc[loan_data['dti'] > 20])
n_low_dti = len(loan_data.loc[loan_data['dti'] <= 20])

# Calculating sample standard deviations
std_high_dti = np.std(loan_data.loc[loan_data['dti'] > 20, 'loan_status_numerical'], ddof=1)
std_low_dti = np.std(loan_data.loc[loan_data['dti'] <= 20, 'loan_status_numerical'], ddof=1)

# Calculate the pooled standard deviation
pooled_std = np.sqrt(((n_high_dti - 1) * std_high_dti ** 2 + (n_low_dti - 1) * std_low_dti ** 2) / (n_high_dti + n_low_dti - 2))

# Perform independent two-tailed t-test
t_stat, p_value = ttest_ind(
    loan_data.loc[loan_data['dti'] > 20, 'loan_status_numerical'],
    loan_data.loc[loan_data['dti'] <= 20, 'loan_status_numerical']
)

# Calculate the degrees of freedom
degrees_of_freedom = len(loan_data[(loan_data['dti'] > 20) & (loan_data['loan_status'] == 'Charged Off')]) + len(loan_data[(loan_data['dti'] <= 20) & (loan_data['loan_status'] == 'Charged Off')]) - 2

# Calculate the confidence level
confidence_level = 0.95

# Calculate the confidence interval
ci_lower, ci_upper = t.interval(confidence_level, df=degrees_of_freedom, loc=t_stat, scale=pooled_std)

# Calculate alpha
alpha = 1 - confidence_level

# Print the results
print("High DTI Default Rate:", high_dti_default_rate)
print("Low DTI Default Rate:", low_dti_default_rate)
print("t-statistic:", t_stat)
print("p-value:", p_value)
print("Confidence Level:", confidence_level)
print("Confidence Interval ({}%): ({}, {})".format(confidence_level * 100, ci_lower, ci_upper))
print("Alpha value:", alpha)

if p_value < alpha:
    print("There is a significant difference in default rates between high DTI and low DTI loans at", confidence_level * 100, "% confidence level.")
    print("The probability of a difference in default rates is statistically significant.")
else:
    print("There is no significant difference in default rates between high DTI and low DTI loans at", confidence_level * 100, "% confidence level.")
    print("The probability of a difference in default rates is not statistically significant.")

### Analysis 4: To detrmine if business is profitable or not?
#Approach:
#1. Time series analysis is performed to evaluate loan volume, default of loans and net profit over time.
#2. From tis data 1-pair t-test is performed to determine if business is profitable or not. 
    
# Time Series Analysis
# Ploting total loan volume over time
loan_data['issue_d'] = pd.to_datetime(loan_data['issue_d'], format='%b-%y')  # Specify date format
loan_volume = loan_data.groupby(loan_data['issue_d'].dt.to_period('M')).size()
loan_volume.plot(title='Total Loan Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Total Loan Volume')
plt.show()

# Ploting default rates over time
default_rates = loan_data.groupby(loan_data['issue_d'].dt.to_period('M'))['loan_status'].apply(lambda x: (x == 'Charged Off').mean())
default_rates.plot(title='Default Rates Over Time')
plt.xlabel('Date')
plt.ylabel('Default Rate')
plt.show()

# Profitability Analysis
# Calculating net profit over time - Example: Net Profit = Total Payments - Total Loan Amount
loan_data['total_payments'] = loan_data['installment'] * loan_data['installment']
loan_data['net_profit'] = loan_data['total_payments'] - loan_data['loan_amnt']
net_profit_mean = loan_data['net_profit'].mean()
net_profit = loan_data.groupby(loan_data['issue_d'].dt.to_period('M'))['net_profit'].sum()
net_profit.plot(title='Net Profit Over Time')
plt.xlabel('Date')
plt.ylabel('Net Profit')
plt.show()


# Statistical Tests
# Performing statistical tests to determine if profitability metrics are significantly different from zero
t_stat, p_value = ttest_1samp(loan_data['net_profit'], 0)

# Printing the results
print("One-sample t-test results for net profit:")
print("Mean net profit:", net_profit_mean)
print("t-statistic:", t_stat)
print("p-value:", p_value)

# Interpreting the results
alpha = 0.05  # Set the significance level
if p_value < alpha:
    print("The net profit is significantly different from zero.")
else:
    print("The net profit is not significantly different from zero.")
