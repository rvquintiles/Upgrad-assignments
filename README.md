# Lending Club Assingment
> To evaluate if lending business is profitable and what proportion of loan are getting default


## Table of Contents
* [General Info] (#general-info)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Provide general information about your project here.
    - Loan data contains data of loan lending since 2007. Project is to evaluate if based on current data is lending business profitable.
- What is the background of your project?
    - Project implement following steps:
        1. Data cleaning: 
            - Columns having more than 30% mising data was identified and removed.
            - Columns not having any duplicate value was identified and removed. This is because if columns have all uniqie values then it is not possible to confer any analysis.
            - Lastly set of columns were manually identified and removed from analysis.
        2. Data manupilation: Once data cleaning was performed 2 columns were added by converting loan status columns and loan verified column into numerical values so that co-relation analysis can be performed on these data
        3. EDA: Categorial columns and numerical columns list was created to perform EDA. 
        4. Inferential statistics is used to determine following:
            - To evaluate significant difference between interest rates of fully paid loan and charged off loans.
            - To evalaute significant difference between default rates of laon between verified vs. unverified loans at 95.0 % confidence level.
            - To evalaute significant different between default rate of loan with high DTI vs low DTI. (DTI = Debt to Income ratio)
            - To evalaute if business is profitable or not?

- What is the business probem that project is trying to solve?
    - Project evaluates the data to determine if interest rates and loan verification process has any impact on loan getting defauls. Analysis is performed to find 
        1. To find co-relation if higher interest rate is resulting in loan default or not?
        2. to evaluate if more number of unverified loan are getting default?
        3. To evaluate if loan with high DTI has higher default rate or not?
        4. To evalute if loan business is profitable or not
- What is the dataset that is being used?
    - Loan.csv is used for analysing the data.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Conclusion 1: categorical EDA: 
    - More than 70% of toal loan approved are of shorter duration of 36 months.
    - More than 50% of total loan approved are of grade A and B.
    - More than 40% of loan applicant have mortegage on their home. Also more than 40% of loan applicant are living in rented house.
    - More than 50% of loan approved are verified or source verified. And more than 40% of loan approved are not verified.
    - Approx 14% of total loan are charged off. 
    - More than 40% of loan approved are availed for debt consolidation.
- Conclusion 2: Numerical EDA:
    - Majority of loan aproved are of less than 30000 
    - Interest rate of loan approved is in range of 5%-25%. Majority of loan approved is between 5-15%
    - Majority of loan have installment less than 400
- Conclusion 3: Comparative EDA:
    - Interest rate is directly proportial to grade of loan. Interest rate of loan having grade A is least and interest rate for loan with grade G is highest.
    - Number of installment is strongly co-related with loan amount.
    - Interest rate have higher co-relation with all parameter of loan amount, installment, loan status and verification.
- Conclusion 4: Inferential statistics
    - Statistical significant difference is observed in interest rates between fully paid and charged off loans is observed. 
    - Statistical significant difference was observed in default rates between verified and unverified loans at 95.0 % confidence level.
    - Statistical significant difference was observed in default rates between high DTI and low DTI loans at 95.0 % confidence level.
    - Business is profitable from net zero.

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
- NumPy version: 1.26.2
- Pandas version: 2.2.0
- Seaborn version: 0.13.2
- Matplotlib version: 3.8.3
- Statsmodels version: 0.14.1

<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements
- This project was based on [Upgrade course of EDA and inferential statistics]


## Contact
Created by [https://github.com/rvquintiles] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->