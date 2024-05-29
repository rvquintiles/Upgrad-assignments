# Linear Regression
> Boombikes is bike sharing provider in US which allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system. Due to COVID-19 pandemic company has suffered significant reduction in revenue. Company wants to understand demand of bikes and attributing factors for revenue so that post pandemic when business gets to normal company can improve its revenue and normalize the businss. Company wants to analyse following:
1. To model the demand for shared bikes with the available independent variables. 
2. The model should be built taking this 'cnt' as the target variable.


## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information
- Boombikes is bike sharing provider in US which allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.
- Due to COVID-19 pandemic company has suffered significant reduction in revenue. Company wants to understand what atrribute contributes to business and how can company revive busienss once situation is normalize after COVID pandemic.
- Boombike data "day.csv" is used for analysis

<!-- You don't have to answer all the questions - just the ones relevant to your project. -->

## Conclusions
- Conclusion 1: Linear Regression:
    - Data was prepared using pairplot, boxplot of categorical analysis and correlation was performed to find most correlation between the parameters.
    - Data of season and weathersit eventhough present in numerical value are categorical dataset. Dummies was created to convert data into numerical.
    - From the data it is observed that there is multicolinerity in the data. Data column 'temp' and 'atemp' have significant correlation because only difference between both columns is how people feel the external temperature. 
    - Similarly 'season', 'weathersit' and 'temp' is expected to have huge correlation because all these parameters are interdependnet. 
    - Additionally 'casual' and 'registered' column cumulatively is subset of 'cnt' dataset. 
    - Since data independency is one of the main requirement for linear regression it is import handle such dependency and eliminate it.
    - Using Recursive Feature Elimination (RFE) mulitcolinear data was eliminated by removing column of workingday (because workingday is subset of weekday data it can be drop from the analysis), atemp (because atemp and temp are similar and hence can be removed)
    - Calculated R2 and adjusted R2 of analysis from RFE  is 0.828 and 0.825 respectively.
    - After RFE we calcuated VIF and based on that dataset of 'hum' was dropped due to high VIF value of approx 15. After droping 'hum' new linear modeling was performed and new R2 and adjusted R2 is 0.824 and 0.821 respectively. 
    - Error term of this training dataset is normally distributed and based on this we can infer multicolinearity issues is addressed
    - Train dataset is used to evaluate test dataset which confirms that attributes of 'yr', 'mnth', 'holiday', 'weekday', 'temp', 'windspeed', 'Season_2', 'weathersit_2' and 'weathersit_3' are important attributes which can help BoomBikes to review the business.  
- Conclusion 2: Exploratory Data Analysis:
    - Univariate analysis: From categorical data of "season", "year", "month", "weekday" and "weather" only weather data has significant contribution in revenue of boombikes. During clear weather company generates aprpox more than 60% of revenue.
    - Bivairate analysis: From analysis between 'casual', 'registered', 'count' vs. 'season', 'month', 'weekday', 'weather' it is observed that count of casual users are higher during weekend whereas during weekday use of boombikes is high among registered users. It is know fact that summer season will have clear weather and hence usage of boombikes is significant high during summer season.
    - Time series evaluation: this evaluation also affirms that revenue of company has increased in 2019 from 2018 and highest earning days for the company is Saturday and weekday. Based on season highest earning months for the company is between May - October
- Conclsuion 3: Based on above analysis complete data can be explain in as below:
    - Impact of weather conditions on boombikes revenuw: variables such as temperature, humidity and wind speed have strong influence on boombikes customer to bikes. 
    - Impact of time: Attributes associated with time ex: day, weekday, workingday, month, season etc. have significant impact on usage of bike by customer of boombikes. Time and weather to-gether have visible and distinct impact on company business.
    - Additional atributes of holiday brings additional revenue from casual users of boombikes. From the data it is visible that usage of bikes by casual users is highest during holidays. While registered users provides steady revenue durign workding and weekdays, causal users bring good influx of revenuw during holiday.

## Technologies Used
- NumPy version: 1.26.2
- Pandas version: 2.2.0
- Seaborn version: 0.13.2
- Matplotlib version: 3.8.3
- Scikit-learn version: 1.4.2
- Statsmodels version: 0.14.1

## Acknowledgements
- This project was based on tutorial (https://learn.upgrad.com/course/5798/segment/49277/291047/885651/4426825) 

## Contact
Created by [https://github.com/rvquintiles] - feel free to contact me!

## License : Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.
<!-- This project is open source and available under the [... License](). -->