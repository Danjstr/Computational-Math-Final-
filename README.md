DAV 5300 Final Paper – Daniel Strauss Jiaqi Min

Our project is about COVID-19, which has been a significant problem all over the world for the past year. We focus on worldwide COVID-19 data and applied statistical analysis and machine learning methods to study and think about how the disease has impacted people all over the world. We use a few basic methods throughout our project to get a look at the data. Box plots and histograms were used to get a look at the spread of the data. We used a heat map to determine the relationship between different variables. We did several statistical tests to draw insights from the data as well. Including a linear model, T-test, and paired wise T-test. We further did some basic machine learning and implemented an auto Regressive machine learning model to allow us to see if we could predict future cases and deaths. We learned we learned that we can distinctly measure the change in cases and deaths per 100,000 million people as well as predict with reasonable certainty infection rates going forward

The global pandemic has torn our world asunder. We have felt its impact on both the largest and smallest scale. From personal stories of not being allowed to visiting a dying family member in the hospital to a global financial crisis. It has impacted all of us and fundamentally changed the world. Its impact is oftentimes indirect and has brought about more pain and sadness than just the people it has made sick. Stopping the disease overcoming this problem is of paramount importance. It is impossible to fight an enemy you know nothing about. Every bit of information on covid-19 and its spread is valuable. Some of the impacts and our motivations for this project are provided below. It is by no means an extensive list. 

•	By the fourth quarter of 2020, tens of millions of people were at risk of falling into extreme poverty. 
•	The number of undernourished people, currently estimated at nearly 690 million, could increase by up to 132 million by the end of the year.
•	Nearly half of the world’s 3.3 billion global workforces are at risk of losing their livelihoods.
•	Millions of agricultural workers face high levels of working poverty, malnutrition, and poor health, and suffer from a lack of safety and labor protection.
•	Trade around the world has fundamentally changed and has entirely halted in some areas.
•	Most notably Millions of deaths as a direct result of infection.

Our dataset is called “owid-covid-data”. It is maintained by an organization called Our World in Data and can be found on their GitHub homepage. It is updated daily and includes data on confirmed cases, deaths, hospitalizations, testing, and vaccinations as well as other variables of potential interest. They have collected data from authorized organizations all over the world and built up this overall dataset. Since the original dataset has too many columns, we dropped some of them to simplify further coding. We had a serious singular issue with this data set that we discovered while performing EDA on this dataset. Tails. Almost all the columns suffered from the most ridiculous tails. It was not a single column either. Best we can guess most reports of cases are made in a timely manner. But as reports are delayed sometimes by months the number of unreported cases rises until all of them are reported at once. This meant we had the equivalent of 30,000 or so reports of just a few dozen to a few hundred deaths. While at the same time we received dozens of reports with tens of thousands of deaths. This might I add was the case after outliers were removed. (Appendix 1)

Considering the nature of the problem we could not just clip off the outliers for each column individually. All the columns impact and relate to one another. For example, New cases relate to new deaths. so they cant have their tails managed in different methods. This was in addition to the fact that we didn’t want to remove the tails entirely despite the skewed data as they represented relevant insights. We could also be removing large-scale outbreaks if the tails represented a larger number of cases being reported all at once because there was a large-scale increase in the number of infections. 

As such we decided to implement the following code as our solution. We removed all data that was more than 1.8 standard deviations from the mean. We played around with different standard deviations including 1, 2, 2.5, and 3 all with lesser results. (Appendix 2)

We applied a linear regression model to the analysis to see if there is a linear relationship between the number of cases and some other variables for each country. Two major linear models we have done are population density versus total cases and new smoothed cases versus GDP. We were trying to determine if some social circumstances like economics or population would affect the spread of the disease. For both, we find just the slightest linear relationship between them. The coefficient of determination was 0.00088. This data does not necessarily fit the linear models. As such no real valuable insights were gleaned. We had hoped to determine a solid relationship between population density, GDP, and the spread of the disease. (Appendix 3 and 4)

Next, we generated a hypothesis test using new smoothed cases and GDP to see if there is a relationship between GDP  and new cases. The conditions for this hypothesis test were as follows.
•	The Null hypothesis: There is a linear relationship between new smoothed cases and GDP. 
•	Alternative hypothesis: There is no linear relationship between new smoothed cases and GDP.

The results of this test are interesting We have a 0 p-value so we can reject the null hypothesis, but since the r-square value is relatively low. This indicates that the additional input variables are not adding value to the model. There may be a slight linear relationship even if the results are largely inconclusive. We still found that both had a very slight R-squared value.
This is interesting because people would assume that the virus would spread more quickly due to higher population density. A good domestic economy also leads to more traveling which we thought may be a contributing factor. We also thought a higher GDP may lead to fewer deaths as there was a possibility that a higher GDP may have led to better treatment. However, again these results are inconclusive. 

           As we did not have conclusive results from the last statistical test we performed two additional paired-sample T-tests. To see if we could gain any valuable insights. Both these tests were done to determine if there was a change in the number of cases and deaths in North America as a result of the pandemic. The first pairwise T-test was done to determine if there was a difference in cases in North America when comparing 3/1/2021 and 4/1/2021. The result of the pairwise T-test was 0.230 as such we accept the null hypothesis. (Appendix  5)
•	H0: There is no difference in the number of new cases in North America between 3/1/2021 and 4/1/2021.
•	H1: There is a difference in the number of new cases in North America between 3/1/2021 and 4/1/2021.

The second pairwise T-test was done to determine if there was a difference in deaths in North America when comparing 3/1/2021 and 4/1/2021. The result of the pairwise T-test was 0.339 as such we accept the null hypothesis. (Appendix  6)
•	H0: There is no difference in the number of new deaths in North America between 3/1/2021 and 4/1/2021. 
•	H1: There is a difference in the number of new deaths in North America between 3/1/2021 and 4/1/2021.

As a result of both of these Pairwise t-tests we can see that we fail to reject the Null hypothesis. As such we can safely see there was likely no significant change in the number of cases and deaths between the two dates.

After these basic statistical analyses, we did some machine learning our hope was to be able to use current data to predict future values. For this purpose, we deployed two Autoregressive models. We did this for both new cases per million and new deaths per million. Both Autoregressive models follow the same format. 

The autoregressive integrated moving average model is a form of regression analysis that gauges the strength of one dependent variable relative to other changing variables. The model is typically deployed to predict future securities or financial market moves by examining the differences between values in the series instead of through actual values.
As such, we thought that the Arima Model was uniquely placed to help us build a predictive model of the pandemic data. The data were are working with also fits the same time series format as financial institutions. The only real difference would be the large spikes in the data brought about as a result of the tail. Where a larger number of cases and deaths were reported all at once. Fortunately, Arima is dependent on the mean of the data as long as the mean remains relatively constant then there was no problem deploying the Arima model.

So while deploying Arima as I mentioned above we have to make sure that the data is stationary. We need to make sure that it has both constant mean and constant variance. We did this and we determined that the data did meet these requirements. This was done by using an ADF Test that was structured as follows. (Appendix  7)
•	H0: Time Series is Stationary
•	H1: Time Series is Non-Stationary
 
The result of the test is as follows. As we can see there is a P-value of 0 as such we fail to reject the null hypothesis. Thus we can determine that the Time series data that we are working with is in fact stationery.

(Appendix  8)

Next, we needed to determine how many lags to use. In an Autoregressive model, Yt is a function of its own lags as such Yt-1 , Yt-2, Yt-3 and so on. We used a partial Autocorrelation plot to solve this problem and determined we needed just one lag to proceed with the Arima model. This was determined by the fact that Yt is perfectly correlated with itself and almost perfectly correlated with Yt-1. (Appendix  9)

This leaves us with the equation Yt = a+B1Yt-1+E1. Now we need to determine what the Alpha(a) value and Beta(B) value are in our equation. To determine Yt-1we use the .shift function in pandas to create a duplicate of the column of our time series data shifted one row down. (Appendix  10)

Next, we use train test split and drop Yt into Y and Yt-1 into X from here it follows the same process as any normal train test split. Each of Yt and Yt-1 are both 1 dimension so we reshape  them both too (-1,1) so that they have two dimensions in the X_Train and X_test data.
Next, we fit the Linear regression model. Then we print the .coef_ and .intercept_ of the newly fit linear regression. These represent our Alpha value and our B1 value. (Appendix  11, and 12)
So for our two Arima models, we have the following equations.

New Cases Smoothed Per Millions: Yt = 0.982 + 0.698 Yt-1+E1
New Deaths Smoothed Per Millions: Yt = 0.978 + 0.015 Yt-1+E1

After that, we simply run our predictions against our testing values and we see in the plot that we have great results. The Prediction largely follows the testing data. This indicates that our model could reasonably predict new cases and new deaths. This is taking out of the equation any real-world circumstances that may impact the results of the prediction. (Appendix  13 and 14)

Still, we can determine that this model would serve reasonably well as a method for getting ahead of the virus. It may be functionally used for issuing social distancing orders before predicted spikes or dropping those orders when predictions indicate that cases and deaths are going down. The model summary of the two Arima Models are as follows. (Appendix 15 and 16)

The results for some of our models were inconclusive and did not really yield usable results. These inconclusive results include our attempts to find a linear relationship between population density and Total cases per million. As well as our attempt to find a linear relationship between GDP per capita and new cases per million using a hypothesis test. 

We had much better results from our two hypothesis tests that we used to determine if there was a difference in the number of cases and deaths on a certain date. These results seem to indicate that there was no difference between those two days. It should be noted that there is a serious limitation of this hypothesis test. Mainly being that you can only compare two dates at a time. It is a poor means of determining overall change across all dates. Still when comparing just two dates this hypothesis test performs wonderfully at determining changes in cases and deaths.

Our most important results were from our Arima model. We successfully trained a time-series model and used it to create reasonably accurate predictions of our test data. This means we can with reasonable accuracy predict when and how many new cases and new deaths we will see in the future. It should be noted that the limitation of this model is that it is dependent on the infection rate of this covid variant. Meaning that if this model was used to predict the spread of any other disease it will suffer by comparison. 

Additionally, the model would immediately fail if an additional factor came into play. If for some reason, there was a sudden decrease in new cases and new deaths as a result of political intervention this model would not be able to predict such an occurrence. Its value is in predicting new cases and new deaths exclusively in an environment where the status quo continues. 

In short, we learned that we can distinctly measure the change in cases and deaths per million people as well as predict with reasonable certainty infection rates going forward. We learned a great deal about how to apply an Arima model in python. We struggled with it at first as we could not quite understand the math behind how it works. If we had more time then we would have implemented several additional steps. Including using confidence intervals in our statistical analysis. As well as employing k-fold cross-validation with our Arima models. I think we could have increased the overall confidence in our analysis if we had employed such methods.
