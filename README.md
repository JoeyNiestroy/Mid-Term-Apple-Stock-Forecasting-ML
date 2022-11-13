# Mid-Term-Apple-Stock-Forecasting-ML
The goal of the project was to attempt to forecast Apple stock performance at close after 2 days using ML, a dataframe was built using possibly predictive features

The dataframe was build using historical data for Apple Stock, S&P500, and NASDAQ 100 Technology Sector Index.
Features inlcude the performance of SP500 and Tech to date at 3 time points [7 days,14 days,30 days] measured in precent changed.
As well as APPLE stock performance to data relative to the mean price off the previous [7 days,14 days,30 days] also measured in precent changed.
Trading Volume means were added as features.
Target column is the exact change in price in percent at close two trading days.
Model_CSV.csv is this dataset

Features were built off hypothesis on retail invenstor patterns as well traditional mean reversion theory

Initially regression modeling was tried but quickly proved ineffective and the model was moved to binary classification for postive change and negative change in stock value after 2 trading days

Model_Simulating.py shows the building of the model, a random forrest was used (logistic regression with polynomial kernals were explored but not as effective), and the model being deployed on 30 days of sequential clean test data to simulate unknown future market outcomes. Stock purchases and short positions were based off exact Open price to exact close price two days later, and an equal amount was invested daily based off prediction. At a random state 8 (Chosen at random for repeatability) the total gain was roughly 1%. However, factoring in the high False positive (38 samples out of 842 total) rates from the training data and adding in a probability condition to the simulation to account for this, we push preformance to an average of 1.5% gain. 

Post inital simulating the test data was explored thoroughly and the model had also had high False Positive rate, which explains the preformance boost of the probability conditonal. In a few weeks new clean test data will be explored taking into account the high false positive rate and I'll see if this pattern countinues. The simulating was replicated with no set random state and preformance was as good as 5%+ and bad as 6%+

Future Changes: Neural Nets will be explored as a possible model, as well as expansions to the inital dataset created to inlucde more features. A more in depth risk model will be created based off the FN and FP from training data
