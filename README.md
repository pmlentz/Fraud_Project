# Credit Card Fraud Detection Project: Finding the Needle in a Haystack
![image](https://user-images.githubusercontent.com/86200382/150909664-2fa4682b-c8cb-463d-8072-5bb6713d8084.png)
## Introduction
When looking for a side project, there was always something about trying to find fraud that appealed to me. Whether it was the challenge of finding the needle in a haystack or the opportunity to work on preventing a problem that according to Nilson Report data caused $28.65 billion in losses in 2019 alone, I was drawn to working on this type of problem.

Utilizing R, I tackled many issues to identify fraudulent transactions such as accounting for the rarity of fraud within the data as well as building and testing many different machine learning models. Ultimately, I chose an XGBoost model to attempt to predict fraud within my test dataset, detecting over 90% of the 116 fraud cases cases while still correctly identifying non-frauds 99.78% of the time!

## Project Goals
1. Learn about and implement ways to try and correct imbalanced data to account for rare event bias
2. Practice tuning and scoring machine learning models to best identify fraudulent credit card transactions
3. Have fun playing with data!

## Data Used
This project uses data from a data set from Kaggle. It contained 198365 credit card transaction observations from European credit card users in September 2013. The data was primarily formatted in 28 principal components to protect anonymity, as well as the variables Amount (amount of transaction), Time (seconds between each transaction and the first in the dataset), and Class (indicator variable for fraud (1) or not fraud (0).  The data set contained 197982 non-fraudulent and 383 fradulent transactions making it extremely imbalanced.
The data was split into a train and test data set with a 70-30 split. The training data set contained 138589 non-fraudulent transactions and 267 fraudulent transactions
or 0.1923% of the data classified as fraud. The test data set contained 59393 non-fraudulent transactions and 116 fraudulent transactions or .1949% of the data classified as fraud.

## Methods

### Correcting Imbalance
In approaching the rare event problem, I first calculated the minimum number of synthetic fraud observations that would need to be created to get above a 5% threshold for rare events. This was calculated to be at least 7309 synthetic fraud observations.
To create these synthetic observations two oversampling techniques were tested.
1. Probability density function estimation based oversampling (PDFOS)
2. Majority weighted minority oversampling (MWMOTE)

After plotting the original and synthetic observations on their first and second principal components, MWMOTE was chosen as a more promising technique. Below I have shown the plots used in this determination.

<img width="1368" alt="First Synthetic Graphic" src="https://user-images.githubusercontent.com/86200382/150915528-72b8a303-8897-4d7e-a3d4-59e517d15184.png">

Not all synthetic observations are created equally though and so a cleaning algorithm using kmeans and game theory was used to attempt to find the observations most resembling the real data. To do this, more observations had to be created with the MWMOTE method and filtered down to above 7309 new synthetic frauds. Below we see the original, unfiltered synthetic, and filtered synthetic plots that was used for machine learning.

<img width="1368" alt="Second Synthetic Graph" src="https://user-images.githubusercontent.com/86200382/150915546-b0a5ecd7-78b8-48ac-9f58-e5d9ff5394dd.png">

### Machine Learning Techniques


