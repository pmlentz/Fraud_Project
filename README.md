# Credit Card Fraud Detection Project: Finding the Needle in a Haystack
<p align="center">
  <img src="https://user-images.githubusercontent.com/86200382/150909664-2fa4682b-c8cb-463d-8072-5bb6713d8084.png" alt="Banner Image">
</p>

## Introduction
When looking for a side project, there was always something about trying to find fraud that appealed to me. Whether it was the challenge of finding the needle in a haystack or the opportunity to work on preventing a problem that according to Nilson Report data caused $28.65 billion in losses in 2019 alone[^1], I was drawn to working on this type of problem.

Utilizing R, I tackled many issues to identify fraudulent transactions such as accounting for the rarity of fraud within the data as well as building and testing many different machine learning models. Ultimately, I chose an XGBoost model to attempt to predict fraud within my test dataset, detecting over 90% of the 116 fraud cases cases while still correctly identifying non-frauds 99.78% of the time!

## Project Goals
1. Learn about and implement ways to try and correct imbalanced data to account for rare event bias
2. Practice tuning and scoring machine learning models to best identify fraudulent credit card transactions
3. Have fun playing with data!

## Data Used
This project uses data from a data set from Kaggle. It contained 198,365 credit card transaction observations from European credit card users in September 2013. The data was primarily formatted in 28 principal components to protect anonymity, as well as the variables Amount (amount of transaction), Time (seconds between each transaction and the first in the dataset), and Class (indicator variable for fraud (1) or not fraud (0)). The data set contained 197,982 non-fraudulent and 383 fradulent transactions making it extremely imbalanced.
The data was split into a train and test data set with a 70-30 split. The training data set contained 138,589 non-fraudulent transactions and 267 fraudulent transactions
or 0.1923% of the data classified as fraud. The test data set contained 59,393 non-fraudulent transactions and 116 fraudulent transactions or 0.1949% of the data classified as fraud.
The dataset can be downloaded from Kaggle [here](https://www.kaggle.com/c/1056lab-credit-card-fraud-detection/data?select=train.csv) (note: only the training dataset from this competition was used).

## Methods

### Correcting Imbalance
To solve the rare event problem, I calculated the minimum number of synthetic fraud observations that would need to be created to get above a 5% threshold for rare events. This was calculated to be at least 7,309 synthetic fraud observations.
To create these synthetic observations, two oversampling techniques were tested.
1. Probability density function estimation based oversampling (PDFOS)
2. Majority weighted minority oversampling (MWMOTE)

After plotting the original and synthetic observations on their first and second principal components, MWMOTE was chosen as a more promising technique. Below I have shown the plots used in making this determination.

<p align ="center"><b>Plots Used in Selection of Synthetic Creation Technique</b></p>
<p align="center">
  <img width="1368" alt="First Synthetic Graphic" src="https://user-images.githubusercontent.com/86200382/150915528-72b8a303-8897-4d7e-a3d4-59e517d15184.png">
</p>

Since not all synthetic observations are created equally, I used a cleaning algorithm that uses kmeans and game theory to attempt to find the observations most resembling the real fraudulent observations. To do this, more observations had to be created with the MWMOTE method and filtered down, preserving at least 7,309 new synthetic frauds. Below are the plots of the original, unfiltered 45,000 synthetic frauds, and the filtered 9,108 synthetic frauds that I used for machine learning. The proportion of fraudulent observations in the training data was 6.336% (so slightly more than the 7,309 new data points needed) after filtering out observations that didn't quite match the original fraud observations well.

<p align ="center"><b>Plots of Original, Non-Filtered, and Filtered Synthetic Fraud Observations</b></p>
<p align="center">
  <img width="1368" alt="First Synthetic Graphic" src="https://user-images.githubusercontent.com/86200382/150915546-b0a5ecd7-78b8-48ac-9f58-e5d9ff5394dd.png">
</p>

### Machine Learning Techniques
I tried multiple machine learning models to attempt to predict the fraudulent transactions in the training dataset. I built a Logistic Regression, Neural Network, Random Forest, XGBoost, and Naive Bayes model. These were scored using area under the precision recall curve(AUPRC), area under the receiver operator characteristic curve (AUROC), Kolmogorov-Smirnov statistic (KS), sensitivity (correct fraud identification rate), and specificity (correct not-fraud identification rate). The table below shows the scores for each model.

![image](https://user-images.githubusercontent.com/86200382/151007277-795fd2ab-9aa5-4f8d-af3c-df040e84e53d.png)

While intending to make final decision based on the AUPRC (as it's more appropriate to use with imbalanced data then AUROC), the XGBoost model did not misclassify a single value making the choice easy.

### Model Testing
The XGBoost model was used to predict fraudulent observations in the test data set. To find an optimal threshold for classifying an observation as fraud or not fraud, I started by using the Youden's Index to find the split that maximizes sensitivity and specificity. While this mathmatically is the optimal solution, it did not make as much sense from the context of the financial cost of a false positive or a false negative. But what is the cost of a fraudulent transaction? What is the cost of declining a customers purchase if it is non-fraudulent? While researching this question, I found that surprisingly false positives (classifying a fraud where there isn't one) cost financial institutions up to 13 times more then false negatives[^2]. For example, a customer who has a transaction classified falsely as fraud, often will just stop using the card forever. The absence of this customer's card utilization is a large loss of potential income. So while fraud is certainly costly for a bank, it is important to balance both the sensitivity and specificity for this problem.

I found that when using the Youden's Index classification threshold of 0.000513, 10 out of 116 frauds were misclassified and 675 of 59,393 non-frauds were misclassified. If this threshold is adjusted to 0.00255, one more fraudulent observation is misclassified but 545 more non-fraudulents are correctly classified. This threshold could be further refined based on an institutions cost for each type of misclassification. The final model has a sensitivity of 0.9052 and a specificity of 0.9978.

## Conclusion

### Takeaways
Predicting fraudulent transactions is an important task in the utilization of credit cards. Customers not only expect the bank's protection in not allowing their cards to be used for fraudulent transactions, but also don't want the inconvenience of their card being declined for legit transactions. Customers rightfully have high expectations of their bank and we can assist in meeting these expectations using data science.

### 👋
This project was a great opportunity to practice many different machine learning skills as well as practice working with an imbalanced data set. If you have any feedback, comments, or concerns, please feel free to contact me on [LinkedIn](https://www.linkedin.com/in/paul-lentz/). To view my code for this project, please check it out [here](https://github.com/pmlentz/Fraud_Project/blob/main/R%20code%20Weighted.R)! Thank you for taking the time to check my project out. I hope you enjoyed it!


[^1]: https://www.cnbc.com/2021/01/27/credit-card-fraud-is-on-the-rise-due-to-covid-pandemic.html\
[^2]: https://www.vesta.io/blog/false-positives-credit-card-fraud-detection
