# Twitter-Data-Analysis
Analysis of twitter data

- __Dataset: Customer Support Twitts__

- __Group__: Jinjiang Lian (20740282) and Junyi Yang (20572021)

- __Dimension__: 2,811,774 obeservations, 7 features

- __Method and Tools__: Python with Jupyter notebook

- __Dataset link__: https://www.kaggle.com/thoughtvector/customer-support-on-twitter/download

- __Objective__: 
- __Main__: Training a model to identify whether feedback would be given to a specific twit. 
  - __Step 1__: Exploratory data analysis and data cleaning
  - __Step 2__: Data preprocessing (feature extraction, train/test split, word tokenizer, stop words removal etc..)
  - __Step 3__: Training classification algorithms on the model (random forest, SVM and GDBT - LightGBM, gradient boosting tree etc...)
  - __Step 4__: Model ensemble and evaluation (majority weighted vote, ROC/AUC)
  - __Step 5__: Findings and possible further improvement

- __Side__: 
  - Summary statistics about the dataset (count of twits group by companies)
  - Adding labels for each twit for further sentiment analysis
  - Discover other interesting facts about the data



#Time vs response rate
#Average time between response


I.  Abstract

In this report, we are going to explore millions of customer support tweets to see what is the most efficient way of writing a tweet that could have a higher chance of getting responses. 

The first section of the report dives into the background and incentive of making this project, including some summary statistics about the dataset we are using. The second part of the report digs into the detailed introduction of the technical details about how we preprocess our data and focus on how to optimize the corresponding preprocessed data in order to better feed into our machine learning algorithm. The third part of the report includes the modeling process, we are going to use several machine learning algorithms such as naive Bayes, decision trees and NNs etc to ensemble them and build our model pipeline. In the fourth part of the report, we will head on to the result of our model and corresponding conclusions and future improvements, as well as results and some sample tests on our pipelines. 

II. Background Introduction

Nowadays, as social networks become more and more popular, instead of waiting in line for several hours to conduct a live chat with an agent, people sometimes seek help through their social network with company customer support such as with Twitter, Facebook, etc… Most individuals, due to time limit, would inform customer support by notifying them of the questions and will proceed to their own business while waiting and hoping for a response from the customer support. However, getting a response from customer support is not always the case since they are receiving tons of news feeds per day from customers and would inevitably unintentional or intentionally ignore some of them. In this report, we are going to explore the efficient way from millions of Tweets (postings through Twitter) to ask questions in order to get a higher reply rate from customer support. Moreover, the resulting model could also be used as a filter to help reduce customer supports’ workload to only reply tweets that need special attention.  

The data set comes from Kaggle.com collected in the year 2017 consists of 2,811,774 of tweets, with 1,537,843 (54.69%) tweets comes from customers and 1,273,931 (45.31%) comes from customer supports. Amongst these 1.5 million customer tweets, about 1.27 million of them received a reply from customer support and 0.23 million of them did not. In the following paragraphs, we are going to walk through the process of data preprocessing and data modeling and interesting findings along the process. 

III. Data Processing

3.1 Insights from our dataset:

3.1.1  Ordered Count of Tweet Replies by Company
 
Figure 1. Ordered Count
From our dataset, we grouped the top 10 most customer support related tweets companies. The topmost customer support related tweets associated company is Amazon, followed by Apple and Uber. The tweets associated with Amazon (rank 1) is almost 8 times more than that of Chipotle (Rank 10) tweets. Suggesting that in the year 2017, Amazon is probably the most popular (also the most-valued) company compared to all other companies in this dataset. We did a little research on company stock price and found out that indeed Amazon’s stock price was the highest amongst all the other companies in our dataset for around 980 USD. 

3.1.2. World Cloud of Word Frequencies

 
Figure 2
This word cloud is a conjugation of all words that appeared in the tweet. The larger the word is in this graph, the more frequent the word appears in the tweet. For customer support related tweets: DM (short for “Direct Message”) is the most common message that showed up in our data. Most of the messages are requests such as “send us”, “please DM” and “let us know”, while some of them are replies and apologies from the customer support team such as  “happy help”, “sorry hear” and “feel free”. 

3.2 Data Preprocessing

From our data preprocessing steps, we found that there is no missing data in our dataset that needs our attention. The main target column for latter modeling is the text column which contains the main body of the corresponding tweet. And it is the column that needs the most time to be pre-processed. 

To process our twitter text data, we made some modifications to the existing stopping words from the NLTK package to better fit our dataset. (For example, we remove the abbreviation “u”, user_id after and the @ sign since they do not actually mean anything and would bring some noise to our model. In addition, posting a link started with HTTP might also not helping our model so we removed those as well). In addition, we used the Potter Stemmer to strip the suffix of particular words so that we can avoid multiple counting of the same word.

We were thinking of using lemmatizers initially intended to remove the tenses and the different forms of the same word, but then we decided not to proceed with this procedure since people were mainly talking to the customer support about things that happened in the past, they would likely to use past tense and thus we might only deviate our result slightly in this case. The resulting sample looks like in the following:

Input: normalizer("@115714 whenever I contact customer support, they tell me I have shortcode enabled on my account, but I have never in the 4 years I've tried https://t.co/0G98RtNxPK.")

Output: 'whenev contact custom support tell shortcod enabl account never year tri'

We also label encoded our target (predicting) column as “have_reponse” where 1 represents that there is a response to the corresponding tweet and 0 represents that there is not a corresponding response associated with this particular twitter. 

After processing all our 1,530,000 data points, we found that there are empty strings coming from tweets solely composed of stop words. These tweets do not have any specific meaning and thus would not be used as training features in our model. There are 34,193 of such data points, within these data, there is 30.23% of them that have no response and about 68.77% that does receive a response. Due to their specialness, we decide to remove them from our data set. Latter when we design our model predicting pipeline, after processing the original data, if the data points are empty strings, they will be randomized based on this probability to decided whether there will be a response corresponding to these specific tweets. 

3.3 Evaluation Metrics

For supervised classification of an unbalanced dataset, accuracy (True Positive / (True Positive + False Positive)) / AUC is not optimal. However, as we will mention in the next section, we are going to sample a roughly balanced data set from the original dataset thus achieved a relatively balanced one. Thus, accuracy score would be our main evaluation metric for our data. In addition, we would provide other metrics for some models such as F-1 score, etc. 

2.3 Optimization

2.3.1 Data Sampling

Since we are using an imbalanced dataset: After preprocessing, we roughly have about 1,280,000 tweets that had a response and about 220,000 tweets that had no response. The ratio is about 6 to 1 which might cause various issues for our classification algorithm. For example, if we go ahead and build a model that classifies everything into the “will have a response” category, we would achieve around 85.7% accuracy but are not gaining any insights about our data nor are we able to tell what keywords might cause customer supports to refuse to reply at the same time. We decide to use all the 223, 675 no response data points paired with 335,513 data points that had a response (ratio about 4 to 6) randomly sampled from our 1,280,000 data points to feed into our machine learning algorithm, this way we make sure that our data is balanced while maximizing the usage of the number of our dataset. 

2.3.2 Memory Usage
Since we have a relatively large dataset, in order for our model to have better performances, we would like to reduce our dataframe’s memory usage by changing some column type (for example we might not need int64 for an integer with the maximum being around 1000). After applying the helper function we successfully reduced our memory usage for almost 20%:
 
Figure 3

IV. Data Modeling

4.1 Model selection
Since we have a relatively large amount of data points. The selection of machine learning algorithms would need special care. 

We planned to use SVM but then we figured out that it was not scalable since “storing the kernel matrix requires memory that scales quadratically with the number of data points. Training time for traditional SVM algorithms also scales superlinearly with the number of data points. So, these algorithms aren't feasible for large data sets” (Haitao Du, 2017). We then ran some experimentation and found out that we can divide the sample up to different chunks by bootstrapping of 1/9 of our sample at a time and do a mini-ensemble of all these SVM models to preserve the majority vote. 

We also decided to use random forest and decision trees as one of our classifiers for the reason that they can run the model and build tree in parallel to each other that would improve our running time on our large data set. However, since they are using relatively the same algorithm, their individual weights in our entire ensembling structure are relatively low. In addition, since the decision tree is a less complex model compared to a random forest, it will have significantly lower weights than random forest. 

One other model that we would like to add to our ensemble is naive Bayes model since it is the fastest model that can give us a result and its accuracy is within higher range compared to all other models. 

Finally, we also added two new recent hot topic of deep learning related neural networks for training and prediction purposes. These neural networks could take advantage of GPUs and provide a fast, scalable solution for predicting purposes. 

4.2 Individual model performances

We divide our data into 2 streams of approaches for training: TF - IDF (Term frequency-inverse document frequency) and simple tokenizing. TF - IDF is used to post-processing our data and feed them into our ensemble of SVMs, naive Bayes, and tree structures while simple tokenizing is used to feed to our neural network.

For TF-IDF, we took the top 5000 features and vectorize them then fit transformed on our training instances and feed the resulting transformed data to our models. Naive Bayes (default hyperparameter setting) achieved 68.56% accuracy on the test set:
 
 
Figure 4
while decision tree (max-depth = 40) achieved 65.48% accuracy:
 
 
Figure 5
and random forest (120 estimators) achieved 69.18% accuracy. 
 
Figure 6: Random Forest Performance
The ensembled SVM (9 components, majority votes) achieved 70% accuracy:

 
Figure 7: SVM Performance
For simple tokenizing, we use the GPU power to constructed 2 neural networks to train our model. Recurrent Neural Network (RNN) and Convolutional Neural Network (CNN). The model structures are as follows:

 
Figure 8: RNN model

 
Figure 9: CNN model
RNN (with LSTM) achieved 60% accuracy over 10 epochs:
 
Figure 10: RNN Accuracy
While CNN achieved 59% accuracy over the 10 epochs:
 
Figure 11: CNN Accuracy
Since most of our predictors are weak learners, our plan up to this point is to ensemble the final prediction by models based on their performances. Since random forest and SVM ensemble had the highest accuracy, we are going to give them more weights in the saying. Followed by Naive Bayes, decision trees and NNs. 

4.3 Ensembling Structure
 
Figure 12: Ensembled Structure
As demonstrated in Figure 12, this is the final structure that we formalized for our dataset. The raw input data is normalized first and then, based on their method, is divided into 2 streams for ensembling. The upper stream used TF-IDF to post-process and then modeled with decision tree, random forest, ensembled SVMs and Naive Bayes. The lower stream used tokenizing to post-process the data and model with convolutional neural network (CNN) and recurrent neural network (RNN). The upper steam is then ensembled with the lower stream to predict the final output. 

4.4 Result and Conclusion

The ensembled model, created and weighted voted by 6 weak learners, each with accuracy less than 70%, achieved an overall astonishing result of 91.34% accuracy on the final prediction on the test data set. Each model, due to the difference in their algorithms, contributes differently to the final prediction (eg: the random forest is good at classifying the false positive while naive Bayes is good at classifying the true positives) We ran some sample tests on the final model pipeline and found out that the results are as expected: 
 
The 1s and 0s in the list showed are the votes of the algorithms. The dominate one would be selected as the final result. 
At last, we added the random factor that we mentioned in part 3.2 (Data Preprocessing). We randomly generate a uniform number between 0 and 1, and if the number is less than 0.3023 it would be labeled as 0 and else it would be labeled as 1. This way that if the normalized data is an empty string, there would be a 30.23% chance you would not get a reply and 69.77% chance that you would receive a reply. Here is a sampled test that we ran on our final pipelined model:
 

4.5 Possible Improvement In the future

Although we achieved 91% accuracy on our final ensembled model. There are still some future improvements that could be done for our project. 

For the purpose of improving accuracy, one possible improvement is that we could have used the entire 500,000 pre-processed sampled datasets to cross-validate train our SVM models so that the performance of our SVM ensembled model could be improved by a significant amount. However, as we mentioned in section 4.1 model selection, SVMs generally do not scale well as the number of data doubles or triples or even more. It will take days to train the model for a situation like this. 

Another possible improvement is that we could have done more hyperparameter tuning for all our base weak classifiers. For example: by using either grid search or random search for tuning hyperparameters in the random forest, SVM and decision trees models;  Adding more layers to the neural networks and etc. In our situation, due to time and resource limitations, we decided to ensemble models instead of explicitly tuning for each individual model. 

For the purpose of handling multiple test cases, we could have added a language filter to detect whether the language is English or not and predict whether there would be a reply or not based on the corresponding language. Or we could add a filter to detect spelling errors to better improve our data quality. 

V. Tools Used and References

5.1 Tools Used:

This analysis was conducted using python 3.7 via Jupyter Notebook. Along with common open-source libraries such as pandas, numpy etc… 

We also utilized deep learning related techniques with GPUs from Google Colab for better training performances. 

5.2 References:
Bedi, G. 2018. A guide to Text Classification(NLP) using SVM and Naive Bayes with Python
https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

Brownlee, J. 2016. Save and Load Machine Learning Models in Python with scikit-learn.
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

CNN. 2019. U.S. Stock Market Data.
https://money.cnn.com/data/us_markets/

Du, H. 2017. Can support vector machine be used in large data?.
https://stats.stackexchange.com/questions/314329/can-support-vector-machine-be-used-in-large-data

Jabeen, H. 2018. Stemming and Lemmatization in Python.
https://www.datacamp.com/community/tutorials/stemming-lemmatization-python

Pathak, M. 2018. Joining DataFrames in Pandas.
https://www.datacamp.com/community/tutorials/joining-dataframes-pandas
