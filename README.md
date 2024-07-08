# Email-spam-classifier
Author
Saroj Kumar

College
IIT Jodhpur

Description
This project involves building an email spam classifier using multiple machine learning models including Logistic Regression, Support Vector Machine (SVM), XGBoost, AdaBoost, and RandomForest.

Dataset
The dataset used is spam.csv which contains labeled email messages. The labels indicate whether the messages are spam or not.

Preprocessing Steps
1. Loading and Cleaning Data:
. Removed unnecessary columns.
. Renamed columns for clarity.
. Encoded target labels.
. Removed duplicates.

2. Exploratory Data Analysis (EDA)
. Distribution of target labels.
. Analyzed the number of characters, words, and sentences in the emails.
. Generated word clouds for spam and ham emails.
. Created a heatmap to show correlations between features.

3. Text Preprocessing:
. Converted text to lowercase.
. Removed stop words and special characters.
. Applied stemming to words.

Feature Extraction
Used TF-IDF Vectorizer to convert text data into numerical features.

Model Training and Evaluation
Implemented various models and evaluated their performance based on accuracy and precision:

Naive Bayes (Gaussian, Bernoulli, Multinomial)
Logistic Regression
Support Vector Machine (SVM)
Decision Tree
K-Nearest Neighbors
Random Forest
AdaBoost
Bagging Classifier
Extra Trees Classifier
Gradient Boosting Classifier
XGBoost
Model Performance
The performance of each model was measured and compared:

Multinomial Naive Bayes achieved the best performance with an accuracy of 95.94% and precision of 1.0.
Ensemble Methods
Implemented Voting and Stacking Classifiers to further improve performance.

Final Model
The final model chosen for deployment is Multinomial Naive Bayes due to its high precision.

Results
Accuracy: 95.94%
Precision: 1.0
Conclusion
Precision is the most important metric for this project as false positives (classifying a ham email as spam) can have serious consequences. The Multinomial Naive Bayes model was selected for its perfect precision score.

Libraries Used
numpy
pandas
seaborn
matplotlib
nltk
scikit-learn
wordcloud
xgboost
Usage
To run the project, ensure you have the necessary libraries installed. Load the dataset, preprocess the data, train the models, and evaluate their performance.

Acknowledgments
This project was developed as part of coursework at IIT Jodhpur.
