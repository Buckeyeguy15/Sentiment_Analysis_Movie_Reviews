# import libraries
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import time
import matplotlib.pyplot as plt


#################################################################################################################

# Data Pull from .txt files

# get data from .txt files, and concatenate all dataframes into 1 dataframe
# imdb data was not working well with read_csv data, so read line by line and made a dataframe from a list
imdb_sentence = []
imdb_score = []
with open("imdb_labelled.txt", 'r') as f:
    content = f.readlines()
    content = [x.strip() for x in content] 

for review in content:
    imdb_sentence.append(review.split("\t")[0])
    imdb_score.append(int(review.split("\t")[1]))

df_imdb = pd.DataFrame(list(zip(imdb_sentence, imdb_score))) 
df_yelp = pd.read_csv("yelp_labelled.txt", delimiter="\t", header=None)
df_amazon = pd.read_csv("amazon_cells_labelled.txt", delimiter="\t", header=None)
frames = [df_imdb, df_yelp, df_amazon]
df = pd.concat(frames, ignore_index=True)
df.columns = ['Sentence', 'Score']

#################################################################################################################
# STEP 1: Feature Construction

# Function:   process_text
# Purpose:    To take in each review and preprocess the data. It will accomplish the folowing:
#             1. Remove punctuations. Punctuation removed are [!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]
#             2. Remove Stop words. Stop words in NLP are useless words such as "the", "a", "and", etc.
#             3: Stemming text. Words such as "like" and "liked" will be trimmed and treated as the same word
# Input:      
#     text:   A single review, which is a string
# Output:
#     finished_words:   A list of post processed data for the single review
#
def process_text(text):
    
    #1 Remove Punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2 Remove Stop Words
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

    #3: Stemming text
    stemmer = PorterStemmer()
    finished_words = [stemmer.stem(word) for word in clean_words]
    
    #4 Return a list of data cleaned words
    return finished_words


# create a vectorizer object, using the process_text function as the means to tokenize the data, and transform the data to an M x N matrix. M being records and N being the number of words
vectorizer = CountVectorizer(analyzer=process_text)
tokenized_matrix = vectorizer.fit_transform(df.Sentence)

#################################################################################################################
# Step 2: Dataset Split
# Will divide data into a training, testing, and split set
# Training: 60%, Testing: 20%, Validation: 20%
X = tokenized_matrix
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# For validation, 80% * 25% = 20%, giving us the desired 60, 20, 20 split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

#################################################################################################################
# Step 3: Feature Selection

cv = CountVectorizer(analyzer=process_text,max_features=1000)
tokenized_matrix_1000 = cv.fit_transform(df.Sentence)

# train, test, and validation split for the top 1000 feature vector
X_1000_train, X_1000_test, y_1000_train, y_1000_test = train_test_split(X, y, test_size=0.2, random_state=1)

# For validation, 80% * 25% = 20%, giving us the desired 60, 20, 20 split
X_1000_train, X_1000_val, y_1000_train, y_1000_val = train_test_split(X_1000_train, y_1000_train, test_size=0.25, random_state=1)

#################################################################################################################
# Step 4: Classification Algorithms


# Function:   run_classifier
# Purpose:    To train, predict, and report scoring for a given classifier and its respective data
#             
# Input:      
#     model:   The classifier that will be runned
#     X_train:  training data for sentences
#     y_train:  training data for labels 
#     X_test:   testing data for sentences
#     y_test:   testing data for labels 
#     model_name:   Name of the model, which is a string
# Output:
#     finished_words:   A list of post processed data for the single review
#
def run_classifier(model, X_train, y_train, X_test, y_test, model_name):

    offline_start = time.perf_counter()
    model.fit(X_train, y_train)
    offline_end = time.perf_counter()

    online_start = time.perf_counter()
    prediction = model.predict(X_test)
    online_end = time.perf_counter()

    tp, fp, fn, tn = confusion_matrix(y_test, prediction).ravel()
    specificity = tn / (tn+fp)

    print('Classification Report \n', classification_report(y_test, prediction))
    print()
    print('Confusion Matrix \n', confusion_matrix(y_test, prediction))
    print()
    print('Accuracy: ', accuracy_score(y_test, prediction)*100, '%')
    print()
    print('AUROC: ', roc_auc_score(y_test, prediction))
    print('Specificity: ', specificity)
    print()
    print('Offline efficiency cost: ', offline_end- offline_start, 'seconds')
    print('Online efficiency cost: ', online_end- online_start, 'seconds')
    print()
    # create and print graph - maybe should make this it's own function
    y_score = model.predict_proba(X_test)[:,1]
    fpr, tpr, threshold = roc_curve(y_test, y_score)
    plt.title('ROC - ' + model_name)
    plt.plot(fpr, tpr, label = 'ROC - ' + model_name)
    plt.plot([0,1], ls = '--', label = 'No Skill')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.show()

    print("------------------------------------------------------------------------------------")
    print()

#################################################################################################################
# Step 5: Experimental Results and Comparisons

# Classification Algorithm #1: Naive Bayes -- All features
print('Analysis of Naive Bayes -- All Features -- Training Data')
run_classifier(MultinomialNB(), X_train, y_train, X_train, y_train, 'Naive Bayes')
print()
print('Analysis of Naive Bayes -- All Features -- Testing Data')
run_classifier(MultinomialNB(), X_train, y_train, X_test, y_test, 'Naive Bayes')
print()
print('Analysis of Naive Bayes -- All Features -- Validation Data')
run_classifier(MultinomialNB(), X_train, y_train, X_val, y_val, 'Naive Bayes')

# Classification Algorithm #1: Naive Bayes -- Top 1000 features
print('Analysis of Naive Bayes -- Top 1000 features -- Training Data')
run_classifier(MultinomialNB(), X_1000_train, y_1000_train, X_1000_train, y_1000_train, 'Naive Bayes')
print()
print('Analysis of Naive Bayes -- Top 1000 features -- Testing Data')
run_classifier(MultinomialNB(), X_1000_train, y_1000_train, X_1000_test, y_1000_test, 'Naive Bayes')
print()
print('Analysis of Naive Bayes -- Top 1000 features -- Validation Data')
run_classifier(MultinomialNB(), X_1000_train, y_1000_train, X_1000_val, y_1000_val, 'Naive Bayes')



# Classification Algorithm #2: Random Forest Classifier -- All features
print('Analysis of Random Forest Classifier -- All Features -- Training Data')
run_classifier(RandomForestClassifier(), X_train, y_train, X_train, y_train, 'Random Forest')
print()
print('Analysis of Random Forest Classifier -- All Features -- Testing Data')
run_classifier(RandomForestClassifier(), X_train, y_train, X_test, y_test, 'Random Forest')
print()
print('Analysis of Random Forest Classifier -- All Features -- Validation Data')
run_classifier(RandomForestClassifier(), X_train, y_train, X_val, y_val, 'Random Forest')


# Classification Algorithm #2: Random Forest Classifier -- Top 1000 Features
print('Analysis of Random Forest Classifier -- Top 1000 Features -- Training Data')
run_classifier(RandomForestClassifier(), X_1000_train, y_1000_train, X_1000_train, y_1000_train, 'Random Forest')
print()
print('Analysis of Random Forest Classifier -- Top 1000 Features -- Testing Data')
run_classifier(RandomForestClassifier(), X_1000_train, y_1000_train, X_1000_test, y_1000_test, 'Random Forest')
print()
print('Analysis of Random Forest Classifier -- Top 1000 Features -- Validation Data')
run_classifier(RandomForestClassifier(), X_1000_train, y_1000_train, X_1000_val, y_1000_val, 'Random Forest')


# Classification Algorithm #3: K Nearest Neighbors -- All Features
print('Analysis of K Nearest Neighbors -- All Features -- Training Data')
run_classifier(KNeighborsClassifier(n_neighbors=5), X_train, y_train, X_train, y_train, 'KNN')
print()
print('Analysis of K Nearest Neighbors -- All Features -- Testing Data')
run_classifier(KNeighborsClassifier(n_neighbors=5), X_train, y_train, X_test, y_test, 'KNN')
print()
print('Analysis of K Nearest Neighbors -- All Features -- Validation Data')
run_classifier(KNeighborsClassifier(n_neighbors=5), X_train, y_train, X_val, y_val, 'KNN')

# Classification Algorithm #3: K Nearest Neighbors -- Top 1000 Features
print('Analysis of K Nearest Neighbors -- Top 1000 Features -- Training Data')
run_classifier(KNeighborsClassifier(n_neighbors=5), X_1000_train, y_1000_train, X_1000_train, y_1000_train, 'KNN')
print()
print('Analysis of K Nearest Neighbors -- Top 1000 Features -- Testing Data')
run_classifier(KNeighborsClassifier(n_neighbors=5), X_1000_train, y_1000_train, X_1000_test, y_1000_test, 'KNN')
print()
print('Analysis of K Nearest Neighbors -- Top 1000 Features -- Validation Data')
run_classifier(KNeighborsClassifier(n_neighbors=5), X_1000_train, y_1000_train, X_1000_val, y_1000_val, 'KNN')