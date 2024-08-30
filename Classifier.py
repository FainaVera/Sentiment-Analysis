import sklearn # machine learning
import pandas as pd
import os
import csv
import re
import plotly.graph_objects as go

import Preprocessor as prep

from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier # k-NN ensemble method
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df=prep.df

df['vader_label'] = df['vader_comp_sentiment'].apply(lambda score: 'positive' if score > 0 else ('negative' if score < 0 else 'neutral'))
# Split the dataframe into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(df["cleaned_comment"], df["vader_label"], test_size=0.2, random_state=42)

# Vectorize the cleaned comments
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, Y_train)

# Evaluate the Naive Bayes classifier
mnb_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, mnb_pred)
precision = precision_score(Y_test, mnb_pred, average="macro")
recall = recall_score(Y_test, mnb_pred, average="macro")
f1 = f1_score(Y_test, mnb_pred, average="macro")

print("Multinomial Naive Bayes")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

# Evaluate the KNN classifier
knn_pred = knn.predict(X_test)
accuracy = accuracy_score(Y_test, knn_pred)
precision = precision_score(Y_test, knn_pred, average="macro")
recall = recall_score(Y_test, knn_pred, average="macro")
f1 = f1_score(Y_test, knn_pred, average="macro")

print("\nK-Nearest Neighbors")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

def calculate_ratios(predictions):
    positive_ratio = sum(predictions == 'positive') / len(predictions)
    negative_ratio = sum(predictions == 'negative') / len(predictions)
    neutral_ratio = sum(predictions == 'neutral') / len(predictions)
    return positive_ratio, negative_ratio, neutral_ratio
    
# Calculate ratios for Naive Bayes predictions
mnb_positive, mnb_negative, mnb_neutral = calculate_ratios(mnb_pred)
# Calculate ratios for KNN predictions
knn_positive, knn_negative, knn_neutral = calculate_ratios(knn_pred)
labels = ['Positive', 'Negative', 'Neutral']
mnb_ratios = [mnb_positive, mnb_negative, mnb_neutral]
knn_ratios = [knn_positive, knn_negative, knn_neutral]

colors = ['#d90429', '#2b2d42', '#edeff4']
labels = ['Positive', 'Negative', 'Neutral']

def generate_mnb_pie_chart():
    fig = go.Figure(data=[go.Pie(labels=labels, values=mnb_ratios, marker=dict(colors=colors))])
    fig.update_layout(title='Sentiment ratios predicted by Naive Bayes')
    return fig

def generate_knn_pie_chart():
    fig = go.Figure(data=[go.Pie(labels=labels, values=knn_ratios, marker=dict(colors=colors))])
    fig.update_layout(title='Sentiment ratios predicted by KNN')
    return fig