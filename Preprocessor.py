import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from emoji import UNICODE_EMOJI
from textblob import TextBlob
from wordcloud import WordCloud,STOPWORDS
from collections import Counter
#!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#df=pd.read_csv("C:\\Users\\Dell\\Desktop\\Sentiment Analysis Main\\friedrice_comments.csv", encoding='utf-8', on_bad_lines='skip')
df=pd.read_csv("C:\\Users\\Dell\\Documents\\Sentiment Analysis\\Sentiment Analysis Main\\lookback2023_comments.csv", encoding='utf-8', on_bad_lines='skip')

polarity=[]
for i in df['comment']:
    try:
        polarity.append(TextBlob(i).sentiment.polarity)
    except:
        polarity.append(0)
df['polarity']=polarity
#print(df)


# Import the stopwords collection and the word_tokenize function
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from cleantext import clean # to remove emojis
import re

# Define the node class for the binary search tree
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

# Define the binary search tree class
class BST:
    def __init__(self):
        self.root = None

    # Insert a new word into the tree
    def insert(self, word):
        # If the tree is empty, make the word the root node
        if self.root is None:
            self.root = Node(word)
        else:
            # Start from the root node
            current = self.root
            # Loop until the word is inserted
            while True:
                # If the word is smaller than the current node's data, go to the left subtree
                if word < current.data:
                    # If the left child is empty, make the word the left child
                    if current.left is None:
                        current.left = Node(word)
                        break
                    # Otherwise, go to the left child
                    else:
                        current = current.left
                # If the word is larger than the current node's data, go to the right subtree
                elif word > current.data:
                    # If the right child is empty, make the word the right child
                    if current.right is None:
                        current.right = Node(word)
                        break
                    # Otherwise, go to the right child
                    else:
                        current = current.right
                # If the word is equal to the current node's data, do nothing
                else:
                    break

    # Display the tree in a level-order traversal
    def display(self):
        # If the tree is empty, print nothing
        if self.root is None:
            return
        else:
            # Create a queue to store the nodes at each level
            queue = []
            # Enqueue the root node
            queue.append(self.root)
            # Loop until the queue is empty
            while queue:
                # Get the number of nodes at the current level
                level_size = len(queue)
                # Loop for each node at the current level
                for _ in range(level_size):
                    # Dequeue the first node in the queue
                    node = queue.pop(0)
                    # Print the node's data
                    print(node.data, end=" ")
                    # Enqueue the left child if it exists
                    if node.left:
                        queue.append(node.left)
                    # Enqueue the right child if it exists
                    if node.right:
                        queue.append(node.right)
                # Print a newline after each level
                print()

    # Return a list of all the words in the tree in an in-order traversal
    def to_list(self):
        # Create an empty list to store the words
        words = []
        # Define a helper function to recursively traverse the tree
        def traverse(node):
            # If the node is not empty, visit the left subtree, the node, and the right subtree
            if node:
                traverse(node.left)
                words.append(node.data)
                traverse(node.right)
        # Call the helper function starting from the root node
        traverse(self.root)
        # Return the list of words
        return words

cleaned_comment_list=[]

# Convert the list of stopwords to a set for faster membership checks
stopwords_set = set(stopwords.words("english"))

# Define the string to process
for string in df['comment']:
    # Create an instance of the binary search tree i.e., new tree object for each comment
    tree = BST()

    # Remove symbols and keep only alphanumeric characters
    string = re.sub(r'[^a-zA-Z0-9\s]', '', string)
    # Remove numbers from the string
    string = re.sub(r'\d+', '', string)
    #Remove punctuations from string
    string = re.sub(r'[^\w\s]', '', string)
    
    # Tokenize the string into a list of words
    words = word_tokenize(string.lower())
    # Create an instance of the WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Loop through the words
    for word in words:
        # Lemmatize the word
        word = lemmatizer.lemmatize(word)
        # Check if the word is a stopword
        if word not in stopwords.words('english') and not re.match(r'[\U0001F300-\U0001F64F]', word):
            if word.startswith('http'):
                word = 'http'
            # Insert the word into the binary search tree
            tree.insert(word)
        
    # Append the list of words in the tree to the cleaned comment list
    cleaned_comment_list.append(' '.join(filter(None, tree.to_list())))

#print(cleaned_comment_list)

df.dropna(subset=['comment'], inplace=True)
# Reset the index of the dataframe
df.reset_index(drop=True, inplace=True)
df=df.assign(cleaned_comment=cleaned_comment_list)

df['comment_len'] = df['comment'].astype(str).apply(len)
df['word_count'] = df['comment'].apply(lambda x: len(str(x).split()))

df['textblob_polarity'] = df['cleaned_comment'].map(lambda text: TextBlob(text).sentiment.polarity)

sentiment = SentimentIntensityAnalyzer()
sentiment.polarity_scores(df.cleaned_comment[8])
df['vader_sentiment'] = df.cleaned_comment.apply(lambda x: sentiment.polarity_scores(x))
df['vader_neg_sentiment'] = df.vader_sentiment.apply(lambda x: x['neg'])
df['vader_pos_sentiment'] = df.vader_sentiment.apply(lambda x: x['pos'])
df['vader_comp_sentiment'] = df.vader_sentiment.apply(lambda x: x['compound'])

def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'
        
df['vader_category'] = df['vader_comp_sentiment'].apply(categorize_sentiment)
df['textblob_category'] = df['textblob_polarity'].apply(categorize_sentiment)

vader_counts = df['vader_category'].value_counts()
textblob_counts = df['textblob_category'].value_counts()


most_positive = df.sort_values(by=['vader_comp_sentiment'], ascending=False)[['comment']].head(10)
# print(most_positive)

# df.loc[575, 'comment']

most_negative = df.sort_values(by=['vader_comp_sentiment'], ascending=True)[['comment']].head(10)
#print(most_negative)

# need to check if to put before or after
df["vader_comp_sentiment"] = df["vader_comp_sentiment"].apply(np.sign)

# most neutral
neutral = df[df['vader_comp_sentiment'] == 0]
most_neutral=neutral['comment']
#print(most_neutral)

# Convert the cleaned_comments column into a list of words
words = []
for comment in df['cleaned_comment']:
    words.extend(comment.split())
# Create a Counter object and get the 10 most frequent words
word_counts = Counter(words)
top10 = word_counts.most_common(20)

def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'
        
df['vader_category'] = df['vader_comp_sentiment'].apply(categorize_sentiment)
df['textblob_category'] = df['textblob_polarity'].apply(categorize_sentiment)

vader_counts = df['vader_category'].value_counts()
textblob_counts = df['textblob_category'].value_counts()

def wordcloudall():
    wordcloudall = WordCloud(collocations=False, colormap='magma', width=1000, height=500, stopwords=set(STOPWORDS), background_color='white',  random_state=42).generate(' '.join(df['cleaned_comment']))
    #plt.figure(figsize=(40,10))
    #plt.axis('off')
    return wordcloudall.to_image()

def wordcloudpos():
    wordcloudpos = WordCloud(collocations=False, colormap='Reds', width=1000, height=500, stopwords=set(STOPWORDS), background_color='white',  random_state=42).generate(' '.join(most_positive['comment']))
    #plt.figure(figsize=(40,10))
    #plt.axis('off')
    return wordcloudpos.to_image()

def wordcloudneg():
    wordcloudneg = WordCloud(collocations=False, colormap='bone', width=1000, height=500, stopwords=set(STOPWORDS), background_color='white',  random_state=42).generate(' '.join(most_negative['comment']))
    #plt.figure(figsize=(40,10))
    #plt.axis('off')
    return wordcloudneg.to_image()

def wordcloudneu():
    wordcloudneu = WordCloud(collocations=False, colormap='gist_grey', width=1000, height=500, stopwords=set(STOPWORDS), background_color='white',  random_state=42).generate(' '.join(most_neutral))
    #plt.figure(figsize=(40,10))
    #plt.axis('off')
    return wordcloudneu.to_image()
