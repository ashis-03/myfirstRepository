import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Initialize NLTK resources
nltk.download('stopwords')

# Initialize SnowballStemmer and stopwords
stemmer = SnowballStemmer("english")
stopword = set(stopwords.words("english"))

# Load the dataset
df = pd.read_csv("hatememe_data.csv")  # Adjust file name and path if necessary
print(df.head())

# Map class labels
df['labels'] = df['class'].map({0: "Hate Speech Detected", 1: "Offensive language detected", 3: "No hate and offensive speech"})
print(df.head())

# Select relevant columns
df = df[['text', 'labels']]  # Adjust column names if necessary
print(df.head())

# Define text cleaning function
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text

# Apply text cleaning function to 'text' column
df["text"] = df["text"].apply(clean)
print(df.head())

# Prepare data for modeling
x = np.array(df["text"])
y = np.array(df["labels"])

# Vectorize text data
CV = CountVectorizer()
x = CV.fit_transform(x)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Initialize and train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)