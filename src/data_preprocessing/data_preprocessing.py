import os
import sys
import re
import string
from bs4 import BeautifulSoup
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_path, get_logger


logger = get_logger(__name__)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define stopwords and punctuation
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

# Functions for preprocessing
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)


def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered_words = [word for word in tokens if word.lower() not in stop and word.isalpha()]
    return ' '.join(filtered_words)

def denoise_text(text):
    text = strip_html(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text


def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    return ' '.join(lemmatized_tokens) 


def preprocess_data(file_path):
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    df.drop_duplicates(ignore_index=True, inplace=True)
    logger.info("Removed duplicates")

    logger.info("Starting text preprocessing")
    df['cleaned_text'] = df['review'].apply(denoise_text)
    df['lemmatize_text'] = df['cleaned_text'].apply(lemmatize_text)
    logger.info("Text preprocessing completed")

    return df


def main():
    train_file_path = get_data_path('raw/train.csv')
    test_file_path = get_data_path('raw/test.csv')
    processed_output_dir = get_data_path('processed')

    train_df = preprocess_data(train_file_path)
    test_df = preprocess_data(test_file_path)

    # Saving the preprocessed data
    train_output_path = os.path.join(processed_output_dir, 'train_processed.csv')
    test_output_path = os.path.join(processed_output_dir, 'test_processed.csv')
    
    os.makedirs(processed_output_dir, exist_ok=True)
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)
    logger.info(f"Preprocessed data saved to {processed_output_dir}")


if __name__ == '__main__':
    main()

