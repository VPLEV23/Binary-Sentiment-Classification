import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import os
import nltk
from utils import get_data_path, get_output_path, get_logger

logger = get_logger(__name__)

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to generate and save word cloud
def generate_and_save_word_cloud(text, title, output_dir, file_name):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis("off")
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()

# Function to plot and save histogram
def plot_and_save_histogram(series, title, xlabel, ylabel, output_dir, file_name, color):
    plt.figure(figsize=(12, 6))
    sns.histplot(series, bins=50, color=color, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()

# Function to find common words
def find_common_words(reviews, n=100):
    all_reviews_text = ' '.join(reviews).lower()
    all_words = all_reviews_text.split()
    most_common_words = Counter(all_words).most_common(n)
    return [word for word, count in most_common_words]

# Function to count dynamic stopwords
def count_dynamic_stopwords(review, stopwords_list):
    review_tokens = review.split()
    return sum(token.lower() in stopwords_list for token in review_tokens)

# Function to plot word frequency
def plot_word_frequency(reviews, n, output_dir, file_name):
    all_reviews_text = ' '.join(reviews).lower()
    all_words = all_reviews_text.split()
    word_freq = Counter(all_words)
    common_words = word_freq.most_common(n)
    words = [word for word, freq in common_words]
    counts = [freq for word, freq in common_words]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=words, y=counts, palette='viridis')
    plt.xticks(rotation=45)
    plt.title(f'Top {n} Most Common Words in Reviews')
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()

# def main():
#     logger.info("Starting EDA script")
#     try:
#         data_file_path = get_data_path('raw/train.csv')
#         output_dir = get_output_path('figures/')
#         # Rest of your script
#     except Exception as e:
#         logger.error(f"An error occurred: {e}")
#         raise
#     os.makedirs(output_dir, exist_ok=True)

#     df = load_data(data_file_path)
#     stop_words = set(stopwords.words('english'))

#     # Separate reviews based on sentiment
#     positive_reviews = df[df['sentiment'] == 'positive']['review']
#     negative_reviews = df[df['sentiment'] == 'negative']['review']

#     # Generate and save word clouds
#     generate_and_save_word_cloud(" ".join(positive_reviews), "Word Cloud for Positive Reviews", output_dir, 'positive_wordcloud.png')
#     generate_and_save_word_cloud(" ".join(negative_reviews), "Word Cloud for Negative Reviews", output_dir, 'negative_wordcloud.png')

#     # Plot word count distribution
#     plot_and_save_histogram(positive_reviews.apply(lambda x: len(x.split())), 'Word Count Distribution for Positive Reviews', 'Word Count', 'Frequency', output_dir, 'positive_word_count.png', 'green')
#     plot_and_save_histogram(negative_reviews.apply(lambda x: len(x.split())), 'Word Count Distribution for Negative Reviews', 'Word Count', 'Frequency', output_dir, 'negative_word_count.png', 'red')

#     # Dynamic stopwords analysis
#     common_stopwords = find_common_words(pd.concat([positive_reviews, negative_reviews], ignore_index=True))
#     df['dynamic_stopword_count'] = df['review'].apply(lambda review: count_dynamic_stopwords(review, common_stopwords))

#     positive_reviews_with_stopwords = df[df['sentiment'] == 'positive']
#     negative_reviews_with_stopwords = df[df['sentiment'] == 'negative']

#     # Plot and save dynamic stopwords distribution
#     plot_and_save_histogram(positive_reviews_with_stopwords['dynamic_stopword_count'], 'Dynamic Stopword Count Distribution for Positive Reviews', 'Dynamic Stopword Count', 'Frequency', output_dir, 'positive_dynamic_stopwords.png', 'green')
#     plot_and_save_histogram(negative_reviews_with_stopwords['dynamic_stopword_count'], 'Dynamic Stopword Count Distribution for Negative Reviews', 'Dynamic Stopword Count', 'Frequency', output_dir, 'negative_dynamic_stopwords.png', 'red')

#     # Plot and save common word frequency
#     all_reviews = pd.concat([positive_reviews, negative_reviews], ignore_index=True)
#     plot_word_frequency(all_reviews, 20, output_dir, 'common_word_frequency.png')

# if __name__ == '__main__':
#     main()


def main():
    logger.info("Starting EDA script")
    try:
        data_file_path = get_data_path('raw/train.csv')
        output_dir = get_output_path('figures/')
        logger.info(f"Data file path: {data_file_path}")
        logger.info(f"Output directory: {output_dir}")

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory created at {output_dir}")

        df = load_data(data_file_path)
        logger.info("Data loaded successfully")

        stop_words = set(stopwords.words('english'))
        logger.info("Stop words set created")

        # Separate reviews based on sentiment
        positive_reviews = df[df['sentiment'] == 'positive']['review']
        negative_reviews = df[df['sentiment'] == 'negative']['review']
        logger.info("Reviews separated based on sentiment")

        # Generate and save word clouds
        generate_and_save_word_cloud(" ".join(positive_reviews), "Word Cloud for Positive Reviews", output_dir, 'positive_wordcloud.png')
        generate_and_save_word_cloud(" ".join(negative_reviews), "Word Cloud for Negative Reviews", output_dir, 'negative_wordcloud.png')
        logger.info("Word clouds generated and saved")

        # Plot word count distribution
        plot_and_save_histogram(positive_reviews.apply(lambda x: len(x.split())), 'Word Count Distribution for Positive Reviews', 'Word Count', 'Frequency', output_dir, 'positive_word_count.png', 'green')
        plot_and_save_histogram(negative_reviews.apply(lambda x: len(x.split())), 'Word Count Distribution for Negative Reviews', 'Word Count', 'Frequency', output_dir, 'negative_word_count.png', 'red')
        logger.info("Word count distributions plotted and saved")

        # Dynamic stopwords analysis
        common_stopwords = find_common_words(pd.concat([positive_reviews, negative_reviews], ignore_index=True))
        df['dynamic_stopword_count'] = df['review'].apply(lambda review: count_dynamic_stopwords(review, common_stopwords))
        logger.info("Dynamic stopwords analysis completed")

        
        positive_reviews_with_stopwords = df[df['sentiment'] == 'positive']
        negative_reviews_with_stopwords = df[df['sentiment'] == 'negative']
        

        # Plot and save dynamic stopwords distribution
        plot_and_save_histogram(positive_reviews_with_stopwords['dynamic_stopword_count'], 'Dynamic Stopword Count Distribution for Positive Reviews', 'Dynamic Stopword Count', 'Frequency', output_dir, 'positive_dynamic_stopwords.png', 'green')
        plot_and_save_histogram(negative_reviews_with_stopwords['dynamic_stopword_count'], 'Dynamic Stopword Count Distribution for Negative Reviews', 'Dynamic Stopword Count', 'Frequency', output_dir, 'negative_dynamic_stopwords.png', 'red')
        logger.info("Dynamic stopwords distributions plotted and saved")

        # Plot and save common word frequency
        all_reviews = pd.concat([positive_reviews, negative_reviews], ignore_index=True)
        plot_word_frequency(all_reviews, 20, output_dir, 'common_word_frequency.png')
        logger.info("Common word frequency plotted and saved")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()