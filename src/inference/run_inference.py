import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow.sklearn
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_path, get_output_path, get_logger

logger = get_logger(__name__)

def load_model(model_path):
    return joblib.load(model_path)

def main():
    # Load the best model
    model_path = os.path.join(get_output_path('models'), 'LinearSVC_best_model.pkl')
    model = load_model(model_path)

    vectorizer_path = os.path.join(get_output_path('models'), 'tfidf_vectorizer.pkl')
    tfidf_vectorizer = joblib.load(vectorizer_path)

    # Load the processed test data (which includes true labels)
    processed_test_data_path = get_data_path('processed/test_processed.csv')
    test_df = pd.read_csv(processed_test_data_path)

    # Assuming 'lemmatize_text' is the processed text and 'sentiment' is the label
    X_test = tfidf_vectorizer.transform(test_df['lemmatize_text'])
    y_true = test_df['sentiment']

    # Making predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, predictions)
    logger.info(f"Accuracy: {accuracy}")

    # Log or process predictions as needed
    logger.info(f"First few predictions: {predictions[:5]}")

    # Optionally, save the predictions to a file
    predictions_output_path = os.path.join(get_output_path('predictions'), 'predictions.csv')
    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)
    pd.DataFrame(predictions, columns=['Predictions']).to_csv(predictions_output_path, index=False)

    logger.info(f"Predictions saved to {predictions_output_path}")

if __name__ == '__main__':
    main()