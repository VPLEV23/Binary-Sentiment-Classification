import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, RocCurveDisplay
import mlflow
import mlflow.sklearn
import matplotlib
import joblib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_path, get_output_path, get_logger

logger = get_logger(__name__)

def train_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, figures_output_dir):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log parameters, metrics, and model
        mlflow.log_params({"model_type": model_name})
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        # Classification report
        report = classification_report(y_test, predictions)
        logger.info(f"Classification Report for {model_name}:\n{report}")

        # RocCurveDisplay
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        roc_curve_path = os.path.join(figures_output_dir, f'roc_curve_{model_name}.png')
        plt.savefig(roc_curve_path)
        mlflow.log_artifact(roc_curve_path)
        plt.close()

        return accuracy
    

def main():
    # Set up MLflow
    mlruns_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "mlruns"))
    formatted_mlruns_dir = mlruns_dir.replace('\\', '/')
    mlflow.set_tracking_uri(f"file:///{formatted_mlruns_dir}")
    mlflow.set_experiment("Text Classification Experiment")

    # Load preprocessed data
    processed_data_path = get_data_path('processed/train_processed.csv')
    train_df = pd.read_csv(processed_data_path)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(
        train_df['lemmatize_text'], train_df['sentiment'],
        test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Define models
    models = {
        "LinearSVC": LinearSVC(random_state=42),
        "MultinomialNB": MultinomialNB(),
        "SGDClassifier": SGDClassifier(random_state=42)
    }

    # Directory for saving figures
    figures_output_dir = get_output_path('figures')
    os.makedirs(figures_output_dir, exist_ok=True)

    best_accuracy = 0
    best_model_name = ""

    # Train and evaluate each model
    for name, model in models.items():
        accuracy = train_evaluate_model(model, name, X_train_tfidf, X_test_tfidf, y_train, y_test, figures_output_dir)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

    logger.info(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")
    
    vectorizer_path = os.path.join(get_output_path('models'), 'tfidf_vectorizer.pkl')
    joblib.dump(tfidf_vectorizer, vectorizer_path)



if __name__ == '__main__':
    main()
