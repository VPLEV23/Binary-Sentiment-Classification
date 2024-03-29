
# Binary Sentiment Classification

# Project Directory Structure 
Below is an outline of the directory structure for the Binary-Sentiment-Classification, providing an overview of how the files and folders are organized:
```
Binary-Sentiment-Classification/
│
├── .github/ # GitHub Actions configurations
│ └── workflows/
│ └── python-app.yml # Workflow configuration file
│
├── data/ # Data directory (ignored by Git, generated by script)
│ ├── processed/ # Processed data ready for modeling
│ │ ├── test_processed.csv
│ │ └── train_processed.csv
│ └── raw/ # Raw data as downloaded
│ ├── test.csv
│ └── train.csv
│
├── myenv/ # Virtual environment directory (ignored by Git)
│
├── notebooks/
│ └── results/
│ └── experiments.ipynb # Jupyter notebook for experiments
│
├── outputs/ # Outputs directory (ignored by Git, generated by script)
│ ├── figures/ # Generated figures and plots
│ ├── models/ # Trained model files
│ └── predictions/ # Output predictions from the model
│
└── src/ # Source code for the project
├── data_preprocessing/ # Data preprocessing scripts
│ ├── init.py
│ └── data_preprocessing.py
│
├── inference/ # Inference scripts
│ ├── init.py
│ ├── Dockerfile
│ └── run_inference.py
│
├── tests/ # Unit tests for the application
│ ├── init.py
│ └── test_data_preprocessing.py
│
└── train/ # Training scripts
├── init.py
├── Dockerfile
└── train.py
│
└── data_load.py # Script to load data into the project
└── eda.py # Exploratory Data Analysis script
└── main.py # Main script to run the entire pipeline
└── # Python dependencies required for the projectutils.py # Utility functions used across the project
├── .dockerignore # Specifies files to ignore in Docker builds
├── .gitignore # Specifies files to be ignored by Git
├── README.md # Project README with detailed instructions
├── requirements.txt # Python dependencies required for the project
```
  ## Table of Contents 
- [Overview](#overview)
- [Data](#data)
- [DS part Report](#ds-part-report) 
	- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda) 
	- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing) 
	- [Model Building and Evaluation](#model-building-and-evaluation)
   	- [Potential Buisness Applications](#potential-business-applications)
	- [Downloading GloVe Word Embeddings](#downloading-glove-word-embeddings)
- [MLE Part](#mle-part) 
	- [Getting Started](#getting-started) 
	- [Setting Up a Local Environment](#setting-up-a-local-environment)
	- [Running the Application](#running-the-application)
	- [Using Docker](#using-docker) 
		- [Building the Docker Containers](#building-the-docker-containers) 
		- [Running the Docker Containers](#running-the-docker-containers) 
	- [Continuous Integration](#continuous-integration)


## Overview

This project focuses on binary sentiment classification, aiming to classify movie reviews into positive or negative sentiments. The dataset used contains 50,000 polar movie reviews, providing a balanced binary classification task.

  
### Data

-  **Type**: Binary Classification

-  **Target**: Sentiment (Positive/Negative)

-  **Size**: 50,000 movie reviews
## DS part Report

### Exploratory Data Analysis (EDA) 

Initial data analysis was conducted to better understand the dataset's characteristics and guide subsequent data preprocessing and modeling steps.

### Sentiment Distribution in Training Data 

The sentiment distribution in the training data is evenly split between positive and negative reviews, which is ideal for a balanced binary classification task.

### Review Length Distribution in Training Data 

The review length distribution is right-skewed, indicating that most reviews are moderate, with some exceptionally long reviews. This suggests that most reviews provide enough content for analysis without being overly verbose.

### Word Clouds 

Word clouds were generated for both positive and negative reviews to visualize the most frequently occurring words, providing insight into common themes and terms used in different sentiments.

### Word Count Distributions 

Histograms were plotted to show the distribution of word counts within the reviews, highlighting the commonality of moderate-length reviews and the presence of some outliers with a very high word count.

### Dynamic Stopword Count Distribution

Dynamic stopwords are those that are most frequent and may not contribute much to the sentiment of the review. The distribution of these stopwords was analyzed for both positive and negative reviews.

### Common Words Frequency 

A bar chart was created to show the frequency of the top 20 most common words found in the reviews. This chart helps to identify the most dominant words in the dataset and understand their impact on model performance.

### Review Text Examples 

A few examples of the reviews and their sentiments:
```text
"I caught this little gem totally by accident but am so glad I did..." Sentiment: Positive "I can't believe that I let myself into this movie..." Sentiment: Negative
```

###  Data Cleaning and Preprocessing 

To ensure the quality of the text data for our models, we performed several cleaning steps:
1.  **HTML Tag Removal**: We removed HTML content, which could be present due to web scraping. 
2.  **Removing Text Inside Square Brackets**: Any text within square brackets often includes non-relevant information like citations and was removed. 
3.  **URL Removal**: URLs do not contribute to sentiment and were removed. 
4. **Stopword Removal**: Common English words that do not carry much meaning were removed to reduce noise in the data. 
5. **Noise Removal**: Any non-textual elements were removed to leave only meaningful textual data.
6. **Label Encoding**: The target variable (sentiment) was converted from a string to a numerical representation for model training purposes.

### Visualization Post-Cleaning 

After cleaning, we visualized the data again with word clouds to ensure that the most frequent words were indeed relevant to the sentiment analysis task.

### Feature Extraction  

#### Word Frequency Analysis

We performed a word frequency analysis to identify the most common words in the reviews after cleaning. Here are the top 10 words: 
```text 
'movie': 48898 occurrences 'film': 44086 occurrences 'one': 35910 occurrences 'like': 29519 occurrences ...
```
#### N-Gram Analysis

N-gram analysis was conducted to understand the common patterns in the text data, particularly the frequency of word sequences.

### Stemming vs. Lemmatization

We compared stemming and lemmatization to understand their impact on the model's performance:

1.  **Tokenization**: We first tokenized the text to split it into individual words.
2.  **Stemming**: Each word was reduced to its stem, revealing that while stemming is faster, it might not always produce semantically accurate results.
3.  **Lemmatization**: This approach considered the Part-of-Speech of each word, resulting in more meaningful word forms and potentially better model performance.

### Text Vectorization

We explored two vectorization techniques:

1.  **TF-IDF Vectorization**: This method highlighted the importance of words in reviews relative to their uniqueness across the corpus.
2.  **Word Embeddings**:
    -   We used GloVe pre-trained word embeddings to capture semantic relationships between words based on a large corpus they were trained on.

### Comparison of Vectorization Techniques

The impact of these vectorization techniques was evaluated on the model's performance, considering accuracy, precision, recall, and F1-score. TF-IDF generally provided more discriminative features for our models compared to GloVe embeddings.

## Model Building and Evaluation 

Several models were trained and evaluated, with the best performing model selected based on accuracy and AUC score.

### Model Performance Here's a summary of the model performance:
#### Multinomial Naive Bayes 
 -  **Accuracy**: 87.53%
 -  **AUC**: 0.94

#### LinearSVC  
-  **Accuracy**: 88.42%
 -  **AUC**: 0.95
 
 #### SGDClassifier 
 -  **Accuracy**: 88.57%
 -  **AUC**: 0.95
 
### Best Model 
After evaluating the models, the `LinearSVC` was chosen as the best model based on the following metrics:
 -  **Accuracy**: 88.42% 
 -  **AUC**: 0.95
 
## Conclusion 
The `LinearSVC` model demonstrated the highest AUC score and accuracy, making it the preferred model for deployment. Its robust performance indicates its effectiveness in handling the feature space and making accurate predictions.
## Potential business applications
Potential business applications of binary sentiment classification, especially in the context of movie reviews, are vast and varied. 
1. **Product and Service Feedback Analysis**:
	- **Value**: Quickly and efficiently assess customer sentiment from reviews, surveys, and social media. This helps in understanding customer satisfaction and areas of improvement.
2. **Market Research and Competitive Analysis**:
	- **Value**: Monitor public sentiment about products and services, both for your own business and competitors. Use insights to refine marketing strategies and product offerings.
3. **Customer Support Automation**:
	- **Value**: Automatically categorize the sentiment of customer support tickets to prioritize and route them appropriately, improving response times and customer satisfaction.
4. **Brand Monitoring**:
	- **Value**: Track brand reputation in real-time across various platforms to quickly respond to negative sentiments and amplify positive feedback.
5. **Content Personalization and Recommendation**:
	- **Value**: Enhance recommendation systems by incorporating sentiment analysis to suggest products, services, or content that resonates with individual preferences.
6. **Content Moderation**:
	- **Value**: Identify and filter out negative content that could harm community standards on platforms, maintaining a positive environment.
7. **Social Media Analysis**:
	- **Value**: Gauge public opinion on marketing campaigns, product launches, and brand events to measure impact and engagement.
8. **Financial Market Prediction**:
	- **Value**: Analyze sentiment in news articles, reports, and social media to inform financial market predictions and investment strategies.
9. **Cross-Functional Data Insights**:
	- **Value**: Combine sentiment analysis with other data sources for a more comprehensive understanding of customers and business performance.

## Downloading GloVe Word Embeddings

The GloVe (Global Vectors for Word Representation) embeddings are pre-trained word vectors that we use for converting text into numerical form that machine learning models can understand. Due to their large size, these files are not stored directly in the GitHub repository.

To download and use the GloVe embeddings for this project, please follow these steps:

1. Visit the [GloVe website](https://nlp.stanford.edu/projects/glove/) hosted by the Stanford NLP Group.

2. Scroll down to the 'Download pre-trained word vectors' section.

3. Click on the link for the pre-trained vectors you want to use. For example, `glove.6B.zip` is commonly used and contains various embedding dimensions.

4. Unzip the downloaded file to extract the `.txt` file containing the word vectors. This can be done using a file archiver tool or via the command line:

   ```bash
   unzip glove.6B.zip
   ```
   5.  Move the extracted `.txt` file to the directory `glove` in your project structure.
    
5.  Update the file paths in your code to point to the location where you've saved the GloVe vectors.
Please ensure you have sufficient storage space available, as these files can be quite large (e.g., `glove.6B.zip` is about 822MB).

Note: The use of the GloVe embeddings is subject to the terms of use provided by the Stanford NLP Group.

# MLE Part

## Getting Started

  

### Setting Up a Local Environment

To set up a local development environment, follow these steps:


1.  **Clone the Repository**

```bash
git clone https://github.com/VPLEV23/EPAM_final_Project.git

cd EPAM_final_Project
```
-  **Create a Virtual Environment** (Optional, but recommended)
	- For Windows:
	```bash
	python -m venv venv 
	
	.\venv\Scripts\activate
	```
	- For macOS and Linux:
	```bash
	python3 -m venv venv
	
   source venv/bin/activate
	```
2. **Install Dependencies**
```bash 
pip install -r requirements.txt
```
### Running the Application

**Note: Run all comads from root directory of the project !!!!**

###  Dataset Download Instructions 
To download the dataset, please run the `data_load.py` script located in the root of the project. This script will automatically download the dataset and place it in the `data/raw/` directory.
```bash
python .\src\data_load.py
   ```

### To run the application, we have several ways to do this:
From root directory of project run:
1. Run all scripts together 
```bash 
python -m src.main
```
**Note: This will start downloading data automaticly**

2. Run all scripts separately
	- To downlaod data use:
  		```bash
  		python .\src\data_load.py
 		 ```
	-  To run EDA script use:
		  ```bash 
		python .\src\eda.py 
		```
	-  To run data_preprocessing script use
	  	 ```bash 
	  	 python .\src\data_preprocessing\data_preprocessing.py
	  	 ```
	-   To run training script use:
	  	```bash 
		python .\src\train\train.py
		```
	- To run inference script use
		```bash 
		python .\src\inference\run_inference.py
		```
### Running Tests

Execute the following command to run unit tests:
```bash 
python -m unittest discover -s src/tests
```
## Using Docker

The following instructions will guide you on how to use Docker to run the training and inference processes for this project.

### Prerequisites

Ensure you have Docker installed on your system. You can download it from the [official Docker website](https://www.docker.com/products/docker-desktop).

### Building the Docker Containers
- For the training process:
```bash 
docker build -t my_train_image -f src/train/Dockerfile .
```
- For the inference process:
 ```bash 
docker build -t my_inference_image -f src/inference/Dockerfile .
```

## Running the Docker Containers

### Running the Training Process

To run the training process using Docker, you'll need to mount the `data` and `outputs` directories to the Docker container. This allows the container to access the data for training and save the outputs (such as the trained model and figures) back to your local filesystem.

The command for running the training container depends on your operating system:

#### For Linux/macOS:

```bash
docker run -v $(pwd)/data:/usr/src/app/data -v $(pwd)/outputs:/usr/src/app/outputs my_train_image
```
#### For Windows Command Prompt:
```bash
docker run -v %cd%/data:/usr/src/app/data -v %cd%/outputs:/usr/src/app/outputs my_train_image
```
#### For Windows PowerShell:
```bash
docker run -v ${PWD}/data:/usr/src/app/data -v ${PWD}/outputs:/usr/src/app/outputs my_train_image
```
The `$(pwd)` (or `%cd%` on Windows Command Prompt and `${PWD}` in PowerShell) is a shell variable that holds the path to the current working directory. This variable is used to provide an absolute path to Docker for volume mounting, which is necessary to ensure that Docker correctly maps the local directories to the container's file system regardless of where the `docker run` command is invoked from.
### Running the Inference Process

Similarly, to run the inference process using Docker:
#### For Linux/macOS:
```bash 
docker run -v $(pwd)/data:/usr/src/app/data -v $(pwd)/outputs:/usr/src/app/outputs my_inference_image
```
#### For Windows Command Prompt:
```bash
docker run -v %cd%/data:/usr/src/app/data -v %cd%/outputs:/usr/src/app/outputs my_inference_image
```
#### For Windows PowerShell:
```bash
docker run -v ${PWD}/data:/usr/src/app/data -v ${PWD}/outputs:/usr/src/app/outputs my_inference_image
```
After running these commands, you will find the results of the training or inference in the `outputs` directory on your local machine.

Please ensure that you replace `my_train_image` and `my_inference_image` with the actual names of your Docker images.

The benefit of using `PWD` (Print Working Directory) is to ensure that the Docker command is portable across different environments and the host's file system structure. This approach avoids hard-coded paths, making the Docker command adaptable to any directory from which it is run. This is particularly useful when sharing your Docker container with others who may have a different directory structure.

## Advantages of Using Volume Mounting in Docker

Volume mounting in Docker is a powerful feature that binds a directory on the host machine to a directory within the container. This has several benefits, particularly when dealing with large datasets or when scalability is a concern:

### Data Persistence

When containers are removed, any data that was written inside the container is lost. By mounting volumes, data can persist beyond the life of a single container. This is crucial for training models and running inference where the output needs to be accessed or stored for long-term use.

### Data Separation from Container's Lifecycle

By keeping data in volumes, the container's lifecycle (creation, running, and deletion) doesn't affect the actual data. This separation allows you to update, upgrade, or even switch containers while keeping your data intact.

### Efficient Data Handling

For machine learning tasks that involve large datasets, it is inefficient to build the data into the image or to copy it into the container every time it starts. Volume mounting allows you to access large datasets from the host system without unnecessary duplication of data, saving both time and storage space.

### Improved Performance

Accessing data directly from the host filesystem can be faster than accessing it from within a container's layered filesystem. This is especially true for I/O intensive applications such as data processing and model training.

### Scalability

Volume mounting is conducive to scaling applications. In a distributed system or cloud environment, you can mount storage from different hosts to different containers in a way that is transparent to the application. This means you can scale up your data processing or inference tasks across multiple containers without each container needing its own copy of the data.

### Ease of Backup and Recovery

With data in volumes, backing up is as simple as backing up the directories on the host system. This also simplifies the process of data recovery in case of any failures, as the data is not tied to the container's lifecycle.

In summary, using volume mounting with Docker ensures data is managed effectively, performance is optimized, and your machine learning workflows are set up for scalability.

Remember to replace the placeholder image names (`my_train_image` and `my_inference_image`) with the actual names of your Docker images when running the commands.

## Continuous Integration

This project uses GitHub Actions for continuous integration. The CI pipeline is triggered by pushes and pull requests to the main branch, ensuring that all tests pass.
