# Anti-Spam Software for an University

This repository contains code for a **spam detection system** developed for **ProfessionAI**, aimed at analyzing email content. The software can **identify spam emails**, **extract main topics** from them, **calculate semantic distances** between topics, and **identify organizations** mentioned in non-spam emails.

## Project Overview

The project addresses four main requirements:

1. **Spam Classification**: Training a classifier to identify spam emails
2. **Topic Identification**: Extracting main topics from spam emails in the dataset
3. **Topic Heterogeneity Analysis**: Calculating semantic distance between topics to evaluate their diversity
4. **Organization Extraction**: Identifying organizations mentioned in non-spam emails

## Dataset

The project uses a dataset of emails labeled as **spam** or **ham** (non-spam), with the following columns:
- **`label`**: String indicating whether the email is 'spam' or 'ham'
- **`text`**: The full text content of the email
- **`label_num`**: Numeric representation of the label (1 for spam, 0 for ham)

## Implementation

### Data Preparation and Exploration

- **Initial data cleaning** (removing duplicates and unnecessary columns)
- **Basic exploratory data analysis**
- **Train-test split** (70/30) while maintaining class distribution

### Text Preprocessing

The preprocessing pipeline includes:
- **Converting text to lowercase**
- **Removing punctuation**
- **Lemmatization** using spaCy
- **Removing stopwords**
- **Removing digits**
- **TF-IDF vectorization**

### Model Selection

Two classification models were evaluated:
- **Logistic Regression**: Achieved an F1 score of 0.9727
- **Multi-layer Perceptron**: Achieved an F1 score of 0.9700

**Logistic Regression** was selected as the final model due to its slightly better performance.

### Topic Modeling

For spam emails, the system identifies main topics using **LDA (Latent Dirichlet Allocation)**:
- **Preprocessing and cleaning** of spam emails
- **Dictionary and corpus creation** with filtering of extreme terms
- **Training an LDA model** with 5 topics
- **Extraction of keywords** for each topic

The **5 main topics** identified in spam emails are:
1. **Company/Business**: "company", "statements", "information", "securities", "within"
2. **Account/Financial**: "account", "price", "online", "microsoft", "windows"
3. **HTML content**: "nbsp", "email", "million", "company", "voip"
4. **Formatting/Pills**: "font", "pills", "color", "align", "size"
5. **Technology/Contact**: "computron", "email", "free", "contact", "epson"

### Semantic Distance Analysis

The system calculates **semantic distances** between topics using word embeddings:
- **Loading a pre-trained FastText model**
- **Converting topic keywords to vectors**
- **Computing cosine similarity** between topic vectors
- **Analyzing the heterogeneity** of topics

The analysis shows a **moderate level of heterogeneity** among topics, with similarity scores ranging from approximately 0.50 to 0.73.

### Organization Extraction

For non-spam emails, the system extracts organization names using:
- **Named Entity Recognition** with spaCy's large English model
- **Filtering** to keep only 'ORG' entities
- **Additional cleaning** to remove noise
- **Frequency counting** of organizations

## Results

1. **Spam Classification**: The Logistic Regression model achieved an **F1 score of 0.9705** on the test set
2. **Topic Modeling**: **5 distinct topics** were identified in spam emails
3. **Semantic Distance**: The topics show **moderate heterogeneity**, with similarity scores suggesting related but distinct topics
4. **Organization Extraction**: Successfully extracted **organization names** from non-spam emails, with frequency counts

## Dependencies

- **pandas**
- **numpy**
- **scikit-learn**
- **spaCy** (with `en_core_web_sm` and `en_core_web_lg` models)
- **gensim**
- **nltk**
- **scipy**

## Usage

1. **Clone the repository**
2. **Install the required dependencies**
3. **Run the Jupyter notebook** to train the model and analyze the results
4. For **inference on new emails**, use the trained model with the preprocessing pipeline

## Project Structure

- **`spam_detection.ipynb`**: Main Jupyter notebook with all code and analysis
- **`models/`**: Directory containing trained models
- **`data/`**: Dataset directory
- **`README.md`**: This file

## Future Work

- Implement a **real-time filtering system**
- Explore more advanced **deep learning models**
- Add a **web interface** for user interaction
- Extend **organization extraction** with relationship mapping
- Improve **topic modeling** with more advanced techniques
