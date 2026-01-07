# AutoJudge - Predicting Programming Problem Difficulty:

AutoJudge is a project that predicts (a) the difficulty class of a programming problem (Easy / Medium / Hard) and (b) a numerical difficulty score, using only the textual problem statement (title, description, input/output).  
The project includes data preprocessing, feature extraction, training (classification + regression)and a Streamlit web UI.

## Project Overview:

Take the text fields of a programming problem, clean & combine them into a single text input, extract features (TF-IDF + other text features), and train:
1. A classification model to predict Easy / Medium / Hard
2. A regression model to predict a numerical difficulty score
Provide a simple Streamlit web app so a user can paste a problem statement and get predictions.

## Dataset used:
The dataset used to train the models was the provided dataset (originally in .jsonl format, was later converted to .csv)
Each data sample contains:
1. title
2. description
3. input_description
4. output_description
5. problem_class → Easy / Medium / Hard
6. problem_score → numerical value (0–10)

## Approach & Models used:
### 1. Feature engineering:

- Clean text: lowercase, remove punctuation, basic tokenization, Combine all text fields into   combined_text.
- TF-IDF vectorization:
    - used TfidfVectorizer to convert text features to numeric values
    - used unigrams and bigrams inside TfidfVectorizer
- Other numeric features:
    - text length (word count)
    - number of occurrences of math symbols 
    - keyword frequency (domain keywords like dp, graph, dfs, bfs, recursion, greedy, ...)
- Combine sparse TF-IDF matrix + dense other features into a final feature matrix (scipy sparse hstack).

### 2. Models:

- Classification:
    1. Baseline: LogisticRegression
    2. Stronger: RandomForestClassifier (used)
    3. Optional: LinearSVC for comparison

- Regression:
    1. Baseline: LinearRegression
    2. Stronger: RandomForestRegressor (used)
    3. Optional: GradientBoostingRegressor for comparison

### 3. Web UI:
- Loaded trained and saved models (Vectorizer, Classifier and Regressor) in app.py using joblib
- Built the Web Interface using Streamlit

## Evaluation Metrics:
1. Classification:
    - Accuracy
    - F1-score-macro (useful for class imbalance)
    - Confusion matrix

2. Regression:
    - MAE
    - RMSE 
    - R2 (indicates proportion of variance explained) 

The evaluation results for various classifiers and regressors have been reported.

## How to run locally?
(Ensure your Python is up to date.)
Open the test folder in command prompt and run these commands:
1. Clone the GitHub Repository.
    * git clone "https://github.com/unknowndx1651/AutoJudge".git
2. Create and active a virtual environment. Enter the repository.
    * python -m venv venv
    * venv\Scripts\activate
    * cd AutoJudge
4. Install dependencies like:  
    * pip install --upgrade pip  
    * pip install -r requirements.txt
5. (OPTIONAL) Place "dataset.csv" inside data/ folder (OR place "dataset.jsonl" and run "data_loader.py" from src/ to convert it to .csv format)
6. (OPTIONAL) Train models again by running src/classifier.py and src/regressor.py
    - run using  
    "python -m src.file"  
    from AutoJudge/ directory to train and save models in models/
7. From AutoJudge/ run   
    * python -m streamlit run app.py
    to open the web interface

## Web Interface (Built using Streamlit):
The Streamlit UI is minimal and focused:  

* Input fields (3 Textboxes):
    * Problem description
    * Input description
    * Output description  

* Predict button — click after pasting text

* Outputs (2 columns):
    * Predicted Difficulty Class (Easy / Medium / Hard)
    * Predicted score (0 - 10)

Internals:
- The UI sends the combined text to the saved vectoriser and models (models/vectoriser.pkl, models/RFC.pkl, models/RFR.pkl) to get predictions.
- If models are missing, the app prompts you to run training scripts or place saved models in models/.

## Details:
Name : Daksh Shah  
Enrollment number : 24115132  
Email : dakshshah71006@gmail.com  
GitHub : https://github.com/unknowndx1651

## Demo Video:




