# TrustLens AI - Fake Review Detection System

A small machine learning project that tries to tell whether a product
review on an e-commerce site looks genuine or suspicious (likely fake).
I built this while learning the basics of NLP and scikit-learn.

## Why I built this

I shop online a lot and I always end up reading product reviews before
buying anything. I noticed that some reviews feel a little off - too
many exclamation marks, repeated phrases like "best ever", "must buy",
"100% genuine", etc. I wanted to see if a simple ML model could pick
up on those patterns.

This is a learning project. The goal was to practice the full ML
workflow - from loading a dataset to deploying a small app - and not
to build a perfect detector.

## What it does

You paste a product review into the app and it tells you:
- whether the review looks genuine or suspicious
- a confidence score for the prediction
- which words in the review pushed the model towards "suspicious"

## Tech stack

- Python
- pandas (for loading and cleaning the data)
- scikit-learn (TF-IDF + Logistic Regression)
- Streamlit (for the UI)
- joblib (for saving and loading the model)

## Features

- Train a simple text classification model from a CSV file
- Save the trained model and vectorizer to disk
- Streamlit app that loads the model and predicts on user input
- Shows which words in the review the model considered suspicious
- Prints accuracy, precision, recall and F1-score during training

## How it works

1. **Data**: a CSV of reviews with a label - `0` for genuine and
   `1` for suspicious. I wrote the sample dataset myself with examples
   of both styles.
2. **Cleaning**: lowercase the text, strip extra spaces and remove
   stray symbols. Nothing fancy.
3. **Features**: `TfidfVectorizer` with unigrams and bigrams. Bigrams
   matter because phrases like "must buy" or "best ever" are stronger
   signals than the individual words.
4. **Model**: Logistic Regression with `class_weight="balanced"`. It
   is a simple model but it works well on text and the coefficients
   are easy to interpret, which is how I show the suspicious words.
5. **Evaluation**: a normal 80/20 train-test split and the four common
   metrics for classification.

## How to run locally

Clone the repo and go into the folder:

```bash
git clone https://github.com/<your-username>/TrustLens-AI.git
cd TrustLens-AI
```

(Optional) Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # on Mac/Linux
# venv\Scripts\activate    # on Windows
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train_model.py
```

This will save `model.pkl` and `vectorizer.pkl` inside the `models/`
folder.

Run the Streamlit app:

```bash
streamlit run app.py
```

Open the link it prints (usually `http://localhost:8501`).

## Screenshots

Screenshots go in the `screenshots/` folder. Once you run the app you
can add them like this:

![Home page](screenshots/home.png)
![Suspicious review example](screenshots/suspicious_example.png)

## What I learned

- How TF-IDF turns text into numerical features and why it weights
  rare-but-meaningful words higher than common ones.
- Why Logistic Regression is a good first model for text - it is fast,
  the coefficients are interpretable, and it does not overfit easily
  on small datasets.
- The difference between accuracy, precision, recall and F1, and why
  accuracy alone can be misleading when the dataset is imbalanced.
- How to save and load models with `joblib` so the training and the
  app can be separate scripts.
- How to make a quick UI with Streamlit without writing any frontend
  code.

## Limitations

I want to be honest about what this project is and is not:

- The dataset is small (around 100 sample reviews). A real detector
  would need thousands of labelled examples.
- The model only looks at the **text** of the review. It does not use
  the reviewer's history, ratings distribution, purchase verification,
  account age, or product metadata - all of which matter a lot in
  practice.
- The "suspicious" examples I wrote are exaggerated. Real fake reviews
  are often more subtle, and this model would likely miss them.
- The keyword highlighting is based on Logistic Regression coefficients.
  It is a rough explanation, not a guaranteed reason.
- Performance numbers depend on the random split, so they can move a
  little between runs.

## Future improvements

A few things I would try if I had more time:

- Collect or use a larger public dataset (for example the Yelp fake
  review dataset or the Amazon review dataset).
- Try other models like Naive Bayes, SVM, or a small neural network
  and compare them.
- Add features beyond the text - review length, punctuation density,
  reviewer history if available.
- Use cross-validation instead of a single train-test split for more
  stable metrics.
- Deploy the app on Streamlit Community Cloud so others can try it
  without cloning the repo.

## Disclaimer

This is an educational project built for learning purposes. It should
**not** be used to actually judge reviews on real shopping sites. The
predictions can be wrong and the model has no understanding of context
beyond the patterns in the small training set.
