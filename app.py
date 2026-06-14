"""
Streamlit app for TrustLens AI.
Loads the saved model and vectorizer, takes a review as input
and shows whether it looks genuine or suspicious.
"""

import os
import re
import joblib
import streamlit as st

MODEL_PATH = "models/model.pkl"
VEC_PATH = "models/vectorizer.pkl"


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VEC_PATH)
    return model, vectorizer


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_top_suspicious_words(text, model, vectorizer, top_k=5):
    """Pick out the words from the input that push the prediction
    most strongly towards the 'suspicious' class. Just uses the
    Logistic Regression coefficients."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    # find which features are actually present in this review
    nonzero_idx = vec.nonzero()[1]
    word_scores = [(feature_names[i], coefs[i]) for i in nonzero_idx]
    # higher coef means more "suspicious" in our setup
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in word_scores[:top_k] if s > 0]


def main():
    st.set_page_config(page_title="TrustLens AI", page_icon="🔍")
    st.title("TrustLens AI")
    st.caption("A simple fake review detector built as a learning project.")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH)):
        st.error(
            "Model files not found. Please run `python train_model.py` first "
            "to generate them."
        )
        return

    model, vectorizer = load_artifacts()

    st.subheader("Paste a product review")
    review = st.text_area(
        "Review text",
        height=150,
        placeholder="Example: The product arrived on time. Battery is okay but not great...",
    )

    if st.button("Check Review"):
        if not review.strip():
            st.warning("Please enter some text first.")
            return

        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        st.markdown("---")
        if pred == 1:
            st.error(f"Prediction: Suspicious / likely fake")
            confidence = proba[1]
        else:
            st.success(f"Prediction: Looks genuine")
            confidence = proba[0]

        st.write(f"Confidence: **{confidence * 100:.1f}%**")

        # show the suspicious-leaning words from the review
        sus_words = get_top_suspicious_words(review, model, vectorizer)
        if sus_words:
            st.write("Words that pushed it toward 'suspicious':")
            st.write(", ".join(f"`{w}`" for w in sus_words))
        else:
            st.write("No strong suspicious words found in this review.")

    st.markdown("---")
    st.info(
        "Note: This is an educational student project. The model is trained on "
        "a small dataset and only looks at the review text. It does not consider "
        "reviewer history, ratings, or product metadata, so predictions are not "
        "always reliable."
    )


if __name__ == "__main__":
    main()
