from src import feature_extraction,preprocessing
import pandas as pd
from scipy.sparse import hstack
import joblib as jb
import streamlit as st

tfidf=jb.load("models/vectoriser.pkl")
classifier=jb.load("models/RFC.pkl")
regressor=jb.load("models/RFR.pkl")

def get_difficuly_class(difficulty)->str:
    mapping={0:"EASY", 1:"MEDIUM", 2:"HARD"}
    return mapping[difficulty];

def get_difficulty_score(difficulty,score)->float:
    match(difficulty):
        case 0:
            return 0.7*score + 1.5
        case 1:
            return 0.7*score + 3.5
        case 2:
            return min(10.0, 0.7*score + 6.0)

def predict_difficulty(text):
    data=pd.DataFrame({"title": [""], 
         "description": [text],
         "input_description": [""],
         "output_description": [""],
         "sample_io": [""]})
    text_df = preprocessing.preprocessing(data)

    combined_text = text_df["combined_text"]

    X_tfidf = tfidf.transform(combined_text)
    X_others = feature_extraction.features(combined_text)
    X_final = hstack([X_tfidf, X_others])

    pred_class = classifier.predict(X_final)[0]
    pred_difficulty=regressor.predict(X_final)[0]
    
    return get_difficuly_class(pred_class), get_difficulty_score(pred_class,pred_difficulty)

# WEB UI -----------------------------------------------------------------------------------------
st.markdown("""
<style>
textarea {
    font-size: 18px !important;
}
label {
    font-size: 18px !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="AUTOJUDGE",
    layout="centered"
)

st.title("AUTOJUDGE",text_alignment="center")
st.subheader("Programming Problem Difficulty Predictor",text_alignment="center")

st.markdown("### Problem Details",text_alignment='center')

problem_desc = st.text_area(
    "Problem Description",
    height=200,
    placeholder="Describe the problem statement..."
)

input_desc = st.text_area(
    "Input Description",
    height=120,
    placeholder="Describe the input format..."
)

output_desc = st.text_area(
    "Output Description",
    height=120,
    placeholder="Describe the output format..."
)
st.markdown("---")

if st.button("Predict Difficulty",width="stretch" ):
    if not problem_desc.strip():
        st.warning("Please enter the problem description.")
    else:
        full_text = (
            problem_desc + " " +
            input_desc + " " +
            output_desc
        )

        difficulty, score = predict_difficulty(full_text)
        st.markdown("### Prediction Result", text_alignment="center")

        col1, col2 = st.columns(2)

        with col1:
            st.success(f"**DIFFICULTY CLASS**\n\n{difficulty}")

        with col2:
            st.info(f"**DIFFICULTY SCORE**\n\n{score:.1f} / 10")
   
st.markdown("---")
st.caption(
    "Difficulty score is an estimate derived from the predicted class. ",text_alignment="center"
)
