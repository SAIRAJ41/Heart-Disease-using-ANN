# app.py  ->  python -m streamlit run app.py
# Entry point: page setup, header, sidebar and tabs. Each tab's content lives in its own page_*.py file.
import streamlit as st
from config import CSS
import page_predict
import page_performance

st.set_page_config(page_title="Heart Disease Risk Screening", page_icon="🫀", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🫀 Heart Disease Risk Screening</h1>
<p>Artificial Neural Network trained on 918 patient records · Neural Networks case study</p></div>""",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("About")
    st.write("Enter a patient's clinical details. The ANN (21 → 24 → 12 → 1) returns the probability of heart disease.")
    st.warning("Educational project only. This is a screening aid, not a medical diagnosis.")
    st.subheader("Decision threshold")
    thr = st.slider("Flag as 'heart disease' when probability ≥", 0.30, 0.70, 0.50, 0.05,
                    help="0.50 is the default. 0.40 is a 'screening mode' that catches more sick patients "
                         "but raises more false alarms.")

tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Model performance"])
with tab1:
    page_predict.render(thr)
with tab2:
    page_performance.render()
with tab3:
    page_limitations.render()
