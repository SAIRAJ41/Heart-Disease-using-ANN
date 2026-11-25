# page_limitations.py - "Limitations & ethics" page
import streamlit as st
from model_utils import metrics


def render():
    m = metrics()
    st.subheader("Limitations")
    st.markdown(f"""
- **Small dataset:** {m['rows']} patients, {m['test_rows']} in the test set, so every score has roughly ±5 points of uncertainty.
- **Data quality matters most:** an earlier merged dataset contained 298 duplicate patients with opposite labels, which held accuracy near 70%. Using the clean source alone raised it to about 86%.
- **Missing values:** {m['chol_missing']} cholesterol values ({m['chol_missing'] / m['rows'] * 100:.0f}%) were recorded as 0 and filled with the average.
- **Mixed hospitals:** the data combines patients from several hospitals, which may record measurements differently.
- **Basic inputs only:** no medical history, imaging or full lab results.""")
    st.subheader("Ethics")
    st.markdown("""
- **Not a diagnosis:** a clinician must make every decision.
- **Cost of errors:** a missed patient can delay care; a false alarm causes worry and extra tests.
- **Fairness:** about 79% of the patients are male, so the model may be less reliable for women; ethnicity is not recorded, so it cannot be checked.
- **Privacy:** inputs are used only for this prediction and are not stored.""")
