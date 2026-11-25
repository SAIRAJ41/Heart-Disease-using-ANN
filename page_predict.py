# page_predict.py - "Predict" page: patient form, probability card and input signals
import numpy as np
import streamlit as st
from config import C0, C1, CHEST_PAIN, ST_SLOPE, RESTING_ECG
from model_utils import predict, signals, metrics


def render(thr):
    with st.form("patient"):
        st.subheader("Patient details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 20, 90, 54)
            sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
            chest_pain = st.selectbox("Chest pain type", CHEST_PAIN)
            exercise_angina = st.radio("Chest pain during exercise?", ["No", "Yes"], horizontal=True)
        with c2:
            resting_bp = st.number_input("Resting blood pressure (mm Hg)", 80, 200, 130)
            chol_known = st.checkbox("Cholesterol is known", value=True)
            cholesterol = st.number_input("Cholesterol (mg/dl)", 85, 603, 240, disabled=not chol_known)
            fasting_bs = st.radio("Fasting blood sugar > 120 mg/dl?", ["No", "Yes"], horizontal=True)
        with c3:
            max_hr = st.number_input("Maximum heart rate achieved", 60, 202, 142)
            oldpeak = st.number_input("Oldpeak (ST depression)", -2.6, 6.2, 0.6, step=0.1)
            st_slope = st.selectbox("ST slope", ST_SLOPE)
            resting_ecg = st.selectbox("Resting ECG", RESTING_ECG)
        go = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

    if not go:
        return
    row = dict(age=age, sex=sex, chest_pain=chest_pain, resting_bp=float(resting_bp),
               cholesterol=float(cholesterol) if chol_known else np.nan,      # missing -> filled with the mean
               fasting_bs=fasting_bs, resting_ecg=resting_ecg, max_hr=max_hr,
               exercise_angina=exercise_angina, oldpeak=float(oldpeak), st_slope=st_slope)
    with st.spinner("Running the neural network..."):
        p = predict(row)
    flagged = p >= thr
    colour = C1 if flagged else C0
    label = "Heart disease likely" if flagged else "Heart disease unlikely"
    advice = ("Recommend a clinical check-up." if flagged
              else "No flag at this threshold. Symptoms still need a doctor's opinion.")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown(f"""<div class="card">
          <div style="color:#5b6b7f">Predicted probability</div>
          <div class="risk-num" style="color:{colour}">{p * 100:.0f}%</div>
          <div class="bar"><div style="width:{p * 100:.1f}%;background:{colour}"></div></div>
          <div style="font-size:.8rem;color:#5b6b7f">threshold {thr:.2f}</div>
          <h3 style="color:{colour};margin:.6rem 0 0">{label}</h3>
          <p style="color:#5b6b7f;margin:.3rem 0 0">{advice}</p></div>""", unsafe_allow_html=True)
    with right:
        risk, ok = signals(row)
        tags = "".join(f'<span class="tag">▲ {s}</span>' for s in risk)
        tags += "".join(f'<span class="tag ok">▼ {s}</span>' for s in ok)
        st.markdown(f"""<div class="card"><b>Signals in this input</b>
          <p style="color:#5b6b7f;font-size:.85rem;margin:.2rem 0 .6rem">
          ▲ linked with higher disease rates in the data · ▼ linked with lower rates</p>{tags}
          <p style="color:#5b6b7f;font-size:.8rem;margin-top:.8rem">Patterns from the dataset,
          not an explanation of the network's exact reasoning.</p></div>""", unsafe_allow_html=True)
    acc = metrics()["accuracy"]
    st.caption(f"Model test accuracy is about {acc * 100:.0f}%, so roughly {round((1 - acc) * 100)} in 100 "
               "predictions are wrong. Treat this as a prompt for a check-up, never as a result.")
