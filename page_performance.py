# page_performance.py - "Model performance" page: test metrics and result graphs
import os
import streamlit as st
from config import ASSETS
from model_utils import metrics


def img(name):
    return os.path.join(ASSETS, name)


def render():
    m = metrics()                                       # numbers saved by model_utils.train()
    st.subheader(f"Test-set results ({m['test_rows']} unseen patients, threshold 0.5)")
    cards = [("Accuracy", m["accuracy"]), ("Recall", m["recall"]), ("Precision", m["precision"]),
             ("F1 score", m["f1"]), ("ROC-AUC", m["roc_auc"])]
    for col, (name, val) in zip(st.columns(5), cards):
        col.markdown(f'<div class="card metric"><b>{val * 100:.1f}%</b><span>{name}</span></div>',
                     unsafe_allow_html=True)
    st.info(f"Across 10 random seeds the ANN averages {m['seed_mean'] * 100:.1f}% ± {m['seed_std'] * 100:.1f}% "
            "accuracy, so a single run can be a little above or below this. Recall (catching sick patients) "
            "is the priority metric for screening.")

    a, b = st.columns(2)
    a.image(img("04_confusion_matrix.png"),
            caption=f"Confusion matrix: {m['fn']} sick patients missed, {m['fp']} false alarms")
    b.image(img("02_training_curves.png"), caption="Training vs validation curves (early stopping)")
    st.image(img("04_roc_pr_curves.png"), caption="ROC and precision-recall curves: ANN vs baseline models")
    a, b = st.columns(2)
    a.image(img("06_feature_importance.png"), caption="Which inputs the ANN relies on most")
    b.image(img("06_threshold_tradeoff.png"), caption="Effect of the decision threshold")
    st.image(img("01_class_balance_chest_pain.png"), caption="Dataset: class balance and chest pain type")
