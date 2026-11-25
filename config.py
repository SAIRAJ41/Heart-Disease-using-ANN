# config.py - shared settings: paths, colours, input options and page styling
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "heart-disease.csv")          # 918 patients, 11 features
MODEL = os.path.join(BASE, "model", "heart_disease_model.keras")
PRE = os.path.join(BASE, "model", "preprocessor.pkl")
METRICS_FILE = os.path.join(BASE, "model", "metrics.json")       # test results written by model_utils.train()
ASSETS = os.path.join(BASE, "assets")
SEED = 42

C0, C1 = "#2a78d6", "#eb6834"            # blue = no disease, orange = heart disease

# Input choices (must match the categories the preprocessor was trained on)
CHEST_PAIN = ["Asymptomatic", "Atypical angina", "Non-anginal pain", "Typical angina"]
ST_SLOPE = ["Upsloping", "Flat", "Downsloping"]
RESTING_ECG = ["Normal", "ST-T abnormality", "LVH"]

CSS = f"""
<style>
.block-container {{padding-top: 2rem; max-width: 1150px;}}
.hero {{background: linear-gradient(120deg, {C0} 0%, #1b4f8f 100%); color: white;
        padding: 1.6rem 2rem; border-radius: 16px; margin-bottom: 1.2rem;}}
.hero h1 {{color: white; margin: 0; font-size: 2rem;}}
.hero p {{margin: .35rem 0 0; opacity: .9;}}
.card {{background: white; border-radius: 14px; padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 3px rgba(20,40,70,.08); border: 1px solid #e6ebf2;}}
.risk-num {{font-size: 3rem; font-weight: 800; line-height: 1;}}
.bar {{height: 14px; border-radius: 7px; background: #e6ebf2; margin: .8rem 0 .3rem;}}
.bar > div {{height: 14px; border-radius: 7px;}}
.tag {{display: inline-block; padding: .2rem .6rem; border-radius: 999px; font-size: .85rem;
       margin: .15rem .25rem .15rem 0; background: #fdebe2; color: #a8431a;}}
.tag.ok {{background: #e3eefb; color: #1b4f8f;}}
.metric {{text-align: center;}}
.metric b {{font-size: 1.7rem; display: block;}}
.metric span {{color: #5b6b7f; font-size: .9rem;}}
</style>
"""
