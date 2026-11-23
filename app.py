# app.py
# Streamlit interface for the Breast Cancer Tumor Classification project.
# Educational demo - NOT a medical diagnostic tool.
#
# Single-page app: 30 feature inputs, live prediction and comparison
# graphics on one styled page (no tabs, no button - the prediction
# recomputes on every input change). Arabic/English language selector
# with full RTL layout for Arabic.
# Same model files and prediction logic as the original app.
#
# First run:   python train_model.py      (once, to create the model files)
# Then start:  streamlit run app.py   (or double-click run_app.bat)

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "stacking_model.joblib"
META_PATH = "model_meta.json"
DATA_PATH = "brca.csv"
BASE_MODELS_PATH = "base_models.joblib"
BASE_METRICS_PATH = "base_models_metrics.json"

# ------------------------------------------------------------ color palette
COLOR_PRIMARY = "#D6496E"    # rose pink (project accent)
COLOR_PRIMARY_DARK = "#A82D51"
COLOR_BENIGN = "#2E9E6B"     # green
COLOR_MALIGNANT = "#E5484D"  # red
COLOR_TEXT = "#33262E"
COLOR_SOFT = "#FFF3F7"       # soft pink tint

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon=":microscope:",
    layout="wide",
)

# ------------------------------------------------------------ translations
STR = {
    "en": {
        "lang_label": "Language",
        "hero_title": "Breast Cancer Tumor Classification",
        "hero_sub": "Stacking ensemble (Random Forest + Gradient Boosting + SVM, Logistic Regression meta-model)",
        "sec1_title": "1 · Enter the sample measurements",
        "sec1_sub": "The 30 features of a cell sample, as computed from a digitized FNA image. "
                    "Fields start with typical values from the training data, and the prediction "
                    "below updates automatically as you change any value.",
        "group_mean": "Mean features",
        "group_se": "Standard-error features",
        "group_worst": "Worst features",
        "sec2_title": "2 · Prediction",
        "sec2_sub": "Live model output for the values currently entered above.",
        "verdict_prefix": "Model output",
        "benign": "Benign",
        "malignant": "Malignant",
        "verdict_note": "Model output scores (predict_proba) for the current sample:",
        "sec3_title": "3 · What each model predicts",
        "sec3_sub": "The six individual models from the notebook and the final stacking model, "
                    "all evaluated on the sample entered above.",
        "final_badge": "Final model",
        "agreement_note": "Outputs for the current sample. Agreement between models on one sample is not a "
                          "measure of accuracy; the final stacking model was chosen by cross-validated tuning "
                          "(GridSearchCV on F1) over the training data, and assessed once on a held-out test set.",
        "metrics_expander": "Cross-validation results on the training data (as in the notebook)",
        "model_col": "Model",
        "sec4_title": "4 · How this sample compares",
        "sec4_sub": "Your entered values against the typical benign and malignant samples in the "
                    "dataset (mean features, each scaled to its largest value so they fit on one chart).",
        "y_axis": "Relative scale",
        "series_input": "Your input",
        "series_benign": "Typical benign",
        "series_malignant": "Typical malignant",
        "footer": "Stacking ensemble trained on the Wisconsin Diagnostic Breast Cancer dataset · "
                  "scaling happens inside the model pipelines, so inputs are the raw feature values.",
    },
    "ar": {
        "lang_label": "اللغة",
        "hero_title": "تصنيف أورام الثدي",
        "hero_sub": "نموذج تجميعي (Stacking): غابات عشوائية + تعزيز تدرّجي + SVM، ونموذج نهائي بالانحدار اللوجستي",
        "sec1_title": "١ · أدخل قياسات العينة",
        "sec1_sub": "الـ30 خاصية لعينة خلايا، كما تُحسب من صورة رقمية لشفط بإبرة رفيعة (FNA). "
                    "الحقول تبدأ بقيم نموذجية من بيانات التدريب، والتوقع بالأسفل يتحدّث تلقائيًا مع أي تغيير.",
        "group_mean": "خواص المتوسط",
        "group_se": "خواص الخطأ المعياري",
        "group_worst": "خواص القيم الأسوأ",
        "sec2_title": "٢ · التوقع",
        "sec2_sub": "ناتج الموديل مباشرة للقيم المدخلة حاليًا بالأعلى.",
        "verdict_prefix": "ناتج الموديل",
        "benign": "حميد",
        "malignant": "خبيث",
        "verdict_note": "درجات الموديل (predict_proba) للعينة الحالية:",
        "sec3_title": "٣ · توقع كل موديل",
        "sec3_sub": "الموديلات الستة الفردية من النوتبوك وموديل الـStacking النهائي، "
                    "كلها متقيّمة على العينة المدخلة فوق.",
        "final_badge": "الموديل النهائي",
        "agreement_note": "دي مخرجات الموديلات للعينة الحالية. اتفاق الموديلات على عينة واحدة مش مقياس للدقة؛ "
                          "موديل الـStacking النهائي اتختار بضبط مُقيّم بالتحقق المتقاطع (GridSearchCV على F1) "
                          "على بيانات التدريب، واتقيّم مرة واحدة على بيانات اختبار محجوزة.",
        "metrics_expander": "نتائج التحقق المتقاطع على بيانات التدريب (زي النوتبوك)",
        "model_col": "الموديل",
        "sec4_title": "٤ · مقارنة العينة بالبيانات",
        "sec4_sub": "القيم التي أدخلتها مقابل القيم النموذجية للحالات الحميدة والخبيثة في البيانات "
                    "(خواص المتوسط، كل خاصية مُعاد قياسها إلى أكبر قيمة لها لتظهر في رسم واحد).",
        "y_axis": "المقياس النسبي",
        "series_input": "إدخالك",
        "series_benign": "نموذجي حميد",
        "series_malignant": "نموذجي خبيث",
        "footer": "نموذج تجميعي مدرّب على بيانات Wisconsin Diagnostic Breast Cancer · "
                  "التقييس يحدث داخل خطوط الموديل، لذا المدخلات هي القيم الخام.",
    },
}

FEATURE_NAMES_EN = {
    "radius": "Radius",
    "texture": "Texture",
    "perimeter": "Perimeter",
    "area": "Area",
    "smoothness": "Smoothness",
    "compactness": "Compactness",
    "concavity": "Concavity",
    "concave_pts": "Concave points",
    "symmetry": "Symmetry",
    "fractal_dim": "Fractal dimension",
}

FEATURE_NAMES_AR = {
    "radius": "نصف القطر",
    "texture": "الملمس",
    "perimeter": "المحيط",
    "area": "المساحة",
    "smoothness": "النعومة",
    "compactness": "التراص",
    "concavity": "التقعر",
    "concave_pts": "النقاط المقعرة",
    "symmetry": "التماثل",
    "fractal_dim": "البعد الكسوري",
}

# ------------------------------------------------------------ styling
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, {COLOR_SOFT} 0%, #FFFFFF 320px);
    }}
    .block-container {{
        padding-top: 2.2rem;
        max-width: 1150px;
    }}
    .hero {{
        background: linear-gradient(120deg, {COLOR_PRIMARY} 0%, {COLOR_PRIMARY_DARK} 100%);
        border-radius: 18px;
        padding: 28px 32px;
        color: white;
        margin-bottom: 6px;
    }}
    .hero h1 {{
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }}
    .hero p {{
        color: #FFD9E4;
        margin: 6px 0 0 0;
        font-size: 1rem;
    }}
    .section-title {{
        color: {COLOR_PRIMARY_DARK};
        font-size: 1.25rem;
        font-weight: 700;
        margin: 26px 0 4px 0;
    }}
    .section-sub {{
        color: #7A6670;
        font-size: 0.92rem;
        margin-bottom: 12px;
    }}
    div[data-testid="stHorizontalBlock"] {{
        background: white;
        border: 1px solid #F2D3DF;
        border-radius: 16px;
        padding: 16px 20px 6px 20px;
        box-shadow: 0 2px 10px rgba(214, 73, 110, 0.06);
    }}
    .verdict {{
        border-radius: 16px;
        padding: 20px 26px;
        margin-bottom: 14px;
    }}
    .verdict-benign {{
        background: #E9F7F0;
        border: 2px solid {COLOR_BENIGN};
    }}
    .verdict-malignant {{
        background: #FDECEC;
        border: 2px solid {COLOR_MALIGNANT};
    }}
    .verdict-title {{
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
    }}
    .verdict-note {{
        font-size: 0.9rem;
        color: #6E5B65;
        margin-top: 4px;
    }}
    .score-chip {{
        display: inline-block;
        border-radius: 999px;
        padding: 6px 16px;
        font-weight: 700;
        font-size: 1rem;
        margin-right: 10px;
    }}
    .model-card {{
        background: white;
        border: 1px solid #F2D3DF;
        border-radius: 14px;
        padding: 14px 10px;
        text-align: center;
    }}
    .model-card-final {{
        border: 2px solid {COLOR_PRIMARY};
        background: #FFF7FA;
    }}
    .model-name {{
        font-size: 0.85rem;
        color: #7A6670;
        font-weight: 600;
    }}
    .model-verdict {{
        font-size: 1.15rem;
        font-weight: 800;
        margin: 4px 0 2px 0;
    }}
    .model-score {{
        font-size: 0.85rem;
        color: #6E5B65;
    }}
    .badge-final {{
        display: inline-block;
        background: {COLOR_PRIMARY};
        color: white;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 700;
        margin-top: 6px;
    }}
    .footer-note {{
        color: #9A8590;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 30px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------ artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return model, meta


@st.cache_resource
def load_base_models():
    """Standalone base models (RF/GB/SVM/KNN) with their test-set metrics, if present."""
    if not os.path.exists(BASE_MODELS_PATH):
        return None, None
    models = joblib.load(BASE_MODELS_PATH)
    metrics = None
    if os.path.exists(BASE_METRICS_PATH):
        with open(BASE_METRICS_PATH) as f:
            metrics = json.load(f)
    return models, metrics


@st.cache_data
def load_class_profiles():
    """Typical (median) feature values per diagnosis, for the comparison chart."""
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Unnamed: 0"])
    df["y"] = df["y"].map({"B": 0, "M": 1})
    return df.groupby("y").median(numeric_only=True)


if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
    st.error(
        "Model files not found. Run `python train_model.py` first in this folder, "
        "then restart the app with `streamlit run app.py`."
    )
    st.stop()

model, meta = load_artifacts()
FEATURES = meta["feature_names"]
DEFAULTS = meta["defaults"]

# ------------------------------------------------------------ language
lang_choice = st.radio(
    "Language · اللغة",
    ["العربية", "English"],
    horizontal=True,
    label_visibility="collapsed",
)
lang = "ar" if lang_choice == "العربية" else "en"
T = STR[lang]
FEATURE_NAMES = FEATURE_NAMES_AR if lang == "ar" else FEATURE_NAMES_EN

if lang == "ar":
    st.markdown(
        """
        <style>
        .block-container, .hero, .section-title, .section-sub, .verdict, .footer-note,
        div[data-testid="stMarkdownContainer"], div[data-testid="stRadio"] {
            direction: rtl;
            text-align: right;
        }
        div[data-testid="stNumberInput"], div[data-testid="stNumberInput"] input,
        div[data-testid="stVegaLiteChart"] {
            direction: ltr;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def feature_label(feat):
    base = feat.replace("x.", "")
    for suffix in ("_mean", "_se", "_worst"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return FEATURE_NAMES.get(base, base.replace("_", " ").title())


# ------------------------------------------------------------ header
st.markdown(
    f"""
    <div class="hero">
        <h1>{T["hero_title"]}</h1>
        <p>{T["hero_sub"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ inputs
st.markdown(f'<div class="section-title">{T["sec1_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-sub">{T["sec1_sub"]}</div>', unsafe_allow_html=True)

groups = [(T["group_mean"], "_mean"), (T["group_se"], "_se"), (T["group_worst"], "_worst")]

cols = st.columns(3)
values = {}
for (title, suffix), col in zip(groups, cols):
    with col:
        st.markdown(f"**{title}**")
        for feat in [f for f in FEATURES if f.endswith(suffix)]:
            values[feat] = st.number_input(
                feature_label(feat),
                value=float(DEFAULTS[feat]),
                format="%.4f",
                key=f"in_{feat}",
            )

# ------------------------------------------------------------ live prediction
# No form and no button: every input change reruns the script and the
# prediction below is recomputed from the values on screen, so the result
# can never be stale relative to the current inputs.
row = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
proba = model.predict_proba(row)[0]
pred = int(model.predict(row)[0])
p_benign, p_malignant = float(proba[0]), float(proba[1])

st.markdown(f'<div class="section-title">{T["sec2_title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-sub">{T["sec2_sub"]}</div>', unsafe_allow_html=True)

if pred == 1:
    card_class, color, word = "verdict-malignant", COLOR_MALIGNANT, T["malignant"]
else:
    card_class, color, word = "verdict-benign", COLOR_BENIGN, T["benign"]

st.markdown(
    f"""
    <div class="verdict {card_class}">
        <p class="verdict-title" style="color:{color};">{T["verdict_prefix"]}: {word}</p>
        <p class="verdict-note">{T["verdict_note"]}</p>
        <div style="margin-top:10px;">
            <span class="score-chip" style="background:{COLOR_BENIGN}1A;color:{COLOR_BENIGN};">
                {T["benign"]} · {p_benign * 100:.1f}%</span>
            <span class="score-chip" style="background:{COLOR_MALIGNANT}1A;color:{COLOR_MALIGNANT};">
                {T["malignant"]} · {p_malignant * 100:.1f}%</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.vega_lite_chart(
    {
        "data": {
            "values": [
                {"class": T["benign"], "p": p_benign},
                {"class": T["malignant"], "p": p_malignant},
            ]
        },
        "mark": {"type": "bar", "height": 38, "cornerRadiusEnd": 8},
        "encoding": {
            "y": {
                "field": "class",
                "type": "nominal",
                "title": None,
                "sort": [T["benign"], T["malignant"]],
                "axis": {"labelFontSize": 13},
            },
            "x": {
                "field": "p",
                "type": "quantitative",
                "title": None,
                "scale": {"domain": [0, 1]},
                "axis": {"format": ".0%"},
            },
            "color": {
                "field": "class",
                "type": "nominal",
                "scale": {
                    "domain": [T["benign"], T["malignant"]],
                    "range": [COLOR_BENIGN, COLOR_MALIGNANT],
                },
                "legend": None,
            },
            "tooltip": [
                {"field": "class", "type": "nominal"},
                {"field": "p", "type": "quantitative", "format": ".1%", "title": "Score"},
            ],
        },
        "height": 150,
    },
    use_container_width=True,
)

# ------------------------------------------------------------ per-model predictions
base_models, base_metrics = load_base_models()

if base_models is not None:
    st.markdown(f'<div class="section-title">{T["sec3_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{T["sec3_sub"]}</div>', unsafe_allow_html=True)

    model_outputs = []
    for name, bm in base_models.items():
        bp = bm.predict_proba(row)[0]
        model_outputs.append((name, int(bm.predict(row)[0]), float(bp[1])))

    def model_card(name, bpred, p_mal, is_final=False):
        word = T["malignant"] if bpred == 1 else T["benign"]
        color = COLOR_MALIGNANT if bpred == 1 else COLOR_BENIGN
        css = "model-card model-card-final" if is_final else "model-card"
        badge = f'<span class="badge-final">{T["final_badge"]}</span>' if is_final else ""
        st.markdown(
            f"""
            <div class="{css}">
                <div class="model-name">{name}</div>
                <div class="model-verdict" style="color:{color};">{word}</div>
                <div class="model-score">{T["malignant"]} {p_mal * 100:.1f}%</div>
                {badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # notebook order: the six individual models, then the final stacking model
    row1, row2 = model_outputs[:4], model_outputs[4:]
    for col, (name, bpred, p_mal) in zip(st.columns(4), row1):
        with col:
            model_card(name, bpred, p_mal)
    cols2 = st.columns(3)
    for col, (name, bpred, p_mal) in zip(cols2[:2], row2):
        with col:
            model_card(name, bpred, p_mal)
    with cols2[2]:
        model_card("Stacking", pred, p_malignant, is_final=True)

    st.caption(T["agreement_note"])

    if base_metrics is not None:
        with st.expander(T["metrics_expander"]):
            table = {T["model_col"]: [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}
            for name, mm in base_metrics.items():
                label = f'{name} ({T["final_badge"]})' if name == "Stacking" else name
                table[T["model_col"]].append(label)
                table["Accuracy"].append(mm["accuracy"])
                table["Precision"].append(mm["precision"])
                table["Recall"].append(mm["recall"])
                table["F1"].append(mm["f1"])
            st.table(pd.DataFrame(table))

# ------------------------------------------------------------ comparison chart
profiles = load_class_profiles()
mean_feats = [f for f in FEATURES if f.endswith("_mean")]

if profiles is not None:
    st.markdown(f'<div class="section-title">{T["sec4_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">{T["sec4_sub"]}</div>', unsafe_allow_html=True)

    mean_labels = [feature_label(f) for f in mean_feats]
    rows = []
    for feat in mean_feats:
        entered = float(values[feat])
        typical_b = float(profiles.loc[0, feat])
        typical_m = float(profiles.loc[1, feat])
        scale = max(entered, typical_b, typical_m) or 1.0
        label = feature_label(feat)
        rows.extend(
            [
                {"feature": label, "series": T["series_input"], "value": entered / scale},
                {"feature": label, "series": T["series_benign"], "value": typical_b / scale},
                {"feature": label, "series": T["series_malignant"], "value": typical_m / scale},
            ]
        )

    st.vega_lite_chart(
        {
            "data": {"values": rows},
            "mark": {"type": "bar", "cornerRadiusEnd": 3},
            "encoding": {
                "x": {
                    "field": "feature",
                    "type": "nominal",
                    "title": None,
                    "sort": mean_labels,
                    "axis": {"labelAngle": -30, "labelFontSize": 12},
                },
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "title": T["y_axis"],
                    "axis": {"format": ".0%"},
                },
                "xOffset": {"field": "series"},
                "color": {
                    "field": "series",
                    "type": "nominal",
                    "scale": {
                        "domain": [T["series_input"], T["series_benign"], T["series_malignant"]],
                        "range": [COLOR_PRIMARY, COLOR_BENIGN, COLOR_MALIGNANT],
                    },
                    "legend": {"title": None, "orient": "top"},
                },
                "tooltip": [
                    {"field": "feature", "type": "nominal"},
                    {"field": "series", "type": "nominal"},
                    {"field": "value", "type": "quantitative", "format": ".2f", "title": "Relative value"},
                ],
            },
            "height": 300,
        },
        use_container_width=True,
    )

st.markdown(
    f'<div class="footer-note">{T["footer"]}</div>',
    unsafe_allow_html=True,
)
