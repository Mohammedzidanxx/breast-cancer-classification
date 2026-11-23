# app.py
# Streamlit interface for the Breast Cancer Tumor Classification project.
# Educational demo - NOT a medical diagnostic tool.
#
# Single-page app: 30 feature inputs, live prediction, notebook insights and comparison
# graphics on one styled page (no tabs, no button - the prediction
# recomputes on every input change). Arabic/English language selector
# with full RTL layout for Arabic.
# Same model files and prediction logic as the original app.
#
# First run:   python train_model.py      (once, to create the model files)
# Then start:  streamlit run app.py   (or double-click run_app.bat)

import base64
import io
import json
from html.parser import HTMLParser
from pathlib import Path
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

# Render live scores directly in HTML so charts cannot retain an old Vega dataset.
probability_bars = []
for label, score, bar_color in [
    (T["benign"], p_benign, COLOR_BENIGN),
    (T["malignant"], p_malignant, COLOR_MALIGNANT),
]:
    probability_bars.append(
        f'<div style="display:flex;align-items:center;gap:12px;margin:14px 0;">'
        f'<span style="width:90px;font-size:13px;">{label}</span>'
        f'<div style="flex:1;background:#F2F2F5;border-radius:6px;overflow:hidden;">'
        f'<div style="width:{score * 100:.6f}%;height:36px;background:{bar_color};"></div></div>'
        f'<span style="width:62px;font-size:13px;">{score * 100:.1f}%</span></div>'
    )
st.markdown('<div class="live-probability-chart">' + ''.join(probability_bars) + '</div>', unsafe_allow_html=True)

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

    # Static SVG is rebuilt from the current values on every Streamlit rerun.
    chart_width, chart_height = 1000, 380
    plot_top, plot_bottom = 38, 285
    plot_height = plot_bottom - plot_top
    svg = [f'<svg class="live-profile-chart" viewBox="0 0 {chart_width} {chart_height}" '
           'style="width:100%;height:auto;direction:ltr;" role="img" aria-label="Current sample feature comparison">']
    palette = [COLOR_PRIMARY, COLOR_BENIGN, COLOR_MALIGNANT]
    legends = [T["series_input"], T["series_benign"], T["series_malignant"]]
    for j, (legend, fill) in enumerate(zip(legends, palette)):
        x = 100 + j * 290
        svg.append(f'<rect x="{x}" y="5" width="12" height="12" fill="{fill}"/>'
                   f'<text x="{x + 20}" y="16" font-size="12" direction="ltr" unicode-bidi="plaintext" text-anchor="start" fill="{COLOR_TEXT}">{legend}</text>')
    for tick in range(6):
        y = plot_bottom - tick * plot_height / 5
        svg.append(f'<line x1="48" y1="{y}" x2="990" y2="{y}" stroke="#E5E7EB"/>'
                   f'<text x="40" y="{y + 4}" text-anchor="end" font-size="11" fill="{COLOR_TEXT}">{tick * 20}%</text>')
    for i, label in enumerate(mean_labels):
        center = 94 + i * 94
        for j in range(3):
            item = rows[i * 3 + j]
            value = item["value"]
            height = value * plot_height
            x, y = center - 30 + j * 20, plot_bottom - height
            svg.append(f'<rect x="{x}" y="{y:.3f}" width="18" height="{height:.3f}" fill="{palette[j]}">'
                       f'<title>{label}: {legends[j]} {value * 100:.1f}%</title></rect>')
        svg.append(f'<text transform="translate({center - 25},305) rotate(30)" '
                   f'font-size="11" direction="ltr" unicode-bidi="plaintext" text-anchor="start" fill="{COLOR_TEXT}">{label}</text>')
    svg.append('</svg>')
    st.markdown(''.join(svg), unsafe_allow_html=True)

# ------------------------------------------------------------ notebook insights
# Original notebook outputs are read locally, never executed or refitted.
# No plotting packages or new assets are required. Keep Breast_Cancer.ipynb
# and brca.csv beside app.py, as they already are in the project repository.
NOTEBOOK_PATH = Path(__file__).resolve().parent / "Breast_Cancer.ipynb"


class NotebookTableParser(HTMLParser):
    """Read the saved pandas table output without an HTML-parser dependency."""
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


@st.cache_data
def load_notebook_insights():
    with NOTEBOOK_PATH.open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    plots, tables = {}, {}
    # These are zero-based cells in the project's original notebook.
    for index in (4, 6, 7, 9, 11, 46, 49, 51):
        for output in notebook["cells"][index].get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                plots[index] = base64.b64decode("".join(encoded) if isinstance(encoded, list) else encoded)
    for index in (21, 23, 26, 29, 32, 35, 39, 40, 41, 48):
        for output in notebook["cells"][index].get("outputs", []):
            html = output.get("data", {}).get("text/html")
            if html:
                parser = NotebookTableParser()
                parser.feed("".join(html) if isinstance(html, list) else html)
                if parser.rows and len(parser.rows) > 1:
                    header, *body = parser.rows
                    # First column is pandas' saved row index, not a feature.
                    frame = pd.DataFrame([r[1:] for r in body], columns=header[1:])
                    for column in frame.columns:
                        if column != "Model":
                            try:
                                frame[column] = pd.to_numeric(frame[column])
                            except (ValueError, TypeError):
                                pass
                    if index == 40:
                        frame.insert(0, "Statistic", [r[0] for r in body])
                    tables[index] = frame
    return plots, tables


@st.cache_data
def load_analysis_data():
    return pd.read_csv(Path(__file__).resolve().parent / DATA_PATH)


def render_notebook_insights():
    ar = lang == "ar"

    def tr(arabic, english):
        return arabic if ar else english

    st.markdown('<div class="section-title">' + tr(
        "٥ · تحليل البيانات ورؤى النوتبوك", "5 · Data analysis & notebook insights") + '</div>',
        unsafe_allow_html=True)
    st.caption(tr(
        "نفس رسومات ونتائج النوتبوك الأصلي، بجانب التوقعات. التحليل هنا للبيانات كاملة، وليس للعينة المدخلة فوق.",
        "The original notebook's plots and results, alongside predictions. This analysis describes the full dataset, not the sample entered above."))
    st.info(tr(
        "مشروع تعليمي فقط، وليس أداة للتشخيص الطبي. نسب التوقع ليست تأكيدًا طبيًا.",
        "Educational project only, not a medical diagnostic tool. Prediction scores are not medical confirmation."))
    if not NOTEBOOK_PATH.exists() or not Path(DATA_PATH).exists():
        st.warning(tr(
            "لعرض التحليل، ضع Breast_Cancer.ipynb وbrca.csv في نفس مجلد app.py.",
            "Keep Breast_Cancer.ipynb and brca.csv beside app.py to display the analysis."))
        return
    plots, tables = load_notebook_insights()
    raw = load_analysis_data()
    features = raw.drop(columns=["Unnamed: 0", "y"], errors="ignore")
    encoded = raw["y"].map({"B": 0, "M": 1})
    counts = raw["y"].value_counts()
    a, b, c, d = st.columns(4)
    a.metric(tr("العينات", "Samples"), len(raw))
    b.metric(tr("الخواص الرقمية", "Numeric features"), features.shape[1])
    c.metric(tr("القيم المفقودة", "Missing values"), int(raw.isna().sum().sum()))
    d.metric(tr("الصفوف المكررة", "Duplicate rows"), int(raw.duplicated().sum()))
    names = [tr("نظرة عامة", "Overview"), tr("توزيع الخواص", "Feature distributions"),
             tr("الارتباطات", "Correlations"), tr("القيم المتطرفة", "Outliers"),
             tr("تقييم الموديلات", "Model evaluation")]
    selected = st.radio(tr("استكشف التحليل", "Explore the analysis"), range(len(names)),
                        format_func=lambda i: names[i], horizontal=True, key="insights_view")

    def heading(arabic, english):
        st.subheader(tr(arabic, english))

    def plot(index, arabic, english):
        st.image(io.BytesIO(plots[index]), caption=tr(arabic, english), width="stretch")

    def table(frame):
        st.dataframe(frame, hide_index=True, width="stretch")

    if selected == 0:
        heading("توزيع التشخيص في البيانات", "Diagnosis distribution")
        table(pd.DataFrame({tr("التشخيص", "Diagnosis"): [T["benign"], T["malignant"]],
                            tr("العدد", "Count"): [int(counts.get("B", 0)), int(counts.get("M", 0))],
                            tr("النسبة %", "Percentage %"): [round(counts.get("B", 0) / len(raw) * 100, 2),
                                                              round(counts.get("M", 0) / len(raw) * 100, 2)]}))
        plot(4, "الرسم الأصلي: B = حميد، M = خبيث", "Original plot: B = benign, M = malignant")
        st.write(tr(
            "البيانات فيها حالات حميدة أكثر من الخبيثة. لا توجد قيم مفقودة أو صفوف مكررة في ملف المشروع. عمود Unnamed: 0 هو فهرس، وليس من خواص الموديل.",
            "Benign cases outnumber malignant cases. The project file has no missing values or duplicate rows. Unnamed: 0 is an index, not a model feature."))
        with st.expander(tr("عينة من البيانات والإحصاء الوصفي", "Data preview & descriptive statistics")):
            table(raw.drop(columns=["Unnamed: 0"], errors="ignore").head(10))
            table(features.describe().rename_axis("Statistic").reset_index())
            table(pd.DataFrame({tr("العمود", "Column"): raw.columns,
                                tr("قيم مفقودة", "Missing"): raw.isna().sum().values,
                                tr("نوع البيانات", "Data type"): raw.dtypes.astype(str).values}))
    elif selected == 1:
        heading("توزيع الـ30 خاصية رقمية", "Distributions of all 30 numeric features")
        st.write(tr(
            "نفس إعداد النوتبوك: 30 فئة لكل هيستوجرام. المقاييس مختلفة، وبعض التوزيعات لها ذيل طويل؛ لا تقارن ارتفاع أو نطاق خاصيتين كأن وحداتهما واحدة.",
            "Same notebook setting: 30 bins per histogram. Feature scales differ and some distributions have long tails; do not treat different feature ranges as the same units."))
        plot(6, "توزيعات الخواص الأصلية، افتح الصورة بملء الشاشة لقراءة التفاصيل", "Original feature histograms. Open the image fullscreen to inspect details.")
    elif selected == 2:
        heading("الارتباط بين الخواص", "Feature correlations")
        plot(7, "مصفوفة ارتباط Pearson لجميع الخواص", "Pearson correlation matrix for all features")
        corr = features.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = upper.stack().reset_index()
        pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
        pairs = pairs[pairs["Correlation"].abs() > 0.90].sort_values("Correlation", key=abs, ascending=False)
        st.write(tr("أزواج الخواص ذات |الارتباط| أكبر من 0.90، بدون تكرار:", "Feature pairs with |correlation| above 0.90, without duplicate pairs:"))
        table(pairs.round(3))
        heading("ارتباط الخواص بالتشخيص", "Feature correlation with diagnosis")
        plot(9, "ترميز التشخيص للتحليل فقط: B=0 وM=1", "Diagnosis encoding for analysis only: B=0 and M=1")
        target_corr = features.corrwith(encoded).sort_values(ascending=False, key=abs)
        table(target_corr.rename("Correlation").rename_axis("Feature").reset_index().round(4))
        st.caption(tr(
            "الارتباط لا يثبت السببية، وليس أهمية الخاصية داخل الموديل. هذه علاقات وصفية على كامل البيانات، وليست نتائج اختبار على بيانات جديدة.",
            "Correlation is not causation or model feature importance. These are descriptive relationships across the full dataset, not held-out predictive results."))
    elif selected == 3:
        heading("فحص القيم المتطرفة", "Outlier inspection")
        plot(11, "Boxplots الأصلية لجميع الخواص", "Original boxplots for all features")
        q1, q3 = features.quantile(0.25), features.quantile(0.75)
        iqr = q3 - q1
        outliers = ((features < q1 - 1.5 * iqr) | (features > q3 + 1.5 * iqr)).sum()
        result = pd.DataFrame({"Feature": features.columns, "Outliers Count": outliers.values,
                               "Outliers Percentage": (outliers / len(features) * 100).round(2).values})
        table(result.sort_values("Outliers Count", ascending=False))
        st.caption(tr(
            "قاعدة النوتبوك: خارج Q1 − 1.5×IQR أو Q3 + 1.5×IQR. ده رصد وصفي، مش دليل إن القياس غلط. لم نحذف أي صف أو نغيّر الموديل بسبب هذا العرض.",
            "Notebook rule: outside Q1 − 1.5×IQR or Q3 + 1.5×IQR. A flag does not prove a measurement is wrong. This display removes no rows and changes no model."))
    else:
        heading("مقارنة الموديلات في النوتبوك", "Notebook model comparison")
        st.caption(tr(
            "النتائج المحفوظة من النوتبوك، وليست إعادة تدريب داخل التطبيق. المقارنة هنا متوسط تحقق متقاطع من 5 طيات على 455 عينة تدريب، بترميز الخبيث = 1.",
            "Saved notebook results, not app-time retraining. This comparison uses mean 5-fold cross-validation on 455 training samples, with malignant encoded as 1."))
        plot(49, "مقارنة F1 للموديلات الـ12", "F1 comparison across the 12 models")
        table(tables[48])
        st.write(tr(
            "Stacking B سجّل F1 = 96.75%، مقابل 96.71% للانحدار اللوجستي. الفارق صغير جدًا، وليس دليلًا على تفوق مهم إحصائيًا.",
            "Stacking B recorded F1 = 96.75%, compared with 96.71% for Logistic Regression. The gap is tiny, not evidence of a statistically meaningful advantage."))
        heading("اختبار الموديل النهائي على بيانات محجوزة", "Final model on the held-out test set")
        st.caption(tr(
            "114 عينة اختبار منفصلة. الأرقام التالية من النوتبوك الأصلي، وليست درجات التوقع للعينة الحالية.",
            "114 separate test samples. The following numbers are from the original notebook, not prediction scores for the current input."))
        for col, label, value in zip(st.columns(4), ["Accuracy", "Precision", "Recall", "F1"],
                                     ["96.49%", "100.00%", "90.48%", "95.00%"]):
            col.metric(label, value)
        plot(46, "مصفوفة الالتباس للموديل النهائي، صفوف = الحقيقي وأعمدة = المتوقع", "Final model confusion matrix. Rows = actual, columns = predicted.")
        st.warning(tr(
            "في الاختبار: 72 حالة حميدة صحيحة، و38 خبيثة صحيحة، و4 حالات خبيثة صُنّفت حميدة، و0 إنذارات خبيثة خاطئة. وجود الحالات الأربع الفائتة يؤكد أن المشروع ليس للتشخيص الطبي.",
            "Test results: 72 true benign, 38 true malignant, 4 malignant cases classified as benign, and 0 false malignant alerts. Those four missed cases reinforce that this project is not for medical diagnosis."))
        with st.expander(tr("مصفوفات الالتباس لكل الموديلات", "Confusion matrices for all models")):
            st.caption(tr(
                "توقعات خارج الطية (OOF) على بيانات التدريب. ليست نفس مجموعة الاختبار في المصفوفة السابقة.",
                "Out-of-fold predictions on the training set. These do not use the test set shown above."))
            plot(51, "مصفوفات الموديلات الـ12 من النوتبوك", "The notebook's 12 model confusion matrices")
        with st.expander(tr("تجارب البذور ووقت التدريب", "Random-seed experiments & training time")):
            table(tables[39])
            table(tables[40])
            table(tables[41])
            st.caption(tr(
                "أزمنة التدريب من تشغيل النوتبوك على جهاز واحد، وليست قياسًا لسرعة التطبيق. تباين البذور من إعداد التجربة الأصلي.",
                "Training times are from one notebook run on one machine, not an app speed benchmark. Seed variation follows the original experiment setup."))
    st.caption(tr(
        "المصدر: Breast_Cancer.ipynb وbrca.csv من نفس المشروع. الرسومات الأصلية تحتفظ بعناوين محاورها الإنجليزية. عرض التحليل لا يعيد تدريب الموديل ولا يغيّر التوقعات.",
        "Source: Breast_Cancer.ipynb and brca.csv from this project. Original plots retain their English axis labels. Viewing analysis neither retrains the model nor changes predictions."))


try:
    render_notebook_insights()
except (OSError, ValueError, KeyError, IndexError) as error:
    st.warning("تعذّر تحميل تحليل النوتبوك. التوقعات بالأعلى مستقلة عنه. / Notebook insights could not load; predictions above are independent.")
    st.caption(type(error).__name__)



st.markdown(
    f'<div class="footer-note">{T["footer"]}</div>',
    unsafe_allow_html=True,
)
