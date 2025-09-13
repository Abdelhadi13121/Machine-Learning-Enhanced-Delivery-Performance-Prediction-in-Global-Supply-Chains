# app.py
import io, os, joblib, shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve
)
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ---------- Global visuals & font ----------
st.set_page_config(page_title="ESI2025 — Late Delivery Risk ML (Pro)",
                   layout="wide", page_icon="🚚")
plt.rcParams["figure.dpi"] = 600
plt.rcParams['font.family'] = 'Palatino Linotype'
sns.set_style("whitegrid")

# Inject CSS
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Palatino Linotype', 'serif' !important;
    }
    h1, h2, h3, h4, p, div, span, label {
        font-family: 'Palatino Linotype', 'serif' !important;
    }
    .title-red { color: #b30000; font-weight: 700; font-style: italic; text-align: center; margin-bottom: 6px; }
    .article-title { text-align: center; font-weight: 600; margin-bottom: 8px; }
    .authors { text-align: center; line-height: 1.4; margin-bottom: 12px; }
    .logo-note { text-align:center; font-size:12px; color:#666; margin-bottom:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

SIGNATURE = "Developed by Dr. BENGHALEM Abdelhadi — ESI2025"

def add_signature(fig):
    fig.text(0.99, 0.01, SIGNATURE, fontsize=5, ha="right", alpha=0.6, style="italic")

# ---------- SIDEBAR: conference info & abstract ----------
st.sidebar.markdown("### 📄 Abstract")
st.sidebar.info(
    "This paper develops a machine-learning pipeline to predict late deliveries in global supply chains. "
    "We integrate multi-modal features (temporal, categorical, numerical) and compare models (Logistic Regression, "
    "Random Forest, XGBoost). Results highlight the potential to flag high-risk shipments, enabling proactive "
    "logistics interventions."
)


# ---------- LOGOS (Centered & on one line, bigger, no description) ----------
import base64

logo_paths = [
    "images/qatar_university.png",
    "images/oran_logo.jpeg",
    "images/tlemcen_logo.png"
]

# Build HTML dynamically for all logos
logo_html = "<div style='display: flex; justify-content: center; align-items: center; gap: 50px; margin-bottom: 20px;'>"
for p in logo_paths:
    if os.path.exists(p):
        with open(p, "rb") as f:
            img_bytes = f.read()
            img_b64 = base64.b64encode(img_bytes).decode()
            logo_html += f"<img src='data:image/png;base64,{img_b64}' style='height:120px;'>"
logo_html += "</div>"

st.markdown(logo_html, unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("<div class='title-red'><h1>8th International Conference on Entrepreneurship for Sustainability & Impact (ESI2025)</h1></div>", unsafe_allow_html=True)

# Updated article title: italic + blue
st.markdown(
    "<div class='article-title'><h2 style='color:#0066cc; font-style:italic;'>"
    "Machine Learning-Enhanced Delivery Performance Prediction in Global Supply Chains:<br>"
    "A Comprehensive Analytics Framework Using Multi-Modal Data Integration for Operational Excellence"
    "</h2></div>",
    unsafe_allow_html=True
)

# ---------- AUTHORS ----------
st.markdown(
    """
    <div class='authors'>
    <strong>Authors</strong><br><br>
    <strong>Dr. Abdelhadi BENGHALEM</strong> — Oran Graduate School of Economics, ERF Research Associate<br>
    <a href="mailto:abdelhadi.benghalem@ese-oran.dz" style="color:#0066cc;">abdelhadi.benghalem@ese-oran.dz</a><br><br>

    <strong>Prof. Mohammed BENBOUZIANE</strong> — Abou-Bekr Belkaid University of Tlemcen, MIFMA Lab<br>
    <a href="mailto:mohamed.benbouziane@univ-tlemcen.dz" style="color:#0066cc;">mohamed.benbouziane@univ-tlemcen.dz</a><br><br>

    <strong>Dr. Abdelbasset BENMAMMAR</strong> — Abou-Bekr Belkaid University of Tlemcen<br>
    <a href="mailto:abdelbasset.benmammar@univ-tlemcen.dz" style="color:#0066cc;">abdelbasset.benmammar@univ-tlemcen.dz</a>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")


# ---------- rest of app (keeps your working logic) ----------
def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for c in df.columns:
        nc = c.strip().lower()
        nc = nc.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        new_cols.append(nc)
    df.columns = new_cols
    return df

def find_target_col(df):
    candidates = ["late_delivery_risk", "latedeliveryrisk", "late_delivery", "late_delivery_risk"]
    for cand in candidates:
        if cand in df.columns:
            return cand
    for c in df.columns:
        if "late" in c and "deliver" in c:
            return c
    return None

def engineer_time_features(df):
    df = df.copy()
    order_col = None
    ship_col = None
    for c in df.columns:
        if "order" in c and "date" in c:
            order_col = c
        if ("ship" in c or "shipping" in c) and "date" in c:
            ship_col = c
    if order_col and ship_col:
        try:
            df[order_col] = pd.to_datetime(df[order_col], errors="coerce")
            df[ship_col] = pd.to_datetime(df[ship_col], errors="coerce")
            delta_hours = (df[ship_col] - df[order_col]).dt.total_seconds() / 3600.0
            df["order_to_shipment_hours"] = delta_hours.fillna(0).astype(int)
        except Exception:
            df["order_to_shipment_hours"] = 0
    else:
        df["order_to_shipment_hours"] = 0
    return df

def fit_preprocessors(X_train: pd.DataFrame, categorical_limit_unique=40):
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_low_card = [c for c in cat_cols if X_train[c].nunique() <= categorical_limit_unique]
    cat_high_card = [c for c in cat_cols if X_train[c].nunique() > categorical_limit_unique]

    imputer_num = SimpleImputer(strategy="median") if num_cols else None
    imputer_cat = SimpleImputer(strategy="most_frequent") if cat_low_card else None

    if imputer_num:
        imputer_num.fit(X_train[num_cols])
    if imputer_cat:
        imputer_cat.fit(X_train[cat_low_card])

    ord_enc = None
    if cat_low_card:
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        ord_enc.fit(imputer_cat.transform(X_train[cat_low_card]) if imputer_cat else X_train[cat_low_card])

    scaler = None
    if num_cols:
        scaler = StandardScaler()
        X_num_imputed = imputer_num.transform(X_train[num_cols]) if imputer_num else X_train[num_cols].values
        scaler.fit(X_num_imputed)

    return {
        "num_cols": num_cols,
        "cat_low_card": cat_low_card,
        "cat_high_card": cat_high_card,
        "imputer_num": imputer_num,
        "imputer_cat": imputer_cat,
        "ord_enc": ord_enc,
        "scaler": scaler
    }

def transform_with_preprocessors(X: pd.DataFrame, preprocessors: dict):
    X = X.copy()
    num_cols = preprocessors["num_cols"]
    cat_low_card = preprocessors["cat_low_card"]
    cat_high_card = preprocessors["cat_high_card"]
    imputer_num = preprocessors["imputer_num"]
    imputer_cat = preprocessors["imputer_cat"]
    ord_enc = preprocessors["ord_enc"]
    scaler = preprocessors["scaler"]

    if num_cols and imputer_num:
        X_num = pd.DataFrame(imputer_num.transform(X[num_cols]), columns=num_cols, index=X.index)
    elif num_cols:
        X_num = X[num_cols].fillna(0)
    else:
        X_num = pd.DataFrame(index=X.index)

    if cat_low_card and imputer_cat and ord_enc:
        X_cat_imputed = pd.DataFrame(imputer_cat.transform(X[cat_low_card]), columns=cat_low_card, index=X.index)
        X_cat_encoded = pd.DataFrame(ord_enc.transform(X_cat_imputed), columns=cat_low_card, index=X.index)
    elif cat_low_card:
        X_cat_encoded = X[cat_low_card].fillna("missing")
    else:
        X_cat_encoded = pd.DataFrame(index=X.index)

    if cat_high_card:
        X = X.drop(columns=[c for c in cat_high_card if c in X.columns], errors="ignore")

    X_proc = pd.concat([X_num, X_cat_encoded], axis=1)
    if scaler and not X_proc.empty:
        X_proc[num_cols] = scaler.transform(X_proc[num_cols])
    return X_proc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {}

    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["Random Forest"] = rf

    if XGB_AVAILABLE:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        models["XGBoost"] = xgb

    results = {}
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        results[name] = {
            "model": model,
            "train_pred": y_train_pred,
            "test_pred": y_test_pred,
            "train_metrics": {
                "accuracy": accuracy_score(y_train, y_train_pred),
                "precision": precision_score(y_train, y_train_pred, average="binary", zero_division=0),
                "recall": recall_score(y_train, y_train_pred, average="binary", zero_division=0),
                "f1": f1_score(y_train, y_train_pred, average="binary", zero_division=0)
            },
            "test_metrics": {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred, average="binary", zero_division=0),
                "recall": recall_score(y_test, y_test_pred, average="binary", zero_division=0),
                "f1": f1_score(y_test, y_test_pred, average="binary", zero_division=0)
            }
        }
    return results

def plot_confusion_side_by_side(y_train, y_train_pred, y_test, y_test_pred, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title(f"{title_prefix} - Train")
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[1], cmap="Oranges", colorbar=False)
    axes[1].set_title(f"{title_prefix} - Test")
    plt.suptitle(f"{title_prefix} — Confusion Matrices", fontsize=12, fontweight="bold")
    add_signature(fig)
    st.pyplot(fig)

def plot_feature_importances(model, feature_names, top_k=20, title="Feature Importance"):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(top_k)
        fig, ax = plt.subplots(figsize=(6, max(3, top_k/4)))
        imp.plot(kind="barh", ax=ax)
        ax.set_title(title)
        add_signature(fig)
        st.pyplot(fig)
    elif hasattr(model, "coef_"):
        coefs = model.coef_.ravel()
        imp = pd.Series(coefs, index=feature_names).abs().sort_values(ascending=True).tail(top_k)
        fig, ax = plt.subplots(figsize=(6, max(3, top_k/4)))
        imp.plot(kind="barh", ax=ax)
        ax.set_title(title + " (abs coefficients)")
        add_signature(fig)
        st.pyplot(fig)
    else:
        st.info("Model does not expose feature importances / coefficients.")

def plot_roc_pr(model, X_test, y_test, model_name):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        probs = model.decision_function(X_test)
    else:
        st.info(f"No probability scores for {model_name}")
        return

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_title(f"{model_name} ROC")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].legend()

    axes[1].plot(rec, prec, label=f"AUPR={pr_auc:.3f}")
    axes[1].set_title(f"{model_name} Precision-Recall")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    add_signature(fig)
    st.pyplot(fig)

# ---------- App flow ----------
st.markdown("Upload the DataCo CSV below (the app will try to detect the target `late_delivery_risk`).")

uploaded = st.file_uploader("Upload CSV (DataCo dataset)", type=["csv"])
sample_button = st.button("Use built-in sample (first 50k rows) - if local file present")

if uploaded is None and not sample_button:
    st.info("Upload a CSV or press 'Use built-in sample'.")
    st.stop()

if uploaded:
    try:
        raw = pd.read_csv(uploaded, encoding="latin1")
    except Exception:
        raw = pd.read_csv(uploaded, encoding="ISO-8859-1")
else:
    # try local file
    local_path = "DataCoSupplyChainDataset.csv"
    if os.path.exists(local_path):
        try:
            raw = pd.read_csv(local_path, encoding="latin1")
        except Exception:
            raw = pd.read_csv(local_path, encoding="ISO-8859-1")
    else:
        st.error("No sample file found locally. Please upload your CSV.")
        st.stop()

raw = normalize_column_names(raw)
raw = engineer_time_features(raw)

target_col = find_target_col(raw)
if target_col is None:
    st.error("❌ Column 'late_delivery_risk' not found. Available columns:\n\n" + ", ".join(raw.columns))
    st.stop()

st.write(f"Using target column: **{target_col}**")
raw.rename(columns={target_col: "late_delivery_risk"}, inplace=True)

with st.expander("Dataset preview & info"):
    st.dataframe(raw.head(10))
    st.write("Shape:", raw.shape)
    st.write("Columns:", list(raw.columns))

counts = raw["late_delivery_risk"].value_counts()
fig, ax = plt.subplots(figsize=(4, 3))
ax.pie(counts.values, labels=[f"{i} ({n})" for i, n in zip(counts.index, counts.values)],
       autopct="%.1f%%", startangle=90, colors=["#4daf4a", "#e41a1c"])
ax.set_title("Late Delivery Distribution (counts)")
add_signature(fig)
st.pyplot(fig)

test_size = st.sidebar.slider("Test set proportion", 0.05, 0.4, 0.2, 0.05)
fast = st.sidebar.checkbox("Fast mode (sample 50k rows if large)", value=True)
if fast and len(raw) > 50000:
    raw = raw.sample(50000, random_state=42).reset_index(drop=True)

drop_extra = [
    "customer_email", "customer_password", "customer_fname", "customer_lname",
    "product_image", "product_description"
]
raw = raw.drop(columns=[c for c in drop_extra if c in raw.columns], errors="ignore")

X = raw.drop(columns=["late_delivery_risk"])
y = raw["late_delivery_risk"].astype(int)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

st.sidebar.markdown("### Preprocessing options")
categorical_limit = st.sidebar.slider("Categorical card limit (keep as categorical if unique <=)", 5, 200, 40, 5)

preprocessors = fit_preprocessors(X_train_raw, categorical_limit_unique=categorical_limit)
X_train = transform_with_preprocessors(X_train_raw, preprocessors)
X_test = transform_with_preprocessors(X_test_raw, preprocessors)

for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
for col in X_test.columns:
    if col not in X_train.columns:
        X_train[col] = 0
X_train = X_train[X_train.columns.sort_values()]
X_test = X_test[X_train.columns]

st.sidebar.markdown("### Training options")
train_button = st.sidebar.button("Train Models")

if not train_button:
    st.info("Click 'Train Models' in the sidebar to train and evaluate models on your uploaded dataset.")
    st.stop()

with st.spinner("Training models..."):
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
st.success("Training complete ✅")

summary_rows = []
for name, res in results.items():
    tm = res["test_metrics"]
    summary_rows.append({
        "Model": name,
        "Accuracy": tm["accuracy"],
        "Precision": tm["precision"],
        "Recall": tm["recall"],
        "F1": tm["f1"]
    })
summary_df = pd.DataFrame(summary_rows).set_index("Model")
st.subheader("Model comparison (test set)")
st.dataframe(summary_df.style.format("{:.3f}"))

fig, ax = plt.subplots(figsize=(6, 3))
summary_df["Accuracy"].plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlim(0, 1)
ax.set_xlabel("Accuracy (test)")
add_signature(fig)
st.pyplot(fig)

for name, res in results.items():
    st.markdown(f"---\n### {name}")
    model = res["model"]
    plot_confusion_side_by_side(y_train, res["train_pred"], y_test, res["test_pred"], title_prefix=name)
    plot_roc_pr(model, X_test, y_test, name)
    st.write("Feature importance / coefficients (top 20):")
    plot_feature_importances(model, X_train.columns.tolist(), top_k=20, title=f"{name} feature importance")

best_name = max(results.keys(), key=lambda k: results[k]["test_metrics"]["f1"])
st.subheader(f"Permutation importance (test) — best model: {best_name}")
perm = permutation_importance(results[best_name]["model"], X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
perm_imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(6, max(3, len(perm_imp)/4)))
perm_imp.plot(kind="barh", ax=ax)
ax.set_xlabel("Permutation importance (mean decrease in score)")
add_signature(fig)
st.pyplot(fig)

st.subheader("Save & Export")
save_model_name = st.text_input("Filename prefix for artifacts", value="late_delivery_model")
if st.button("Export best model and preprocessors"):
    artifact = {
        "model": results[best_name]["model"],
        "preprocessors": preprocessors,
        "feature_columns": X_train.columns.tolist()
    }
    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    buf.seek(0)
    st.download_button("Download model artifact (.pkl)", buf, file_name=f"{save_model_name}_{best_name}.pkl", mime="application/octet-stream")

import streamlit as st
import pandas as pd
import numpy as np

st.subheader("Single-row prediction (manual input)")

# Get only the variables used in the trained model
model_features = X_train.columns.tolist()

# Use first row of X_train_raw as default values (if available)
defaults = X_train_raw.iloc[0].to_dict() if not X_train_raw.empty else {}

# --- Create form ---
with st.form("single_form"):
    cols = st.columns(3)
    inputs = {}

    for i, col in enumerate(model_features):
        val = defaults.get(col, "")
        if np.issubdtype(type(val), np.number):
            inputs[col] = cols[i % 3].number_input(col, value=float(val) if val != "" else 0.0)
        else:
            inputs[col] = cols[i % 3].text_input(col, value=str(val))

    submitted = st.form_submit_button("Predict single row")

# --- Handle prediction & store in session_state ---
if submitted:
    new_df = pd.DataFrame([inputs])
    new_df = normalize_column_names(new_df)
    new_df = engineer_time_features(new_df)

    X_like = transform_with_preprocessors(new_df, preprocessors)

    # Align columns with training data
    for col in model_features:
        if col not in X_like.columns:
            X_like[col] = 0
    X_like = X_like[model_features]

    # Predict
    model = results[best_name]["model"]
    pred = model.predict(X_like)[0]
    proba = model.predict_proba(X_like)[:, 1][0] if hasattr(model, "predict_proba") else None

    # Store result in session_state to persist after rerun
    st.session_state["last_prediction"] = {
        "label": "Late Delivery" if pred == 1 else "On Time",
        "proba": proba,
        "model": best_name
    }

# --- Display last prediction if available ---
if "last_prediction" in st.session_state:
    last_pred = st.session_state["last_prediction"]
    if last_pred["proba"] is not None:
        st.success(f"Prediction ({last_pred['model']}): **{last_pred['label']}** — Probability: {last_pred['proba']:.3f}")
    else:
        st.success(f"Prediction: **{last_pred['label']}**")


st.subheader("Batch inference (upload CSV of new orders)")
batch_file = st.file_uploader("Upload new orders CSV for batch prediction (optional)", type=["csv"], key="batch")
if batch_file is not None:
    try:
        batch_df = pd.read_csv(batch_file, encoding="latin1")
    except Exception:
        batch_df = pd.read_csv(batch_file, encoding="ISO-8859-1")
    batch_df = normalize_column_names(batch_df)
    batch_df = engineer_time_features(batch_df)
    X_batch = transform_with_preprocessors(batch_df, preprocessors)
    for col in X_train.columns:
        if col not in X_batch.columns:
            X_batch[col] = 0
    X_batch = X_batch[X_train.columns]
    preds = results[best_name]["model"].predict(X_batch).astype(int)
    probas = results[best_name]["model"].predict_proba(X_batch)[:, 1] if hasattr(results[best_name]["model"], "predict_proba") else [None]*len(preds)
    out = batch_df.copy()
    out["predicted_late_delivery"] = preds
    out["predicted_probability"] = np.round(probas, 4)
    st.write("Sample predictions:")
    st.dataframe(out.head(), use_container_width=True)
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.header("How companies can deploy & benefit from this model")
st.markdown("""
**Short guidance (operational):**
- Use the exported model artifact in a microservice. Flag high-risk shipments for proactive intervention (e.g. priority handling, customer notification, rerouting).
- Retrain regularly (weekly/monthly) depending on volume and seasonality.
- Monitor precision/recall and false negative rate; set alerts for data drift.

**Technical checklist for production:**
1. Save artifact and serve via Flask/FastAPI or serverless function.
2. Provide input validation and missing-value handling consistent with training.
3. Log inputs, predictions and outcomes for monitoring & retraining.
4. Implement A/B tests for recommended interventions before full rollout.
""")


