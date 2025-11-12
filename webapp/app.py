from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template, request
import joblib
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
DATA_PATH = BASE_DIR.parent / "data_set.csv"
GRAPHS_DIR = BASE_DIR / "static" / "graphs"

GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError as exc:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}") from exc

FEATURE_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_COLUMNS = {
    "customerID": int,
    "SeniorCitizen": int,
    "tenure": int,
    "MonthlyCharges": float,
    "TotalCharges": float,
}

CATEGORICAL_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}

LABEL_ENCODERS = {
    column: LabelEncoder().fit(values) for column, values in CATEGORICAL_VALUES.items()
}

OPTIONAL_COLUMNS = {"customerID"}

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]


sns.set_theme(style="whitegrid")


def _humanize_feature(name: str) -> str:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    spaced = spaced.replace("_", " ")
    return spaced.strip().title()


def _load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(str)
    for column in NUMERIC_FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _palette() -> dict[str, str]:
    return {"Yes": "#e63946", "No": "#14a44d"}


def _generate_categorical_chart(df: pd.DataFrame, feature: str) -> tuple[str, str] | None:
    plot_df = df[[feature, "Churn"]].copy()
    if plot_df.empty:
        return None

    plot_df[feature] = plot_df[feature].fillna("Unknown").astype(str)
    plot_df["Churn"] = plot_df["Churn"].fillna("Unknown")
    plot_df = plot_df[plot_df["Churn"].isin(["Yes", "No"])]

    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=plot_df,
        x=feature,
        hue="Churn",
        palette=_palette(),
        ax=ax,
        order=sorted(plot_df[feature].unique(), key=lambda x: str(x)),
        hue_order=["No", "Yes"],
    )
    ax.set_title(f"{feature} vs Churn", fontsize=14, fontweight="semibold")
    ax.set_xlabel(feature)
    ax.set_ylabel("Customer count")
    ax.legend(title="Churn")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    filename = f"cat_{feature.lower()}_churn.png".replace(" ", "_")
    output_path = GRAPHS_DIR / filename
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    insight = _categorical_insight(plot_df, feature)
    return filename, insight


def _generate_numeric_chart(df: pd.DataFrame, feature: str) -> tuple[str, str] | None:
    plot_df = df[[feature, "Churn"]].dropna().copy()
    plot_df = plot_df[plot_df["Churn"].isin(["Yes", "No"])]
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=plot_df,
        x="Churn",
        y=feature,
        hue="Churn",
        palette=_palette(),
        dodge=False,
        ax=ax,
    )
    ax.set_title(f"{feature} vs Churn", fontsize=14, fontweight="semibold")
    ax.set_xlabel("Churn")
    ax.set_ylabel(feature)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.tight_layout()

    filename = f"num_{feature.lower()}_churn.png".replace(" ", "_")
    output_path = GRAPHS_DIR / filename
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    insight = _numeric_insight(plot_df, feature)
    return filename, insight


def _categorical_insight(df: pd.DataFrame, feature: str) -> str:
    pivot = (
        df.groupby([feature, "Churn"])
        .size()
        .unstack("Churn", fill_value=0)
        .assign(total=lambda x: x.sum(axis=1))
    )

    if pivot["total"].eq(0).all():
        return f"No churn data available for {feature}."

    yes_counts = pivot["Yes"] if "Yes" in pivot.columns else pd.Series(0, index=pivot.index)
    pivot["churn_rate"] = yes_counts / pivot["total"]
    highest = pivot["churn_rate"].idxmax()
    lowest = pivot["churn_rate"].idxmin()
    high_rate = pivot.loc[highest, "churn_rate"]
    low_rate = pivot.loc[lowest, "churn_rate"]

    if high_rate == low_rate:
        return f"Churn is evenly distributed across {feature} categories."

    return (
        f"Customers with {feature} = {highest} have the highest churn rate "
        f"({high_rate:.0%}), while {feature} = {lowest} shows the lowest churn "
        f"rate ({low_rate:.0%})."
    )


def _numeric_insight(df: pd.DataFrame, feature: str) -> str:
    stats = df.groupby("Churn")[feature].agg(["mean", "median"]).round(2)
    yes_mean = stats.loc["Yes", "mean"] if "Yes" in stats.index else None
    no_mean = stats.loc["No", "mean"] if "No" in stats.index else None

    if yes_mean is None or no_mean is None:
        return f"Unable to compute churn insight for {feature}."

    if pd.isna(yes_mean) or pd.isna(no_mean):
        return f"Insufficient numeric data for {feature} to extract insights."

    if yes_mean > no_mean:
        return (
            f"Customers who churn have higher average {feature} ({yes_mean}) compared "
            f"to retained customers ({no_mean})."
        )
    if yes_mean < no_mean:
        return (
            f"Retained customers show higher average {feature} ({no_mean}) than those "
            f"who churn ({yes_mean})."
        )
    return f"Average {feature} is similar for both churned and retained customers."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        payload = request.get_json() or {}
    else:
        payload = request.form.to_dict()

    missing_columns = [
        col
        for col in FEATURE_COLUMNS
        if col not in OPTIONAL_COLUMNS and (col not in payload or payload[col] in ("", None))
    ]
    if missing_columns:
        return (
            jsonify(
                {
                    "error": "Missing required fields",
                    "details": missing_columns,
                }
            ),
            400,
        )

    cleaned_payload: dict[str, str | int | float] = {}
    encoded_payload: dict[str, int | float] = {}
    for column in FEATURE_COLUMNS:
        value = payload.get(column, "" if column not in NUMERIC_COLUMNS else 0)
        if column in NUMERIC_COLUMNS:
            try:
                numeric_value = NUMERIC_COLUMNS[column](value)
            except (TypeError, ValueError):
                return (
                    jsonify(
                        {
                            "error": f"Invalid value for {column}",
                            "details": value,
                        }
                    ),
                    400,
                )
            cleaned_payload[column] = numeric_value
            encoded_payload[column] = numeric_value
        else:
            text_value = str(value)
            cleaned_payload[column] = text_value
            if column not in CATEGORICAL_VALUES:
                return (
                    jsonify(
                        {
                            "error": f"Unsupported categorical feature: {column}",
                        }
                    ),
                    400,
                )
            if text_value not in CATEGORICAL_VALUES[column]:
                return (
                    jsonify(
                        {
                            "error": f"Invalid value for {column}",
                            "details": text_value,
                            "allowed": CATEGORICAL_VALUES[column],
                        }
                    ),
                    400,
                )
            encoder = LABEL_ENCODERS[column]
            encoded_payload[column] = int(encoder.transform([text_value])[0])

    input_df = pd.DataFrame([encoded_payload], columns=FEATURE_COLUMNS)

    try:
        raw_prediction = model.predict(input_df)[0]
    except Exception as err:  # pylint: disable=broad-except
        app.logger.exception("Prediction failed: %s", err)
        return (
            jsonify(
                {
                    "error": "Prediction failed",
                }
            ),
            500,
        )

    prediction_str = str(raw_prediction).strip()
    if prediction_str in {"1", "Yes", "True"}:
        prediction = "Yes"
    elif prediction_str in {"0", "No", "False"}:
        prediction = "No"
    else:
        prediction = prediction_str

    return jsonify({"prediction": prediction})


@app.route("/visualization")
def visualization():
    try:
        df = _load_dataset()
    except FileNotFoundError as error:
        app.logger.error("%s", error)
        return render_template(
            "visualization.html",
            error_message=str(error),
            charts=[],
        )

    ordered_assets: list[dict[str, str]] = [
        {"feature": "SeniorCitizen", "file": "cat_seniorcitizen_churn.png", "type": "categorical"},
        {"feature": "Contract", "file": "cat_contract_churn.png", "type": "categorical"},
        {"feature": "tenure", "file": "num_tenure_churn.png", "type": "numeric"},
        {"feature": "MonthlyCharges", "file": "num_monthlycharges_churn.png", "type": "numeric"},
        {"feature": "TotalCharges", "file": "num_totalcharges_churn.png", "type": "numeric"},
        {"feature": "InternetService", "file": "cat_internetservice_churn.png", "type": "categorical"},
        {"feature": "TechSupport", "file": "cat_techsupport_churn.png", "type": "categorical"},
        {"feature": "PaymentMethod", "file": "cat_paymentmethod_churn.png", "type": "categorical"},
        {"feature": "PaperlessBilling", "file": "cat_paperlessbilling_churn.png", "type": "categorical"},
        {"feature": "gender", "file": "cat_gender_churn.png", "type": "categorical"},
        {"feature": "Partner", "file": "cat_partner_churn.png", "type": "categorical"},
        {"feature": "Dependents", "file": "cat_dependents_churn.png", "type": "categorical"},
        {"feature": "MultipleLines", "file": "cat_multiplelines_churn.png", "type": "categorical"},
        {"feature": "PhoneService", "file": "cat_phoneservice_churn.png", "type": "categorical"},
        {"feature": "OnlineSecurity", "file": "cat_onlinesecurity_churn.png", "type": "categorical"},
        {"feature": "OnlineBackup", "file": "cat_onlinebackup_churn.png", "type": "categorical"},
        {"feature": "DeviceProtection", "file": "cat_deviceprotection_churn.png", "type": "categorical"},
        {"feature": "StreamingTV", "file": "cat_streamingtv_churn.png", "type": "categorical"},
        {"feature": "StreamingMovies", "file": "cat_streamingmovies_churn.png", "type": "categorical"},
    ]

    charts: list[dict[str, Any]] = []

    for asset in ordered_assets:
        feature = asset["feature"]
        file_name = asset["file"]
        chart_type = asset["type"]

        if feature not in df.columns:
            continue

        file_path = GRAPHS_DIR / file_name
        if not file_path.exists():
            continue

        data_subset = df[[feature, "Churn"]].dropna().copy()

        if chart_type == "numeric":
            data_subset[feature] = pd.to_numeric(data_subset[feature], errors="coerce")
            data_subset = data_subset.dropna(subset=[feature])
            insight = _numeric_insight(data_subset, feature)
        else:
            insight = _categorical_insight(data_subset, feature)

        charts.append(
            {
                "title": f"{_humanize_feature(feature)} vs Churn",
                "image": file_name,
                "insight": insight,
                "type": chart_type,
            }
        )

    return render_template(
        "visualization.html",
        charts=charts,
        error_message=None,
    )


if __name__ == "__main__":
    app.run(debug=True)
