import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import plotly.graph_objects as go
import plotly.express as px
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os


def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))


def detect_date_columns(df):
    date_cols = []
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            date_cols.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            sample = df[col].dropna().astype(str).head(100)
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                date_cols.append(col)
        except Exception:
            pass
    return date_cols


def detect_forecast_type(df):
    date_cols = detect_date_columns(df)
    num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if not c.lower().endswith("_id")
        and c.lower() not in ["id", "index", "row_number"]
    ]
    if date_cols and num_cols:
        return "timeseries", date_cols, num_cols
    elif num_cols and len(num_cols) >= 2:
        return "regression", [], num_cols
    return None, [], num_cols


def run_timeseries(df, date_col, target_col, periods=6):
    ts = df[[date_col, target_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts = ts.dropna(subset=[date_col, target_col])
    
# Use weekly aggregation if less than 100 monthly points
    ts_monthly = ts.set_index(date_col).resample("ME")[target_col].sum().reset_index()
    if len(ts_monthly) < 100:
        ts = ts.set_index(date_col).resample("W")[target_col].sum().reset_index()
    else:
        ts = ts_monthly

    ts.columns = ["date", "value"]
    ts = ts.sort_values("date").reset_index(drop=True)
    if len(ts) < 6:
        return None, None, None, None
    ts["month"]         = ts["date"].dt.month
    ts["quarter"]       = ts["date"].dt.quarter
    ts["year"]          = ts["date"].dt.year
    ts["month_sin"]     = np.sin(2 * np.pi * ts["month"] / 12)
    ts["month_cos"]     = np.cos(2 * np.pi * ts["month"] / 12)
    ts["lag_1"]         = ts["value"].shift(1)
    ts["lag_3"]         = ts["value"].shift(3)
    ts["rolling_mean3"] = ts["value"].shift(1).rolling(3).mean()
    ts                  = ts.dropna().reset_index(drop=True)
    features = ["month", "quarter", "year", "month_sin", "month_cos",
                "lag_1", "lag_3", "rolling_mean3"]
    X, y = ts[features], ts["value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2  = round(r2_score(y_test, y_pred), 3)
    last_date = ts["date"].max()
    history   = ts["value"].tolist()
    future_rows = []
    for i in range(1, periods + 1):
        future_date   = last_date + pd.DateOffset(months=i)
        lag_1         = history[-1]
        lag_3         = history[-3] if len(history) >= 3 else lag_1
        rolling_mean3 = np.mean(history[-3:]) if len(history) >= 3 else lag_1
        month         = future_date.month
        row = {
            "date": future_date, "month": month, "quarter": future_date.quarter,
            "year": future_date.year, "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12), "lag_1": lag_1,
            "lag_3": lag_3, "rolling_mean3": rolling_mean3
        }
        future_rows.append(row)
        predicted = model.predict(pd.DataFrame([row])[features])[0]
        history.append(max(0, predicted))
    future_df = pd.DataFrame(future_rows)
    future_df["forecast"] = [max(0, v) for v in history[-periods:]]
    metrics = {"MAE": mae, "R2": r2, "Training rows": len(X_train), "Test rows": len(X_test)}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["date"], y=ts["value"], mode="lines+markers",
        name="Historical", line=dict(color="#2E75B6", width=2)))
    fig.add_trace(go.Scatter(x=future_df["date"], y=future_df["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#FF6B35", width=2, dash="dash"), marker=dict(symbol="diamond")))
    fig.update_layout(title=f"{target_col} � Historical + {periods}-Month Forecast",
        xaxis_title="Date", yaxis_title=target_col, legend=dict(orientation="h"),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white")
    return ts, future_df, metrics, fig


def run_regression(df, target_col, feature_cols):
    data = df[feature_cols + [target_col]].dropna()
    if len(data) < 20:
        return None, None, None
    X, y = data[feature_cols], data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    r2  = round(r2_score(y_test, y_pred), 3)
    metrics = {"MAE": mae, "R2 Score": r2, "Training rows": len(X_train), "Test rows": len(X_test)}
    importance_df = pd.DataFrame({
        "Feature": feature_cols, "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h",
        title=f"Feature Importance for predicting '{target_col}'",
        color="Importance", color_continuous_scale="Blues")
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    results_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).reset_index(drop=True)
    return results_df, metrics, fig


FORECAST_INTERPRET_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Business Intelligence analyst interpreting a machine learning forecast.
Write a clear 3-4 sentence interpretation for a non-technical business stakeholder.
Cover: what the model predicts, how reliable it is (R2/MAE), and one business recommendation.
Do NOT mention code, algorithms, or technical ML terms."""),
    ("human", "Model type: {model_type}\nTarget: {target_col}\nMetrics: {metrics}\nResult: {key_result}\nDataset: {dataset_name}")
])


def interpret_forecast(model_type, target_col, metrics, key_result, dataset_name):
    chain = FORECAST_INTERPRET_PROMPT | get_llm()
    response = chain.invoke({"model_type": model_type, "target_col": target_col,
        "metrics": str(metrics), "key_result": key_result, "dataset_name": dataset_name})
    return response.content
