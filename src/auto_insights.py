from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import numpy as np
import os


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )


def compute_dataset_profile(df):
    """Compute statistical profile of the dataset to feed into the LLM."""
    profile = {}
    total_rows = len(df)
    profile["total_rows"] = total_rows
    profile["total_columns"] = len(df.columns)
    profile["column_names"] = list(df.columns)

    # Null summary
    nulls = df.isnull().sum()
    profile["null_summary"] = {
        col: {"count": int(count), "pct": round(count / total_rows * 100, 1)}
        for col, count in nulls.items() if count > 0
    }

    # Duplicate count
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # Numeric column stats
    num_cols = df.select_dtypes(include="number").columns.tolist()
    profile["numeric_stats"] = {}
    for col in num_cols[:8]:
        profile["numeric_stats"][col] = {
            "min":    round(float(df[col].min()), 2),
            "max":    round(float(df[col].max()), 2),
            "mean":   round(float(df[col].mean()), 2),
            "median": round(float(df[col].median()), 2),
            "std":    round(float(df[col].std()), 2),
        }

    # Categorical column top values
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    profile["categorical_summary"] = {}
    for col in cat_cols[:6]:
        top = df[col].value_counts().head(5).to_dict()
        profile["categorical_summary"][col] = {
            "unique_values": int(df[col].nunique()),
            "top_5": top
        }

    # Correlations between numeric columns
    if len(num_cols) >= 2:
        corr = df[num_cols].corr().abs()
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                val = corr.iloc[i, j]
                if not np.isnan(val) and val > 0.5:
                    pairs.append({
                        "col_a": num_cols[i],
                        "col_b": num_cols[j],
                        "correlation": round(float(val), 2)
                    })
        pairs.sort(key=lambda x: -x["correlation"])
        profile["strong_correlations"] = pairs[:5]

    # Outlier detection using IQR
    profile["outliers"] = {}
    for col in num_cols[:6]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_count = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        if outlier_count > 0:
            profile["outliers"][col] = {
                "count": outlier_count,
                "pct": round(outlier_count / total_rows * 100, 1)
            }

    return profile


AUTO_INSIGHTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Business Intelligence analyst.
    You have been given a statistical profile of a dataset.
    Generate exactly 5 key insights a business stakeholder would care about.

    Rules:
    - Each insight must be specific — use actual numbers, column names, and category values from the profile
    - Do NOT use generic statements like "the data has nulls" — say exactly which column, how many, and what it means
    - Infer the actual subject matter from the column names (sales, wildlife, fitness, real estate, etc.)
    - Each insight should be actionable or decision-relevant
    - Format each insight as a single clear sentence starting with an emoji:
      📊 for data patterns
      ⚠️ for data quality issues
      🔗 for correlations
      🏆 for top performers
      💡 for recommendations

    Dataset profile:
    {profile}

    Dataset name: {dataset_name}"""),
    ("human", "Generate 5 key insights from this dataset.")
])


def generate_auto_insights(df, dataset_name="the dataset"):
    """Main function — profile the data and generate insights."""
    profile = compute_dataset_profile(df)
    chain = AUTO_INSIGHTS_PROMPT | get_llm()
    response = chain.invoke({
        "profile": str(profile),
        "dataset_name": dataset_name
    })
    return response.content, profile

def compute_quality_score(df, profile):
    """
    Compute a 0-100 data quality score based on:
    - Null values (-30 max penalty)
    - Duplicate rows (-20 max penalty)
    - Outliers (-20 max penalty)
    - Column type variety (-15 max penalty)
    - Row count adequacy (-15 max penalty)
    """
    score = 100
    total_rows = profile.get("total_rows", 1)

    # Null penalty
    total_nulls = sum(v["count"] for v in profile.get("null_summary", {}).values())
    null_pct = total_nulls / (total_rows * len(df.columns)) * 100
    null_penalty = min(30, null_pct * 3)
    score -= null_penalty

    # Duplicate penalty
    dup_pct = profile.get("duplicate_rows", 0) / total_rows * 100
    dup_penalty = min(20, dup_pct * 2)
    score -= dup_penalty

    # Outlier penalty
    outliers = profile.get("outliers", {})
    if outliers:
        avg_outlier_pct = sum(v["pct"] for v in outliers.values()) / len(outliers)
        outlier_penalty = min(20, avg_outlier_pct * 2)
        score -= outlier_penalty

    # Column type variety (penalise if all columns are same type)
    num_cols = len(df.select_dtypes(include="number").columns)
    cat_cols = len(df.select_dtypes(include="object").columns)
    if num_cols == 0 or cat_cols == 0:
        score -= 15

    # Row count adequacy
    if total_rows < 100:
        score -= 15
    elif total_rows < 1000:
        score -= 7

    score = max(0, min(100, round(score)))

    # Grade
    if score >= 90:
        grade, color = "A — Excellent", "green"
    elif score >= 75:
        grade, color = "B — Good", "blue"
    elif score >= 60:
        grade, color = "C — Fair", "orange"
    elif score >= 40:
        grade, color = "D — Poor", "red"
    else:
        grade, color = "F — Critical", "red"

    breakdown = {
        "Null penalty":      round(null_penalty, 1),
        "Duplicate penalty": round(dup_penalty, 1),
        "Outlier penalty":   round(outliers and min(20, sum(v["pct"] for v in outliers.values()) / len(outliers) * 2) or 0, 1),
    }

    return score, grade, color, breakdown

