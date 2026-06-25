from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import os


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )


def summarize_location(df, location_col, location_value):
    """Filter the dataframe to rows matching the location and build a summary dict."""
    filtered = df[df[location_col].astype(str).str.lower() == str(location_value).lower()]

    if filtered.empty:
        return None, filtered

    summary = {
        "location_column": location_col,
        "location_value": location_value,
        "total_matching_rows": len(filtered),
    }

    # Numeric column summaries
    num_cols = filtered.select_dtypes(include="number").columns.tolist()
    for col in num_cols[:6]:
        summary[f"{col} (sum)"] = round(filtered[col].sum(), 2)
        summary[f"{col} (avg)"] = round(filtered[col].mean(), 2)

    # Categorical column top values
    cat_cols = filtered.select_dtypes(include="object").columns.tolist()
    for col in cat_cols[:6]:
        if col != location_col:
            top_vals = filtered[col].value_counts().head(5).to_dict()
            summary[f"{col} (top values)"] = top_vals

    return summary, filtered


def dynamic_geo_analysis(df, location_col, location_value, question, dataset_name="the uploaded dataset"):
    """Generic geo/location analysis that adapts to any dataset's domain."""
    summary, filtered = summarize_location(df, location_col, location_value)

    if summary is None:
        return f"No rows found where '{location_col}' equals '{location_value}'.", None

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a data analyst providing location-based insights for {dataset_name}.

        You will be given a statistical summary of the data filtered to a specific location/category value.
        Do NOT assume this is sales or business revenue data unless the summary clearly shows that.
        Infer the actual subject matter (e.g. wildlife observations, fitness data, real estate, sales)
        from the column names and values provided.

        Structure your response as:
        - DATA INSIGHT: what the numbers show for this location, using actual figures from the summary
        - CONTEXT: relevant domain, geographic, cultural, or seasonal context that explains the pattern
        - RECOMMENDATION: specific, actionable next steps relevant to the data's actual subject matter

        Data summary:
        {{data_summary}}
        """),
        ("human", "{question}")
    ])

    chain = prompt | get_llm()

    response = chain.invoke({
        "data_summary": str(summary),
        "question": question
    })

    return response.content, filtered


def guess_geo_column(columns):
    """Guess which column represents location/category based on common naming patterns."""
    geo_keywords = [
        "country", "state", "region", "location", "city",
        "admin_unit", "area", "zone", "district", "province", "unit_code"
    ]
    for col in columns:
        col_lower = col.lower()
        for kw in geo_keywords:
            if kw in col_lower:
                return col
    return columns[0] if columns else None