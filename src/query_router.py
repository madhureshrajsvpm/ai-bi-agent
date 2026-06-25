from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import os


def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query router for a business intelligence app.
    Given a user's question and the dataframe's column names, decide HOW to answer it.

    Return ONLY valid JSON with these keys:
    - route: "aggregate" or "exploratory"
    - operation: one of ["sum", "mean", "count", "max", "min", "nunique", "value_counts", null]
    - group_by_column: the column to group by, or null
    - target_column: the numeric column to aggregate, or null
    - top_n: integer, how many top results to return
    - ascending: true if asking for least/lowest, false otherwise

    Available columns: {columns}

    No markdown, no explanation, just the JSON object."""),
    ("human", "{question}")
])


AGGREGATE_EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Business Intelligence analyst presenting results to a business stakeholder.
    An exact pandas calculation was already run on the FULL dataset of {total_rows} rows.
    The result is given to you below — do NOT re-explain how to calculate it, do NOT show code,
    do NOT use hypothetical examples.
    Just present the finding clearly in 2-3 sentences using the exact numbers provided.
    End with: 'Source: Exact calculation over all {total_rows} rows.'"""),
    ("human", """Question: {question}
    Calculation performed: {calc_description}
    Exact result: {result}""")
])


def classify_query(question, columns):
    parser = JsonOutputParser()
    chain = CLASSIFY_PROMPT | get_llm() | parser
    return chain.invoke({"question": question, "columns": columns})


def run_aggregate(df, classification):
    operation = classification.get("operation")
    group_col = classification.get("group_by_column")
    target_col = classification.get("target_column")
    top_n = classification.get("top_n") or 5
    ascending = classification.get("ascending", False)
    total_rows = len(df)

    if group_col and group_col not in df.columns:
        group_col = None
    if target_col and target_col not in df.columns:
        target_col = None

    if group_col and target_col and operation in ["sum", "mean", "max", "min"]:
        grouped = getattr(df.groupby(group_col)[target_col], operation)()
        grouped = grouped.sort_values(ascending=ascending).head(top_n)
        result = grouped.to_dict()
        calc_description = f"{operation}('{target_col}') grouped by '{group_col}', top {top_n}"
        return result, calc_description, total_rows

    if group_col and operation in ["count", "value_counts", "nunique", None]:
        counts = df[group_col].value_counts(ascending=ascending if ascending else False)
        top_n = top_n if top_n else 5
        result = counts.head(top_n).to_dict()
        calc_description = f"row count grouped by '{group_col}', top {top_n}"
        return result, calc_description, total_rows

    if target_col and operation in ["sum", "mean", "max", "min", "nunique"]:
        value = getattr(df[target_col], operation)()
        result = {target_col: round(float(value), 2) if pd.notna(value) else None}
        calc_description = f"{operation}('{target_col}') across all rows"
        return result, calc_description, total_rows

    result = {"total_rows": total_rows}
    calc_description = "count of all rows"
    return result, calc_description, total_rows


def answer_aggregate_query(df, question, classification):
    result, calc_description, total_rows = run_aggregate(df, classification)
    if result is None:
        return None
    chain = AGGREGATE_EXPLAIN_PROMPT | get_llm()
    response = chain.invoke({
        "question": question,
        "calc_description": calc_description,
        "result": str(result),
        "total_rows": total_rows
    })
    return response.content


def smart_query(df, question, rag_chain=None):
    columns = list(df.columns)
    try:
        classification = classify_query(question, columns)
    except Exception:
        classification = {"route": "exploratory"}

    route = classification.get("route", "exploratory")

    if route == "aggregate":
        answer = answer_aggregate_query(df, question, classification)
        if answer is not None:
            return answer, "aggregate"

    if rag_chain is not None:
        response = rag_chain.invoke(question)
        return response.content, "exploratory"

    return "I couldn't determine how to answer this question.", "none"