from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

parser = JsonOutputParser()

CLEANING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a data cleaning assistant.
     The user will describe a cleaning operation in plain English.
     You will receive the column names and the first 3  rows of the dataframe.
     
     Return only a valid JSON object with exactly these keys:
     - operation: one of [drop_duplicates, fill_nulls, drop_column, rename_column, filter_rows, conver_type]
     - column: the column name to apply the operation to (null if not applicable
     - value: the value to use for fill_nulls or filter_rows (null if not applicable)
     - explanation: one sentence explaining what will be done
     No markdown, no extra text, just the JSON object."""),
     ("human", """DataFrame info:
      Columns: {columns}
      Sample rows: {sample}
      
      User instruction: {instruction}""")
])

cleaning_chain = CLEANING_PROMPT | llm | parser

def apply_cleaning(df, instruction):
    columns = list(df.columns)
    sample = df.head(3).to_dict(orient="records")

    print(f"\nAnalyzing instruction: '{instruction}'")

    # Intercept null-dropping instructions directly
    if "null" in instruction.lower() and any(
        word in instruction.lower() for word in ["drop", "remove", "delete"]
    ):
        original_shape = df.shape
        # Check if a specific column is mentioned
        matched_col = None
        for col in columns:
            if col.lower() in instruction.lower():
                matched_col = col
                break
        if matched_col:
            df = df.dropna(subset=[matched_col])
        else:
            df = df.dropna()
        new_shape = df.shape
        removed = original_shape[0] - new_shape[0]
        print(f"Shape changed: {original_shape} → {new_shape}")
        return df, f"Dropped {removed} rows with null values."

    result = cleaning_chain.invoke({
        "columns": columns,
        "sample": sample,
        "instruction": instruction
    })

    print(f"Operation detected: {result}")

    original_shape = df.shape
    operation = result.get("operation")
    column = result.get("column")
    value = result.get("value")

    if operation == "drop_duplicates":
        if column:
            df = df.drop_duplicates(subset=[column])
        else:
            df = df.drop_duplicates()

    elif operation == "fill_nulls":
        if column and value is not None:
            df[column] = df[column].fillna(value)
        elif column:
            df[column] = df[column].fillna("Unknown")

    elif operation == "drop_column":
        if column and column in df.columns:
            df = df.drop(columns=[column])

    elif operation == "rename_column":
        if column and value:
            df = df.rename(columns={column: value})

    elif operation == "filter_rows":
        if column and value is not None:
            df = df[df[column] != value]

    elif operation == "convert_type":
        if column and value:
            df[column] = df[column].astype(value)

    new_shape = df.shape
    print(f"Shape changed: {original_shape} → {new_shape}")
    print(f"Explanation: {result.get('explanation')}")

    return df, result.get("explanation")