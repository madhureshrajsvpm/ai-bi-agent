from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import pandas as pd
import os

_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings

def dataframe_to_documents(df, max_rows=2000):
    """Convert any dataframe into RAG-ready documents, one per row."""
    docs = []
    columns = list(df.columns)

    # Sample if too large (keeps embedding fast)
    if len(df) > max_rows:
        df_sample = df.sample(n=max_rows, random_state=42)
    else:
        df_sample = df

    for idx, row in df_sample.iterrows():
        parts = [f"{col}: {row[col]}" for col in columns if pd.notna(row[col])]
        text = "Record: " + ", ".join(parts)
        docs.append(Document(
            page_content=text,
            metadata={"row_index": int(idx)}
        ))

    return docs

def build_dynamic_index(df):
    """Build a fresh FAISS index in memory from the current dataframe."""
    docs = dataframe_to_documents(df)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 10})

def build_dynamic_chain(retriever, dataset_name="the uploaded dataset"):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a Business Intelligence analyst.
        Answer the user's question using ONLY the data context provided below,
        which comes from {dataset_name}.
        Always mention specific numbers, names, or categories from the data.
        End your answer with: 'Sources: [list the relevant records used]'

        Context:
        {{context}}
        """),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return chain
