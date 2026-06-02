from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

load_dotenv()

FAISS_PATH = "faiss_index"

def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"        
    )
    vectorstore = FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True        
    )
    return vectorstore.as_retriever(search_kwargs={"k":10})

def build_chain(retriever):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")        
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Business Intelligence Analyst.
         Answer the user's question using ONLY the data context provided below.
         Always mention specific numbers, countries, genres, or artist name from the data.
         End your answer with: 'Sources: [list the relevant sales/customer records used]'
         
         Context:
         {context}
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

if __name__ == "__main__":
    print("Loading retriever...")
    retriever = load_retriever()
    chain = build_chain(retriever)

    questions = [
        "Which country has the most customers?",
        "What is the best selling genre?",
        "Which artist genereates the most revenue?"
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print("-" * 50)
        response = chain.invoke(q)
        print(response.content)
        print()

