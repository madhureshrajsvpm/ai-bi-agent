from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

load_dotenv()

llm=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
                      
)

prompt = ChatPromptTemplate.from_messages([
    ("system, You are a BI analyst. Answer questions about data concisely."),
    ("human", "{question}")
])

chain = prompt | llm

response = chain.invoke ({
    "question": "What KPIs matter the most for a music streaming industry?"
})

print(response.content)