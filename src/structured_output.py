from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os

load_dotenv()

llm=ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

parser= JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", """Return your answer Only as valid JSON exactly with these keys:
     answer, confidence (high/mediium/low), suggested_followup,
     No explanation, no markdown, just the JSON object."""),
     ("human", "{question}")
])

chain = prompt | llm | parser

result = chain.invoke({
    "question": "Which music genre typically generates the most revenue per track?"
})

print(result)
print(type(result))

