from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY")
)

GEO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a geo-cultural business intelligence analyst.
     When given sales data and a target location, you provide:
     1. What the data shows about that location specifically
     2. Cultural and market context that explains the patterns
     3. Specific business recommendations tailored to that location
     
     Always structure your response as:
     - DATA INSIGHT: what the numbers show
     - CULTURAL CONTEXT: why this pattern exists culturally/demographically
     - RECOMMENDATION: specific action the business should take for this location
     
     Be specific - mention actual city names, cultural events, local preferences."""),
     ("human", """Sales data summary for {location}:
      {data_summary}
      
      Question: {question}""")
])

geo_chain = GEO_PROMPT | llm

def get_location_summary(invoice_df, customer_df, location, level="country"):
    if level == "country":
        filtered = invoice_df[
            invoice_df["billing_country"].str.lower() == location.lower()            
        ]
        customers = customer_df[
            customer_df["country"].str.lower() == location.lower()            
        ]
    else:
        filtered = invoice_df[
            invoice_df["billing_city"].str.lower() == location.lower()
        ]
        customers = customer_df[
            customer_df["city"].str.lower() == location.lower()            
        ]
    
    if filtered.empty:
        return None
    
    summary = {
        "location": location,
        "total_revenue": round(filtered["total"].sum(), 2),
        "total_invoices": len(filtered),
        "average_order_value": round(filtered["total"].mean(), 2),
        "total_customers": len(customers),
        "top_cities": filtered["billing_city"].value_counts().head(3).to_dict()
        if level == "country" else {},
        "data_range": f"{filtered['invoice_date'].min()} to {filtered['invoice_date'].max()}"
    }
    return summary

def geo_analysis(location, question, level="country"):
    invoice = pd.read_csv("data/itunes/invoice.csv")
    customer = pd.read_csv("data/itunes/customer.csv")

    summary = get_location_summary(invoice, customer, location, level)

    if not summary:
        return f"No data found for {location}."
    
    response = geo_chain.invoke({
        "location": location,
        "data_summary": str(summary),
        "question": question
    })

    return response.content

if __name__ == "__main__":
    print("=" * 60)
    print("COUNTRY LEVEL: USA")
    print("=" * 60)
    result = geo_analysis(
        location="USA",
        question="How should we improve music sales in this market?",
        level="country"
    )
    print(result)

    print("\n" + "=" * 60)
    print("COUNTRY LEVEL: Brazil")
    print("=" * 60)
    result = geo_analysis(
        location="Brazil",
        question="What music genres should we promote here and why?",
        level="country"
    )
    print(result)

    print("\n" + "=" * 60)
    print("CITY LEVEL: Paris")
    print("=" * 60)
    result = geo_analysis(
        location="Paris",
        question="What is the revenue potential and what should we do differently here?",
        level="city"
    )
    print(result)

