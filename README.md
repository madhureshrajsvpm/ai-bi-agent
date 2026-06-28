# 📊 AI Business Intelligence Agent

> **An end-to-end AI-powered BI platform** that lets you upload any dataset, ask business questions in plain English, clean data automatically, forecast trends, and get geo-cultural recommendations — all in one app.

🔗 **Live Demo:** [ai-bi-agent.streamlit.app](https://ai-bi-agent-lsodkky8sbb9fc83ub3rg9.streamlit.app/)
*(Request access password via [LinkedIn](https://linkedin.com/in/madhuresh-raj-selvaraj) or [Email](mailto:madhureshraj@gmail.com))*

---

## 🧠 What This Project Demonstrates

This project was built to showcase the skills most in-demand for **AI-Augmented Business Analyst** roles in 2026:

| Skill | How it's demonstrated |
|---|---|
| RAG pipeline development | FAISS vector store + HuggingFace embeddings + Groq LLM |
| LLM integration | LangChain chains, prompt engineering, structured outputs |
| Smart query routing | Auto-detects aggregate vs exploratory questions, routes accordingly |
| Data engineering | 11-table relational merge, multi-file upload, union/join operations |
| ML forecasting | XGBoost time-series and regression with auto-detection |
| BI & visualization | Plotly charts, choropleth maps, auto-insights generation |
| Geo-cultural intelligence | Location-aware recommendations adapting to any dataset domain |
| Consulting output | Exportable Word reports with insights, stats, correlations and charts |

---

## 🚀 Features

### 📋 Data Explorer
- Upload single or multiple CSV/Excel files
- Merge datasets with configurable join types (inner, left, right, outer, union)
- Auto-detect common columns for merging
- Interactive Plotly charts (bar, line, scatter, box)
- Choropleth world map for geographic data
- Data quality score (0–100) with breakdown

### 🤖 Auto Insights
- One-click generation of 5 key business findings from any dataset
- Detects top performers, outliers, correlations, null patterns
- Domain-aware — adapts language to sales, wildlife, real estate, fitness data

### 🧹 AI-Powered Data Cleaning
- Plain English cleaning instructions (e.g. "fill empty composer fields with Unknown")
- Supports: fill nulls, drop duplicates, drop columns, rename, filter rows, convert types
- Download cleaned CSV after each operation

### 📄 Executive Report Export
- One-click Word document generation (.docx)
- Includes: dataset overview, key insights, numeric stats, correlation table, chart
- Branded with author name and date — ready to share with stakeholders

### 💬 BI Chat (RAG + Smart Router)
- Ask business questions in plain English
- **Smart Query Router** detects aggregate questions and runs exact pandas calculations over the full dataset — not just retrieved samples
- **Conversational memory** — follow-up questions like "break that down by city" work correctly
- Answers labeled by method: 🧮 exact calculation or 🔍 retrieved records

### 🌍 Geo / Category Insights
- Select any location or category column from your data
- LLM generates DATA INSIGHT → CULTURAL CONTEXT → RECOMMENDATION
- Auto-generates relevant bar/line charts for the selected location
- Works on any domain — not just sales data

### 📈 AI Forecasting
- **Auto-detects** whether dataset has a date column (time-series) or not (regression)
- Time-series: XGBoost with lag features, seasonality encoding, rolling averages
- Regression: XGBoost with feature importance visualization
- LLM-generated plain English interpretation of results for stakeholders

---

## 🏗️ Architecture

```
User Input (CSV / Excel / Question)
         │
         ▼
   File Processor
   (pandas, pdfplumber, auto-profiling)
         │
         ▼
┌────────────────────────────────────┐
│         AI Agent Core              │
│  ┌──────────┐  ┌───────────────┐  │
│  │   RAG    │  │ Smart Router  │  │
│  │  FAISS   │  │ Pandas/LLM    │  │
│  └──────────┘  └───────────────┘  │
│  ┌──────────┐  ┌───────────────┐  │
│  │Cleaning  │  │  Geo-Culture  │  │
│  │  Agent   │  │     Layer     │  │
│  └──────────┘  └───────────────┘  │
└────────────────────────────────────┘
         │
         ▼
   Output Engine
   (Plotly charts, Word report, CSV export, Forecast)
```

---

## 🛠️ Tech Stack (100% Free & Open Source)

| Component | Tool | Purpose |
|---|---|---|
| LLM | Groq `llama-3.1-8b-instant` | Fast, free inference |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Local, no API cost |
| Vector Store | FAISS | Local similarity search |
| ML Models | XGBoost + Scikit-learn | Forecasting + regression |
| UI | Streamlit | Web interface + deployment |
| Charts | Plotly Express | Interactive visualizations |
| Data | Pandas | Data processing + cleaning |
| Reports | python-docx | Word document generation |
| Orchestration | LangChain | LLM chains + RAG pipeline |

---

## 📁 Project Structure

```
ai-bi-agent/
├── src/
│   ├── app.py               # Main Streamlit application
│   ├── ingest.py            # FAISS index builder
│   ├── retriever_test.py    # RAG retrieval chain
│   ├── dynamic_rag.py       # Dynamic index from any uploaded dataset
│   ├── dynamic_geo.py       # Geo-cultural analysis for any dataset
│   ├── cleaning_agent.py    # AI-powered data cleaning
│   ├── geo_agent.py         # iTunes geo analysis
│   ├── query_router.py      # Smart aggregate vs exploratory router
│   ├── auto_insights.py     # Auto insight + quality score generation
│   ├── report_generator.py  # Word report generation
│   └── forecasting.py       # XGBoost time-series + regression
├── data/
│   └── itunes/              # 11-table Chinook music store dataset
├── faiss_index/             # Pre-built FAISS index for iTunes data
├── requirements.txt
├── LICENSE.txt
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repo
git clone https://github.com/madhureshrajsvpm/ai-bi-agent.git
cd ai-bi-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo GROQ_API_KEY=your_key_here > .env
echo APP_PASSWORD=your_password > .env

# Build FAISS index (first time only)
python src/ingest.py

# Run the app
streamlit run src/app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## 📊 Demo Datasets

The app comes preloaded with the **Chinook iTunes dataset** (11 relational tables, 4,757 merged rows) for instant demo. You can also upload:

- Sales transaction CSVs
- Real estate datasets
- Wildlife monitoring data
- Any structured CSV or Excel file

---

## 👤 Author

**Madhuresh Raj Selvaraj**
Business Consultant | BI & Generative AI Analyst

🔗 [LinkedIn](https://linkedin.com/in/madhuresh-raj-selvaraj)
📧 [madhureshraj@gmail.com](mailto:madhureshraj@gmail.com)
💻 [GitHub](https://github.com/madhureshrajsvpm)

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License**.
You may share and adapt this work for non-commercial purposes with attribution.
See [LICENSE.txt](LICENSE.txt) for full terms.
