import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys
from dotenv import load_dotenv
from dynamic_rag import build_dynamic_index, build_dynamic_chain
from dynamic_geo import dynamic_geo_analysis, guess_geo_column
from query_router import smart_query
from auto_insights import generate_auto_insights
from report_generator import generate_report
from forecasting import detect_forecast_type, run_timeseries, run_regression, interpret_forecast
from auto_insights import generate_auto_insights, compute_quality_score

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cleaning_agent import apply_cleaning
from geo_agent import geo_analysis
from retriever_test import load_retriever, build_chain

st.set_page_config(
    page_title="AI BI Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Business Intelligence Agent")
st.caption("Upload any dataset — ask questions, clean data, and get geo-aware recommendations.")

# ── Session state ──────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "df"            not in st.session_state: st.session_state.df            = None
if "df_name"       not in st.session_state: st.session_state.df_name       = None
if "all_dfs"       not in st.session_state: st.session_state.all_dfs       = {}
if "rag_chain"     not in st.session_state: st.session_state.rag_chain     = None
if "rag_ready"     not in st.session_state: st.session_state.rag_ready     = False
if "last_insights"     not in st.session_state: st.session_state.last_insights     = None
if "last_profile"      not in st.session_state: st.session_state.last_profile      = None
if "quality_score"     not in st.session_state: st.session_state.quality_score     = None
if "quality_grade"     not in st.session_state: st.session_state.quality_grade     = None
if "quality_breakdown" not in st.session_state: st.session_state.quality_breakdown = {}

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Data Source")

    data_source = st.radio(
        "Choose data source",
        ["Upload my own file", "Use iTunes demo data"]
    )

    if data_source == "Upload my own file":
        uploaded_files = st.file_uploader(
            "Upload CSV or Excel files",
            type=["csv", "xlsx"],
            accept_multiple_files=True
        )

        if uploaded_files:
            dfs = {}
            for uploaded in uploaded_files:
                try:
                    if uploaded.name.lower().endswith(".xlsx"):
                        dfs[uploaded.name] = pd.read_excel(uploaded)
                    else:
                        try:
                            dfs[uploaded.name] = pd.read_csv(uploaded, encoding="utf-8")
                        except UnicodeDecodeError:
                            uploaded.seek(0)
                            try:
                                dfs[uploaded.name] = pd.read_csv(uploaded, encoding="latin1")
                            except UnicodeDecodeError:
                                uploaded.seek(0)
                                dfs[uploaded.name] = pd.read_csv(uploaded, encoding="cp1252")
                except Exception as e:
                    st.error(f"Could not load {uploaded.name}: {e}")

            if dfs:
                st.success(f"Loaded {len(dfs)} file(s)")

                if len(dfs) == 1:
                    name = list(dfs.keys())[0]
                    st.session_state.df = dfs[name]
                    st.session_state.df_name = name
                    st.session_state.all_dfs = dfs
                else:
                    st.session_state.all_dfs = dfs
                    file_names = list(dfs.keys())

                    st.write("**Choose how to work with multiple files:**")
                    mode = st.radio(
                        "Mode",
                        ["Analyze one at a time", "Merge all into one table"]
                    )

                    if mode == "Analyze one at a time":
                        selected = st.selectbox("Select file to analyze", file_names)
                        st.session_state.df = dfs[selected]
                        st.session_state.df_name = selected

                    else:
                        # Find columns common to ALL uploaded files
                        all_columns = [set(df.columns) for df in dfs.values()]
                        common_cols = set.intersection(*all_columns)

                        # Find columns shared by AT LEAST 2 files (useful for partial merges)
                        from collections import Counter
                        col_counts = Counter()
                        for cols in all_columns:
                            col_counts.update(cols)
                        partial_cols = sorted(
                            [col for col, count in col_counts.items() if count >= 2],
                            key=lambda c: -col_counts[c]
                        )

                        merge_options = [
                            "-- Stack rows (no merge key) --",
                            "-- Union (stack + remove duplicate rows) --"
                        ] + sorted(common_cols) + [
                            c for c in partial_cols if c not in common_cols
                        ]

                        merge_on_choice = st.selectbox(
                            "Select column to merge on",
                            merge_options
                        )

                        merge_on = None if merge_on_choice.startswith("--") else merge_on_choice
                        is_union = merge_on_choice == "-- Union (stack + remove duplicate rows) --"

                        join_type = None
                        if merge_on:
                            join_type = st.selectbox(
                                "Join type",
                                ["left", "right", "inner", "outer"],
                                index=2,
                                help=(
                                    "Inner: only rows with matching values in all files. "
                                    "Left: keep all rows from the first file. "
                                    "Right: keep all rows from the last file. "
                                    "Outer: keep all rows from every file, filling gaps with blanks."
                                )
                            )

                        if st.button("Merge Files"):
                            try:
                                if merge_on:
                                    merged = list(dfs.values())[0]
                                    for df_next in list(dfs.values())[1:]:
                                        merged = merged.merge(df_next, on=merge_on, how=join_type)
                                elif is_union:
                                    merged = pd.concat(list(dfs.values()), ignore_index=True)
                                    before = len(merged)
                                    merged = merged.drop_duplicates().reset_index(drop=True)
                                    removed = before - len(merged)
                                    st.info(f"Removed {removed:,} duplicate row(s) during union.")
                                else:
                                    merged = pd.concat(list(dfs.values()), ignore_index=True)

                                st.session_state.df = merged
                                st.session_state.df_name = "Merged dataset"
                                st.success(f"Merged into {merged.shape[0]:,} rows × {merged.shape[1]} columns")
                            except Exception as e:
                                st.error(f"Merge failed: {e}")

    else:
        if st.button("Load iTunes Demo Data"):
            try:
                invoice  = pd.read_csv("data/itunes/invoice.csv")
                inv_line = pd.read_csv("data/itunes/invoice_line.csv")
                customer = pd.read_csv("data/itunes/customer.csv")
                track    = pd.read_csv("data/itunes/track.csv")
                album    = pd.read_csv("data/itunes/album.csv")
                artist   = pd.read_csv("data/itunes/artist.csv")
                genre    = pd.read_csv("data/itunes/genre.csv")
                media    = pd.read_csv("data/itunes/media_type.csv")

                artist.rename(columns={"name": "artist_name"}, inplace=True)
                genre.rename(columns={"name": "genre_name"},   inplace=True)
                media.rename(columns={"name": "media_name"},   inplace=True)
                track.rename(columns={"name": "track_name", "unit_price": "track_price"}, inplace=True)
                album.rename(columns={"title": "album_title"}, inplace=True)

                df = (
                    inv_line
                    .merge(invoice,  on="invoice_id")
                    .merge(customer, on="customer_id")
                    .merge(track,    on="track_id")
                    .merge(album,    on="album_id")
                    .merge(artist,   on="artist_id")
                    .merge(genre,    on="genre_id")
                    .merge(media,    on="media_type_id")
                )

                st.session_state.df      = df
                st.session_state.df_name = "iTunes Sales (merged)"
                st.success(f"Loaded iTunes data: {len(df)} rows")
            except Exception as e:
                st.error(f"Error loading iTunes data: {e}")

    st.divider()
    st.header("🤖 RAG Chatbot")

    is_itunes_active = st.session_state.df_name == "iTunes Sales (merged)"

    if is_itunes_active:
        rag_source = st.radio(
            "Index source",
            ["Current dataset (uploaded/merged)", "iTunes demo index (prebuilt)"],
            key="rag_source"
        )
    else:
        rag_source = "Current dataset (uploaded/merged)"

    if st.button("Build / Load RAG Index"):
        if rag_source == "Current dataset (uploaded/merged)":
            if st.session_state.df is None:
                st.error("Load a dataset first.")
            else:
                with st.spinner("Building index from your data (this may take a minute)..."):
                    try:
                        retriever = build_dynamic_index(st.session_state.df)
                        st.session_state.rag_chain = build_dynamic_chain(
                            retriever, dataset_name=st.session_state.df_name
                        )
                        st.session_state.rag_ready = True
                        st.session_state.messages = []  # reset chat history
                        st.success(f"Index built from {st.session_state.df_name}!")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            with st.spinner("Loading prebuilt iTunes index..."):
                try:
                    retriever = load_retriever()
                    st.session_state.rag_chain = build_chain(retriever)
                    st.session_state.rag_ready = True
                    st.session_state.messages = []
                    st.success("iTunes RAG index loaded!")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.header("🌍 Geo Analysis")

    if is_itunes_active:
        geo_source = st.radio(
            "Geo data source",
            ["Current dataset (uploaded/merged)", "iTunes demo data"],
            key="geo_source"
        )
    else:
        geo_source = "Current dataset (uploaded/merged)"

    geo_column = None
    geo_location = None
    geo_level = None

    if geo_source == "Current dataset (uploaded/merged)":
        if st.session_state.df is not None:
            df_cols = st.session_state.df.columns.tolist()
            default_col = guess_geo_column(df_cols)
            default_idx = df_cols.index(default_col) if default_col in df_cols else 0

            geo_column = st.selectbox(
                "Location/category column",
                df_cols,
                index=default_idx,
                key="geo_col"
            )

            unique_vals = (
                st.session_state.df[geo_column]
                .dropna().astype(str).unique().tolist()
            )
            unique_vals = sorted(unique_vals)[:300]

            geo_location = st.selectbox(
                "Value to analyze",
                unique_vals,
                key="geo_loc_val"
            )
        else:
            st.info("Load a dataset first.")
    else:
        geo_location = st.text_input("Location (country or city)", "Brazil")
        geo_level    = st.selectbox("Level", ["country", "city"])

    geo_question = st.text_area(
        "Question",
        "What insights and recommendations can you provide for this?"
    )
    run_geo = st.button("Run Geo Analysis")

# ── Main tabs ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Data Explorer",
    "🧹 Data Cleaning",
    "💬 BI Chat",
    "🌍 Geo Insights",
    "📈 Forecasting"
])

# ── Tab 1: Data Explorer ───────────────────────────────────
with tab1:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader(f"Dataset: {st.session_state.df_name}")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rows",        f"{len(df):,}")
        col2.metric("Columns",     len(df.columns))
        col3.metric("Null values", df.isnull().sum().sum())
        col4.metric("Duplicates",  df.duplicated().sum())

# Data quality score
        if st.session_state.quality_score is not None:
            col5.metric(
                "Quality Score",
                f"{st.session_state.quality_score}/100",
                st.session_state.quality_grade
            )
            with st.expander("📊 Quality Score Breakdown"):
                for k, v in st.session_state.quality_breakdown.items():
                    st.write(f"**{k}:** -{v} points")
        else:
            col5.metric("Quality Score", "—", "Generate insights first")

        # Null breakdown
        nulls = df.isnull().sum()
        nulls = nulls[nulls > 0]
        if not nulls.empty:
            st.warning(f"⚠️ Null values found in {len(nulls)} column(s):")
            null_df = pd.DataFrame({
                "Column": nulls.index,
                "Null Count": nulls.values,
                "% of Rows": [round(c / len(df) * 100, 1) for c in nulls.values]
            }).sort_values("Null Count", ascending=False)

            st.dataframe(
                null_df,
                use_container_width=True,
                height=min(35 * (len(null_df) + 1), 250),
                hide_index=True
            )
        else:
            st.success("✅ No null values found.")

# Auto Insights
        st.divider()
        col_insight, col_btn = st.columns([4, 1])
        with col_insight:
            st.subheader("🤖 Auto Insights")
        with col_btn:
            run_insights = st.button("✨ Generate Insights")

        if run_insights:
            with st.spinner("Analysing your dataset..."):
                try:
                    insights, profile = generate_auto_insights(
                        df, dataset_name=st.session_state.df_name
                    )
                    st.session_state.last_insights = insights
                    st.session_state.last_profile  = profile
                    score, grade, color, breakdown = compute_quality_score(df, profile)
                    st.session_state.quality_score     = score
                    st.session_state.quality_grade     = grade
                    st.session_state.quality_breakdown = breakdown
                    st.rerun()                  
                except Exception as e:
                    st.error(f"Error generating insights: {e}")

        if "last_insights" in st.session_state and st.session_state.last_insights:
            for line in st.session_state.last_insights.strip().split("\n"):
                line = line.strip()
                if line:
                    st.info(line)

        st.divider()
        st.dataframe(df.head(50), use_container_width=True)
        
        st.subheader("📊 Quick Visualizations")
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        if cat_cols and num_cols:
            col_a, col_b = st.columns(2)
            with col_a:
                x_col = st.selectbox("Category (X axis)", cat_cols, key="x")
            with col_b:
                y_col = st.selectbox("Metric (Y axis)", num_cols, key="y")

            chart_type = st.radio(
                "Chart type",
                ["Bar", "Line", "Scatter", "Box"],
                horizontal=True
            )

            agg = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False).head(15)

            if chart_type == "Bar":
                fig = px.bar(agg, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif chart_type == "Line":
                fig = px.line(agg, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif chart_type == "Scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            else:
                fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")

            st.plotly_chart(fig, use_container_width=True)

            # Map chart if country column exists
            country_cols = [c for c in cat_cols if "country" in c.lower()]
            if country_cols and num_cols:
                st.subheader("🗺️ Geographic Distribution")
                map_country = st.selectbox("Country column", country_cols)
                map_metric  = st.selectbox("Metric", num_cols, key="map_y")
                map_data    = df.groupby(map_country)[map_metric].sum().reset_index()
                fig_map     = px.choropleth(
                    map_data,
                    locations=map_country,
                    locationmode="country names",
                    color=map_metric,
                    title=f"{map_metric} by Country",
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("👈 Load a dataset from the sidebar to get started.")

# ── Tab 2: Data Cleaning ───────────────────────────────────
with tab2:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("🧹 AI-Powered Data Cleaning")
        st.write("Describe what you want to clean in plain English.")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Null values per column:**")
            nulls = df.isnull().sum()
            nulls = nulls[nulls > 0]
            if nulls.empty:
                st.success("No null values found!")
            else:
                st.dataframe(nulls.rename("null count"))

        with col2:
            st.write("**Data types:**")
            st.dataframe(df.dtypes.rename("dtype"))

        instruction = st.text_input(
            "Cleaning instruction",
            placeholder="e.g. fill empty composer fields with Unknown"
        )

        if st.button("Apply Cleaning") and instruction:
            with st.spinner("Analyzing and applying..."):
                try:
                    new_df, explanation = apply_cleaning(df, instruction)
                    st.session_state.df = new_df
                    st.success(f"Done: {explanation}")
                    st.dataframe(new_df.head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

        st.divider()
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            st.subheader("⬇️ Download Cleaned Data")
            csv = st.session_state.df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Cleaned CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime="text/csv"
            )

        with col_dl2:
            st.subheader("📄 Export Executive Report")
            if st.button("Generate Word Report"):
                if st.session_state.get("last_insights") is None:
                    st.warning("Generate Auto Insights first from the Data Explorer tab.")
                else:
                    with st.spinner("Building report..."):
                        try:
                            report_bytes = generate_report(
                                df=st.session_state.df,
                                dataset_name=st.session_state.df_name,
                                insights_text=st.session_state.last_insights,
                                profile=st.session_state.last_profile
                            )
                            st.download_button(
                                label="📥 Download Report (.docx)",
                                data=report_bytes,
                                file_name=f"BI_Report_{st.session_state.df_name.replace(' ', '_')}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                        except Exception as e:
                            st.error(f"Error generating report: {e}")
    else:
        st.info("👈 Load a dataset from the sidebar first.")

# ── Tab 3: BI Chat ─────────────────────────────────────────
with tab3:
    st.subheader("💬 Ask Business Questions")

    if not st.session_state.rag_ready:
        st.warning("Load the RAG index from the sidebar first.")
    else:
        col_chat, col_clear = st.columns([5, 1])
        with col_clear:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()

        chat_container = st.container(height=450)
        with chat_container:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

        question = st.chat_input("Ask a business question about your data...")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Build conversation context from history
                        history_context = ""
                        if len(st.session_state.messages) > 1:
                            history_context = "\n".join([
                                f"{m['role'].upper()}: {m['content']}"
                                for m in st.session_state.messages[:-1][-6:]
                            ])
                            question_with_context = f"""Previous conversation:
{history_context}

Current question: {question}

Answer the current question. If it refers to something from the conversation history (like 'that', 'those', 'the same', 'break it down', 'what about'), use the history to understand what it refers to."""
                        else:
                            question_with_context = question

                        answer, route = smart_query(
                            st.session_state.df,
                            question_with_context,
                            rag_chain=st.session_state.rag_chain
                        )
                        if route == "aggregate":
                            st.caption("🧮 Answered using exact calculation over full dataset")
                        elif route == "exploratory":
                            st.caption("🔍 Answered using retrieved sample records")
                    except Exception as e:
                        answer = f"Error: {e}"
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Tab 4: Geo Insights ────────────────────────────────────
with tab4:
    st.subheader("🌍 Geo / Category Insights")

    if run_geo:
        if geo_source == "Current dataset (uploaded/merged)":
            if st.session_state.df is None or geo_column is None:
                st.error("Load a dataset and select a column first.")
            else:
                with st.spinner(f"Analyzing '{geo_location}'..."):
                    try:
                        result, filtered = dynamic_geo_analysis(
                            st.session_state.df,
                            geo_column,
                            geo_location,
                            geo_question,
                            dataset_name=st.session_state.df_name
                        )
                        st.markdown(result)

                        if filtered is not None and not filtered.empty:
                            num_cols = filtered.select_dtypes(include="number").columns.tolist()
                            cat_cols = [
                                c for c in filtered.select_dtypes(include="object").columns
                                if c != geo_column
                            ]

                            if num_cols and cat_cols:
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    metric = st.selectbox("Metric", num_cols, key="geo_metric")
                                with col_b:
                                    group_col = st.selectbox("Group by", cat_cols, key="geo_group")

                                chart_data = (
                                    filtered.groupby(group_col)[metric]
                                    .sum().reset_index()
                                    .sort_values(metric, ascending=False)
                                    .head(15)
                                )
                                fig = px.bar(
                                    chart_data, x=group_col, y=metric,
                                    title=f"{metric} by {group_col} for {geo_location}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")

        else:
            with st.spinner(f"Analyzing {geo_location}..."):
                try:
                    result = geo_analysis(geo_location, geo_question, geo_level)
                    st.markdown(result)

                    invoice = pd.read_csv("data/itunes/invoice.csv")
                    if geo_level == "country":
                        filtered = invoice[
                            invoice["billing_country"].str.lower() == geo_location.lower()
                        ]
                        if not filtered.empty:
                            city_rev = filtered.groupby("billing_city")["total"].sum().reset_index()
                            fig = px.bar(
                                city_rev.sort_values("total", ascending=False),
                                x="billing_city", y="total",
                                title=f"Revenue by City in {geo_location}",
                                labels={"billing_city": "City", "total": "Revenue ($)"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        filtered = invoice[
                            invoice["billing_city"].str.lower() == geo_location.lower()
                        ]
                        if not filtered.empty:
                            filtered["invoice_date"] = pd.to_datetime(filtered["invoice_date"])
                            monthly = filtered.groupby(
                                filtered["invoice_date"].dt.to_period("M").astype(str)
                            )["total"].sum().reset_index()
                            fig = px.line(
                                monthly,
                                x="invoice_date", y="total",
                                title=f"Monthly Revenue in {geo_location}",
                                labels={"invoice_date": "Month", "total": "Revenue ($)"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Select a value in the sidebar and click 'Run Geo Analysis'.")

# ── Tab 5: Forecasting ─────────────────────────────────────
with tab5:
    st.subheader("📈 AI-Powered Forecasting")

    if st.session_state.df is None:
        st.info("👈 Load a dataset from the sidebar first.")
    else:
        df = st.session_state.df
        forecast_type, date_cols, num_cols = detect_forecast_type(df)

        if forecast_type is None:
            st.error("No numeric columns found for forecasting.")
        else:
            if forecast_type == "timeseries":
                st.success("📅 Date column detected — running Time-Series Forecasting")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    date_col = st.selectbox("Date column", date_cols)
                with col_b:
                    target_col = st.selectbox("Value to forecast", num_cols)
                with col_c:
                    periods = st.slider("Months to forecast", 1, 24, 6)
                total_months = len(pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").unique())
                if total_months < 24:
                    st.warning(f"⚠️ Only {total_months} months of data detected. Forecast accuracy may be limited — more historical data improves results.")
                if st.button("Run Forecast"):
                    with st.spinner("Training XGBoost model..."):
                        try:
                            ts, future_df, metrics, fig = run_timeseries(
                                df, date_col, target_col, periods
                            )
                            if ts is None:
                                st.error("Not enough data for time-series forecasting (need at least 6 months).")
                            else:
                                st.plotly_chart(fig, use_container_width=True)

                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                col_m1.metric("MAE",           metrics["MAE"])
                                col_m2.metric("R² Score",      metrics["R2"])
                                col_m3.metric("Training rows", metrics["Training rows"])
                                col_m4.metric("Test rows",     metrics["Test rows"])

                                st.subheader(f"Forecast for next {periods} months")
                                st.dataframe(
                                    future_df[["date", "forecast"]].rename(
                                        columns={"date": "Month", "forecast": f"Predicted {target_col}"}
                                    ),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                key_result = f"Forecast for next {periods} months: {future_df['forecast'].sum():.2f} total predicted {target_col}"
                                with st.spinner("Generating business interpretation..."):
                                    interpretation = interpret_forecast(
                                        "Time-Series XGBoost",
                                        target_col,
                                        metrics,
                                        key_result,
                                        st.session_state.df_name
                                    )
                                st.info(f"💡 {interpretation}")

                        except Exception as e:
                            st.error(f"Forecasting error: {e}")

            else:
                st.success("🔢 No date column detected — running Regression Forecasting")

                col_a, col_b = st.columns(2)
                with col_a:
                    target_col = st.selectbox("Target variable to predict", num_cols)
                with col_b:
                    feature_options = [c for c in num_cols if c != target_col]
                    feature_cols = st.multiselect(
                        "Feature columns (predictors)",
                        feature_options,
                        default=feature_options[:3]
                    )

                if st.button("Run Regression") and feature_cols:
                    with st.spinner("Training XGBoost regression model..."):
                        try:
                            results_df, metrics, fig = run_regression(
                                df, target_col, feature_cols
                            )
                            if results_df is None:
                                st.error("Not enough data for regression (need at least 20 rows).")
                            else:
                                st.plotly_chart(fig, use_container_width=True)

                                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                                col_m1.metric("MAE",           metrics["MAE"])
                                col_m2.metric("R² Score",      metrics["R2 Score"])
                                col_m3.metric("Training rows", metrics["Training rows"])
                                col_m4.metric("Test rows",     metrics["Test rows"])

                                st.subheader("Actual vs Predicted (test set)")
                                st.dataframe(
                                    results_df.head(20),
                                    use_container_width=True,
                                    hide_index=True
                                )

                                key_result = f"Top predictor: {feature_cols[0]}, MAE: {metrics['MAE']}, R2: {metrics['R2 Score']}"
                                with st.spinner("Generating business interpretation..."):
                                    interpretation = interpret_forecast(
                                        "XGBoost Regression",
                                        target_col,
                                        metrics,
                                        key_result,
                                        st.session_state.df_name
                                    )
                                st.info(f"💡 {interpretation}")

                        except Exception as e:
                            st.error(f"Regression error: {e}")

                            