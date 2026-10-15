import os
import re
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 1. Page Configuration
st.set_page_config(
    page_title="Finding Nemo-tron | AI Copilot",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Sleek Custom CSS (Glassmorphism & Sky Blue Glow)
st.markdown(
    """
<style>
    /* Global font & background polish */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Header hero styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(17, 20, 27, 0.8) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
    }
    .hero-tag {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f8fafc 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* KPI Stat Cards */
    .metric-card {
        background: #11141b;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .metric-badge {
        color: #38bdf8;
        font-size: 0.8rem;
        margin-top: 2px;
    }

    /* File uploader outline accent */
    [data-testid="stFileUploader"] {
        border: 1px dashed rgba(56, 189, 248, 0.4);
        border-radius: 12px;
        padding: 10px;
        background: rgba(17, 20, 27, 0.4);
    }
</style>
""",
    unsafe_allow_html=True,
)

# 3. Environment & Nebius Client
load_dotenv()
api_key = os.getenv("NEBIUS_API_KEY")

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=api_key,
)

# Initialize Chat History early so sidebar can access it
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Sidebar Configuration
with st.sidebar:
    st.markdown("### 🪸 Control Center")
    uploaded_file = st.file_uploader(
        "Upload Dataset (.csv, .xlsx)", type=["csv", "xlsx", "xls"], help="Maximum 200MB"
    )

    st.markdown("---")
    st.markdown("### 🧠 Engine Settings")
    selected_model = st.selectbox(
        "Model Architecture",
        [
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            "nvidia/nemotron-3-super-120b-a12b",
            "nvidia/Nemotron-3-Ultra-550b-a55b",
        ],
        index=0,
    )
    # Defaulting temperature to 0.1 for more precise code generation
    temperature = st.slider("Reasoning Focus", 0.0, 1.0, 0.1, step=0.05)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Add Transcript Download Button
    if len(st.session_state.messages) > 0:
        chat_text = "\n\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]
        )
        st.download_button(
            label="📥 Download Audit Transcript",
            data=chat_text,
            file_name="finding_nemotron_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

# 5. Hero Header Section
st.markdown(
    """
<div class="hero-container">
    <span class="hero-tag">Nebius x NVIDIA Token Factory</span>
    <h1 class="hero-title">Finding Nemo-tron</h1>
    <p class="hero-sub">Spotting the anomalies in your anemones.</p>
</div>
""",
    unsafe_allow_html=True,
)

# 6. Main Workflow
if uploaded_file is not None:
    # Detect file type and load accordingly
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">Observations</div>
            <div class="metric-value">{df.shape[0]:,}</div>
            <div class="metric-badge">Total Rows</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">Dimensionality</div>
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-badge">Columns / Features</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">Data Completeness</div>
            <div class="metric-value">{100 - (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%</div>
            <div class="metric-badge">{df.isna().sum().sum():,} Null cells</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">Memory Footprint</div>
            <div class="metric-value">{df.memory_usage().sum() / 1024:.1f} KB</div>
            <div class="metric-badge">Client In-Memory</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Inspection Tabs
    tab_inspect, tab_viz, tab_summary = st.tabs(
        ["📊 Data Preview", "📈 Instant Visualizer", "📑 Numeric Diagnostics"]
    )

    with tab_inspect:
        st.dataframe(df.head(15), use_container_width=True)

    with tab_viz:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if len(numeric_cols) > 0:
            v_col1, v_col2 = st.columns([1, 3])
            with v_col1:
                selected_num = st.selectbox(
                    "Select variable to plot:", numeric_cols
                )
            with v_col2:
                fig = px.histogram(
                    df,
                    x=selected_num,
                    nbins=30,
                    color_discrete_sequence=["#38bdf8"],
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="#08090c",
                    plot_bgcolor="#11141b",
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for automated plotting.")

    with tab_summary:
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")
    st.subheader("💬 Interactive Data Analyst")

    # Quick Suggestion Chips
    q1, q2, q3 = st.columns(3)
    preset_query = None
    if q1.button(
        "✨ Key Trends & Outliers",
        use_container_width=True,
        help="Run general pattern discovery",
    ):
        preset_query = "Summarize the key trends, distributions, and any noticeable outliers."
    if q2.button(
        "🔍 Data Quality Audit",
        use_container_width=True,
        help="Scan for nulls and anomalies",
    ):
        preset_query = (
            "Analyze data completeness, null rates, and potential anomalies."
        )
    if q3.button(
        "🎯 Executive Takeaway",
        use_container_width=True,
        help="Business takeaway summary",
    ):
        preset_query = "What are the 3 most critical business or operational takeaways from this dataset?"

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(
            msg["role"],
            avatar="🐠" if msg["role"] == "assistant" else "👤",
        ):
            st.markdown(msg["content"])

    # Chat Input Box
    user_input = st.chat_input("Ask any question about your dataset...")
    active_prompt = preset_query or user_input

    if active_prompt:
        # 1. Render user message & append to state
        st.session_state.messages.append(
            {"role": "user", "content": active_prompt}
        )
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_prompt)

        # 2. Query Nemotron with Agentic Prompting
        with st.chat_message("assistant", avatar="🐠"):
            with st.spinner("Nemotron is analyzing the dataset..."):
                data_summary = f"""
                Columns & Types: {df.dtypes.to_dict()}
                First 3 Sample Rows: {df.head(3).to_dict(orient='records')}
                """

                # System prompt instructing the model to write Pandas code when needed
                system_prompt = (
                    "You are a Python Data Analyst Agent powered by NVIDIA Nemotron. "
                    "You have a pandas dataframe named `df` containing the user's data. "
                    "If a user asks a general question, answer it directly. "
                    "If a user asks a specific question requiring row-level data (like finding a specific item, "
                    "filtering by category, or calculating an exact max/min), you MUST output a single-line "
                    "Python expression using `df` that evaluates to the answer. "
                    "Wrap your Python expression in exactly this format: ```python\n[your code here]\n```"
                )

                # Build API messages payload with full context
                api_messages = [
                    {
                        "role": "system", 
                        "content": f"{system_prompt}\n\nDataset Summary:\n{data_summary}"
                    }
                ]
                
                # Append all stored chat history
                for m in st.session_state.messages:
                    api_messages.append({"role": m["role"], "content": m["content"]})

                try:
                    response = client.chat.completions.create(
                        model=selected_model,
                        messages=api_messages,
                        temperature=temperature,
                    )
                    answer = response.choices[0].message.content
                    
                    # 3. Agent Execution Loop: Check if the model wrote code
                    code_match = re.search(r'```python\n(.*?)\n```', answer, re.DOTALL)
                    
                    if code_match:
                        code_to_run = code_match.group(1).strip()
                        st.markdown(f"**⚡ Agent generated code:**\n```python\n{code_to_run}\n```")
                        
                        try:
                            # Safely evaluate the pandas expression
                            result = eval(code_to_run, {"df": df, "pd": pd})
                            st.success(f"**Result:**\n{result}")
                            
                            # Save the execution to history
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"*(Ran code: `{code_to_run}`)*\n\n**Answer:** {result}"}
                            )
                        except Exception as eval_err:
                            st.error(f"Code execution failed: {eval_err}")
                            st.session_state.messages.append({"role": "assistant", "content": f"Failed to run code: {eval_err}"})
                    else:
                        # If no code was generated, print the standard text answer
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )
                        
                except Exception as e:
                    st.error(f"Error communicating with Nebius: {e}")

else:
    # Clean Empty State
    st.markdown(
        """
    <div style="text-align: center; padding: 60px 20px; background: #11141b; border-radius: 16px; border: 1px dashed rgba(56, 189, 248, 0.2);">
        <h3 style="color: #f8fafc; margin-bottom: 8px;">No Dataset Loaded</h3>
        <p style="color: #64748b; font-size: 0.95rem; max-width: 480px; margin: 0 auto 20px auto;">
            Upload any CSV spreadsheet using the sidebar control to inspect schemas, visualize distributions, and interrogate your data using Nemotron.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )