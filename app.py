"""
app.py
------------------
Streamlit front-end for the Personal AI Data Analyst.
Upload a CSV, then ask questions about it in plain English.
"""

import streamlit as st
import pandas as pd
from analyst import DataAnalyst

st.set_page_config(page_title="Personal AI Data Analyst", page_icon="📊", layout="wide")

st.title("📊 Personal AI Data Analyst")
st.caption(
    "Upload a CSV and ask questions about it in plain English — "
    "no Pandas syntax required."
)

# ---------------------------------------------------------------
# Sidebar: API key + model
# ---------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Used only for this session. Never stored or logged.",
    )
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    st.markdown("---")
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
    st.markdown("Built with Streamlit, Pandas & the OpenAI API.")

# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: question + code + result/result_fig/error

# ---------------------------------------------------------------
# File upload
# ---------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read this CSV: {e}")
        st.stop()

    st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows x {df.shape[1]} columns")
    with st.expander("Preview data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    question = st.chat_input("Ask a question about your data...")

    if question:
        if not api_key:
            st.warning("Please add your OpenAI API key in the sidebar first.")
        else:
            with st.spinner("Thinking..."):
                analyst = DataAnalyst(api_key=api_key, model=model)
                output = analyst.ask(question, df)
            st.session_state.chat_history.append({"question": question, **output})

    # -----------------------------------------------------------
    # Render chat history (most recent first)
    # -----------------------------------------------------------
    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            if "error" in turn:
                st.error(f"Something went wrong: {turn['error']}")
                with st.expander("Generated code"):
                    st.code(turn["code"], language="python")
            else:
                if "result_fig" in turn:
                    st.pyplot(turn["result_fig"])
                if "result" in turn:
                    result = turn["result"]
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        st.dataframe(result, use_container_width=True)
                    else:
                        st.write(result)
                if "result" not in turn and "result_fig" not in turn:
                    st.info("No explicit result was returned — check the generated code below.")
                with st.expander("Show generated code"):
                    st.code(turn["code"], language="python")
else:
    st.info("👆 Upload a CSV file to get started.")