# Personal AI Data Analyst

We have all been there. You have a messy CSV file, a deadline, and a head
full of questions. But instead of finding answers, you find yourself trying
to remember the exact syntax to reshape a Pandas DataFrame or change the
colour of a Matplotlib bar chart. What if you could just ask your data
questions?

**Personal AI Data Analyst** is a Streamlit app that lets you upload any CSV
and ask questions about it in plain English. Behind the scenes, an LLM reads
a summary of your data and writes the Pandas/Matplotlib code needed to
answer your question — which is then run automatically and shown to you as
a table, a number, or a chart.

## How it works

1. You upload a CSV file.
2. The app summarizes its schema, sample rows, and basic stats.
3. Your question + that summary are sent to an LLM (OpenAI), which writes
   Python code to answer it.
4. The code is executed in a restricted sandbox (only `pandas`, `numpy`,
   and `matplotlib` are available — no file or network access).
5. The result (table, value, or chart) is displayed in a chat-style UI,
   along with the generated code so you can learn from it.

## Setup

```bash
git clone https://github.com/ChennaBrindha/Personal-AI-Data-Analyst.git
cd Personal-AI-Data-Analyst
pip install -r requirements.txt
streamlit run app.py
```

You'll need an [OpenAI API key](https://platform.openai.com/api-keys),
entered in the sidebar at runtime. It's used only for that session and is
never stored.

## Tech stack

- **Streamlit** — chat-style web UI
- **Pandas / NumPy** — data handling
- **Matplotlib** — chart generation
- **OpenAI API** — natural language → code generation

## Project structure

```
.
├── app.py          # Streamlit UI: upload, chat, render results
├── analyst.py       # Core logic: prompt building, LLM calls, sandboxed execution
└── requirements.txt
```