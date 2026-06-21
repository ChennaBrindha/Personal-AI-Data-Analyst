"""
analyst.py
------------------
Core logic for the Personal AI Data Analyst.

Given a pandas DataFrame and a natural-language question, this module:
  1. Summarizes the dataframe (columns, dtypes, sample rows, stats)
  2. Asks an LLM to write the pandas/matplotlib code needed to answer it
  3. Executes that code in a restricted namespace and returns the result
     (a table, a single value, and/or a matplotlib figure)
"""

import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for Streamlit
import matplotlib.pyplot as plt
from openai import OpenAI


class DataAnalyst:
    """Wraps an LLM client around a DataFrame to answer natural-language
    questions by generating and running pandas/matplotlib code."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def summarize_dataframe(df: pd.DataFrame, n_rows: int = 5) -> str:
        """Builds a compact text summary of the dataframe so the LLM
        understands the schema without us sending the full dataset."""
        return (
            f"Columns and dtypes:\n{df.dtypes.to_string()}\n\n"
            f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n\n"
            f"First {n_rows} rows:\n{df.head(n_rows).to_string()}\n\n"
            f"Basic stats:\n{df.describe(include='all').to_string()}"
        )

    def _build_prompt(self, question: str, df_summary: str) -> str:
        return f"""You are a senior data analyst writing Python code.

You are given a pandas DataFrame called `df`. Here is a summary of it:

{df_summary}

The user asked this question about the data:
\"\"\"{question}\"\"\"

Write Python code that answers the question using the dataframe `df`.
Rules:
- Use only pandas, numpy and matplotlib (already imported as pd, np, plt).
- If the answer is a value or table, assign it to a variable called `result`.
- If the answer is best shown as a chart, build it with matplotlib and
  assign the current Figure object to a variable called `result_fig`.
- Do not read/write files, do not use input(), do not import other libraries.
- Do not include explanations, markdown, or comments outside the code.
- Return ONLY raw Python code, with no markdown code fences.
"""

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        code = code.strip()
        if code.startswith("```"):
            code = code.strip("`")
            if code.lower().startswith("python"):
                code = code[len("python"):]
        return code.strip()

    # ---------------------------------------------------------------
    # Core methods
    # ---------------------------------------------------------------

    def generate_code(self, question: str, df: pd.DataFrame) -> str:
        """Asks the LLM to translate a natural-language question into
        runnable pandas/matplotlib code."""
        df_summary = self.summarize_dataframe(df)
        prompt = self._build_prompt(question, df_summary)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You only output runnable Python code."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return self._strip_code_fences(response.choices[0].message.content)

    @staticmethod
    def run_code(code: str, df: pd.DataFrame) -> dict:
        """Executes generated code in a restricted namespace and returns
        whatever it produced: a table/value, a chart, or an error."""
        plt.close("all")  # don't bleed old charts into a new answer

        safe_globals = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "df": df,
            "__builtins__": {
                "len": len, "range": range, "min": min, "max": max,
                "sum": sum, "sorted": sorted, "round": round, "list": list,
                "dict": dict, "str": str, "int": int, "float": float,
                "abs": abs, "enumerate": enumerate, "zip": zip,
            },
        }
        local_vars = {}

        try:
            exec(code, safe_globals, local_vars)
        except Exception as e:
            return {"error": str(e), "code": code}

        output = {"code": code}
        if "result" in local_vars:
            output["result"] = local_vars["result"]
        if "result_fig" in local_vars:
            output["result_fig"] = local_vars["result_fig"]
        elif plt.get_fignums():
            # Model drew a chart but forgot to assign result_fig
            output["result_fig"] = plt.gcf()

        return output

    def ask(self, question: str, df: pd.DataFrame) -> dict:
        """End-to-end: question -> generated code -> executed result."""
        code = self.generate_code(question, df)
        return self.run_code(code, df)