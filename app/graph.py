from __future__ import annotations

from typing import Iterable, List, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from .ontology import EXCEL_PATH, choose_tables_for_query, load_tables_metadata

load_dotenv()


class GraphState(TypedDict):

    messages: List[BaseMessage]
    user_query: str
    selected_tables: List[str]
    table_summaries: str


def _get_llm() -> ChatOpenAI:

    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        streaming=True,
    )


llm = _get_llm()

builder: StateGraph[GraphState] = StateGraph(GraphState)


def select_tables(state: GraphState) -> GraphState:
    query = state["user_query"]
    all_tables = load_tables_metadata()
    chosen = choose_tables_for_query(query, all_tables)
    print(chosen)
    print("--------------------------------")
    return {
        **state,
        "selected_tables": [t.name for t in chosen],
    }


def load_and_summarize_tables(state: GraphState) -> GraphState:
    summaries: List[str] = []
    for sheet_name in state["selected_tables"]:
        try:
            print(sheet_name)
            print(EXCEL_PATH)
            excel_file = pd.ExcelFile(EXCEL_PATH)
            df = excel_file.parse(sheet_name)
            print(df)
            input("dasdadd")
        except Exception as exc:
            summaries.append(f"Table `{sheet_name}` could not be loaded: {exc}")
            continue

        head_str = df.head(5).to_markdown(index=False)
        summaries.append(
            f"Table `{sheet_name}` sample (first 5 rows):\n{head_str}\n"
        )

    table_summaries = "\n\n".join(summaries) if summaries else "No tables loaded."
    print(table_summaries)
    print("--------------------------------")
    return {
        **state,
        "table_summaries": table_summaries,
    }


def run_model(state: GraphState) -> GraphState:

    system_prompt = (
        "You are a data assistant for a brewery utilities and manufacturing team. "
        "You are given:\n"
        "1) A natural-language user question about brewery utilities/energy/production.\n"
        "2) Table samples from an Excel-based data mart.\n\n"
        "Use ONLY the information in the table summaries to answer. "
        "Explain which tables were most important and highlight key KPIs, trends, "
        "or anomalies. If information is missing, say so explicitly."
    )

    table_context = state.get("table_summaries", "")
    user_query = state["user_query"]

    messages = [
        HumanMessage(
            content=(
                f"{system_prompt}\n\n"
                f"User question:\n{user_query}\n\n"
                f"Available table samples:\n{table_context}"
            )
        )
    ]

    response = llm.invoke(messages)
    return {"messages": [*state["messages"], response], "user_query": user_query, "selected_tables": state["selected_tables"], "table_summaries": table_context}


builder.add_node("select_tables", select_tables)
builder.add_node("load_and_summarize_tables", load_and_summarize_tables)
builder.add_node("call_model", run_model)

builder.set_entry_point("select_tables")
builder.add_edge("select_tables", "load_and_summarize_tables")
builder.add_edge("load_and_summarize_tables", "call_model")
builder.add_edge("call_model", END)

compiled_graph = builder.compile()


def stream_prompt(prompt: str) -> Iterable[str]:

    initial_state: GraphState = {
        "messages": [HumanMessage(content=prompt)],
        "user_query": prompt,
        "selected_tables": [],
        "table_summaries": "",
    }
    for update in compiled_graph.stream(initial_state, stream_mode="values"):
        final_message = update["messages"][-1]
        content = final_message.content
        if isinstance(content, str):
            yield content
        else:
            # langchainmight return a list of message parts; string-ify for demo purposes
            yield "".join(part["text"] for part in content if isinstance(part, dict))

