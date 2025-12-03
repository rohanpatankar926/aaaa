from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ONTOLOGY_PATH = DATA_DIR / "ontology.json"
EXCEL_PATH = DATA_DIR / "DataHub-Core.xlsm"

@dataclass
class TableInfo:
    name: str
    description: str
    domain: str
    concepts: List[str]


def _get_selector_llm() -> ChatOpenAI:
    """LLM used specifically for table selection (non-streaming)."""

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        streaming=False,
    )


def _load_raw_ontology() -> Dict[str, Any]:
    """Load the ontology JSON from disk.

    Expected structure is compatible with the example you shared.
    """
    with ONTOLOGY_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_tables_metadata() -> List[TableInfo]:
    """Flatten ontology into a list of TableInfo objects."""
    raw = _load_raw_ontology()
    domain_context = raw.get("domain_context", "")
    tables: List[TableInfo] = []

    for domain in raw.get("domains", []):
        domain_name = domain.get("domain_name", "")
        dom_desc = domain.get("description", "")
        covered = domain.get("covered_tables", []) or []

        concept_names: List[str] = []
        for concept in domain.get("concepts", []) or []:
            cname = concept.get("concept")
            if cname:
                concept_names.append(cname)

        for tname in covered:
            tables.append(
                TableInfo(
                    name=tname,
                    description=f"{domain_context}. {dom_desc}",
                    domain=domain_name,
                    concepts=concept_names,
                )
            )

    return tables


def choose_tables_for_query(query: str, candidates: Iterable[TableInfo]) -> List[TableInfo]:
    """Use an LLM to decide which tables are relevant for this query.

    The model receives a compact description of each table and must return a JSON
    array of table names, e.g.:

        ["datahub - utility consumptions", "powermanager electricity - week"]

    If parsing fails or the model returns nothing useful, we fall back to using
    all available tables.
    """

    tables = list(candidates)
    if not tables:
        return []

    table_blocks: List[str] = []
    for idx, t in enumerate(tables, start=1):
        concepts = ", ".join(t.concepts) if t.concepts else "N/A"
        block = (
            f"{idx}. name: {t.name}\n"
            f"   domain: {t.domain}\n"
            f"   concepts: {concepts}\n"
            f"   description: {t.description}"
        )
        table_blocks.append(block)

    tables_text = "\n\n".join(table_blocks)

    system_msg = SystemMessage(
        content=(
            "You are a data modeling assistant for a brewery's utilities and manufacturing data platform. "
            "You are given a user analytics question and a list of available tables with metadata. "
            "Your job is to choose ONLY the tables that are clearly relevant for answering the question.\n\n"
            "Return STRICTLY a JSON array of table name strings, with no extra text, e.g.:\n"
            '  [\"table_a\", \"table_b\"]'
        )
    )

    human_msg = HumanMessage(
        content=(
            f"User question:\n{query}\n\n"
            f"Available tables:\n{tables_text}\n\n"
            "Respond ONLY with a JSON array of table names that should be used."
        )
    )

    llm = _get_selector_llm()
    # try:
    resp = llm.invoke([system_msg, human_msg])
    content = resp.content
    if isinstance(content, list):
        raw_text = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    else:
        raw_text = str(content)
    
    raw_text = raw_text.replace("```json", "").replace("```", "")
    raw_text = raw_text.replace("json", "").replace("json", "")
    raw_text = raw_text.replace("json", "").replace("json", "")

    selected_names = json.loads(raw_text)
    if not isinstance(selected_names, list):
        raise ValueError("LLM did not return a list")

    norm_names: List[str] = []
    for item in selected_names:
        if isinstance(item, str):
            norm_names.append(item)
        elif isinstance(item, dict) and "table_name" in item:
            norm_names.append(str(item["table_name"]))

    name_to_table = {t.name: t for t in tables}
    chosen = [name_to_table[name] for name in norm_names if name in name_to_table]
    return chosen

    # except Exception:
    #     return tables

