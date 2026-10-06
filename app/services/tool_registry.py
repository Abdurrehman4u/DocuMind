"""
Tool Registry — registers available agent tools and their JSON schemas for LLM function calling.
"""

from __future__ import annotations

from typing import Any
from sqlalchemy.orm import Session
from google.genai import types

from app.tools.search_kb_tool import search_knowledge_base
from app.tools.db_query_tool import query_structured_data
from app.tools.report_tool import generate_report

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "search_knowledge_base",
        "description": "Search the user's uploaded documents for semantic knowledge and relevant text excerpts.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Semantic search query string.",
                },
                "k": {
                    "type": "INTEGER",
                    "description": "Number of context chunks to return (default 4).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_structured_data",
        "description": "Safely query structured document metadata and system metrics using whitelisted queries. Arbitrary SQL is not permitted.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query_type": {
                    "type": "STRING",
                    "description": "Type of query to run: 'document_summary' (totals & status breakdown), 'document_counts_by_status', or 'recent_documents'.",
                },
                "limit": {
                    "type": "INTEGER",
                    "description": "Maximum number of records to return for list queries (default 10).",
                },
            },
            "required": ["query_type"],
        },
    },
    {
        "name": "generate_report",
        "description": "Format synthesized facts, findings, and analysis into a clean, structured Markdown report.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "title": {
                    "type": "STRING",
                    "description": "Title of the report.",
                },
                "summary": {
                    "type": "STRING",
                    "description": "Executive summary or overview.",
                },
                "sections": {
                    "type": "ARRAY",
                    "description": "List of report sections with 'heading' and 'content'.",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "heading": {"type": "STRING", "description": "Section title"},
                            "content": {"type": "STRING", "description": "Section markdown content"},
                        },
                        "required": ["heading", "content"],
                    },
                },
            },
            "required": ["title", "summary", "sections"],
        },
    },
]

TOOL_CALLABLES = {
    "search_knowledge_base": search_knowledge_base,
    "query_structured_data": query_structured_data,
    "generate_report": generate_report,
}


def get_gemini_tools() -> list[types.Tool]:
    """Convert registered tool declarations into google.genai types.Tool."""
    func_decls = []
    for tool_def in TOOL_DEFINITIONS:
        func_decls.append(
            types.FunctionDeclaration(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
            )
        )
    return [types.Tool(function_declarations=func_decls)]


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    user_id: str,
    db: Session,
) -> Any:
    """Dispatch execution of a tool by name with tenant isolation."""
    fn = TOOL_CALLABLES.get(name)
    if not fn:
        return {"error": f"Tool '{name}' is not recognized."}

    try:
        return fn(user_id=user_id, db=db, **arguments)
    except Exception as exc:
        return {"error": f"Tool execution failed: {str(exc)}"}
