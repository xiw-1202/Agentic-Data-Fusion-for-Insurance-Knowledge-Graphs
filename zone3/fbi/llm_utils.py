"""Shared LLM calling helpers for the FBI pipeline."""

from __future__ import annotations

import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

from langchain_ollama import ChatOllama


def llm_call(
    prompt: str,
    model: str | None = None,
    json_mode: bool = False,
    num_predict: int = 4096,
) -> str:
    """Make a single LLM call via ChatOllama, return stripped response text.

    Parameters
    ----------
    prompt : str
        The prompt to send.
    model : str | None
        Ollama model name. Defaults to ``config.OLLAMA_MODEL``.
    json_mode : bool
        If True, request JSON-formatted output from the model.
    num_predict : int
        Maximum tokens to generate.

    Returns
    -------
    str
        The stripped text content of the LLM response.
    """
    kwargs: dict = {
        "model": model or config.OLLAMA_MODEL,
        "base_url": config.OLLAMA_BASE_URL,
        "temperature": 0,
        "num_predict": num_predict,
    }
    if json_mode:
        kwargs["format"] = "json"

    llm = ChatOllama(**kwargs)
    response = llm.invoke(prompt)
    return response.content.strip()


def llm_call_json(
    prompt: str,
    model: str | None = None,
    num_predict: int = 4096,
) -> dict | list:
    """Make an LLM call with json_mode and parse the response.

    Falls back through several extraction strategies:
    1. Direct ``json.loads`` on the full response.
    2. Regex extraction from markdown code blocks (```json ... ```).
    3. Find the first ``{`` or ``[`` and parse from there.

    Parameters
    ----------
    prompt : str
        The prompt to send.
    model : str | None
        Ollama model name. Defaults to ``config.OLLAMA_MODEL``.
    num_predict : int
        Maximum tokens to generate.

    Returns
    -------
    dict | list
        Parsed JSON object.

    Raises
    ------
    json.JSONDecodeError
        If all parsing strategies fail.
    """
    raw = llm_call(prompt, model=model, json_mode=True, num_predict=num_predict)

    # Strategy 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract from markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: find first { or [
    for char in ("{", "["):
        idx = raw.find(char)
        if idx != -1:
            try:
                return json.loads(raw[idx:])
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("No valid JSON found in LLM response", raw, 0)


def parse_arrow_mapping(text: str) -> dict[str, str]:
    """Parse ``KEY -> Value`` or ``KEY → Value`` lines from LLM output.

    Lines that do not contain an arrow token are silently skipped.

    Parameters
    ----------
    text : str
        Raw LLM output containing arrow-separated key-value pairs.

    Returns
    -------
    dict[str, str]
        Mapping of stripped keys to stripped values.
    """
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        # Match both unicode arrow (→) and ASCII arrow (->)
        match = re.match(r"^\s*(.+?)\s*(?:→|->)\s*(.+?)\s*$", line)
        if match:
            mapping[match.group(1)] = match.group(2)
    return mapping
