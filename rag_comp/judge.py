import os
import sys
import json
from typing import Dict, Any, List

# Allow imports from project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import config
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class RAGJudge:
    def __init__(self, model_name: str = None):
        self.model = model_name or config.OLLAMA_MODEL
        self.llm = ChatOllama(
            model=self.model, 
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json"
        )

    def judge_answers(self, question: str, ground_truth_keywords: List[str], answers: Dict[str, str]) -> Dict[str, Any]:
        """
        Evaluate multiple answers to the same question.
        answers: { "doc": "...", "graph": "...", "hybrid": "..." }
        """
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert insurance auditor. Evaluate the following answers to a specific question based on ground truth keywords.
        
        Question: {question}
        Ground Truth Keywords (Must be addressed): {keywords}
        
        Answers to Evaluate:
        {answers_formatted}
        
        Task:
        1. Calculate keyword_coverage for each answer (percentage of ground truth keywords present).
        2. Assign a score (1-10) for each based on accuracy, completeness, and grounding.
        3. Rank the answers from 1 (Best) to 3 (Worst). Use the mode names: {mode_names}.
        4. Detect potential hallucinations.
        
        Output MUST be valid JSON:
        {{
            "keyword_coverage": {{ "mode": float, ... }},
            "scores": {{ "mode": int, ... }},
            "ranking": ["mode_1", "mode_2", "mode_3"],
            "winner": "mode_name",
            "rationale": "...",
            "hallucination_detected": {{ "mode": bool, ... }}
        }}
        """)
        
        formatted_answers = ""
        for mode, ans in answers.items():
            formatted_answers += f"- Answer ({mode}): {ans}\n"
            
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            return chain.invoke({
                "question": question,
                "keywords": ", ".join(ground_truth_keywords),
                "answers_formatted": formatted_answers,
                "mode_names": ", ".join(answers.keys())
            })
        except Exception as e:
            return {
                "error": str(e),
                "winner": "error",
                "rationale": "Evaluation failed due to LLM error."
            }
