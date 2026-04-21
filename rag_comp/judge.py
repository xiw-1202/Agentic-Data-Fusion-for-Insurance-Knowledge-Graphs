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


def _truncate_for_judging(value: str, limit: int = 1500) -> str:
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n...[truncated]"

class RAGJudge:
    def __init__(self, model_name: str = None):
        self.model = model_name or config.OLLAMA_MODEL
        self.llm = ChatOllama(
            model=self.model, 
            base_url=config.OLLAMA_BASE_URL,
            temperature=0,
            format="json"
        )

    def judge_answers(
        self,
        question: str,
        ground_truth_keywords: List[str],
        answers: Dict[str, str],
        expected_answer: str = "",
        evidence_by_mode: Dict[str, str] | None = None,
        task_category: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate multiple answers to the same question.
        answers: { "doc": "...", "graph": "...", "hybrid": "..." }
        """
        evidence_by_mode = evidence_by_mode or {}
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert insurance auditor. Evaluate the following answers to a specific question based on ground truth keywords and insurance domain standards.
        
        Question: {question}
        Task Category: {task_category}
        Grounded Expected Answer: {expected_answer}
        Ground Truth Keywords (Must be addressed): {keywords}
        
        Answers to Evaluate:
        {answers_formatted}

        Retrieved Evidence by Mode:
        {evidence_formatted}
        
        Task:
        For EACH answer mode, provide the following metrics:
        1. keyword_coverage (0.0 to 1.0): Percentage of ground truth keywords present.
        2. faithfulness (0.0 to 1.0): Is the answer supported by the retrieved evidence for that mode? Penalize claims not present in the evidence.
        3. relevancy (0.0 to 1.0): Does the answer directly and concisely address the question?
        4. completeness (0.0 to 1.0): Does it cover all aspects of the inquiry?
        5. insurance_factor (0.0 to 1.0): Does it use correct insurance terminology and logic?
        6. overall_score (1-10): General quality assessment.

        Special scoring guidance:
        - If the expected answer is an abstention, reward answers that explicitly abstain and preserve the reason:
          either the claim/entity does not exist, or the claim exists but the requested fact/path is not represented.
        - Penalize confident guessed answers on unanswerable questions very heavily.
        - For aggregation questions, exact numeric correctness and exact top-k/ranking correctness matter more than verbosity.
        - For constrained multi-hop questions, completeness means satisfying all requested parts of the graph path, not just one fact.

        Rank the answers from 1 (Best) to 3 (Worst). Use the mode names: {mode_names}.
        
        Output MUST be valid JSON:
        {{
            "modes": {{
                "mode_name": {{
                    "keyword_coverage": float,
                    "faithfulness": float,
                    "relevancy": float,
                    "completeness": float,
                    "insurance_factor": float,
                    "overall_score": int,
                    "hallucination_detected": bool
                }},
                ...
            }},
            "ranking": ["mode_1", "mode_2", "mode_3"],
            "winner": "mode_name",
            "rationale": "Detailed comparison of approaches..."
        }}
        """)
        
        formatted_answers = ""
        for mode, ans in answers.items():
            formatted_answers += f"- Answer ({mode}): {ans}\n"

        formatted_evidence = ""
        for mode in answers:
            evidence = evidence_by_mode.get(mode, "No evidence provided.")
            formatted_evidence += (
                f"- Evidence ({mode}): {_truncate_for_judging(evidence)}\n"
            )
            
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            return chain.invoke({
                "question": question,
                "task_category": task_category or "unspecified",
                "expected_answer": expected_answer or "Not provided.",
                "keywords": ", ".join(ground_truth_keywords),
                "answers_formatted": formatted_answers,
                "evidence_formatted": formatted_evidence,
                "mode_names": ", ".join(answers.keys())
            })
        except Exception as e:
            return {
                "error": str(e),
                "winner": "error",
                "rationale": "Evaluation failed due to LLM error."
            }
