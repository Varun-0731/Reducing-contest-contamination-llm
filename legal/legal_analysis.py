"""
Legal Court Case Analysis
-------------------------
Domain-specific output layer on top of the SAME retrieval pipeline.
"""

import re
from retrieval import retrieval_pipeline as rag


def _clean(value):
    return rag.clean_text(value or "")


def _evidence(result):
    return rag.format_evidence(result.get("clean_evidence") or result.get("evidence") or [])


def extract_ipc_sections(text):
    text = _clean(text)
    if not text:
        return []

    patterns = [
        r"\bSection\s+([0-9]+[A-Za-z]?)\s+(?:of\s+)?IPC\b",
        r"\bIPC\s+Section\s+([0-9]+[A-Za-z]?)\b",
        r"\bIPC\s+([0-9]+[A-Za-z]?)\b",
        r"\bSec\.?\s*([0-9]+[A-Za-z]?)\s+IPC\b",
    ]
    found = []
    for pattern in patterns:
        for number in re.findall(pattern, text, re.I):
            section = f"Section {number} IPC"
            if section not in found:
                found.append(section)
    return found


def extract_case_facts(text):
    text = _clean(text)
    if not text:
        return []

    keywords = [
        "facts", "alleged", "allegation", "accused", "complaint",
        "incident", "case", "prosecution", "petitioner", "respondent"
    ]
    results = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if any(k in sentence.lower() for k in keywords):
            sentence = _clean(sentence)
            if sentence and sentence not in results:
                results.append(sentence)
    return results[:15]


def extract_punishment_sentence(text):
    text = _clean(text)
    if not text:
        return []

    keywords = [
        "sentence", "sentenced", "punishment", "imprisonment",
        "rigorous imprisonment", "simple imprisonment", "fine",
        "convicted", "acquitted", "life imprisonment"
    ]
    results = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if any(k in sentence.lower() for k in keywords):
            sentence = _clean(sentence)
            if sentence and sentence not in results:
                results.append(sentence)
    return results[:12]


def extract_case_title(text):
    text = _clean(text)
    if not text:
        return ""
    lines = [x.strip() for x in text.split(".") if x.strip()]
    return lines[0][:250] if lines else ""


def classify_legal_question(question):
    q = _clean(question).lower()
    if any(x in q for x in ["ipc", "section", "sections", "penal code"]):
        return "IPC_SECTIONS"
    if any(x in q for x in ["punishment", "sentence", "sentenced",
                            "imprisonment", "fine", "penalty"]):
        return "PUNISHMENT_SENTENCE"
    if any(x in q for x in ["facts", "allegation", "allegations",
                            "cause", "incident", "accused", "case"]):
        return "CASE_FACTS"
    return "GENERAL_LEGAL"


def build_legal_prompt(question, evidence_text):
    return f"""
You are analyzing a legal court-case document.

Question:
{question}

Document evidence:
{evidence_text}

Rules:
1. Use ONLY information supported by the supplied court document.
2. Preserve the legal terminology used in the document.
3. Clearly distinguish allegations from findings/judgment when the document does.
4. Do not invent facts.
5. Do not infer punishment merely from an IPC section.
6. Report a sentence/punishment only when it is actually documented.
7. If the requested information is not supported, say so.
8. Return a clear, concise answer.
"""


def generate_legal_analysis(question, result):
    evidence = _evidence(result)
    if not evidence:
        return "The requested legal information is not available in the provided document."

    answer = rag.call_nim(
        build_legal_prompt(question, evidence)
    )

    return rag.clean_answer(
        answer or result.get("proposed_answer") or result.get("baseline_answer") or ""
    )


def analyze_legal_question(question, save_to_csv=True):
    result = rag.compare_baseline_and_proposed(
        question,
        save_to_csv=save_to_csv,
    )
    result["legal_analysis"] = generate_legal_analysis(question, result)
    result["legal_question_type"] = classify_legal_question(question)
    return result


def answer_legal_question(question, return_metadata=False):
    result = analyze_legal_question(question)
    if return_metadata:
        return result
    return result["legal_analysis"]


def get_legal_summary():
    info = rag.get_current_pdf_info()
    evidence = rag.get_beginning_chunks(12)
    text = rag.format_evidence(evidence)
    if not text:
        return {"pdf_name": info["pdf_name"], "summary": ""}

    prompt = f"""
Summarize only the legal information explicitly documented in this court
case document. Do not infer facts, liability, or punishment.

Evidence:
{text}

Return:
- case title, if documented
- IPC sections, if documented
- material facts/allegations
- documented judgment/sentence/punishment, if present
"""
    answer = rag.call_nim(prompt)
    return {
        "pdf_name": info["pdf_name"],
        "summary": rag.clean_answer(answer or ""),
    }


if __name__ == "__main__":
    print("LEGAL CASE ANALYSIS MODULE")
    print("Module loaded successfully.")
    print("Available functions:")
    print("  analyze_legal_question()")
    print("  answer_legal_question()")
    print("  extract_ipc_sections()")
    print("  extract_case_facts()")
    print("  extract_punishment_sentence()")
    print("  get_legal_summary()")
