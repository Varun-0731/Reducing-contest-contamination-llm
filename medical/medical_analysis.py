"""
Medical Report Analysis
-----------------------
Domain-specific output layer on top of the SAME retrieval pipeline.

The module does not create a second FAISS index or second RAG system.
It uses retrieval.retrieval_pipeline for PDF evidence and generation.
"""

import re
from retrieval import retrieval_pipeline as rag


def _clean(value):
    return rag.clean_text(value or "")


def _evidence_text(result):
    return rag.format_evidence(result.get("clean_evidence") or result.get("evidence") or [])


def extract_disease_conditions(text):
    text = _clean(text)
    if not text:
        return []

    patterns = [
        r"\b(?:diagnosis|diagnosed with|condition|disease|impression)\s*[:\-]\s*([^.;\n]+)",
        r"\b(?:suggestive of|consistent with|findings of)\s+([^.;\n]+)",
    ]
    found = []
    for pattern in patterns:
        for match in re.findall(pattern, text, re.I):
            value = _clean(match)
            if value and value not in found:
                found.append(value)
    return found[:10]


def extract_diagnosis_findings(text):
    text = _clean(text)
    if not text:
        return []

    labels = [
        "diagnosis", "findings", "impression", "clinical findings",
        "laboratory findings", "assessment"
    ]
    results = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        low = sentence.lower()
        if any(label in low for label in labels):
            sentence = _clean(sentence)
            if sentence and sentence not in results:
                results.append(sentence)
    return results[:12]


def extract_precautions_recommendations(text):
    text = _clean(text)
    if not text:
        return []

    results = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keywords = [
        "recommend", "recommended", "precaution", "precautions",
        "advised", "avoid", "follow-up", "follow up", "monitor",
        "treatment", "management", "consult", "review"
    ]
    for sentence in sentences:
        if any(k in sentence.lower() for k in keywords):
            sentence = _clean(sentence)
            if sentence and sentence not in results:
                results.append(sentence)
    return results[:12]


def classify_medical_question(question):
    q = _clean(question).lower()
    if any(x in q for x in ["precaution", "precautions", "recommendation",
                            "recommendations", "what should", "advised", "avoid"]):
        return "PRECAUTIONS"
    if any(x in q for x in ["diagnosis", "diagnosed", "finding", "findings",
                            "impression", "result", "results"]):
        return "DIAGNOSIS_FINDINGS"
    if any(x in q for x in ["disease", "condition", "disorder", "illness"]):
        return "DISEASE_CONDITION"
    return "GENERAL_MEDICAL"


def build_medical_prompt(question, evidence_text):
    return f"""
You are analyzing a medical report.

Question:
{question}

Document evidence:
{evidence_text}

Rules:
1. Use ONLY information supported by the supplied medical document.
2. Do not independently diagnose the patient.
3. Do not invent a disease, finding, treatment, precaution, dosage, or prognosis.
4. Distinguish documented findings from interpretations.
5. Preserve the terminology and numbers used in the report.
6. If the report does not support the requested information, say so.
7. Return a clear, concise answer.
"""


def generate_medical_analysis(question, result):
    evidence = _evidence_text(result)
    if not evidence:
        return "The requested medical information is not available in the provided report."

    prompt = build_medical_prompt(question, evidence)
    answer = rag.call_nim(prompt)

    if answer:
        return rag.clean_answer(answer)

    # Safe fallback: return the core RAG answer rather than inventing medical content.
    return rag.clean_answer(result.get("proposed_answer") or result.get("baseline_answer") or "")


def analyze_medical_question(question, save_to_csv=True):
    result = rag.compare_baseline_and_proposed(
        question,
        save_to_csv=save_to_csv,
    )
    result["medical_analysis"] = generate_medical_analysis(question, result)
    result["medical_question_type"] = classify_medical_question(question)
    return result


def answer_medical_question(question, return_metadata=False):
    result = analyze_medical_question(question)
    if return_metadata:
        return result
    return result["medical_analysis"]


def get_medical_summary():
    info = rag.get_current_pdf_info()
    evidence = rag.get_beginning_chunks(12)
    text = rag.format_evidence(evidence)
    if not text:
        return {"pdf_name": info["pdf_name"], "summary": ""}
    prompt = f"""
Summarize only the medically relevant information explicitly documented
in this report. Do not diagnose independently.

Report evidence:
{text}

Return a concise summary covering:
- documented disease/condition, if present
- documented diagnosis/findings
- documented precautions/recommendations
"""
    answer = rag.call_nim(prompt)
    return {
        "pdf_name": info["pdf_name"],
        "summary": rag.clean_answer(answer or ""),
    }


if __name__ == "__main__":
    print("MEDICAL CASE ANALYSIS MODULE")
    print("Module loaded successfully.")
    print("Available functions:")
    print("  analyze_medical_question()")
    print("  answer_medical_question()")
    print("  get_medical_summary()")
