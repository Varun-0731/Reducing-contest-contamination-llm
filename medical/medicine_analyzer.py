"""
Medicine Analyzer
-----------------
Given a medical PDF containing information about a medicine/drug, this module:
  1. Extracts the primary medicine name from the document.
  2. Generates:
     - Alternative / substitute medicines
     - Side effects
     - Use cases / indications
     - Mechanism of action (how it helps cure the condition)

It reuses the same FAISS RAG pipeline -- no second vector store is created.
"""

import re
from retrieval import retrieval_pipeline as rag


# --------------------------------------------------
# Helper utilities
# --------------------------------------------------

def _clean(value: str) -> str:
    return rag.clean_text(value or "")


def _evidence_text(result: dict) -> str:
    return rag.format_evidence(
        result.get("clean_evidence") or result.get("evidence") or []
    )


def _call_llm(prompt: str) -> str:
    """Wrapper around the shared NVIDIA NIM call."""
    raw = rag.call_gemini(prompt)
    if not raw:
        return ""
    # Note: Do NOT use rag.clean_answer here because it strips newlines,
    # headers, and bullet formatting into a single flat paragraph.
    return raw.strip()


def _fetch_evidence_for_drug(drug_name: str) -> str:
    """
    Retrieve focused evidence chunks for a specific drug name.
    Used by the 'Specific Drug' mode to stay fast — only fetches
    chunks most relevant to that drug instead of the whole PDF.
    """
    queries = [
        f"{drug_name} side effects contraindications",
        f"{drug_name} uses indications treatment",
        f"{drug_name} mechanism of action",
        f"{drug_name} alternative substitute",
    ]
    all_evidence = []
    seen = set()
    for q in queries:
        try:
            result = rag.answer_question(q, return_metadata=True)
            chunks = result.get("clean_evidence") or result.get("evidence") or []
            for chunk in chunks:
                text = rag.get_chunk_text(chunk).strip()
                if text and text not in seen:
                    seen.add(text)
                    all_evidence.append(chunk)
        except Exception:
            pass
    if all_evidence:
        return rag.format_evidence(all_evidence[:12])
    # Fallback: return beginning chunks
    return rag.format_evidence(rag.get_beginning_chunks(8))



# --------------------------------------------------
# Medicine name extraction
# --------------------------------------------------

def extract_medicine_name(text: str) -> str:
    """
    Heuristically extract the primary drug/medicine name from raw PDF text.
    Falls back to asking the LLM if pattern matching yields nothing.
    """
    text = _clean(text)

    patterns = [
        r"(?:brand name|trade name|drug name|medicine name|product name)\s*[:\-]\s*([A-Za-z0-9 \-]+)",
        r"(?:contains?|active ingredient|active substance)\s*[:\-]\s*([A-Za-z0-9 \-]+)",
        r"^([A-Z][A-Za-z0-9\-]+(?: [A-Z][A-Za-z0-9\-]+){0,3})\s+(?:\d+\s*mg|\d+\s*ml|\d+\s*mcg)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.I | re.M)
        if match:
            candidate = _clean(match.group(1))
            if 2 < len(candidate) < 60:
                return candidate

    # Fallback: ask the LLM
    beginning_chunks = rag.get_beginning_chunks(6)
    evidence = rag.format_evidence(beginning_chunks)
    if not evidence:
        return "Unknown Medicine"

    prompt = (
        "You are a pharmaceutical expert.\n"
        "Read the following excerpts from a medicine document and identify the PRIMARY drug/medicine name.\n"
        "Return ONLY the medicine name - no explanation, no brand/generic prefix, just the name.\n\n"
        f"Document excerpts:\n{evidence}\n\nMedicine name:"
    )
    answer = _call_llm(prompt)
    return answer.strip().split("\n")[0] if answer else "Unknown Medicine"


# --------------------------------------------------
# Section generators
# --------------------------------------------------

def generate_alternative_medicines(medicine_name: str, evidence: str) -> str:
    prompt = (
        f"You are a licensed pharmacist and medical expert.\n\n"
        f"The medicine being analyzed is: {medicine_name}\n\n"
        f"Document evidence about this medicine:\n{evidence}\n\n"
        f"Task:\n"
        f"List all ALTERNATIVE or SUBSTITUTE medicines for {medicine_name}.\n"
        f"For each alternative, provide:\n"
        f"- The alternative medicine name\n"
        f"- Drug class / category it belongs to\n"
        f"- Key difference or advantage over {medicine_name}\n\n"
        f"Rules:\n"
        f"- Prioritize alternatives mentioned in the document.\n"
        f"- If the document does not mention alternatives, use your expert knowledge but explicitly state "
        f"that these are general pharmacological alternatives not mentioned in the document.\n"
        f"- Format your answer as a numbered list.\n"
        f"- Be concise and medically accurate.\n\n"
        f"Alternative medicines:"
    )
    return _call_llm(prompt)


def generate_side_effects(medicine_name: str, evidence: str) -> str:
    prompt = (
        f"You are a licensed pharmacist and medical expert.\n\n"
        f"The medicine being analyzed is: {medicine_name}\n\n"
        f"Document evidence:\n{evidence}\n\n"
        f"Task:\n"
        f"Provide a COMPREHENSIVE list of SIDE EFFECTS for {medicine_name}, categorized as:\n"
        f"### 1. Common Side Effects (greater than 10% of patients)\n"
        f"### 2. Less Common Side Effects (1-10%)\n"
        f"### 3. Rare / Serious Side Effects (less than 1%, but clinically significant)\n"
        f"### 4. Contraindications (conditions where the medicine must NOT be used)\n\n"
        f"Rules:\n"
        f"- Output EVERY side effect and contraindication as a distinct bullet point (* or -) on its own line.\n"
        f"- Absolutely DO NOT output as a continuous paragraph or combine multiple side effects into a single block of text.\n"
        f"- Prioritize side effects explicitly documented in the provided text.\n"
        f"- If the document is sparse, supplement with known pharmacological side effects and clearly indicate the source.\n"
        f"- Use clear, patient-friendly language alongside medical terms.\n\n"
        f"Side effects:"
    )
    return _call_llm(prompt)


def generate_use_cases(medicine_name: str, evidence: str) -> str:
    prompt = (
        f"You are a licensed pharmacist and medical expert.\n\n"
        f"The medicine being analyzed is: {medicine_name}\n\n"
        f"Document evidence:\n{evidence}\n\n"
        f"Task:\n"
        f"List all USE CASES and INDICATIONS for {medicine_name}, categorized under:\n"
        f"### 1. Primary / Approved Indications\n"
        f"### 2. Off-Label Uses\n"
        f"### 3. Target Patient Population\n"
        f"### 4. Dosage & Administration Highlights\n\n"
        f"Rules:\n"
        f"- Output EVERY point as a distinct bullet point (* or -) on its own separate line.\n"
        f"- DO NOT output as a continuous paragraph.\n"
        f"- Base your answer primarily on the document. Supplement gaps with expert knowledge and label it clearly.\n\n"
        f"Use cases and indications:"
    )
    return _call_llm(prompt)


def generate_mechanism_and_cure(medicine_name: str, evidence: str) -> str:
    prompt = (
        f"You are a licensed pharmacist and medical expert.\n\n"
        f"The medicine being analyzed is: {medicine_name}\n\n"
        f"Document evidence:\n{evidence}\n\n"
        f"Task:\n"
        f"Explain HOW {medicine_name} works and how it helps in treating/curing the condition it is prescribed for:\n"
        f"### 1. Mechanism of Action (Biochemical / Physiological)\n"
        f"### 2. Therapeutic Effect & Clinical Benefit\n"
        f"### 3. Disease Modification (Cures, Manages, or Suppresses)\n"
        f"### 4. Duration of Treatment\n"
        f"### 5. Expected Clinical Outcomes\n\n"
        f"Rules:\n"
        f"- Output all explanations with clear sub-bullets (* or -) on separate lines.\n"
        f"- Do NOT write dense, hard-to-read paragraphs.\n"
        f"- Explain in a way that is understandable to both medical professionals and patients.\n"
        f"- Prioritize document-sourced information; supplement with pharmacological knowledge where needed.\n\n"
        f"Mechanism and therapeutic action:"
    )
    return _call_llm(prompt)


# --------------------------------------------------
# Full medicine profile builder
# --------------------------------------------------

def _extract_name_from_question(question: str) -> str:
    q = question.strip()
    patterns = [
        r"(?:about|for|of|on)\s+([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,2})",
        r"([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,2})\s+(?:medicine|drug|tablet|capsule|injection|syrup)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            candidate = match.group(1).strip()
            if 2 < len(candidate) < 60:
                return candidate
    return ""


def build_medicine_profile(question: str, result: dict) -> dict:
    """
    Given a RAG result, build a complete medicine information profile.
    Returns a dict with: medicine_name, alternatives, side_effects, use_cases, mechanism
    """
    evidence = _evidence_text(result)
    medicine_name = _extract_name_from_question(question)
    if not medicine_name:
        beginning_chunks = rag.get_beginning_chunks(6)
        beginning_text = rag.format_evidence(beginning_chunks)
        medicine_name = extract_medicine_name(beginning_text or evidence)

    return {
        "medicine_name": medicine_name,
        "alternatives": generate_alternative_medicines(medicine_name, evidence),
        "side_effects": generate_side_effects(medicine_name, evidence),
        "use_cases": generate_use_cases(medicine_name, evidence),
        "mechanism": generate_mechanism_and_cure(medicine_name, evidence),
    }


# --------------------------------------------------
# Stepwise & Streaming Medicine Profile Generator
# --------------------------------------------------

def stream_medicine_profile(medicine_name: str = None, evidence: str = None):
    """
    Generator that produces medicine analysis step-by-step.
    Yields each section immediately as soon as it is generated,
    allowing the UI to display each card progressively without
    making the user wait for all 4 sections at once.
    """
    info = rag.get_current_pdf_info()
    pdf_name = info.get("pdf_name", "Uploaded PDF")

    # If no medicine name supplied, detect from PDF
    if not medicine_name or medicine_name.strip().lower() in ("unknown", "unknown medicine"):
        beginning_chunks = rag.get_beginning_chunks(10)
        evidence_text = rag.format_evidence(beginning_chunks)
        if not evidence_text:
            medicine_name = "Unknown Medicine"
        else:
            medicine_name = extract_medicine_name(evidence_text)
    else:
        medicine_name = medicine_name.strip()

    # If evidence not provided, retrieve targeted evidence for this medicine
    if not evidence:
        evidence = _fetch_evidence_for_drug(medicine_name)

    # 1. First yield metadata so UI can show the header badge immediately
    yield {
        "type": "metadata",
        "medicine_name": medicine_name,
        "pdf_name": pdf_name,
        "total": 4,
    }

    # 2. Sequential sections list
    sections = [
        ("alternatives", "Alternative Medicines", "🔄", generate_alternative_medicines),
        ("side_effects", "Side Effects & Contraindications", "⚠️", generate_side_effects),
        ("use_cases", "Use Cases & Indications", "✅", generate_use_cases),
        ("mechanism", "Mechanism & How It Cures", "🧬", generate_mechanism_and_cure),
    ]

    total = len(sections)
    for i, (key, title, icon, func) in enumerate(sections, start=1):
        try:
            content = func(medicine_name, evidence)
        except Exception as err:
            content = f"*(Information for {title} could not be retrieved: {err})*"

        yield {
            "type": "section",
            "key": key,
            "title": title,
            "icon": icon,
            "content": content,
            "step": i,
            "total": total,
            "medicine_name": medicine_name,
            "pdf_name": pdf_name,
        }


def auto_analyze_medicine_pdf() -> dict:
    """
    Automatically analyze the currently loaded PDF as a medicine document.
    Collects all sections generated by stream_medicine_profile into a complete dict.
    """
    profile = {
        "medicine_name": "Unknown Medicine",
        "pdf_name": "Uploaded PDF",
        "alternatives": "N/A",
        "side_effects": "N/A",
        "use_cases": "N/A",
        "mechanism": "N/A",
    }
    for item in stream_medicine_profile():
        if item["type"] == "metadata":
            profile["medicine_name"] = item["medicine_name"]
            profile["pdf_name"] = item["pdf_name"]
        elif item["type"] == "section":
            profile[item["key"]] = item["content"]
    return profile


# --------------------------------------------------
# Answer a direct medicine question
# --------------------------------------------------

def answer_medicine_question(question: str, result: dict) -> str:
    """
    Given a RAG result and a user question about a medicine, generate a focused answer.
    """
    evidence = _evidence_text(result)
    if not evidence:
        return "The medicine information requested is not available in the provided document."

    medicine_name = _extract_name_from_question(question)
    if not medicine_name:
        medicine_name = "the medicine described in this document"

    prompt = (
        f"You are a licensed pharmacist and medical expert analyzing a medicine document.\n\n"
        f"Medicine: {medicine_name}\n"
        f"Question: {question}\n\n"
        f"Document evidence:\n{evidence}\n\n"
        f"Rules:\n"
        f"1. Answer ONLY what is asked.\n"
        f"2. Use information from the document first; supplement with expert pharmacological knowledge "
        f"where the document is silent (and label those additions clearly).\n"
        f"3. Be concise, accurate, and use a structured format (numbered lists, bullet points).\n"
        f"4. Do NOT invent drug names, dosages, or clinical data.\n"
        f"5. If the document has no relevant information, say so clearly.\n\n"
        f"Answer:"
    )
    return _call_llm(prompt)


# --------------------------------------------------
# Module test
# --------------------------------------------------

if __name__ == "__main__":
    print("MEDICINE ANALYZER MODULE")
    print("Module loaded successfully.")
    print("Available functions:")
    print("  auto_analyze_medicine_pdf()  -- full profile from loaded PDF")
    print("  build_medicine_profile(question, result)  -- profile from RAG result")
    print("  answer_medicine_question(question, result)  -- single question")
    print("  extract_medicine_name(text)  -- extract drug name from text")
