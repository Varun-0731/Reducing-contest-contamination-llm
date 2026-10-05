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

    answer = rag.call_gemini(
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
def _call_llm(prompt: str) -> str:
    """Shared LLM caller that preserves markdown structure and bullet points."""
    raw = rag.call_gemini(prompt)
    return raw.strip() if raw else ""


# --------------------------------------------------
# Legal Defense Strategy Generators
# --------------------------------------------------

def generate_charged_sections_and_facts(evidence: str) -> str:
    prompt = (
        "You are an expert senior criminal defense advocate and legal scholar.\n\n"
        f"Case evidence from court record:\n{evidence}\n\n"
        "Task:\n"
        "Extract and analyze all INVOLVED STATUTORY SECTIONS and CORE FACTS from the case.\n"
        "Provide:\n"
        "### 1. Case Identity & Jurisdiction\n"
        "- Case Title, Reference Number, Court, and Date (if recorded)\n"
        "- Complainant / Prosecution vs Accused parties\n\n"
        "### 2. Charged Statutory Sections & Legal Ingredients\n"
        "- List every charged section (e.g., IPC/BNS/CrPC sections)\n"
        "- Key statutory ingredients the prosecution is legally required to prove for each section\n\n"
        "### 3. Allegations vs Documented Findings\n"
        "- Prosecution's primary allegations\n"
        "- Specific bodily harm, weapons, or property involved\n\n"
        "Rules:\n"
        "- Format strictly as clear bullet points (* or -) under each heading.\n"
        "- Do NOT write running paragraphs.\n"
        "- Base findings on the document evidence.\n\n"
        "Charged sections & facts:"
    )
    return _call_llm(prompt)


def generate_best_defense_section(evidence: str) -> str:
    prompt = (
        "You are an expert senior criminal defense advocate.\n\n"
        f"Case evidence:\n{evidence}\n\n"
        "Task:\n"
        "Identify the SINGLE BEST STATUTORY SECTION / LEGAL DOCTRINE to defend the accused in this case.\n"
        "Provide:\n"
        "### 1. Primary Defense Section & Act\n"
        "- Exact Section number, Title, and Statute (e.g. IPC, CrPC, Evidence Act)\n\n"
        "### 2. Why This is the Strongest Defense (Best Use Case)\n"
        "- How this section directly dismantles or negates the prosecution's charges\n"
        "- Evidence in the record supporting this section (e.g. absence of intention, mutual struggle, weapon nature)\n\n"
        "### 3. Evidentiary Burden & Legal Standard\n"
        "- Burden of proof required to establish this defense (preponderance of probabilities vs reasonable doubt)\n"
        "- Key judicial holding or principle that supports the accused\n\n"
        "Rules:\n"
        "- Use clean, structured bullet points (* or -) on separate lines.\n"
        "- Keep it sharp, authoritative, and actionable for defense counsel.\n\n"
        "Best defense section:"
    )
    return _call_llm(prompt)


def generate_defense_hierarchy(evidence: str) -> str:
    prompt = (
        "You are an expert senior criminal defense advocate preparing a defense brief.\n\n"
        f"Case evidence:\n{evidence}\n\n"
        "Task:\n"
        "Structure the defense arguments in STRICT DECREASING ORDER OF HIERARCHY (from strongest complete defense down to procedural/mitigating alternatives).\n\n"
        "Categorize into exactly 4 hierarchical tiers:\n"
        "### 🥇 Rank 1 (Primary Shield — Complete Defense & Full Acquittal)\n"
        "- Statutory provisions offering complete immunity, self-defense, or total negation of mens rea (e.g., Section 96-106 IPC Private Defense, Section 80 Accident, or complete failure of Section 307 intention)\n"
        "- Why this rank carries the highest legal value\n\n"
        "### 🥈 Rank 2 (Substantive Statutory Mitigation — Reduction to Lesser Offense)\n"
        "- Statutory provisions that reduce culpability (e.g., Section 334 IPC sudden provocation, Section 300 Exception 4 sudden fight, or reduction from heinous to simple hurt)\n"
        "- How this shields the client from severe imprisonment\n\n"
        "### 🥉 Rank 3 (Non-Custodial & Compounding Relief)\n"
        "- Provisions allowing settlement or non-incarceration (e.g., Section 320 CrPC Compounding of offenses with victim, Section 360 CrPC or Probation of Offenders Act Section 4 release on good conduct)\n"
        "- Practical legal leverage for avoiding jail time\n\n"
        "### 🎖️ Rank 4 (Procedural & Evidentiary Vulnerabilities of Prosecution)\n"
        "- Procedural infirmities, delay in FIR, lack of independent corroboration, chain of custody of weapon, Section 101/102 Evidence Act reasonable doubt\n"
        "- Supplemental pressure points in trial or appeal\n\n"
        "Rules:\n"
        "- Output strictly in clean, detailed bullet points under each Rank.\n"
        "- Do NOT merge sections into paragraphs.\n"
        "- Highlight exact section numbers in bold.\n\n"
        "Defense hierarchy:"
    )
    return _call_llm(prompt)


def generate_procedural_strategy(evidence: str) -> str:
    prompt = (
        "You are an expert criminal defense advocate.\n\n"
        f"Case evidence:\n{evidence}\n\n"
        "Task:\n"
        "Provide actionable procedural, appellate, and trial strategies for defending this case:\n"
        "### 1. Appellate / Pre-Trial Relief\n"
        "- Bail / Suspension of sentence under Section 389 CrPC or appeal under Section 374 CrPC\n"
        "- Grounds for challenging any conviction or harsh sentence\n\n"
        "### 2. Cross-Examination & Evidentiary Pitfalls\n"
        "- Key witness contradictions to expose (discrepancy in injuries vs medical report, lighting, mutual struggle)\n"
        "- Forensic/recovery challenges regarding the weapon\n\n"
        "### 3. Compounding / Settlement Viability\n"
        "- Feasibility of mediation or compounding under Section 320 CrPC\n\n"
        "Rules:\n"
        "- Structure with distinct bullet points (* or -) on separate lines.\n"
        "- Be actionable and practical.\n\n"
        "Procedural strategy:"
    )
    return _call_llm(prompt)


def generate_defense_executive_summary(evidence: str) -> str:
    prompt = (
        "You are a senior criminal defense counsel.\n\n"
        f"Case evidence:\n{evidence}\n\n"
        "Task:\n"
        "Provide a SINGLE, SIMPLE, CONCISE executive summary paragraph (3 to 5 sentences) for the defense posture.\n"
        "Summarize:\n"
        "- The core charges and the current status of the accused,\n"
        "- The strongest legal defense section and why it effectively shields the accused,\n"
        "- The primary defense objective (complete acquittal vs probation/compounding) and recommended counsel next step.\n\n"
        "Rules:\n"
        "- Strictly output as ONE clear, cohesive paragraph. No bullet points or headers.\n"
        "- Highly professional legal counsel tone.\n\n"
        "Executive summary:"
    )
    return _call_llm(prompt)


def _fetch_evidence_for_legal_section(section_name: str) -> str:
    """Retrieve focused evidence chunks for a specific legal section or charge."""
    queries = [
        f"{section_name} allegations ingredients charge",
        f"{section_name} defense exceptions acquittal",
        f"{section_name} evidence witness weapon injury",
        f"{section_name} court findings judgment sentence",
    ]
    seen = set()
    collected = []
    for q in queries:
        try:
            res = rag.answer_question(q, return_metadata=True)
            chunks = res.get("clean_evidence") or res.get("evidence") or []
            for c in chunks:
                t = rag.get_chunk_text(c).strip()
                if t and t not in seen:
                    seen.add(t)
                    collected.append(t)
        except Exception:
            pass
    return "\n\n".join(collected) if collected else ""


# --------------------------------------------------
# Stepwise Streaming Generator for Legal Strategy
# --------------------------------------------------

def stream_legal_defense_strategy(target_section: str = None, evidence: str = None):
    """
    Generator that produces comprehensive legal defense analysis step-by-step.
    Yields each section immediately as soon as it is generated:
    1. Case Details & Charged Sections
    2. Primary Defense Section (Best Use Case)
    3. Defense Hierarchy (Decreasing Order of Strength)
    4. Procedural & Appellate Remedies
    5. Executive Counsel Summary (1 simple paragraph)
    """
    info = rag.get_current_pdf_info()
    pdf_name = info.get("pdf_name", "Court Case PDF")

    if not evidence:
        if target_section and target_section.strip():
            evidence = _fetch_evidence_for_legal_section(target_section.strip())
        if not evidence:
            beginning_chunks = rag.get_beginning_chunks(12)
            evidence = rag.format_evidence(beginning_chunks)
        if not evidence:
            evidence = "No document evidence available."

    # Extract case title
    case_title = extract_case_title(evidence)
    if not case_title or len(case_title) < 5:
        case_title = "Court Case Matter"

    if target_section and target_section.strip():
        display_title = f"{case_title} — Defense for {target_section.strip()}"
    else:
        display_title = case_title

    # Step 0: Metadata
    yield {
        "type": "metadata",
        "case_title": display_title,
        "pdf_name": pdf_name,
        "total": 5,
    }

    # Stepwise sections
    sections = [
        ("charged_sections", "Involved Sections & Case Allegations", "📜", generate_charged_sections_and_facts),
        ("primary_defense", "Best Supporting Defense Section (Rank 1)", "🛡️", generate_best_defense_section),
        ("defense_hierarchy", "Defense Hierarchy (Decreasing Order of Strength)", "📊", generate_defense_hierarchy),
        ("procedural_strategy", "Appellate & Trial Strategy", "⚖️", generate_procedural_strategy),
        ("summary", "Executive Counsel Summary", "📋", generate_defense_executive_summary),
    ]

    total = len(sections)
    for i, (key, title, icon, func) in enumerate(sections, start=1):
        try:
            content = func(evidence)
        except Exception as err:
            content = f"*(Information for {title} could not be generated: {err})*"

        yield {
            "type": "section",
            "key": key,
            "title": title,
            "icon": icon,
            "content": content,
            "step": i,
            "total": total,
            "case_title": display_title,
            "pdf_name": pdf_name,
        }


def auto_analyze_legal_case() -> dict:
    """Full synchronous runner that aggregates all sections into a dictionary."""
    profile = {
        "case_title": "Court Case Matter",
        "pdf_name": "Court Case PDF",
        "charged_sections": "N/A",
        "primary_defense": "N/A",
        "defense_hierarchy": "N/A",
        "procedural_strategy": "N/A",
        "summary": "N/A",
    }
    for item in stream_legal_defense_strategy():
        if item["type"] == "metadata":
            profile["case_title"] = item["case_title"]
            profile["pdf_name"] = item["pdf_name"]
        elif item["type"] == "section":
            profile[item["key"]] = item["content"]
    return profile


# --------------------------------------------------
# Factual Situation / Incident Legal Classifier
# --------------------------------------------------

def generate_situation_applicable_sections(situation_text: str) -> str:
    prompt = (
        "You are an expert senior criminal defense advocate and legal jurist.\n\n"
        f"Factual Situation / Incident:\n{situation_text}\n\n"
        "Task:\n"
        "Analyze this factual situation and DECIDE ALL APPLICABLE STATUTORY SECTIONS it comes under.\n"
        "Provide:\n"
        "### 1. Factual Elements & Legal Act Matrix\n"
        "- Key physical acts, verbal threats, property damage, or injuries in this situation\n"
        "- Presence or absence of criminal intent (Mens Rea)\n\n"
        "### 2. Applicable Statutory Offense Sections (Which Section It Comes Under)\n"
        "For each potential section under the Indian Penal Code (IPC) / Bharatiya Nyaya Sanhita (BNS) or special statutes:\n"
        "- Exact Section Number & Title (e.g., Section 323, 325, 341, 354, 448, 420, 506 IPC)\n"
        "- Why it applies: exact factual act that matches the required statutory ingredients\n"
        "- Classification: Cognizable/Non-cognizable, Bailable/Non-bailable, Compoundable status, and Maximum Punishment\n\n"
        "Rules:\n"
        "- Format strictly in clear bullet points (* or -) under each heading.\n"
        "- Highlight exact section numbers in bold.\n"
        "- Do NOT write running paragraphs.\n\n"
        "Applicable sections:"
    )
    return _call_llm(prompt)


def generate_situation_best_defense(situation_text: str) -> str:
    prompt = (
        "You are a senior criminal defense counsel advising a client involved in this situation.\n\n"
        f"Factual Situation / Incident:\n{situation_text}\n\n"
        "Task:\n"
        "If we are defending the person involved in this situation, identify the SINGLE BEST SECTION / STATUTORY DEFENSE that supports their best use case.\n"
        "Provide:\n"
        "### 1. Primary Shield Section & Act\n"
        "- Exact Section number, Title, and Statute (e.g., Section 96/97/100 IPC Private Defense, Section 80 Accident, Section 334/335 Provocation, Section 84, etc.)\n\n"
        "### 2. Why This is the Strongest Defense for this Situation\n"
        "- Exact factual justification from the situation showing how this section protects the client\n"
        "- How it legally refutes or exonerates the alleged offenses\n\n"
        "### 3. Legal Standard & Requisite Burden\n"
        "- Burden of proof required to establish this defense (preponderance of probabilities under Section 105 Evidence Act)\n\n"
        "Rules:\n"
        "- Format in distinct bullet points (* or -) on separate lines.\n"
        "- Keep it sharp, practical, and authoritative.\n\n"
        "Best defense section:"
    )
    return _call_llm(prompt)


def generate_situation_defense_hierarchy(situation_text: str) -> str:
    prompt = (
        "You are an expert senior criminal defense advocate preparing a defense strategy brief.\n\n"
        f"Factual Situation / Incident:\n{situation_text}\n\n"
        "Task:\n"
        "Structure all defense arguments for this situation in STRICT DECREASING ORDER OF HIERARCHY (from strongest complete defense down to procedural/mitigating alternatives).\n\n"
        "Categorize into exactly 4 tiers:\n"
        "### 🥇 Rank 1 (Primary Shield — Complete Defense & Exoneration)\n"
        "- Statutory provisions offering complete immunity, self-defense, lack of mens rea, or justified action (e.g. Sections 96-106 IPC, Section 80 Accident, consent)\n"
        "- Why this rank carries the highest legal value for this situation\n\n"
        "### 🥈 Rank 2 (Substantive Statutory Mitigation — Reduction of Culpability)\n"
        "- Statutory provisions reducing the offense from serious/cognizable to a minor or lesser offense (e.g., Section 334/335 provocation, sudden fight, absence of dangerous weapon)\n"
        "- How this shields the client from severe imprisonment\n\n"
        "### 🥉 Rank 3 (Non-Custodial & Compounding Relief)\n"
        "- Statutory compounding with the other party (Section 320 CrPC), mediation, or probation of good conduct (Section 360 CrPC / Section 4 Probation of Offenders Act)\n"
        "- Practical legal leverage to avoid jail time\n\n"
        "### 🎖️ Rank 4 (Procedural & Evidentiary Vulnerabilities)\n"
        "- Inherent evidentiary weaknesses in the scenario (unwitnessed altercation, contradictory injury claims, cross-complaint leverage, benefit of reasonable doubt)\n\n"
        "Rules:\n"
        "- Output strictly in clean, detailed bullet points under each Rank.\n"
        "- Do NOT merge sections into paragraphs.\n"
        "- Highlight exact section numbers in bold.\n\n"
        "Defense hierarchy:"
    )
    return _call_llm(prompt)


def generate_situation_procedural_strategy(situation_text: str) -> str:
    prompt = (
        "You are a criminal defense advocate.\n\n"
        f"Factual Situation / Incident:\n{situation_text}\n\n"
        "Task:\n"
        "Provide immediate procedural next steps and legal safeguards for the person in this situation:\n"
        "### 1. Pre-Arrest / Bail Protection\n"
        "- Section 438 CrPC (Anticipatory Bail) or Section 41A CrPC notice requirements if offense carries <= 7 years\n"
        "- Immediate steps if police complaint or FIR is threatened\n\n"
        "### 2. Cross-Complaint & Evidentiary Preservation\n"
        "- Filing a cross-FIR / counter-complaint or medical MLC to document own injuries\n"
        "- Preservation of CCTV, phone call records, messages, or independent bystander witnesses\n\n"
        "### 3. Settlement / Compounding Feasibility\n"
        "- Whether the offenses can be amicably resolved under Section 320 CrPC\n\n"
        "Rules:\n"
        "- Format in clean bullet points (* or -) on separate lines.\n\n"
        "Procedural strategy:"
    )
    return _call_llm(prompt)


def generate_situation_executive_summary(situation_text: str) -> str:
    prompt = (
        "You are senior criminal defense counsel.\n\n"
        f"Factual Situation / Incident:\n{situation_text}\n\n"
        "Task:\n"
        "Provide a SINGLE, SIMPLE, CONCISE executive summary paragraph (3 to 5 sentences) summarizing:\n"
        "- The primary statutory section(s) this situation falls under,\n"
        "- The strongest defense section that best protects the person,\n"
        "- Counsel's immediate recommended course of action (e.g. anticipatory bail, compounding, or filing counter-complaint).\n\n"
        "Rules:\n"
        "- Strictly output as ONE clear, cohesive paragraph. No bullet points or headers.\n"
        "- Highly professional legal counsel tone.\n\n"
        "Executive summary:"
    )
    return _call_llm(prompt)


def stream_situation_legal_analysis(situation_text: str):
    """
    Generator that produces comprehensive legal section identification
    and defense hierarchy for any user-provided factual situation.
    """
    yield {
        "type": "metadata",
        "case_title": "Factual Incident Analysis",
        "pdf_name": "Custom Factual Scenario",
        "total": 5,
    }

    sections = [
        ("charged_sections", "Applicable Statutory Sections (Offense Classification)", "📜", generate_situation_applicable_sections),
        ("primary_defense", "Best Supporting Defense Section (Rank 1)", "🛡️", generate_situation_best_defense),
        ("defense_hierarchy", "Defense Hierarchy (Decreasing Order of Strength)", "📊", generate_situation_defense_hierarchy),
        ("procedural_strategy", "Immediate Legal Safeguards & Procedural Strategy", "⚖️", generate_situation_procedural_strategy),
        ("summary", "Executive Counsel Legal Summary", "📋", generate_situation_executive_summary),
    ]

    total = len(sections)
    for i, (key, title, icon, func) in enumerate(sections, start=1):
        try:
            content = func(situation_text)
        except Exception as err:
            content = f"*(Information for {title} could not be generated: {err})*"

        yield {
            "type": "section",
            "key": key,
            "title": title,
            "icon": icon,
            "content": content,
            "step": i,
            "total": total,
            "case_title": "Factual Incident Analysis",
            "pdf_name": "Custom Factual Scenario",
        }


if __name__ == "__main__":
    print("LEGAL CASE ANALYSIS MODULE")
    print("Module loaded successfully.")
    print("Available functions:")
    print("  stream_legal_defense_strategy()")
    print("  auto_analyze_legal_case()")
    print("  analyze_legal_question()")
    print("  answer_legal_question()")
    print("  get_legal_summary()")
