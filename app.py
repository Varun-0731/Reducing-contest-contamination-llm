import os
import time
from pathlib import Path
import pandas as pd
import streamlit as st

from retrieval import retrieval_pipeline as rag
import ui_theme as ui


st.set_page_config(
    page_title="PDF RAG Intelligence System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom modern glassmorphic styling
ui.inject_custom_css()


CASE_STUDIES = [
    "General PDF QA",
    "Medical Report Analysis",
    "Legal Court Case Analysis",
]

DEMO_PDFS = {
    "General PDF QA": Path("data/pdfs/RAG.pdf"),
    "Medical Report Analysis": Path("data/pdfs/Sample-Smart-Report-Clinics.pdf"),
    "Legal Court Case Analysis": Path("data/pdfs/legal_case_study_base.pdf"),
}

SUGGESTED_PROMPTS = {
    "General PDF QA": [
        "What is Retrieval-Augmented Generation (RAG)?",
        "How do parametric and non-parametric memory work together?",
        "What benchmark datasets were evaluated in this research?",
        "What is the difference between RAG-Sequence and RAG-Token models?",
    ],
    "Medical Report Analysis": [
        "Summarize the patient's primary lab results and vitals.",
        "Are there any abnormal values or critical clinical indicators?",
        "Provide a comprehensive diagnostic and care summary.",
        "What follow-up recommendations are noted in the report?",
    ],
    "Legal Court Case Analysis": [
        "What was the core holding and ratio decidendi of the judgment?",
        "Which statutory provisions, acts, and precedents are cited?",
        "Summarize the appellant's primary legal arguments.",
        "What relief or final orders were pronounced by the bench?",
    ],
}


def init_state():
    defaults = {
        "case_study": "General PDF QA",
        "uploaded_signature": None,
        "chat_history": [],
        "last_result": None,
        "pending_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def pct(value):
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def signed_pct(value):
    if value is None:
        return "N/A"
    val = float(value)
    sign = "+" if val > 0 else ""
    return f"{sign}{val * 100:.2f}%"


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def load_pdf_if_needed(uploaded_file, case_study):
    if uploaded_file is None:
        return

    signature = (
        uploaded_file.name,
        uploaded_file.size,
        case_study,
    )

    if st.session_state.uploaded_signature == signature:
        return

    with st.spinner(f"Processing PDF '{uploaded_file.name}': extracting text, chunking, embedding and building FAISS..."):
        info = rag.load_uploaded_pdf_bytes(
            uploaded_file.getvalue(),
            uploaded_file.name,
            case_study=case_study,
        )

    st.session_state.uploaded_signature = signature
    st.session_state.chat_history = []
    st.session_state.last_result = None
    st.toast(f"Successfully loaded '{info['pdf_name']}'!", icon="✅")


def load_demo_pdf(case_study):
    demo_path = DEMO_PDFS.get(case_study)
    if not demo_path or not demo_path.exists():
        st.sidebar.error(f"Demo file not found for {case_study}.")
        return

    signature = (str(demo_path), os.path.getsize(demo_path), case_study)
    if st.session_state.uploaded_signature == signature:
        st.toast(f"Demo PDF '{demo_path.name}' is already active.", icon="ℹ️")
        return

    with st.spinner(f"Ingesting pre-packaged Demo PDF: {demo_path.name}..."):
        info = rag.load_uploaded_pdf(
            str(demo_path),
            case_study=case_study,
        )

    st.session_state.uploaded_signature = signature
    st.session_state.chat_history = []
    st.session_state.last_result = None
    st.toast(f"Loaded demo '{info['pdf_name']}' ({info['chunks']} chunks)!", icon="🎉")


def metric_table(result):
    evaluation = result.get("evaluation", {})
    current = evaluation.get("current_question_metrics", {})

    retrieval = evaluation.get("retrieval", {})
    rb = retrieval.get("baseline", {})
    ra = retrieval.get("proposed", {})

    gb = current.get("text_generation", {}).get("baseline", {})
    ga = current.get("text_generation", {}).get("proposed", {})

    b = current.get("baseline", {})
    a = current.get("proposed", {})

    def diff(before, after, lower=False):
        if before is None or after is None:
            return None
        return (float(before) - float(after)) if lower else (float(after) - float(before))

    rows = []

    # Retrieval metrics
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
    ]:
        before = rb.get(key)
        after = ra.get(key)
        rows.append({
            "Metric": label,
            "Retrieval — Before": pct(before),
            "Retrieval — After": pct(after),
            "Retrieval Δ": signed_pct(diff(before, after)),
            "Generation — Before": pct(b.get(key)),
            "Generation — After": pct(a.get(key)),
            "Generation Δ": signed_pct(diff(b.get(key), a.get(key))),
        })

    generation_rows = [
        ("bleu", "BLEU-4"),
        ("rouge1", "ROUGE-1 F1"),
        ("rouge2", "ROUGE-2 F1"),
        ("rougeL", "ROUGE-L F1"),
    ]

    for key, label in generation_rows:
        before = gb.get(key)
        after = ga.get(key)
        rows.append({
            "Metric": label,
            "Retrieval — Before": "—",
            "Retrieval — After": "—",
            "Retrieval Δ": "—",
            "Generation — Before": pct(before),
            "Generation — After": pct(after),
            "Generation Δ": signed_pct(diff(before, after)),
        })

    bp = gb.get("perplexity")
    ap = ga.get("perplexity")

    rows.append({
        "Metric": "Perplexity (Lower is better)",
        "Retrieval — Before": "—",
        "Retrieval — After": "—",
        "Retrieval Δ": "—",
        "Generation — Before": f"{bp:.2f}" if bp is not None else "N/A",
        "Generation — After": f"{ap:.2f}" if ap is not None else "N/A",
        "Generation Δ": (
            f"{((bp - ap) / bp) * 100:+.2f}%"
            if bp is not None and ap is not None and bp != 0
            else "N/A"
        ),
    })

    return pd.DataFrame(rows)


def render_sidebar():
    # Brand Header - Monochromatic
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            <div style="background: #ffffff; color: #0a0a0a; width: 38px; height: 38px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 800; box-shadow: 0 4px 14px rgba(255,255,255,0.15);">
                🛡️
            </div>
            <div>
                <div style="font-weight: 800; font-size: 1.15rem; color: #ffffff; letter-spacing: -0.02em;">PDF RAG Shield</div>
                <div style="font-size: 0.72rem; color: #a1a1aa; font-weight: 500;">Context Contamination Defense</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div style="display: flex; gap: 0.4rem; margin-bottom: 1.25rem;">
            <span class="badge-pill badge-white">Online</span>
            <span class="badge-pill badge-outline">v2.0 Fast</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### 🎯 Case Study Domain")
    case_study = st.sidebar.selectbox(
        "Select Domain",
        CASE_STUDIES,
        index=CASE_STUDIES.index(st.session_state.case_study),
        help="Selects specialized domain output formatting and sample document presets.",
        label_visibility="collapsed",
    )

    if case_study != st.session_state.case_study:
        st.session_state.case_study = case_study
        st.session_state.uploaded_signature = None
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.rerun()

    st.sidebar.divider()

    st.sidebar.markdown("### 📂 Ingestion & Documents")
    
    # 1-Click Demo Document Loader
    demo_path = DEMO_PDFS.get(case_study)
    demo_name = demo_path.name if demo_path else "demo.pdf"
    
    if st.sidebar.button(f"⚡ Load Demo: {demo_name}", use_container_width=True):
        load_demo_pdf(case_study)

    st.sidebar.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.75rem; margin: 0.4rem 0;">
            — OR UPLOAD CUSTOM PDF —
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader(
        "Import Custom PDF",
        type=["pdf"],
        help="Upload the PDF you want this case study to analyze.",
        label_visibility="collapsed",
    )

    if uploaded_file:
        try:
            load_pdf_if_needed(uploaded_file, case_study)
        except Exception as exc:
            st.sidebar.error(f"PDF processing failed: {exc}")

    st.sidebar.divider()

    # Live Vector Store Telemetry Widget
    try:
        info = rag.get_current_pdf_info()
        is_active = info["pdf_name"] not in ["No document selected", "No PDF loaded"]
    except Exception:
        info = {"pdf_name": "No document selected", "chunks": 0, "vectors": 0}
        is_active = False

    st.sidebar.markdown(
        f"""
        <div style="background: #141414; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                <span style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #a1a1aa;">FAISS Vector Store</span>
                <span class="badge-pill {'badge-white' if is_active else 'badge-muted'}">
                    {'● Ready' if is_active else '○ Idle'}
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="font-size: 0.82rem; color: #a1a1aa;">Indexed Chunks:</span>
                <strong style="color: #ffffff; font-family: 'JetBrains Mono', monospace;">{info.get('chunks', 0)}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="font-size: 0.82rem; color: #a1a1aa;">Vector Embeddings:</span>
                <strong style="color: #ffffff; font-family: 'JetBrains Mono', monospace;">{info.get('vectors', 0)}</strong>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-size: 0.82rem; color: #a1a1aa;">Dimension:</span>
                <strong style="color: #ededed; font-family: 'JetBrains Mono', monospace;">384-dim (BGE)</strong>
            </div>
            <div style="font-size: 0.75rem; color: #71717a; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.5rem; word-break: break-all;">
                📄 <strong>{info.get('pdf_name', 'None')}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    
    if st.sidebar.button("🗑️ Reset Chat History", use_container_width=True, type="secondary"):
        st.session_state.chat_history = []
        st.session_state.last_result = None
        st.rerun()

    return case_study


def render_chat(case_study):
    try:
        info = rag.get_current_pdf_info()
    except Exception:
        info = {"pdf_name": "No PDF loaded", "chunks": 0, "vectors": 0}

    is_doc_ready = info["pdf_name"] not in ["No document selected", "No PDF loaded"]

    if not is_doc_ready:
        st.markdown(
            f"""
            <div class="rag-card" style="text-align: center; padding: 2.5rem 2rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">📂</div>
                <h3 style="margin-bottom: 0.5rem; color: #ffffff;">No Active Document Ingested</h3>
                <p style="color: #94a3b8; max-width: 540px; margin: 0 auto 1.5rem auto; line-height: 1.6;">
                    Select <strong>"{case_study}"</strong> and click <strong>"Load Demo PDF"</strong> in the sidebar, or upload a custom PDF document to begin asking questions with contamination shielding.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⚡ Load Research Paper (RAG.pdf)", use_container_width=True):
                st.session_state.case_study = "General PDF QA"
                load_demo_pdf("General PDF QA")
                st.rerun()
        with col2:
            if st.button("🩺 Load Medical Clinical Report", use_container_width=True):
                st.session_state.case_study = "Medical Report Analysis"
                load_demo_pdf("Medical Report Analysis")
                st.rerun()
        with col3:
            if st.button("⚖️ Load Legal Court Precedent", use_container_width=True):
                st.session_state.case_study = "Legal Court Case Analysis"
                load_demo_pdf("Legal Court Case Analysis")
                st.rerun()
        return

    # Suggested Prompts Bar
    st.markdown(
        """
        <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; margin-bottom: 0.4rem;">
            💡 Recommended Questions for this Case Study
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    prompts = SUGGESTED_PROMPTS.get(case_study, [])
    prompt_cols = st.columns(len(prompts))
    for idx, prompt_text in enumerate(prompts):
        if prompt_cols[idx].button(f"“{prompt_text[:34]}...”", key=f"prompt_btn_{idx}", use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # Render Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    query_to_process = None
    if st.session_state.pending_prompt:
        query_to_process = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    else:
        user_input = st.chat_input("Ask any question regarding the active PDF...")
        if user_input:
            query_to_process = user_input

    if not query_to_process:
        return

    # User message
    st.session_state.chat_history.append({
        "role": "user",
        "content": query_to_process,
    })

    with st.chat_message("user"):
        st.markdown(query_to_process)

    # Assistant processing
    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving evidence pool, applying contamination filter, and generating grounded response..."):
            start = time.time()
            result = rag.answer_question(
                query_to_process,
                return_metadata=True,
            )
            elapsed = time.time() - start

        answer = result.get("proposed_answer", result.get("answer", ""))

        if case_study == "Medical Report Analysis":
            from medical.medical_analysis import generate_medical_analysis
            answer = generate_medical_analysis(query_to_process, result)
        elif case_study == "Legal Court Case Analysis":
            from legal.legal_analysis import generate_legal_analysis
            answer = generate_legal_analysis(query_to_process, result)

        st.markdown(answer)

        # Performance & Telemetry Strip
        retrieved_cnt = result.get("retrieved_count", 0)
        clean_cnt = result.get("clean_count", 0)
        removed_cnt = result.get("removed_count", max(0, retrieved_cnt - clean_cnt))
        contam_rate = result.get("contamination_rate", 0)

        st.markdown(
            f"""
            <div style="background: #141414; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 0.75rem 1rem; margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; justify-content: space-between;">
                <div style="display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap;">
                    <span class="badge-pill badge-outline">
                        Retrieved: <strong>{retrieved_cnt} chunks</strong>
                    </span>
                    <span class="badge-pill badge-white">
                        Kept: <strong>{clean_cnt} clean</strong>
                    </span>
                    <span class="badge-pill badge-muted">
                        Filtered: <strong>{removed_cnt} noise</strong>
                    </span>
                    <span class="badge-pill badge-contrast">
                        Shield: <strong>{contam_rate:.1f}% reduced</strong>
                    </span>
                </div>
                <div style="font-size: 0.8rem; color: #a1a1aa; font-family: 'JetBrains Mono', monospace;">
                    ⏱️ Latency: <strong>{elapsed:.2f}s</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Detailed Analysis Expander inside Assistant Message
        with st.expander("⚖️ Inspect Contamination Defense & Evidence Diff"):
            diff_tab1, diff_tab2 = st.tabs(["Side-by-Side Comparison", "Kept Evidence Context"])
            with diff_tab1:
                ui.render_comparison_view(
                    baseline_answer=result.get("baseline_answer", ""),
                    proposed_answer=answer,
                    reference_answer=result.get("reference_answer"),
                )
            with diff_tab2:
                clean_evidence = result.get("clean_evidence", [])
                for i, chunk in enumerate(clean_evidence, 1):
                    ui.render_chunk_card(
                        index=i,
                        text=rag.get_chunk_text(chunk),
                        is_kept=True,
                        page=rag.get_chunk_page(chunk),
                        chunk_id=rag.get_chunk_id(chunk, fallback=f"clean_{i}"),
                    )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
        })
        st.session_state.last_result = result


def render_overview(case_study):
    try:
        info = rag.get_document_info()
    except Exception:
        st.info("Upload or select a PDF first to view document metrics.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.render_kpi_card("Total Chunks", str(info.get("chunks", 0)), icon="🧩")
    with c2:
        ui.render_kpi_card("FAISS Vectors", str(info.get("vectors", 0)), icon="⚡")
    with c3:
        ui.render_kpi_card("Document References", str(info.get("references") or "N/A"), icon="📚")
    with c4:
        ui.render_kpi_card("Pipeline Status", "Ready & Indexed", icon="🟢", delta_color="pos")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    col_meta, col_specs = st.columns(2)
    with col_meta:
        st.markdown(
            f"""
            <div class="rag-card">
                <h4 style="margin-top: 0; color: #ffffff;">📄 Document Metadata</h4>
                <div style="display: flex; flex-direction: column; gap: 0.75rem; margin-top: 1rem;">
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #94a3b8;">Document Filename</div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #f8fafc;">{info.get('pdf_name', 'Unknown')}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #a1a1aa;">Active Domain</div>
                        <div style="font-size: 0.95rem; font-weight: 600; color: #ffffff;">{case_study}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #a1a1aa;">Extracted Title</div>
                        <div style="font-size: 0.9rem; color: #d4d4d8;">{info.get('title') or 'Title not detected in header metadata'}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #a1a1aa;">Authors / Origin</div>
                        <div style="font-size: 0.9rem; color: #d4d4d8;">{info.get('authors') or 'Not specified in header metadata'}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_specs:
        st.markdown(
            """
            <div class="rag-card">
                <h4 style="margin-top: 0; color: #ffffff;">⚙️ Ingestion & RAG Specs</h4>
                <div style="display: flex; flex-direction: column; gap: 0.75rem; margin-top: 1rem;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.5rem;">
                        <span style="color: #a1a1aa; font-size: 0.88rem;">Embedding Architecture</span>
                        <strong style="color: #ffffff; font-size: 0.88rem;">BAAI/bge-small-en-v1.5</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.5rem;">
                        <span style="color: #a1a1aa; font-size: 0.88rem;">Vector Store Index</span>
                        <strong style="color: #ffffff; font-size: 0.88rem;">FAISS (IndexFlatIP)</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.5rem;">
                        <span style="color: #a1a1aa; font-size: 0.88rem;">Chunk Size / Overlap</span>
                        <strong style="color: #ffffff; font-size: 0.88rem;">512 chars / 50 overlap</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 0.5rem;">
                        <span style="color: #a1a1aa; font-size: 0.88rem;">LLM Generator</span>
                        <strong style="color: #ffffff; font-size: 0.88rem;">NVIDIA NIM Llama 3.2</strong>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #a1a1aa; font-size: 0.88rem;">Contamination Shield</span>
                        <strong style="color: #ededed; font-size: 0.88rem;">Adaptive Cosine Thresholding</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_retrieval_analysis():
    if not st.session_state.last_result:
        st.markdown(
            """
            <div class="rag-card" style="text-align: center; padding: 2.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
                <h3 style="color: #ffffff; margin-bottom: 0.5rem;">No Query Evaluated Yet</h3>
                <p style="color: #94a3b8;">Go to the <strong>💬 Chat & QA</strong> tab and ask a question to generate comparative metrics and retrieval telemetry.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    result = st.session_state.last_result

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #94a3b8; margin-bottom: 0.25rem;">Active Evaluated Query</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">“{result.get('question', '')}”</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.render_kpi_card("Retrieved Chunks", str(result.get("retrieved_count", 0)), icon="📥")
    with c2:
        ui.render_kpi_card("Clean Evidence Kept", str(result.get("clean_count", 0)), icon="🛡️")
    with c3:
        ui.render_kpi_card("Noise Stripped", str(result.get("removed_count", 0)), icon="❌")
    with c4:
        rate = result.get("contamination_rate", 0)
        ui.render_kpi_card(
            "Contamination Reduction",
            f"{rate:.1f}%",
            delta=f"-{rate:.1f}% noise",
            icon="✨",
            delta_color="pos",
        )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    st.markdown("### 📊 Contamination Benchmark Metrics (Before vs. After Filtering)")
    st.dataframe(
        metric_table(result),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    st.markdown("### ⚖️ Side-by-Side Generated Answers")
    ui.render_comparison_view(
        baseline_answer=result.get("baseline_answer", ""),
        proposed_answer=result.get("proposed_answer", ""),
        reference_answer=result.get("reference_answer"),
    )

    retrieval = result.get("retrieval_metrics", {})
    if retrieval.get("available"):
        with st.expander("🔬 Deep Retrieval Evidence Scores"):
            st.json(retrieval)


def render_contamination():
    result = st.session_state.last_result
    if not result:
        st.markdown(
            """
            <div class="rag-card" style="text-align: center; padding: 2.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧹</div>
                <h3 style="color: #ffffff; margin-bottom: 0.5rem;">Awaiting Query Execution</h3>
                <p style="color: #94a3b8;">Execute a question in the <strong>💬 Chat & QA</strong> tab to inspect chunk-level contamination filtering.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    evidence = result.get("evidence", [])
    clean_evidence = result.get("clean_evidence", [])
    removed_count = max(0, len(evidence) - len(clean_evidence))

    st.markdown(
        """
        <div class="rag-card" style="border-left: 3px solid #ffffff;">
            <h4 style="margin-top: 0; color: #ffffff;">🛡️ How the Adaptive Contamination Filter Operates</h4>
            <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; margin-bottom: 0;">
                Traditional RAG passes the full Top-K retrieval pool directly to the LLM, contaminating the context window with weak, conflicting, or redundant chunks. 
                Our <strong>Adaptive Shield</strong> calculates dynamic cosine semantic relevance thresholds while enforcing <strong>lexical entity protection</strong>. Only verifiable, high-fidelity context reaches the generator.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        ui.render_kpi_card("Raw Retrieved Pool", f"{len(evidence)} chunks", icon="📥")
    with c2:
        ui.render_kpi_card("Shielded Clean Context", f"{len(clean_evidence)} chunks", icon="🛡️")
    with c3:
        ui.render_kpi_card("Distractors Filtered Out", f"{removed_count} chunks", icon="🧹", delta=f"{result.get('contamination_rate', 0):.1f}% cut", delta_color="pos")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    tab_kept, tab_removed, tab_all = st.tabs([
        f"🛡️ Kept Clean Evidence ({len(clean_evidence)})",
        f"❌ Filtered Out Contaminants ({removed_count})",
        f"📥 All Retrieved Chunks ({len(evidence)})",
    ])

    clean_texts = {rag.get_chunk_text(c).strip() for c in clean_evidence}

    with tab_kept:
        if not clean_evidence:
            st.info("No clean chunks preserved.")
        else:
            for i, chunk in enumerate(clean_evidence, 1):
                ui.render_chunk_card(
                    index=i,
                    text=rag.get_chunk_text(chunk),
                    is_kept=True,
                    page=rag.get_chunk_page(chunk),
                    chunk_id=rag.get_chunk_id(chunk, fallback=f"clean_{i}"),
                )

    with tab_removed:
        removed_chunks = [c for c in evidence if rag.get_chunk_text(c).strip() not in clean_texts]
        if not removed_chunks:
            st.success("No chunks were flagged as contaminants for this query!")
        else:
            for i, chunk in enumerate(removed_chunks, 1):
                ui.render_chunk_card(
                    index=i,
                    text=rag.get_chunk_text(chunk),
                    is_kept=False,
                    page=rag.get_chunk_page(chunk),
                    chunk_id=rag.get_chunk_id(chunk, fallback=f"noise_{i}"),
                )

    with tab_all:
        for i, chunk in enumerate(evidence, 1):
            is_kept = rag.get_chunk_text(chunk).strip() in clean_texts
            ui.render_chunk_card(
                index=i,
                text=rag.get_chunk_text(chunk),
                is_kept=is_kept,
                page=rag.get_chunk_page(chunk),
                chunk_id=rag.get_chunk_id(chunk, fallback=f"retrieved_{i}"),
            )


def render_analytics():
    records = rag.load_saved_evaluation_records()
    if not records:
        st.markdown(
            """
            <div class="rag-card" style="text-align: center; padding: 2.5rem;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
                <h3 style="color: #ffffff; margin-bottom: 0.5rem;">No Historical Records Found</h3>
                <p style="color: #94a3b8;">Run question evaluations to populate benchmark statistics and historical quality deltas.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    metrics = rag.calculate_cumulative_metrics()
    b = metrics.get("baseline", {})
    p = metrics.get("proposed", {})

    acc_diff = (p.get("accuracy", 0) or 0) - (b.get("accuracy", 0) or 0)
    prec_diff = (p.get("precision", 0) or 0) - (b.get("precision", 0) or 0)
    rec_diff = (p.get("recall", 0) or 0) - (b.get("recall", 0) or 0)
    f1_diff = (p.get("f1_score", 0) or 0) - (b.get("f1_score", 0) or 0)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.render_kpi_card("Accuracy Gain", pct(p.get("accuracy")), delta=signed_pct(acc_diff), icon="🎯", delta_color="pos")
    with c2:
        ui.render_kpi_card("Precision Gain", pct(p.get("precision")), delta=signed_pct(prec_diff), icon="✨", delta_color="pos")
    with c3:
        ui.render_kpi_card("Recall Retention", pct(p.get("recall")), delta=signed_pct(rec_diff), icon="📈", delta_color="pos")
    with c4:
        ui.render_kpi_card("F1 Quality Delta", pct(p.get("f1_score")), delta=signed_pct(f1_diff), icon="🏆", delta_color="pos")

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    st.markdown("### 📈 Cumulative Benchmark Performance")
    table = pd.DataFrame([
        {
            "Quality Dimension": "Semantic Accuracy",
            "Baseline (Unfiltered)": pct(b.get("accuracy")),
            "Proposed (Decontaminated)": pct(p.get("accuracy")),
            "Absolute Net Delta": signed_pct(acc_diff),
        },
        {
            "Quality Dimension": "Precision (Noise Rejection)",
            "Baseline (Unfiltered)": pct(b.get("precision")),
            "Proposed (Decontaminated)": pct(p.get("precision")),
            "Absolute Net Delta": signed_pct(prec_diff),
        },
        {
            "Quality Dimension": "Recall (Evidence Preservation)",
            "Baseline (Unfiltered)": pct(b.get("recall")),
            "Proposed (Decontaminated)": pct(p.get("recall")),
            "Absolute Net Delta": signed_pct(rec_diff),
        },
        {
            "Quality Dimension": "Harmonic F1-Score",
            "Baseline (Unfiltered)": pct(b.get("f1_score")),
            "Proposed (Decontaminated)": pct(p.get("f1_score")),
            "Absolute Net Delta": signed_pct(f1_diff),
        },
    ])
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    st.markdown("### 📜 Session Query Evaluation Logs")
    rows = []
    for record in records:
        rows.append({
            "Question": record.get("question", ""),
            "Baseline Score": safe_float(record.get("baseline_score")),
            "Proposed Score": safe_float(record.get("proposed_score")),
            "Retrieved": record.get("retrieved_chunks", ""),
            "Kept": record.get("kept_chunks", ""),
            "Removed": record.get("removed_chunks", ""),
            "Grounding Verified": "✅ True" if str(record.get("verified", "")).lower() == "true" else "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_architecture():
    st.markdown("### 🏗️ Contamination-Shielded Pipeline Architecture")

    st.graphviz_chart(
        """
        digraph {
            bgcolor="transparent";
            rankdir=LR;
            node [shape=box, style="filled,rounded", fontname="Plus Jakarta Sans, sans-serif", fontsize=10, fontcolor="#ffffff", fillcolor="#18181b", color="#3f3f46", penwidth=1.2];
            edge [fontname="Plus Jakarta Sans, sans-serif", fontsize=8, color="#52525b", fontcolor="#a1a1aa", penwidth=1.0];

            PDF [label="📄 Ingested PDF", fillcolor="#18181b", color="#71717a"];
            Chunking [label="✂️ Recursive Chunking\\n(512 chars, 50 ovlp)", fillcolor="#18181b", color="#52525b"];
            BGE [label="🧠 BGE Embeddings\\n(bge-small-en-v1.5)", fillcolor="#18181b", color="#52525b"];
            FAISS [label="⚡ FAISS IndexFlatIP\\n(Cosine Vectors)", fillcolor="#18181b", color="#52525b"];
            Retrieval [label="🔍 Top-K Retrieval\\nRaw Evidence Pool", fillcolor="#27272a", color="#71717a"];
            Filter [label="🛡️ Adaptive Filter\\nCosine + Lexical Guard", fillcolor="#27272a", color="#ffffff", penwidth=2.0];
            GenContam [label="⚠️ Baseline LLM\\n(Contaminated Raw Pool)", fillcolor="#141414", color="#3f3f46", style="dashed,filled,rounded"];
            GenClean [label="✨ Decontaminated LLM\\n(Shielded Grounding)", fillcolor="#27272a", color="#ffffff", penwidth=1.8];
            Domain [label="🩺 Specialized Formatter\\nMedical / Legal Adaptor", fillcolor="#18181b", color="#71717a"];
            Eval [label="📊 Benchmark Analytics\\nBLEU, ROUGE, Perplexity", fillcolor="#18181b", color="#71717a"];

            PDF -> Chunking -> BGE -> FAISS;
            FAISS -> Retrieval;
            Retrieval -> Filter;
            Retrieval -> GenContam;
            Filter -> GenClean [label="Kept evidence", color="#ffffff"];
            GenClean -> Domain;
            GenContam -> Eval [style="dashed", color="#52525b"];
            GenClean -> Eval [color="#ffffff"];
        }
        """
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="rag-card">
                <h4 style="color: #ffffff; margin-top: 0;">1. High-Precision Retrieval Core</h4>
                <p style="color: #a1a1aa; font-size: 0.88rem; line-height: 1.6;">
                    PyMuPDF extracts text streams with preservation of semantic structure. Chunks are embedded with <code>BAAI/bge-small-en-v1.5</code>, producing 384-dimensional dense vectors indexed via FAISS Inner Product for fast nearest-neighbor retrieval.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="rag-card">
                <h4 style="color: #ffffff; margin-top: 0;">2. Adaptive Contamination Shield</h4>
                <p style="color: #a1a1aa; font-size: 0.88rem; line-height: 1.6;">
                    Rather than imposing an arbitrary fixed chunk limit, the filter evaluates cosine distances against an adaptive percentile threshold. Chunks containing critical named entities and query terms are protected, while noisy outliers are discarded.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="rag-card">
                <h4 style="color: #ffffff; margin-top: 0;">3. NVIDIA NIM Llama 3.2 Generation</h4>
                <p style="color: #a1a1aa; font-size: 0.88rem; line-height: 1.6;">
                    Grounded generation runs on Llama 3.2 via NVIDIA NIM with strict temperature boundaries. With irrelevant context removed, the model focuses solely on verified facts, avoiding distraction and hallucination.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="rag-card">
                <h4 style="color: #ffffff; margin-top: 0;">4. Multi-Domain Post-Processing</h4>
                <p style="color: #a1a1aa; font-size: 0.88rem; line-height: 1.6;">
                    Domain modules for Medical and Legal dynamically structure answers without running a fragmented second RAG pipeline. Clinical reports receive diagnostic disclaimers, while legal cases receive holdings and ratio decidendi breakdowns.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    case_study = render_sidebar()

    # Current PDF status for static top hero header
    try:
        info = rag.get_current_pdf_info()
    except Exception:
        info = {"pdf_name": "No document selected", "chunks": 0, "vectors": 0}

    # Static Hero Header fixed at the top of the dashboard
    ui.render_hero_header(
        case_study=case_study,
        pdf_name=info.get("pdf_name"),
        chunks=info.get("chunks", 0),
    )

    pages = [
        "💬 Chat & QA",
        "📄 Document Overview",
        "🔎 Retrieval Analysis",
        "🧹 Contamination Shield",
        "📊 Benchmark Analytics",
        "🏗️ System Architecture",
    ]

    page = st.sidebar.radio("Navigation", pages)

    if page == "💬 Chat & QA":
        render_chat(case_study)
    elif page == "📄 Document Overview":
        render_overview(case_study)
    elif page == "🔎 Retrieval Analysis":
        render_retrieval_analysis()
    elif page == "🧹 Contamination Shield":
        render_contamination()
    elif page == "📊 Benchmark Analytics":
        render_analytics()
    else:
        render_architecture()


if __name__ == "__main__":
    main()
