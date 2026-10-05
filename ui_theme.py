"""
Monochromatic UI Theme and Modern Styling Components for PDF RAG Intelligence System.
Features: Minimalist grayscale palette, sleek obsidian/graphite surfaces, high-contrast typography, and silver accents.
"""

import streamlit as st


CUSTOM_CSS = """
<style>
/* Import Modern Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Monochromatic CSS Variables */
:root {
    --bg-main: #0a0a0a;
    --bg-surface: #141414;
    --bg-card: rgba(22, 22, 22, 0.75);
    --border-color: rgba(255, 255, 255, 0.12);
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-highlight: rgba(255, 255, 255, 0.35);
    --accent-white: #ffffff;
    --accent-gray: #a1a1aa;
    --accent-dark: #27272a;
    --text-primary: #ededed;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --glass-blur: blur(14px);
}

/* Base Body Styling - Strict Monochromatic */
.stApp {
    background-color: var(--bg-main) !important;
    background-image: 
        radial-gradient(circle at 10% 15%, rgba(255, 255, 255, 0.04) 0%, transparent 45%),
        radial-gradient(circle at 85% 85%, rgba(255, 255, 255, 0.02) 0%, transparent 50%),
        radial-gradient(circle at 50% 50%, rgba(255, 255, 255, 0.02) 0%, transparent 60%) !important;
    background-attachment: fixed !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

/* Scrollbar Monochromatic */
::-webkit-scrollbar {
    width: 5px;
    height: 5px;
}
::-webkit-scrollbar-track {
    background: #0a0a0a;
}
::-webkit-scrollbar-thumb {
    background: #27272a;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #52525b;
}

/* Sidebar Monochromatic */
[data-testid="stSidebar"] {
    background: #0e0e0e !important;
    border-right: 1px solid var(--border-color) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* Glassmorphic Obsidian Card Container */
.rag-card {
    background: var(--bg-card);
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}
.rag-card:hover {
    border-color: rgba(255, 255, 255, 0.25);
    box-shadow: 0 10px 36px rgba(0, 0, 0, 0.8);
}

/* Streamlit Header & Toolbar Styling */
[data-testid="stDecoration"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: #0a0a0a !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
    height: 2.75rem !important;
    z-index: 1000 !important;
}

[data-testid="stToolbar"] {
    right: 1.5rem !important;
    top: 0.25rem !important;
}

/* Proper top clearance so toolbar (Fork/GitHub) never overlaps the hero card */
[data-testid="stMainBlockContainer"],
.block-container {
    padding-top: 3.75rem !important;
    padding-bottom: 3.5rem !important;
}

/* Make Hero Header Static & Sticky at the Top across the entire scroll */
div[data-testid="element-container"]:has(.hero-container),
div[data-testid="stVerticalBlock"] > div:has(.hero-container) {
    position: sticky !important;
    top: 2.75rem !important;
    z-index: 995 !important;
    margin-bottom: 1.25rem !important;
    background: transparent !important;
}

.hero-container {
    position: sticky !important;
    top: 2.75rem !important;
    z-index: 995 !important;
    background: rgba(14, 14, 14, 0.97) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    border-radius: 12px !important;
    padding: 1.15rem 1.6rem !important;
    margin-bottom: 0 !important;
    overflow: hidden !important;
    box-shadow: 0 12px 36px rgba(0, 0, 0, 0.8) !important;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #27272a 0%, #ffffff 50%, #27272a 100%);
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin: 0;
    color: #ffffff;
    display: inline-block;
}
.hero-subtitle {
    font-size: 0.88rem;
    color: var(--text-secondary);
    margin-top: 0.35rem;
    margin-bottom: 0.75rem;
    line-height: 1.45;
}
.hero-badges-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    align-items: center;
}

/* Monochromatic Status Badges */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid transparent;
}
.badge-white {
    background: #ffffff;
    color: #0a0a0a !important;
    border-color: #ffffff;
}
.badge-outline {
    background: rgba(255, 255, 255, 0.05);
    color: #e4e4e7;
    border-color: rgba(255, 255, 255, 0.18);
}
.badge-muted {
    background: rgba(255, 255, 255, 0.03);
    color: #a1a1aa;
    border-color: rgba(255, 255, 255, 0.08);
}
.badge-contrast {
    background: #27272a;
    color: #ffffff;
    border-color: #3f3f46;
}

/* Pulsing White Indicator */
.pulsing-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #ffffff;
    box-shadow: 0 0 8px #ffffff;
    display: inline-block;
    animation: pulse-white 2.2s infinite;
}
@keyframes pulse-white {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.7); }
    70% { transform: scale(1.15); box-shadow: 0 0 0 6px rgba(255, 255, 255, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
}

/* KPI Card Monochromatic */
.kpi-card {
    background: rgba(20, 20, 20, 0.8);
    backdrop-filter: var(--glass-blur);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.1rem 1.25rem;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
}
.kpi-card:hover {
    border-color: rgba(255, 255, 255, 0.35);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.7);
}
.kpi-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
}
.kpi-icon-pill {
    font-size: 1.1rem;
    line-height: 1;
    filter: grayscale(100%);
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #ffffff;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}
.kpi-delta {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    margin-top: 0.35rem;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.76rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    background: rgba(255, 255, 255, 0.08);
    color: #e4e4e7;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

/* Comparison Answer Cards */
.comparison-card {
    border-radius: 12px;
    padding: 1.4rem;
    height: 100%;
    backdrop-filter: var(--glass-blur);
    border: 1px solid;
    position: relative;
}
.comparison-baseline {
    background: rgba(18, 18, 18, 0.75);
    border-color: rgba(255, 255, 255, 0.12);
}
.comparison-baseline:hover {
    border-color: rgba(255, 255, 255, 0.25);
}
.comparison-proposed {
    background: rgba(26, 26, 26, 0.85);
    border-color: rgba(255, 255, 255, 0.3);
}
.comparison-proposed:hover {
    border-color: rgba(255, 255, 255, 0.5);
}
.comparison-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.85rem;
    padding-bottom: 0.65rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}
.comparison-title {
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: -0.01em;
}

/* Evidence Chunk Cards */
.chunk-item {
    background: rgba(18, 18, 18, 0.7);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}
.chunk-item:hover {
    background: rgba(25, 25, 25, 0.9);
    border-color: rgba(255, 255, 255, 0.2);
}
.chunk-item-kept {
    border-left: 3px solid #ffffff;
}
.chunk-item-removed {
    border-left: 3px solid #3f3f46;
    opacity: 0.65;
}
.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
}
.chunk-body {
    font-size: 0.87rem;
    color: #d4d4d8;
    line-height: 1.6;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Monochromatic Buttons */
div.stButton > button {
    background: #ededed !important;
    color: #0a0a0a !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid #ffffff !important;
    padding: 0.45rem 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(255, 255, 255, 0.08) !important;
}
div.stButton > button:hover {
    background: #ffffff !important;
    color: #000000 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(255, 255, 255, 0.2) !important;
    border-color: #ffffff !important;
}

/* Secondary Button styling */
div.stButton > button[kind="secondary"] {
    background: #18181b !important;
    color: #e4e4e7 !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: none !important;
}
div.stButton > button[kind="secondary"]:hover {
    background: #27272a !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
    color: #ffffff !important;
}

/* Tabs Overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: #121212;
    padding: 0.35rem;
    border-radius: 10px;
    border: 1px solid var(--border-color);
}
.stTabs [data-baseweb="tab"] {
    height: 2.3rem;
    border-radius: 6px;
    font-weight: 600;
    color: var(--text-secondary);
    padding: 0 1rem;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #27272a !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* Chat Messages Monochromatic */
[data-testid="stChatMessage"] {
    background: rgba(18, 18, 18, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.85rem !important;
    backdrop-filter: var(--glass-blur);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(28, 28, 28, 0.6) !important;
    border-color: rgba(255, 255, 255, 0.16) !important;
}

/* Dataframe table styling */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border-color);
}

/* Expander custom look */
.streamlit-expanderHeader {
    background: #141414 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
}
</style>
"""


def inject_custom_css():
    """Injects high-end monochromatic CSS styling into the Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero_header(case_study: str, pdf_name: str = None, chunks: int = 0):
    """Renders a polished monochromatic header banner with live system status."""
    is_loaded = bool(pdf_name and pdf_name != "No document selected" and pdf_name != "No PDF loaded")

    status_html = f"""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
            <div>
                <div class="badge-pill badge-outline" style="margin-bottom: 0.6rem;">
                    <span>🛡️</span> RAG Contamination Defense
                </div>
                <h1 class="hero-title">PDF RAG Intelligence System</h1>
                <p class="hero-subtitle">
                    Mitigating context contamination, noise injection, and hallucinations with adaptive evidence filtering.
                </p>
                <div class="hero-badges-row">
                    <span class="badge-pill badge-white">
                        Domain: <strong>{case_study}</strong>
                    </span>
                    <span class="badge-pill badge-contrast">
                        Model: <strong>NVIDIA NIM Llama 3.2</strong>
                    </span>
                    <span class="badge-pill badge-outline">
                        Embeddings: <strong>BAAI/bge-small-en-v1.5</strong>
                    </span>
                </div>
            </div>
            <div style="text-align: right; background: #141414; padding: 0.85rem 1.25rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700; margin-bottom: 0.25rem;">
                    Active Document
                </div>
                <div style="font-weight: 700; font-size: 0.95rem; color: #ffffff; max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
                    {pdf_name if is_loaded else 'No document loaded'}
                </div>
                <div style="display: flex; align-items: center; justify-content: flex-end; gap: 0.5rem; margin-top: 0.4rem;">
                    <span class="{"pulsing-dot" if is_loaded else ""}" style="background: {"#ffffff" if is_loaded else "#52525b"}; width: 7px; height: 7px; border-radius: 50%; display: inline-block;"></span>
                    <span style="font-size: 0.75rem; color: {"#e4e4e7" if is_loaded else "#71717a"}; font-weight: 600;">
                        {f"{chunks} FAISS Chunks Indexed" if is_loaded else "Awaiting PDF Import"}
                    </span>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)


def render_kpi_card(label: str, value: str, delta: str = None, icon: str = "📊", delta_color: str = "normal"):
    """Renders a sleek monochromatic KPI card."""
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta">{delta}</div>'

    html = f"""
    <div class="kpi-card">
        <div class="kpi-top">
            <span class="kpi-label">{label}</span>
            <span class="kpi-icon-pill">{icon}</span>
        </div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_comparison_view(baseline_answer: str, proposed_answer: str, reference_answer: str = None):
    """Renders monochromatic side-by-side comparison between contaminated and clean answers."""
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown(
            f"""
            <div class="comparison-card comparison-baseline">
                <div class="comparison-header">
                    <span class="comparison-title" style="color: #a1a1aa;">
                        Baseline (Unfiltered Context)
                    </span>
                    <span class="badge-pill badge-muted">Raw Retrieval</span>
                </div>
                <div style="font-size: 0.88rem; line-height: 1.6; color: #d4d4d8; white-space: pre-wrap;">{baseline_answer or "No baseline answer generated."}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="comparison-card comparison-proposed">
                <div class="comparison-header">
                    <span class="comparison-title" style="color: #ffffff;">
                        Proposed (Decontaminated)
                    </span>
                    <span class="badge-pill badge-white">Clean Context</span>
                </div>
                <div style="font-size: 0.88rem; line-height: 1.6; color: #ffffff; white-space: pre-wrap;">{proposed_answer or "No proposed answer generated."}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if reference_answer:
        st.markdown(
            f"""
            <div style="margin-top: 1rem; background: #141414; border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 10px; padding: 1rem 1.25rem;">
                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.35rem;">
                    Ground Truth Reference Answer
                </div>
                <div style="font-size: 0.88rem; color: #a1a1aa; line-height: 1.5;">{reference_answer}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_chunk_card(index: int, text: str, is_kept: bool = True, page: int = None, chunk_id: str = None):
    """Renders a styled monochromatic evidence chunk card."""
    card_type = "chunk-item-kept" if is_kept else "chunk-item-removed"
    badge_type = "badge-white" if is_kept else "badge-muted"
    badge_label = "Kept Evidence" if is_kept else "Filtered (Contaminant)"
    page_info = f" · Page {page}" if page is not None else ""
    cid_info = f"ID: {chunk_id}" if chunk_id else f"Evidence Chunk #{index}"
    words = len(text.split())

    html = f"""
    <div class="chunk-item {card_type}">
        <div class="chunk-header">
            <div>
                <strong style="color: #ffffff;">{cid_info}</strong>
                <span style="color: #71717a; font-size: 0.75rem; margin-left: 0.4rem;">{words} words{page_info}</span>
            </div>
            <span class="badge-pill {badge_type}">{badge_label}</span>
        </div>
        <div class="chunk-body">{text}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
