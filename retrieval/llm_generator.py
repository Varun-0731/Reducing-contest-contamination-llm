"""
LLM Answer Generator using NVIDIA NIM (OpenAI-compatible client).
Replaces the legacy local model with the active NVIDIA NIM model.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "minimaxai/minimax-m3.1-flash")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
API_KEY = os.getenv("NVIDIA_API_KEY")

_client = None

def get_client():
    global _client
    if _client is None:
        key = API_KEY or os.getenv("NVIDIA_API_KEY")
        if not key:
            try:
                import streamlit as st
                if "NVIDIA_API_KEY" in st.secrets:
                    key = st.secrets["NVIDIA_API_KEY"]
            except Exception:
                pass
        if not key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Please add it to your .env file or environment."
            )
        _client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=key, timeout=35.0)
    return _client


def generate_llm_answer(question: str, context: str) -> str:
    """Generate an answer to a question using the retrieved context via NVIDIA NIM."""
    if not context or not context.strip():
        return "The answer is not available in the document."

    context = context.strip()
    if len(context) > 12000:
        context = context[:12000]

    prompt = f"""You are a document question-answering assistant.

You MUST answer the question using ONLY the information provided in the DOCUMENT CONTEXT.

Do NOT use your own knowledge.
Do NOT invent information.
Do NOT guess.
Do NOT use information that is not present in the context.

If the answer cannot be found in the DOCUMENT CONTEXT, respond exactly:
The answer is not available in the document.

When the answer is available:
- Format your response using clean, structured bullet points (- ) with key terms bolded for maximum readability.
- Give the complete answer.
- Preserve important technical details.
- Do not stop in the middle of a sentence.
- Do not repeat the same information.
- For definition questions, give the definition clearly.
- For method questions, list the relevant methods in bullet points.
- For summary questions, combine the important information into clear bullet points.
- Use simple and clear language.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        client = get_client()
        system_content = (
            "You are a factual, concise document question-answering assistant. "
            "Use ONLY the supplied context. Format your answer using structured bullet points (- ) with key terms bolded for high readability. "
            "Do NOT generate internal thoughts, reasoning steps, monologue, or thinking tags. "
            "Provide ONLY the direct, final, factual answer immediately."
        )
        response = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1536,
        )

        if response and response.choices:
            answer = response.choices[0].message.content
            if answer:
                answer = answer.strip()
                if answer.lower().startswith("answer:"):
                    answer = answer[7:].strip()
                return answer

        return "The answer is not available in the document."

    except Exception as e:
        print("Error during NVIDIA NIM answer generation:", str(e))
        return "The answer is not available in the document."
